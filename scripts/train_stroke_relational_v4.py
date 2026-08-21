"""Train the co-Stroke relational stroke model v4 on Quick, Draw!.

V3 flattened a whole drawing into one causal stream of point actions. V4 uses
one fixed-width embedding per stroke, lets a bidirectional Transformer reason
over the visible set of strokes, then factorizes the next-stroke distribution:

    visible strokes -> context encoder -> next start -> next relative shape

The factorization is designed for interactive completion. The context may be a
normal prefix or a random subset of the source drawing, so the model cannot rely
only on the canonical Quick, Draw! construction order.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


QUICKDRAW_SCALE = 255.0
AUTHOR_DATASET = 0
AUTHOR_HUMAN = 1
AUTHOR_AI = 2


@dataclass
class TrainConfig:
    data: str
    out_dir: str = "runs/stroke-relational-v4-cat"
    max_drawings: int = 70000
    recognized_only: bool = True
    val_fraction: float = 0.04
    max_context_strokes: int = 24
    points_per_stroke: int = 16
    batch_size: int = 192
    epochs: int = 16
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.15
    mixtures: int = 10
    prefix_probability: float = 0.5
    lr: float = 3e-4
    weight_decay: float = 0.01
    stop_loss_weight: float = 0.25
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


class QuickDrawStrokeDrawingDataset:
    """Load recognized Quick, Draw! records as resampled absolute strokes."""

    def __init__(self, ndjson_path: Path, config: TrainConfig) -> None:
        self.drawings: list[list[torch.Tensor]] = []
        self.skipped_too_small = 0

        for drawing in load_quickdraw_drawings(ndjson_path, config):
            strokes = drawing_to_v4_strokes(drawing, config.points_per_stroke)
            if len(strokes) < 2:
                self.skipped_too_small += 1
                continue
            self.drawings.append(strokes)

        if not self.drawings:
            raise ValueError(f"No usable drawings found in {ndjson_path}")

    def __len__(self) -> int:
        return len(self.drawings)


class StrokeCompletionDataset(Dataset):
    """Create prefix and arbitrary-subset completion tasks from drawings."""

    def __init__(
        self,
        drawings: Sequence[list[torch.Tensor]],
        indices: Sequence[int],
        config: TrainConfig,
        *,
        training: bool,
    ) -> None:
        self.drawings = drawings
        self.indices = list(indices)
        self.config = config
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        drawing_index = self.indices[item]
        source = self.drawings[drawing_index]
        rng: random.Random | Any
        if self.training:
            rng = random
        else:
            rng = random.Random(self.config.seed * 1_000_003 + drawing_index)

        strokes = augment_drawing(source, rng) if self.training else [stroke.clone() for stroke in source]
        return make_completion_example(strokes, self.config, rng)


class StrokeRelationalTransformer(nn.Module):
    """Encode complete strokes and predict the next start and relative shape."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        shape_dimensions = config.points_per_stroke * 2
        target_dimensions = (config.points_per_stroke - 1) * 2

        self.shape_encoder = nn.Sequential(
            nn.Linear(shape_dimensions, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(9, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.order_projection = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.author_embedding = nn.Embedding(3, config.d_model)
        self.context_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.context_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relational_transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.context_norm = nn.LayerNorm(config.d_model)

        self.start_head = nn.Linear(config.d_model, config.mixtures * 6)
        self.stop_head = nn.Linear(config.d_model, 1)

        shape_hidden = config.d_model * 2
        self.shape_conditioner = nn.Sequential(
            nn.Linear(config.d_model + 2, shape_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(shape_hidden, config.d_model),
            nn.GELU(),
        )
        self.shape_pi_head = nn.Linear(config.d_model, config.mixtures)
        self.shape_mean_head = nn.Linear(config.d_model, config.mixtures * target_dimensions)
        self.shape_log_sigma_head = nn.Linear(config.d_model, config.mixtures * target_dimensions)

    def encode_context(
        self,
        context_points: torch.Tensor,
        context_authors: torch.Tensor,
        context_order: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, stroke_count, point_count, _ = context_points.shape
        if point_count != self.config.points_per_stroke:
            raise ValueError(
                f"Expected {self.config.points_per_stroke} points per stroke, got {point_count}"
            )
        if stroke_count > self.config.max_context_strokes:
            raise ValueError(
                f"Context has {stroke_count} strokes; maximum is {self.config.max_context_strokes}"
            )

        if stroke_count:
            start = context_points[:, :, :1]
            relative = context_points - start
            shape = self.shape_encoder(relative.reshape(batch_size, stroke_count, -1))
            spatial = self.spatial_projection(stroke_spatial_features(context_points))
            order = self.order_projection(context_order.unsqueeze(-1))
            hidden = shape + spatial + order + self.author_embedding(context_authors)
        else:
            hidden = context_points.new_zeros((batch_size, 0, self.config.d_model))

        context_token = self.context_token.expand(batch_size, -1, -1)
        hidden = torch.cat((context_token, hidden), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=context_mask.device),
                ~context_mask,
            ),
            dim=1,
        )
        encoded = self.relational_transformer(hidden, src_key_padding_mask=padding_mask)
        return self.context_norm(encoded[:, 0])

    def predict_start(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.start_head(context)
        pi_logits, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat = raw.chunk(6, dim=-1)
        return {
            "pi_logits": pi_logits,
            "mu_x": torch.sigmoid(mu_x),
            "mu_y": torch.sigmoid(mu_y),
            "log_sigma_x": log_sigma_x.clamp(-7.0, 1.0),
            "log_sigma_y": log_sigma_y.clamp(-7.0, 1.0),
            "rho": torch.tanh(rho_hat).clamp(-0.999, 0.999),
        }

    def predict_shape(self, context: torch.Tensor, start: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shape_conditioner(torch.cat((context, start), dim=-1))
        batch_size = context.shape[0]
        point_count = self.config.points_per_stroke - 1
        return {
            "pi_logits": self.shape_pi_head(hidden),
            "mean": self.shape_mean_head(hidden).view(
                batch_size, self.config.mixtures, point_count, 2
            ),
            "log_sigma": self.shape_log_sigma_head(hidden).view(
                batch_size, self.config.mixtures, point_count, 2
            ).clamp(-7.0, 1.0),
        }

    def forward(
        self,
        context_points: torch.Tensor,
        context_authors: torch.Tensor,
        context_order: torch.Tensor,
        context_mask: torch.Tensor,
        target_start: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        context = self.encode_context(
            context_points,
            context_authors,
            context_order,
            context_mask,
        )
        return self.predict_start(context), self.predict_shape(context, target_start), self.stop_head(context).squeeze(-1)


def load_quickdraw_drawings(ndjson_path: Path, config: TrainConfig) -> Iterable[Sequence[Any]]:
    loaded = 0
    with ndjson_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if loaded >= config.max_drawings:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if config.recognized_only and not bool(record.get("recognized", False)):
                continue
            drawing = record.get("drawing")
            if not drawing:
                continue
            loaded += 1
            yield drawing


def drawing_to_v4_strokes(drawing: Sequence[Any], points_per_stroke: int) -> list[torch.Tensor]:
    strokes: list[torch.Tensor] = []
    for raw_stroke in drawing:
        if not isinstance(raw_stroke, Sequence) or len(raw_stroke) < 2:
            continue
        xs, ys = raw_stroke[0], raw_stroke[1]
        if len(xs) < 2 or len(xs) != len(ys):
            continue
        points = [
            (float(x) / QUICKDRAW_SCALE, float(y) / QUICKDRAW_SCALE)
            for x, y in zip(xs, ys)
        ]
        strokes.append(resample_stroke(points, points_per_stroke))
    return strokes


def co_stroke_to_v4_tensor(stroke: dict[str, Any], points_per_stroke: int) -> torch.Tensor:
    points = [
        (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
        for point in stroke.get("points", [])
    ]
    return resample_stroke(points, points_per_stroke)


def resample_stroke(points: Sequence[Sequence[float]], point_count: int) -> torch.Tensor:
    """Return exactly ``point_count`` samples at equal arc-length intervals."""
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    if len(points) < 2:
        raise ValueError("A stroke needs at least two points")

    clean = [
        (clamp(float(point[0]), 0.0, 1.0), clamp(float(point[1]), 0.0, 1.0))
        for point in points
    ]
    cumulative = [0.0]
    for previous, point in zip(clean, clean[1:]):
        cumulative.append(cumulative[-1] + math.hypot(point[0] - previous[0], point[1] - previous[1]))
    total = cumulative[-1]
    if total <= 1e-8:
        return torch.tensor([clean[0]] * point_count, dtype=torch.float32)

    sampled: list[tuple[float, float]] = []
    segment = 0
    for sample_index in range(point_count):
        target = total * sample_index / (point_count - 1)
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target:
            segment += 1
        start_distance = cumulative[segment]
        end_distance = cumulative[segment + 1]
        ratio = 0.0 if end_distance <= start_distance else (target - start_distance) / (end_distance - start_distance)
        start = clean[segment]
        end = clean[segment + 1]
        sampled.append((
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        ))
    return torch.tensor(sampled, dtype=torch.float32)


def augment_drawing(strokes: Sequence[torch.Tensor], rng: random.Random | Any) -> list[torch.Tensor]:
    angle = rng.uniform(-0.18, 0.18)
    scale = rng.uniform(0.90, 1.08)
    translation = torch.tensor([rng.uniform(-0.035, 0.035), rng.uniform(-0.035, 0.035)])
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    augmented = []
    for stroke in strokes:
        points = ((stroke - 0.5) @ rotation.T) * scale + 0.5 + translation
        augmented.append(points.clamp(0.01, 0.99))
    return augmented


def make_completion_example(
    strokes: Sequence[torch.Tensor],
    config: TrainConfig,
    rng: random.Random | Any,
) -> dict[str, torch.Tensor]:
    if len(strokes) < 2:
        raise ValueError("Completion examples need at least two strokes")

    target_index = rng.randrange(1, len(strokes))
    if rng.random() < config.prefix_probability:
        context_indices = list(range(target_index))
        if len(context_indices) > config.max_context_strokes:
            context_indices = evenly_spaced_indices(context_indices, config.max_context_strokes)
    else:
        available = [index for index in range(len(strokes)) if index != target_index]
        count = rng.randint(1, min(len(available), config.max_context_strokes))
        context_indices = sorted(rng.sample(available, count))

    context = torch.stack([strokes[index] for index in context_indices])
    target = strokes[target_index]
    denominator = max(len(strokes) - 1, 1)
    order = torch.tensor([index / denominator for index in context_indices], dtype=torch.float32)

    # Source labels are deliberately randomized during synthetic training. The
    # geometry is the supervision; this prevents an untrained author embedding
    # from shifting browser input far away from the dataset representation.
    if rng.random() < 0.75:
        authors = torch.full((len(context_indices),), AUTHOR_DATASET, dtype=torch.long)
    else:
        authors = torch.tensor(
            [AUTHOR_HUMAN if rng.random() < 0.8 else AUTHOR_AI for _ in context_indices],
            dtype=torch.long,
        )

    return {
        "context_points": context,
        "context_authors": authors,
        "context_order": order,
        "target_start": target[0],
        "target_shape": target[1:] - target[0],
        "target_stop": torch.tensor(float(target_index == len(strokes) - 1)),
    }


def collate_batch(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch_size = len(examples)
    max_context = max(example["context_points"].shape[0] for example in examples)
    point_count = examples[0]["context_points"].shape[1]
    context_points = torch.zeros((batch_size, max_context, point_count, 2), dtype=torch.float32)
    context_authors = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_order = torch.zeros((batch_size, max_context), dtype=torch.float32)
    context_mask = torch.zeros((batch_size, max_context), dtype=torch.bool)

    for index, example in enumerate(examples):
        count = example["context_points"].shape[0]
        context_points[index, :count] = example["context_points"]
        context_authors[index, :count] = example["context_authors"]
        context_order[index, :count] = example["context_order"]
        context_mask[index, :count] = True

    return {
        "context_points": context_points,
        "context_authors": context_authors,
        "context_order": context_order,
        "context_mask": context_mask,
        "target_start": torch.stack([example["target_start"] for example in examples]),
        "target_shape": torch.stack([example["target_shape"] for example in examples]),
        "target_stop": torch.stack([example["target_stop"] for example in examples]),
    }


def stroke_spatial_features(points: torch.Tensor) -> torch.Tensor:
    start = points[:, :, 0]
    end = points[:, :, -1]
    minimum = points.amin(dim=2)
    maximum = points.amax(dim=2)
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    length = torch.linalg.vector_norm(points[:, :, 1:] - points[:, :, :-1], dim=-1).sum(dim=-1, keepdim=True)
    return torch.cat((start, end, center, size, length), dim=-1)


def sequence_loss(
    start_distribution: dict[str, torch.Tensor],
    shape_distribution: dict[str, torch.Tensor],
    stop_logits: torch.Tensor,
    target_start: torch.Tensor,
    target_shape: torch.Tensor,
    stop_target: torch.Tensor,
    stop_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=target_start.device.type, enabled=False):
        start_loss = bivariate_gmm_nll(
            {key: value.float() for key, value in start_distribution.items()},
            target_start.float(),
        )
        shape_loss = diagonal_shape_gmm_nll(
            {key: value.float() for key, value in shape_distribution.items()},
            target_shape.float(),
        )
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits.float(), stop_target.float())
        total = start_loss + shape_loss + stop_weight * stop_loss
    return total, start_loss, shape_loss, stop_loss


def bivariate_gmm_nll(distribution: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    sigma_x = distribution["log_sigma_x"].exp()
    sigma_y = distribution["log_sigma_y"].exp()
    rho = distribution["rho"]
    target_x = target[:, 0:1]
    target_y = target[:, 1:2]
    norm_x = (target_x - distribution["mu_x"]) / sigma_x
    norm_y = (target_y - distribution["mu_y"]) / sigma_y
    one_minus_rho_sq = (1.0 - rho.square()).clamp_min(1e-5)
    exponent = (norm_x.square() + norm_y.square() - 2.0 * rho * norm_x * norm_y) / one_minus_rho_sq
    component_log_prob = (
        -math.log(2.0 * math.pi)
        - distribution["log_sigma_x"]
        - distribution["log_sigma_y"]
        - 0.5 * one_minus_rho_sq.log()
        - 0.5 * exponent
    )
    return -torch.logsumexp(F.log_softmax(distribution["pi_logits"], dim=-1) + component_log_prob, dim=-1).mean()


def diagonal_shape_gmm_nll(distribution: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1)
    sigma = distribution["log_sigma"].exp()
    normalized = (target - distribution["mean"]) / sigma
    coordinate_log_prob = -0.5 * normalized.square() - distribution["log_sigma"] - 0.5 * math.log(2.0 * math.pi)
    component_log_prob = coordinate_log_prob.sum(dim=(-1, -2))
    mixture_log_prob = torch.logsumexp(
        F.log_softmax(distribution["pi_logits"], dim=-1) + component_log_prob,
        dim=-1,
    )
    dimensions = max(target.shape[-2] * target.shape[-1], 1)
    return -(mixture_log_prob / dimensions).mean()


@torch.no_grad()
def sample_v4_stroke(
    model: StrokeRelationalTransformer,
    context_points: torch.Tensor,
    context_authors: torch.Tensor,
    context_order: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    temperature: float = 0.55,
) -> tuple[torch.Tensor, float]:
    model.eval()
    context = model.encode_context(context_points, context_authors, context_order, context_mask)
    start_distribution = model.predict_start(context)
    start, start_log_probability = sample_bivariate_gmm(start_distribution, temperature)
    shape_distribution = model.predict_shape(context, start)
    relative_shape, shape_log_probability = sample_shape_gmm(shape_distribution, temperature)
    origin = torch.zeros((start.shape[0], 1, 2), dtype=start.dtype, device=start.device)
    points = start.unsqueeze(1) + torch.cat((origin, relative_shape), dim=1)
    confidence = (start_log_probability + shape_log_probability).mean()
    return points.clamp(0.01, 0.99), float(confidence.item())


def sample_bivariate_gmm(
    distribution: dict[str, torch.Tensor],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = max(float(temperature), 1e-3)
    logits = distribution["pi_logits"] / temperature
    component = torch.distributions.Categorical(logits=logits).sample()
    gather = component.unsqueeze(-1)
    mu_x = distribution["mu_x"].gather(1, gather).squeeze(1)
    mu_y = distribution["mu_y"].gather(1, gather).squeeze(1)
    sigma_x = distribution["log_sigma_x"].gather(1, gather).squeeze(1).exp() * math.sqrt(temperature)
    sigma_y = distribution["log_sigma_y"].gather(1, gather).squeeze(1).exp() * math.sqrt(temperature)
    rho = distribution["rho"].gather(1, gather).squeeze(1)
    epsilon_x = torch.randn_like(mu_x)
    epsilon_y = torch.randn_like(mu_y)
    x = mu_x + sigma_x * epsilon_x
    y = mu_y + sigma_y * (rho * epsilon_x + torch.sqrt((1.0 - rho.square()).clamp_min(1e-5)) * epsilon_y)
    probability = F.log_softmax(logits, dim=-1).gather(1, gather).squeeze(1)
    return torch.stack((x, y), dim=-1).clamp(0.01, 0.99), probability


def sample_shape_gmm(
    distribution: dict[str, torch.Tensor],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = max(float(temperature), 1e-3)
    logits = distribution["pi_logits"] / temperature
    component = torch.distributions.Categorical(logits=logits).sample()
    batch_indices = torch.arange(component.shape[0], device=component.device)
    mean = distribution["mean"][batch_indices, component]
    # The mixture component already supplies most of the useful multimodality.
    # Full independent noise at every control point makes a decoded stroke
    # jagged, so keep only a small residual around the learned component mean.
    sigma = (
        distribution["log_sigma"][batch_indices, component].exp()
        * math.sqrt(temperature)
        * 0.20
    )
    sample = mean + sigma * torch.randn_like(mean)
    probability = F.log_softmax(logits, dim=-1)[batch_indices, component]
    return sample, probability


def evenly_spaced_indices(indices: Sequence[int], count: int) -> list[int]:
    if count >= len(indices):
        return list(indices)
    if count <= 1:
        return [indices[-1]]
    selected = []
    for position in range(count):
        index = round(position * (len(indices) - 1) / (count - 1))
        value = indices[index]
        if value not in selected:
            selected.append(value)
    return selected


def train_epoch(
    model: StrokeRelationalTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    stop_weight: float,
) -> tuple[float, float, float, float]:
    model.train()
    totals = torch.zeros(4, dtype=torch.float64)
    examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(
                batch["context_points"],
                batch["context_authors"],
                batch["context_order"],
                batch["context_mask"],
                batch["target_start"],
            )
            losses = sequence_loss(
                *outputs,
                batch["target_start"],
                batch["target_shape"],
                batch["target_stop"],
                stop_weight,
            )
        scaler.scale(losses[0]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["target_start"].shape[0]
        totals += torch.tensor([float(loss.item()) for loss in losses]) * batch_size
        examples += batch_size
    return tuple(float(value / max(examples, 1)) for value in totals)  # type: ignore[return-value]


@torch.no_grad()
def evaluate(
    model: StrokeRelationalTransformer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    stop_weight: float,
) -> tuple[float, float, float, float]:
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64)
    examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(
                batch["context_points"],
                batch["context_authors"],
                batch["context_order"],
                batch["context_mask"],
                batch["target_start"],
            )
            losses = sequence_loss(
                *outputs,
                batch["target_start"],
                batch["target_shape"],
                batch["target_stop"],
                stop_weight,
            )
        batch_size = batch["target_start"].shape[0]
        totals += torch.tensor([float(loss.item()) for loss in losses]) * batch_size
        examples += batch_size
    return tuple(float(value / max(examples, 1)) for value in totals)  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    model: StrokeRelationalTransformer,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "stroke-relational-v4",
        "model": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def make_sample_json(points: torch.Tensor) -> dict[str, Any]:
    point_list = [
        {"x": float(point[0]), "y": float(point[1]), "t": index * 40, "p": 0.5}
        for index, point in enumerate(points.cpu())
    ]
    return {
        "version": "0.1.0",
        "id": "sample_stroke_relational_v4",
        "title": "stroke-relational-v4-sample",
        "category": "cat",
        "canvas": {"width": 960, "height": 640, "coordinateSystem": "normalized", "background": "#ffffff"},
        "timeline": {"unit": "stroke", "currentStep": 1, "branchPolicy": "truncate-future-on-edit"},
        "strokes": [{
            "id": "ai_v4_sample_001",
            "author": {"type": "ai", "model": "stroke-relational-v4"},
            "tool": "pen",
            "style": {"color": "#ff44aa", "width": 4, "opacity": 1, "lineCap": "round", "lineJoin": "round"},
            "timing": {"startMs": 0, "durationMs": len(point_list) * 40},
            "points": point_list,
            "metadata": {"sample": True},
        }],
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the co-Stroke relational stroke model v4.")
    parser.add_argument("--data", required=True, help="Path to a Quick, Draw! simplified .ndjson file.")
    parser.add_argument("--out-dir", default="runs/stroke-relational-v4-cat")
    parser.add_argument("--max-drawings", type=int, default=70000)
    parser.add_argument("--include-unrecognized", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.04)
    parser.add_argument("--max-context-strokes", type=int, default=24)
    parser.add_argument("--points-per-stroke", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--mixtures", type=int, default=10)
    parser.add_argument("--prefix-probability", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--stop-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        data=args.data,
        out_dir=args.out_dir,
        max_drawings=args.max_drawings,
        recognized_only=not args.include_unrecognized,
        val_fraction=args.val_fraction,
        max_context_strokes=args.max_context_strokes,
        points_per_stroke=args.points_per_stroke,
        batch_size=args.batch_size,
        epochs=args.epochs,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        mixtures=args.mixtures,
        prefix_probability=args.prefix_probability,
        lr=args.lr,
        weight_decay=args.weight_decay,
        stop_loss_weight=args.stop_loss_weight,
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
    )


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(config.device)
    amp_enabled = config.amp and device.type == "cuda"
    source = QuickDrawStrokeDrawingDataset(Path(config.data), config)
    indices = list(range(len(source)))
    random.Random(config.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * config.val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_dataset = StrokeCompletionDataset(source.drawings, train_indices, config, training=True)
    val_dataset = StrokeCompletionDataset(source.drawings, val_indices, config, training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = StrokeRelationalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.lr * 0.1,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded {len(source)} drawings from {config.data}", flush=True)
    print(f"Skipped {source.skipped_too_small} drawings with fewer than two usable strokes", flush=True)
    print(f"Train/validation drawings: {len(train_dataset)}/{len(val_dataset)}", flush=True)
    print(f"Model parameters: {parameter_count:,}", flush=True)
    print(f"Training on {device} (amp={amp_enabled}) -> {out_dir}", flush=True)

    best_val = math.inf
    for epoch in range(1, config.epochs + 1):
        train_losses = train_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled, config.stop_loss_weight
        )
        val_losses = evaluate(model, val_loader, device, amp_enabled, config.stop_loss_weight)
        scheduler.step()
        print(
            f"epoch {epoch:03d} | "
            f"train {train_losses[0]:.4f} (start {train_losses[1]:.4f}, shape {train_losses[2]:.4f}, stop {train_losses[3]:.4f}) | "
            f"val {val_losses[0]:.4f} (start {val_losses[1]:.4f}, shape {val_losses[2]:.4f}, stop {val_losses[3]:.4f}) | "
            f"lr {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        save_checkpoint(out_dir / "latest.pt", model, config, epoch, val_losses[0])
        if val_losses[0] < best_val:
            best_val = val_losses[0]
            save_checkpoint(out_dir / "checkpoint.pt", model, config, epoch, val_losses[0])

    example = val_dataset[0]
    batch = move_batch(collate_batch([example]), device)
    sample, _ = sample_v4_stroke(
        model,
        batch["context_points"],
        batch["context_authors"],
        batch["context_order"],
        batch["context_mask"],
    )
    (out_dir / "sample.json").write_text(
        json.dumps(make_sample_json(sample[0]), indent=2),
        encoding="utf-8",
    )
    print(f"Saved v4 checkpoint and sample to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
