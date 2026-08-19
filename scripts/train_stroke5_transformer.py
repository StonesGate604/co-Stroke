"""Train a continuous stroke-5 autoregressive Transformer on Quick, Draw!.

Unlike the primitive-token baselines, this model keeps the complete drawing
action stream.  Every source point contributes an ``(dx, dy, pen)`` action,
including the pen-up move from one stroke endpoint to the next stroke start.
Coordinates stay continuous and the model predicts their joint distribution
with a bivariate Gaussian mixture density head.

The representation follows the useful part of SketchRNN's stroke-5 format:

    pen = 0  draw/continue after arriving at this point
    pen = 1  lift the pen after arriving at this point
    pen = 2  end of drawing

The pen state on the previous action decides whether the movement to the next
point is visible.  This detail is what preserves independent stroke anchors.
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
from torch.utils.data import DataLoader, Dataset, random_split


PEN_CONTINUE = 0
PEN_LIFT = 1
PEN_END = 2
QUICKDRAW_SCALE = 255.0
BOS_ACTION = (0.0, 0.0, PEN_LIFT)


@dataclass
class TrainConfig:
    data: str
    out_dir: str = "runs/stroke5-transformer-v3-cat"
    max_drawings: int = 70000
    recognized_only: bool = True
    val_fraction: float = 0.04
    max_len: int = 192
    batch_size: int = 64
    epochs: int = 12
    d_model: int = 384
    layers: int = 6
    heads: int = 6
    dropout: float = 0.2
    mixtures: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    coordinate_scale: float = 1.0
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


class QuickDrawStroke5Dataset(Dataset):
    """In-memory QuickDraw actions with a single dataset-wide coordinate scale."""

    def __init__(self, ndjson_path: Path, config: TrainConfig) -> None:
        self.config = config
        self.samples: list[torch.Tensor] = []
        coordinate_values: list[torch.Tensor] = []
        skipped_too_long = 0

        for actions in load_quickdraw_actions(ndjson_path, config):
            if len(actions) > config.max_len:
                skipped_too_long += 1
                continue
            tensor = torch.tensor(actions, dtype=torch.float32)
            self.samples.append(tensor)
            coordinate_values.append(tensor[:-1, :2].reshape(-1))

        if not self.samples:
            raise ValueError(f"No usable drawings found in {ndjson_path}")

        all_coordinates = torch.cat(coordinate_values)
        scale = float(all_coordinates.std(unbiased=False).item())
        if not math.isfinite(scale) or scale <= 1e-6:
            raise ValueError(f"Invalid coordinate scale computed from {ndjson_path}: {scale}")
        config.coordinate_scale = scale
        self.coordinate_scale = scale
        self.skipped_too_long = skipped_too_long

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target = self.samples[index]
        target_xy = target[:, :2] / self.coordinate_scale
        target_pen = target[:, 2].long()

        input_xy = torch.cat((torch.zeros(1, 2), target_xy[:-1]), dim=0)
        input_pen = torch.cat(
            (torch.tensor([PEN_LIFT], dtype=torch.long), target_pen[:-1]),
            dim=0,
        )
        return {
            "input_xy": input_xy,
            "input_pen": input_pen,
            "target_xy": target_xy,
            "target_pen": target_pen,
        }


class Stroke5Transformer(nn.Module):
    """Causal Transformer with a full-covariance bivariate MDN output."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.coordinate_projection = nn.Linear(2, config.d_model, bias=False)
        self.pen_embedding = nn.Embedding(3, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.d_model)
        # pi, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat
        self.mixture_head = nn.Linear(config.d_model, config.mixtures * 6)
        self.pen_head = nn.Linear(config.d_model, 3)

    def forward(
        self,
        xy: torch.Tensor,
        pen: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch_size, seq_len, _ = xy.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.config.max_len}")

        positions = torch.arange(seq_len, device=xy.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (
            self.coordinate_projection(xy)
            + self.pen_embedding(pen)
            + self.position_embedding(positions)
        )
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=xy.device),
            diagonal=1,
        )
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.norm(hidden)

        raw = self.mixture_head(hidden)
        pi_logits, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat = raw.chunk(6, dim=-1)
        mixture = {
            "pi_logits": pi_logits,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "log_sigma_x": log_sigma_x.clamp(-7.0, 3.0),
            "log_sigma_y": log_sigma_y.clamp(-7.0, 3.0),
            "rho": torch.tanh(rho_hat).clamp(-0.999, 0.999),
        }
        return mixture, self.pen_head(hidden)


def load_quickdraw_actions(ndjson_path: Path, config: TrainConfig) -> Iterable[list[tuple[float, float, int]]]:
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
            actions = drawing_to_stroke5(drawing)
            if len(actions) < 3:
                continue
            loaded += 1
            yield actions


def drawing_to_stroke5(drawing: Sequence[Any]) -> list[tuple[float, float, int]]:
    """Convert raw QuickDraw strokes without discarding pen-up movements."""
    actions: list[tuple[float, float, int]] = []
    previous_x = 0.5
    previous_y = 0.5

    for stroke in drawing:
        if len(stroke) != 2:
            continue
        xs, ys = stroke
        if len(xs) != len(ys) or len(xs) < 2:
            continue

        for point_index, (raw_x, raw_y) in enumerate(zip(xs, ys)):
            x = clamp(float(raw_x) / QUICKDRAW_SCALE, 0.0, 1.0)
            y = clamp(float(raw_y) / QUICKDRAW_SCALE, 0.0, 1.0)
            pen = PEN_LIFT if point_index == len(xs) - 1 else PEN_CONTINUE
            actions.append((x - previous_x, y - previous_y, pen))
            previous_x = x
            previous_y = y

    actions.append((0.0, 0.0, PEN_END))
    return actions


def co_strokes_to_stroke5(strokes: Sequence[dict[str, Any]]) -> list[tuple[float, float, int]]:
    """Convert visible co-Stroke polylines into the same global action stream."""
    actions: list[tuple[float, float, int]] = []
    previous_x = 0.5
    previous_y = 0.5

    for stroke in strokes:
        points = stroke.get("points", [])
        if len(points) < 2:
            continue
        for point_index, point in enumerate(points):
            x = clamp(float(point.get("x", previous_x)), 0.0, 1.0)
            y = clamp(float(point.get("y", previous_y)), 0.0, 1.0)
            pen = PEN_LIFT if point_index == len(points) - 1 else PEN_CONTINUE
            actions.append((x - previous_x, y - previous_y, pen))
            previous_x = x
            previous_y = y

    return actions


def stroke5_to_polylines(
    actions: Sequence[tuple[float, float, int]],
    *,
    start_x: float = 0.5,
    start_y: float = 0.5,
    previous_pen: int = PEN_LIFT,
) -> list[list[dict[str, float]]]:
    """Decode actions while using the previous pen state for visible movement."""
    strokes: list[list[dict[str, float]]] = []
    points: list[dict[str, float]] = []
    x = start_x
    y = start_y

    for dx, dy, pen in actions:
        if pen == PEN_END:
            break
        x = clamp(x + float(dx), 0.0, 1.0)
        y = clamp(y + float(dy), 0.0, 1.0)
        point = {"x": x, "y": y}

        if previous_pen == PEN_LIFT:
            points = [point]
        else:
            points.append(point)

        if pen == PEN_LIFT:
            if len(points) >= 2:
                strokes.append(points)
            points = []
        previous_pen = int(pen)

    if len(points) >= 2:
        strokes.append(points)
    return strokes


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_pen"].shape[0] for item in batch)
    result: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0]}
    masks: list[torch.Tensor] = []

    for item in batch:
        length = item["input_pen"].shape[0]
        pad = max_len - length
        result["input_xy"].append(F.pad(item["input_xy"], (0, 0, 0, pad), value=0.0))
        result["input_pen"].append(F.pad(item["input_pen"], (0, pad), value=PEN_END))
        result["target_xy"].append(F.pad(item["target_xy"], (0, 0, 0, pad), value=0.0))
        result["target_pen"].append(F.pad(item["target_pen"], (0, pad), value=PEN_END))
        masks.append(F.pad(torch.ones(length, dtype=torch.bool), (0, pad), value=False))

    stacked = {key: torch.stack(value) for key, value in result.items()}
    stacked["mask"] = torch.stack(masks)
    return stacked


def sequence_loss(
    mixture: dict[str, torch.Tensor],
    pen_logits: torch.Tensor,
    target_xy: torch.Tensor,
    target_pen: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, coordinate NLL, and pen cross entropy."""
    # Keep the probability-density arithmetic in float32 even when the
    # Transformer itself is running under CUDA autocast.
    mixture = {key: value.float() for key, value in mixture.items()}
    pen_logits = pen_logits.float()
    target_xy = target_xy.float()
    coord_mask = mask & target_pen.ne(PEN_END)
    target_x = target_xy[..., 0].unsqueeze(-1)
    target_y = target_xy[..., 1].unsqueeze(-1)

    sigma_x = mixture["log_sigma_x"].exp()
    sigma_y = mixture["log_sigma_y"].exp()
    rho = mixture["rho"]
    one_minus_rho_sq = (1.0 - rho.square()).clamp_min(1e-5)
    norm_x = (target_x - mixture["mu_x"]) / sigma_x
    norm_y = (target_y - mixture["mu_y"]) / sigma_y
    z = norm_x.square() + norm_y.square() - 2.0 * rho * norm_x * norm_y
    component_log_prob = (
        -math.log(2.0 * math.pi)
        - mixture["log_sigma_x"]
        - mixture["log_sigma_y"]
        - 0.5 * torch.log(one_minus_rho_sq)
        - z / (2.0 * one_minus_rho_sq)
    )
    log_prob = torch.logsumexp(
        F.log_softmax(mixture["pi_logits"], dim=-1) + component_log_prob,
        dim=-1,
    )
    coordinate_loss = -log_prob[coord_mask].mean()
    pen_loss = F.cross_entropy(pen_logits[mask], target_pen[mask])
    return coordinate_loss + pen_loss, coordinate_loss, pen_loss


def sample_next_action(
    model: Stroke5Transformer,
    xy: torch.Tensor,
    pen: torch.Tensor,
    temperature: float,
) -> tuple[float, float, int]:
    """Sample one normalized coordinate action from the model's final step."""
    mixture, pen_logits = model(xy, pen)
    temperature = max(float(temperature), 1e-3)
    pi = torch.distributions.Categorical(logits=mixture["pi_logits"][0, -1] / temperature)
    component = int(pi.sample().item())

    mu_x = mixture["mu_x"][0, -1, component]
    mu_y = mixture["mu_y"][0, -1, component]
    sigma_x = mixture["log_sigma_x"][0, -1, component].exp() * math.sqrt(temperature)
    sigma_y = mixture["log_sigma_y"][0, -1, component].exp() * math.sqrt(temperature)
    rho = mixture["rho"][0, -1, component]
    epsilon_x = torch.randn((), device=xy.device)
    epsilon_y = torch.randn((), device=xy.device)
    sampled_x = mu_x + sigma_x * epsilon_x
    sampled_y = mu_y + sigma_y * (
        rho * epsilon_x + torch.sqrt((1.0 - rho.square()).clamp_min(1e-5)) * epsilon_y
    )
    sampled_pen = torch.distributions.Categorical(logits=pen_logits[0, -1] / temperature).sample()
    return float(sampled_x.item()), float(sampled_y.item()), int(sampled_pen.item())


@torch.no_grad()
def sample_drawing(
    model: Stroke5Transformer,
    config: TrainConfig,
    *,
    steps: int = 180,
    temperature: float = 0.55,
) -> list[tuple[float, float, int]]:
    model.eval()
    device = next(model.parameters()).device
    input_xy = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
    input_pen = torch.tensor([[PEN_LIFT]], dtype=torch.long, device=device)
    actions: list[tuple[float, float, int]] = []
    x = 0.5
    y = 0.5

    for _ in range(steps):
        normalized_dx, normalized_dy, pen = sample_next_action(model, input_xy, input_pen, temperature)
        raw_dx = normalized_dx * config.coordinate_scale
        raw_dy = normalized_dy * config.coordinate_scale
        next_x = clamp(x + raw_dx, 0.02, 0.98)
        next_y = clamp(y + raw_dy, 0.02, 0.98)
        raw_dx = next_x - x
        raw_dy = next_y - y
        actions.append((raw_dx, raw_dy, pen))
        x, y = next_x, next_y

        next_xy = torch.tensor(
            [[[raw_dx / config.coordinate_scale, raw_dy / config.coordinate_scale]]],
            dtype=torch.float32,
            device=device,
        )
        next_pen = torch.tensor([[pen]], dtype=torch.long, device=device)
        input_xy = torch.cat((input_xy, next_xy), dim=1)[:, -config.max_len :]
        input_pen = torch.cat((input_pen, next_pen), dim=1)[:, -config.max_len :]
        if pen == PEN_END:
            break
    return actions


def train_epoch(
    model: Stroke5Transformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    model.train()
    totals = torch.zeros(3, dtype=torch.float64)
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            mixture, pen_logits = model(
                batch["input_xy"],
                batch["input_pen"],
                padding_mask=~batch["mask"],
            )
            losses = sequence_loss(
                mixture,
                pen_logits,
                batch["target_xy"],
                batch["target_pen"],
                batch["mask"],
            )
        scaler.scale(losses[0]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        token_count = int(batch["mask"].sum().item())
        totals += torch.tensor([float(loss.item()) for loss in losses]) * token_count
        total_tokens += token_count

    return tuple(float(value / max(total_tokens, 1)) for value in totals)  # type: ignore[return-value]


@torch.no_grad()
def evaluate(
    model: Stroke5Transformer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            mixture, pen_logits = model(
                batch["input_xy"],
                batch["input_pen"],
                padding_mask=~batch["mask"],
            )
            losses = sequence_loss(
                mixture,
                pen_logits,
                batch["target_xy"],
                batch["target_pen"],
                batch["mask"],
            )
        token_count = int(batch["mask"].sum().item())
        totals += torch.tensor([float(loss.item()) for loss in losses]) * token_count
        total_tokens += token_count

    return tuple(float(value / max(total_tokens, 1)) for value in totals)  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    model: Stroke5Transformer,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "stroke5-transformer-v3",
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def make_sample_json(actions: Sequence[tuple[float, float, int]]) -> dict[str, Any]:
    polylines = stroke5_to_polylines(actions)
    strokes = []
    for index, points in enumerate(polylines):
        strokes.append(
            {
                "id": f"ai_v3_sample_{index + 1:03d}",
                "author": {"type": "ai", "model": "stroke5-transformer-v3"},
                "tool": "pen",
                "style": {
                    "color": "#ff44aa",
                    "width": 4,
                    "opacity": 1,
                    "lineCap": "round",
                    "lineJoin": "round",
                },
                "timing": {"startMs": 0, "durationMs": max(120, len(points) * 40)},
                "points": points,
                "metadata": {"sample": True},
            }
        )
    return {
        "schema": "https://co-stroke.local/schema/co-stroke-v0.1.json",
        "version": "0.1.0",
        "id": "sample_stroke5_transformer_v3",
        "title": "stroke5-transformer-v3-sample",
        "category": "cat",
        "canvas": {"width": 960, "height": 640, "coordinateSystem": "normalized", "background": "#ffffff"},
        "timeline": {"unit": "stroke", "currentStep": len(strokes), "branchPolicy": "truncate-future-on-edit"},
        "source": {"type": "model", "name": "scripts/train_stroke5_transformer.py"},
        "strokes": strokes,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the co-Stroke continuous stroke-5 Transformer v3.")
    parser.add_argument("--data", required=True, help="Path to a Quick, Draw! simplified .ndjson file.")
    parser.add_argument("--out-dir", default="runs/stroke5-transformer-v3-cat")
    parser.add_argument("--max-drawings", type=int, default=70000)
    parser.add_argument("--include-unrecognized", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.04)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mixtures", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        mixtures=args.mixtures,
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    data_path = Path(config.data)
    out_dir = Path(config.out_dir)
    device = torch.device(config.device)
    amp_enabled = config.amp and device.type == "cuda"

    dataset = QuickDrawStroke5Dataset(data_path, config)
    val_size = max(1, int(len(dataset) * config.val_fraction))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
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

    model = Stroke5Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1), eta_min=config.lr * 0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_val = math.inf

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded {len(dataset)} recognized drawings from {data_path}", flush=True)
    print(f"Skipped {dataset.skipped_too_long} drawings longer than {config.max_len} actions", flush=True)
    print(f"Coordinate scale: {config.coordinate_scale:.8f}", flush=True)
    print(f"Model parameters: {parameter_count:,}", flush=True)
    print(f"Training on {device} (amp={amp_enabled}) -> {out_dir}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    for epoch in range(1, config.epochs + 1):
        train_losses = train_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
        val_losses = evaluate(model, val_loader, device, amp_enabled)
        scheduler.step()
        print(
            f"epoch {epoch:03d} | "
            f"train {train_losses[0]:.4f} (xy {train_losses[1]:.4f}, pen {train_losses[2]:.4f}) | "
            f"val {val_losses[0]:.4f} (xy {val_losses[1]:.4f}, pen {val_losses[2]:.4f}) | "
            f"lr {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        save_checkpoint(out_dir / "latest.pt", model, config, epoch, val_losses[0])
        if val_losses[0] < best_val:
            best_val = val_losses[0]
            save_checkpoint(out_dir / "checkpoint.pt", model, config, epoch, val_losses[0])

    sample_actions = sample_drawing(model, config)
    (out_dir / "sample.json").write_text(
        json.dumps(make_sample_json(sample_actions), indent=2),
        encoding="utf-8",
    )
    print(f"Saved v3 checkpoint and sample to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
