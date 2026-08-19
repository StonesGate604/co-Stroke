"""Train a small autoregressive stroke transformer on Quick, Draw! ndjson.

This is the first CoStroke training baseline. It reads Google's Quick, Draw!
"simplified drawing" ndjson format:

    {"word":"cat","recognized":true,"drawing":[[[x...],[y...]], ...]}

Each drawing is converted into point-level movement tokens:

    dx_bin, dy_bin, pen_state

Where dx/dy are relative movements quantized into bins and pen_state is:

    0 = continue drawing
    1 = lift pen after this movement
    2 = end drawing

The model predicts the next movement token from previous movement tokens.
It is intentionally small and hackable so the data/model/frontend loop can be
connected before spending time on model quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


PEN_CONTINUE = 0
PEN_LIFT = 1
PEN_END = 2
QUICKDRAW_SCALE = 255.0


@dataclass
class TrainConfig:
    data: str
    out_dir: str = "runs/stroke-transformer"
    max_drawings: int = 50000
    val_fraction: float = 0.08
    max_len: int = 192
    num_bins: int = 121
    batch_size: int = 64
    epochs: int = 8
    d_model: int = 192
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QuickDrawStrokeDataset(Dataset):
    def __init__(self, ndjson_path: Path, config: TrainConfig) -> None:
        self.config = config
        self.samples = list(load_quickdraw_tokens(ndjson_path, config))
        if not self.samples:
            raise ValueError(f"No usable drawings found in {ndjson_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[index]

        x = tokens[:-1]
        y = tokens[1:]

        return {
            "dx": torch.tensor([item[0] for item in x], dtype=torch.long),
            "dy": torch.tensor([item[1] for item in x], dtype=torch.long),
            "pen": torch.tensor([item[2] for item in x], dtype=torch.long),
            "target_dx": torch.tensor([item[0] for item in y], dtype=torch.long),
            "target_dy": torch.tensor([item[1] for item in y], dtype=torch.long),
            "target_pen": torch.tensor([item[2] for item in y], dtype=torch.long),
        }


class StrokeTransformer(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.dx_embedding = nn.Embedding(config.num_bins, config.d_model)
        self.dy_embedding = nn.Embedding(config.num_bins, config.d_model)
        self.pen_embedding = nn.Embedding(3, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.dx_head = nn.Linear(config.d_model, config.num_bins)
        self.dy_head = nn.Linear(config.d_model, config.num_bins)
        self.pen_head = nn.Linear(config.d_model, 3)

    def forward(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        pen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = dx.shape
        positions = torch.arange(seq_len, device=dx.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (
            self.dx_embedding(dx)
            + self.dy_embedding(dy)
            + self.pen_embedding(pen)
            + self.position_embedding(positions)
        )
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=dx.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.norm(hidden)
        return self.dx_head(hidden), self.dy_head(hidden), self.pen_head(hidden)


def load_quickdraw_tokens(ndjson_path: Path, config: TrainConfig) -> Iterable[list[tuple[int, int, int]]]:
    loaded = 0
    with ndjson_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if loaded >= config.max_drawings:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            drawing = record.get("drawing")
            if not drawing:
                continue
            tokens = drawing_to_tokens(drawing, config)
            if len(tokens) >= 3:
                loaded += 1
                yield tokens


def drawing_to_tokens(drawing: list, config: TrainConfig) -> list[tuple[int, int, int]]:
    tokens: list[tuple[int, int, int]] = []
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
            dx = x - previous_x
            dy = y - previous_y
            pen = PEN_LIFT if point_index == len(xs) - 1 else PEN_CONTINUE
            tokens.append((quantize_delta(dx, config.num_bins), quantize_delta(dy, config.num_bins), pen))
            previous_x = x
            previous_y = y

            if len(tokens) >= config.max_len - 1:
                break
        if len(tokens) >= config.max_len - 1:
            break

    zero = quantize_delta(0.0, config.num_bins)
    tokens.append((zero, zero, PEN_END))
    return tokens


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["dx"].shape[0] for item in batch)
    result: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0]}
    mask_items: list[torch.Tensor] = []

    for item in batch:
        length = item["dx"].shape[0]
        pad = max_len - length
        for key, tensor in item.items():
            result[key].append(F.pad(tensor, (0, pad), value=0))
        mask_items.append(F.pad(torch.ones(length, dtype=torch.bool), (0, pad), value=False))

    stacked = {key: torch.stack(value) for key, value in result.items()}
    stacked["mask"] = torch.stack(mask_items)
    return stacked


def train_epoch(
    model: StrokeTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        dx_logits, dy_logits, pen_logits = model(batch["dx"], batch["dy"], batch["pen"])
        loss = sequence_loss(dx_logits, dy_logits, pen_logits, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        token_count = int(batch["mask"].sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate(model: StrokeTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        dx_logits, dy_logits, pen_logits = model(batch["dx"], batch["dy"], batch["pen"])
        loss = sequence_loss(dx_logits, dy_logits, pen_logits, batch)
        token_count = int(batch["mask"].sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(1, total_tokens)


def sequence_loss(
    dx_logits: torch.Tensor,
    dy_logits: torch.Tensor,
    pen_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = batch["mask"].reshape(-1)
    dx_loss = F.cross_entropy(dx_logits.reshape(-1, dx_logits.shape[-1])[mask], batch["target_dx"].reshape(-1)[mask])
    dy_loss = F.cross_entropy(dy_logits.reshape(-1, dy_logits.shape[-1])[mask], batch["target_dy"].reshape(-1)[mask])
    pen_loss = F.cross_entropy(pen_logits.reshape(-1, pen_logits.shape[-1])[mask], batch["target_pen"].reshape(-1)[mask])
    return dx_loss + dy_loss + pen_loss


@torch.no_grad()
def sample_tokens(
    model: StrokeTransformer,
    config: TrainConfig,
    steps: int = 120,
    temperature: float = 0.9,
) -> list[tuple[int, int, int]]:
    model.eval()
    device = next(model.parameters()).device
    zero = quantize_delta(0.0, config.num_bins)
    tokens = [(zero, zero, PEN_LIFT)]

    for _ in range(steps):
        context = tokens[-config.max_len :]
        dx = torch.tensor([[item[0] for item in context]], dtype=torch.long, device=device)
        dy = torch.tensor([[item[1] for item in context]], dtype=torch.long, device=device)
        pen = torch.tensor([[item[2] for item in context]], dtype=torch.long, device=device)
        dx_logits, dy_logits, pen_logits = model(dx, dy, pen)
        next_dx = sample_from_logits(dx_logits[0, -1], temperature)
        next_dy = sample_from_logits(dy_logits[0, -1], temperature)
        next_pen = sample_from_logits(pen_logits[0, -1], temperature)
        tokens.append((next_dx, next_dy, next_pen))
        if next_pen == PEN_END:
            break

    return tokens


def sample_from_logits(logits: torch.Tensor, temperature: float) -> int:
    probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def tokens_to_co_stroke_json(tokens: list[tuple[int, int, int]], config: TrainConfig) -> dict:
    strokes = []
    points = []
    x = 0.5
    y = 0.5

    for dx_bin, dy_bin, pen in tokens:
        if pen == PEN_END:
            break
        x = clamp(x + dequantize_delta(dx_bin, config.num_bins), 0.0, 1.0)
        y = clamp(y + dequantize_delta(dy_bin, config.num_bins), 0.0, 1.0)
        points.append({"x": x, "y": y})
        if pen == PEN_LIFT and len(points) >= 2:
            strokes.append(make_ai_stroke(points, len(strokes)))
            points = []

    if len(points) >= 2:
        strokes.append(make_ai_stroke(points, len(strokes)))

    return {
        "schema": "https://co-stroke.local/schema/co-stroke-v0.1.json",
        "version": "0.1.0",
        "id": "sample_stroke_transformer",
        "title": "stroke-transformer-sample",
        "category": "quickdraw",
        "canvas": {"width": 960, "height": 640, "coordinateSystem": "normalized", "background": "#ffffff"},
        "timeline": {"unit": "stroke", "currentStep": len(strokes), "branchPolicy": "truncate-future-on-edit"},
        "source": {"type": "model", "name": "scripts/train_stroke_transformer.py"},
        "strokes": strokes,
    }


def make_ai_stroke(points: list[dict[str, float]], index: int) -> dict:
    return {
        "id": f"ai_sample_{index + 1:03d}",
        "author": {"type": "ai", "model": "stroke-transformer-baseline"},
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


def save_checkpoint(
    out_dir: Path,
    model: StrokeTransformer,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def quantize_delta(value: float, num_bins: int) -> int:
    value = clamp(value, -1.0, 1.0)
    return int(round((value + 1.0) * 0.5 * (num_bins - 1)))


def dequantize_delta(token: int, num_bins: int) -> float:
    return (float(token) / float(num_bins - 1)) * 2.0 - 1.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CoStroke Quick, Draw! stroke transformer baseline.")
    parser.add_argument("--data", required=True, help="Path to a Quick, Draw! simplified .ndjson file.")
    parser.add_argument("--out-dir", default="runs/stroke-transformer")
    parser.add_argument("--max-drawings", type=int, default=50000)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--num-bins", type=int, default=121)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    data_path = Path(config.data)
    out_dir = Path(config.out_dir)
    device = torch.device(config.device)

    dataset = QuickDrawStrokeDataset(data_path, config)
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = StrokeTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    best_val = math.inf

    print(f"Loaded {len(dataset)} drawings from {data_path}")
    print(f"Training on {device} -> {out_dir}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir, model, config, epoch, val_loss)

    sample = tokens_to_co_stroke_json(sample_tokens(model, config), config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Saved checkpoint and sample to {out_dir}")


if __name__ == "__main__":
    main()
