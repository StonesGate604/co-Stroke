"""Train a SketchGPT-style primitive-token transformer on Quick, Draw!.

This version avoids predicting dx and dy independently. It converts each sketch
into a single token stream. The current v2 encoding combines direction and
segment length in one movement token:

    BOS, direction_12_x_length_3, ..., STROKE_END, ..., EOS

The earlier baseline represented length by repeating a direction token. That
created long runs of identical training targets and led to straight-line loops
at inference time. The v2 direction-length vocabulary preserves geometry while
greatly reducing that repetition bias. Old checkpoints remain loadable by the
model server through ``token_encoding="repeated-direction"``.

这份脚本的学习地图：

- Dataset/DataLoader：把很多张画变成一批批 mini-batch，交给模型训练。
- Batch size：每次 optimizer 更新参数前，一起处理多少张画。
- Epoch：完整看完一遍 training split。
- Training split：真正用来更新模型参数的数据。
- Validation split：留出来不参与训练，只用来检查模型有没有学会泛化的数据。
- Loss curve：每个 epoch 打印出来的 train/validation loss；通常越低越好。
- Learning rate：每次 optimizer 更新参数时，步子迈多大。
- Checkpoint：保存下来的模型快照，之后可以加载来 sampling 或 serving。
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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional visualization dependency
    SummaryWriter = None


QUICKDRAW_SCALE = 255.0


@dataclass
class TrainConfig:
    data: str
    out_dir: str = "runs/sketchgpt-primitives"
    max_drawings: int = 50000
    # Validation data 不参与参数更新。它像模型每个 epoch 后的小测验，
    # 可以帮助我们发现 overfitting。
    val_fraction: float = 0.08
    # max_len 是 transformer 一次能看到的最长 token 序列。
    # context 越长，模型记住的绘画历史越多，但也更耗显存和时间。
    max_len: int = 256
    # 这是绘画的“字母表”：48 个 direction token 表示每一步可以从
    # 48 个类似罗盘方向的移动里选一个。
    num_primitives: int = 48
    # 一个 direction token 会让笔在 normalized canvas 里移动这么远。
    # step 越小，线条越细腻，但需要更长的 token 序列。
    primitive_step: float = 0.015
    max_repeats_per_segment: int = 16
    # v2 combines a direction and a quantized length into one movement token.
    # ``repeated-direction`` is retained only for loading legacy checkpoints.
    token_encoding: str = "direction-length"
    num_length_bins: int = 8
    recognized_only: bool = True
    # Batch size 表示一次 optimizer step 之前，多少个样本一起参与训练。
    # batch 越大，梯度通常越稳定，但更占显存。
    batch_size: int = 96
    # 一个 epoch 表示训练循环完整看完一遍 training data。
    epochs: int = 8
    d_model: int = 192
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    # Learning rate 控制每次参数更新的步子大小。太高可能训练不稳定；
    # 太低则会学得很慢。
    lr: float = 3e-4
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def movement_vocab_size(self) -> int:
        if self.token_encoding == "direction-length":
            return self.num_primitives * self.num_length_bins
        return self.num_primitives

    @property
    def bos_token(self) -> int:
        return self.movement_vocab_size

    @property
    def stroke_end_token(self) -> int:
        return self.movement_vocab_size + 1

    @property
    def eos_token(self) -> int:
        return self.movement_vocab_size + 2

    @property
    def pad_token(self) -> int:
        return self.movement_vocab_size + 3

    @property
    def vocab_size(self) -> int:
        return self.movement_vocab_size + 4


class PrimitiveSketchDataset(Dataset):
    def __init__(self, ndjson_path: Path, config: TrainConfig) -> None:
        # 创建 Dataset 时先把 drawings 转成 token。每个样本会变成类似：
        # BOS, direction, direction, STROKE_END, ..., EOS 的序列。
        self.samples = list(load_primitive_tokens(ndjson_path, config))
        if not self.samples:
            raise ValueError(f"No usable drawings found in {ndjson_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[index]
        # Next-token prediction：
        # input:  [BOS, dir, dir, STROKE_END]
        # target: [dir, dir, STROKE_END, EOS]
        # 模型要学的是：在每个位置预测下一个 token。
        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "target_ids": torch.tensor(tokens[1:], dtype=torch.long),
        }


class PrimitiveTransformer(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        # Token embedding 会把整数 token id 变成可学习的向量。
        # 这和语言模型把文字 token 变成向量是同一类事情。
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Position embedding 告诉模型每个 token 在序列里的位置。
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
        # head 把 transformer 的 hidden vector 转成整个 drawing-token vocabulary
        # 上的分数，也就是每个 token 作为下一个 token 的可能性。
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        # Causal mask 会挡住未来 token。没有它，模型就能偷看自己本该预测的答案。
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.head(self.norm(hidden))


def load_primitive_tokens(ndjson_path: Path, config: TrainConfig) -> Iterable[list[int]]:
    loaded = 0
    with ndjson_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if loaded >= config.max_drawings:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if config.recognized_only and not record.get("recognized", False):
                continue
            drawing = record.get("drawing")
            if not drawing:
                continue
            tokens = drawing_to_primitive_tokens(drawing, config)
            if len(tokens) >= 4:
                loaded += 1
                yield tokens


def drawing_to_primitive_tokens(drawing: list, config: TrainConfig) -> list[int]:
    tokens = [config.bos_token]

    for stroke in drawing:
        if len(stroke) != 2:
            continue
        xs, ys = stroke
        if len(xs) != len(ys) or len(xs) < 2:
            continue

        previous_x = clamp(float(xs[0]) / QUICKDRAW_SCALE, 0.0, 1.0)
        previous_y = clamp(float(ys[0]) / QUICKDRAW_SCALE, 0.0, 1.0)

        for raw_x, raw_y in zip(xs[1:], ys[1:]):
            x = clamp(float(raw_x) / QUICKDRAW_SCALE, 0.0, 1.0)
            y = clamp(float(raw_y) / QUICKDRAW_SCALE, 0.0, 1.0)
            dx = x - previous_x
            dy = y - previous_y
            distance = math.hypot(dx, dy)
            if distance > 1e-5:
                tokens.extend(segment_tokens(dx, dy, config))
            previous_x = x
            previous_y = y

            if len(tokens) >= config.max_len - 2:
                break

        if len(tokens) < config.max_len - 1:
            tokens.append(config.stroke_end_token)
        if len(tokens) >= config.max_len - 1:
            break

    tokens.append(config.eos_token)
    return tokens[: config.max_len]


def direction_token(dx: float, dy: float, num_primitives: int) -> int:
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += math.tau
    return int(round(angle / math.tau * num_primitives)) % num_primitives


def segment_tokens(dx: float, dy: float, config: TrainConfig) -> list[int]:
    """Encode one source segment while retaining both direction and length."""
    direction = direction_token(dx, dy, config.num_primitives)
    distance = math.hypot(dx, dy)
    if distance <= 1e-5:
        return []

    if config.token_encoding == "repeated-direction":
        repeats = max(1, min(config.max_repeats_per_segment, round(distance / config.primitive_step)))
        return [direction] * repeats

    max_token_length = config.primitive_step * config.num_length_bins
    pieces = max(1, math.ceil(distance / max_token_length))
    piece_length = distance / pieces
    length_bin = max(0, min(config.num_length_bins - 1, round(piece_length / config.primitive_step) - 1))
    return [direction * config.num_length_bins + length_bin] * pieces


def movement_direction(token: int, config: TrainConfig) -> int:
    if config.token_encoding == "direction-length":
        return token // config.num_length_bins
    return token


def token_to_delta(token: int, config: TrainConfig) -> tuple[float, float]:
    direction = movement_direction(token, config)
    length = config.primitive_step
    if config.token_encoding == "direction-length":
        length = (token % config.num_length_bins + 1) * config.primitive_step
    angle = math.tau * float(direction) / float(config.num_primitives)
    return math.cos(angle) * length, math.sin(angle) * length


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_token: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    target_ids = []
    mask = []

    for item in batch:
        length = item["input_ids"].shape[0]
        pad = max_len - length
        # 不同 drawing 的 token 长度不同。Padding 会把它们补齐成一个矩形 tensor，
        # 这样 PyTorch 才能高效地按 batch 处理。
        input_ids.append(F.pad(item["input_ids"], (0, pad), value=pad_token))
        target_ids.append(F.pad(item["target_ids"], (0, pad), value=pad_token))
        # mask 记录哪些位置是真 token，哪些位置只是 padding。
        mask.append(F.pad(torch.ones(length, dtype=torch.bool), (0, pad), value=False))

    return {
        "input_ids": torch.stack(input_ids),
        "target_ids": torch.stack(target_ids),
        "mask": torch.stack(mask),
    }


def train_epoch(
    model: PrimitiveTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_token: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"])
        loss = sequence_loss(logits, batch, pad_token)
        # Backpropagation 会计算 gradient：为了降低当前 batch 的 loss，
        # 每个参数应该往哪个方向调整。
        loss.backward()
        # Gradient clipping 防止某个 batch 产生过大的更新，把训练带偏。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer 根据 gradient 真正更新模型参数。
        optimizer.step()

        token_count = int(batch["mask"].sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate(model: PrimitiveTransformer, loader: DataLoader, device: torch.device, pad_token: int) -> float:
    # Evaluation 使用 validation split。它只衡量表现，不调用 backward()
    # 或 optimizer.step()，所以不会训练模型。
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["input_ids"])
        loss = sequence_loss(logits, batch, pad_token)
        token_count = int(batch["mask"].sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(1, total_tokens)


def sequence_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], pad_token: int) -> torch.Tensor:
    # Cross entropy 是“选择正确下一个 token”这类分类任务的常用 loss。
    # ignore_index 让 padding token 不参与 loss 计算。
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["target_ids"].reshape(-1),
        ignore_index=pad_token,
    )


@torch.no_grad()
def sample_tokens(
    model: PrimitiveTransformer,
    config: TrainConfig,
    steps: int = 180,
    temperature: float = 0.85,
    top_k: int = 24,
) -> list[int]:
    model.eval()
    device = next(model.parameters()).device
    tokens = [config.bos_token]

    for _ in range(steps):
        context = tokens[-config.max_len :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1]
        logits = apply_direction_repetition_penalty(logits, tokens[1:], config)
        next_token = sample_from_logits(logits, temperature=temperature, top_k=top_k)
        tokens.append(next_token)
        if next_token == config.eos_token:
            break

    return tokens


def apply_direction_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: list[int],
    config: TrainConfig,
    penalty: float = 0.65,
) -> torch.Tensor:
    """Discourage direction loops after two consecutive generated segments."""
    if config.token_encoding != "direction-length" or len(generated_tokens) < 2:
        return logits
    movement_tokens = [token for token in generated_tokens if token < config.movement_vocab_size]
    if len(movement_tokens) < 2:
        return logits
    last_direction = movement_direction(movement_tokens[-1], config)
    run_length = 1
    for token in reversed(movement_tokens[:-1]):
        if movement_direction(token, config) != last_direction:
            break
        run_length += 1
    if run_length < 2:
        return logits

    adjusted = logits.clone()
    start = last_direction * config.num_length_bins
    end = start + config.num_length_bins
    adjusted[start:end] -= penalty * min(run_length - 1, 3)
    return adjusted


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    # Sampling 会把模型分数变成一个具体选中的 next token。temperature 越低越保守；
    # temperature 越高越意外，也更容易 noisy。
    if top_k > 0 and top_k < logits.numel():
        # top_k 只允许从最可能的 k 个 token 里选，可以减少生成时的混乱。
        values, indices = torch.topk(logits, k=top_k)
        probs = F.softmax(values / max(temperature, 1e-5), dim=-1)
        sampled = int(torch.multinomial(probs, num_samples=1).item())
        return int(indices[sampled].item())
    probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def tokens_to_co_stroke_json(tokens: list[int], config: TrainConfig) -> dict:
    strokes = []
    points = []
    x = 0.5
    y = 0.5

    for token in tokens:
        if token in (config.bos_token, config.pad_token):
            continue
        if token == config.eos_token:
            break
        if token == config.stroke_end_token:
            if len(points) >= 2:
                strokes.append(make_ai_stroke(points, len(strokes)))
            points = []
            continue
        if token >= config.movement_vocab_size:
            continue

        dx, dy = token_to_delta(token, config)
        x = clamp(x + dx, 0.0, 1.0)
        y = clamp(y + dy, 0.0, 1.0)
        point = {"x": round(x, 4), "y": round(y, 4)}
        if not points or points[-1] != point:
            points.append(point)

    if len(points) >= 2:
        strokes.append(make_ai_stroke(points, len(strokes)))

    return {
        "schema": "https://co-stroke.local/schema/co-stroke-v0.1.json",
        "version": "0.1.0",
        "id": "sample_sketchgpt_primitives",
        "title": "sketchgpt-primitive-sample",
        "category": "quickdraw",
        "canvas": {"width": 960, "height": 640, "coordinateSystem": "normalized", "background": "#ffffff"},
        "timeline": {"unit": "stroke", "currentStep": len(strokes), "branchPolicy": "truncate-future-on-edit"},
        "source": {"type": "model", "name": "scripts/train_sketchgpt_primitives.py"},
        "strokes": strokes,
    }


def make_ai_stroke(points: list[dict[str, float]], index: int) -> dict:
    return {
        "id": f"ai_primitive_{index + 1:03d}",
        "author": {"type": "ai", "model": "sketchgpt-primitive-baseline"},
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
    model: PrimitiveTransformer,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
) -> None:
    # checkpoint 是保存下来的训练结果。之后可以加载它做 inference、serving，
    # 或继续实验，不需要从头训练。
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": (
                "sketchgpt-segments-v2"
                if config.token_encoding == "direction-length"
                else "sketchgpt-primitives"
            ),
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a SketchGPT-style primitive-token baseline.")
    parser.add_argument("--data", required=True, help="Path to a Quick, Draw! simplified .ndjson file.")
    parser.add_argument("--out-dir", default="runs/sketchgpt-primitives")
    parser.add_argument("--max-drawings", type=int, default=50000)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--num-primitives", type=int, default=48)
    parser.add_argument("--primitive-step", type=float, default=0.015)
    parser.add_argument("--max-repeats-per-segment", type=int, default=16)
    parser.add_argument("--token-encoding", choices=["direction-length", "repeated-direction"], default="direction-length")
    parser.add_argument("--num-length-bins", type=int, default=8)
    parser.add_argument("--include-unrecognized", action="store_false", dest="recognized_only")
    parser.add_argument("--batch-size", type=int, default=96)
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

    dataset = PrimitiveSketchDataset(data_path, config)
    val_size = max(1, int(len(dataset) * config.val_fraction))
    train_size = len(dataset) - val_size
    # 这里把数据分成 train set 和 validation set。如果 train loss 一直下降，
    # 但 validation loss 变差，模型可能在 overfitting：它在背训练画作，
    # 而不是学习可复用的绘画规律。
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, config.pad_token),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, config.pad_token),
    )

    model = PrimitiveTransformer(config).to(device)
    # AdamW 是 optimizer：它根据 gradient 决定如何更新 weights。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    # 记录最好的 validation loss，这样保存下来的 checkpoint 是泛化最好的一版，
    # 而不只是最后一个 epoch 的版本。
    best_val = math.inf
    writer = SummaryWriter(str(out_dir / "tb")) if SummaryWriter else None

    print(f"Loaded {len(dataset)} drawings from {data_path}")
    print(
        f"Primitive vocab size: {config.vocab_size} "
        f"({config.num_primitives} directions x {config.num_length_bins if config.token_encoding == 'direction-length' else 1} "
        f"length bins + special tokens; encoding={config.token_encoding})"
    )
    print(f"Training on {device} -> {out_dir}")
    if writer:
        writer.add_text("config/json", json.dumps(asdict(config), indent=2), 0)
    else:
        print("TensorBoard is not installed; skipping dashboard logs.")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, config.pad_token)
        val_loss = evaluate(model, val_loader, device, config.pad_token)
        # 这些打印出来的数字就是最简单的 loss curve。到了 TensorBoard 里，
        # 同样的数据会变成真正的曲线图。
        print(f"epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir, model, config, epoch, val_loss)

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("loss/best_val", best_val, epoch)
            writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch)

    sample = tokens_to_co_stroke_json(sample_tokens(model, config), config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    if writer:
        writer.add_text("sample/json", json.dumps(sample, indent=2), config.epochs)
        writer.flush()
        writer.close()
    print(f"Saved checkpoint and sample to {out_dir}")


if __name__ == "__main__":
    main()
