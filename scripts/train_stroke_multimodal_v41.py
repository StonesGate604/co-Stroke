"""Train co-Stroke v4.1 with vector strokes and a rasterized partial canvas.

Every supervision example hides one target stroke. Only the visible context is
rendered, preventing target leakage. A vector Transformer preserves exact
stroke geometry while a small CNN reads the partial drawing as a whole.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_stroke_relational_v4 import (
    AUTHOR_AI,
    QuickDrawStrokeDrawingDataset,
    StrokeCompletionDataset,
    StrokeRelationalTransformer,
    TrainConfig as V4TrainConfig,
    collate_batch as collate_vector_batch,
    make_sample_json,
    move_batch,
    sample_bivariate_gmm,
    sample_shape_gmm,
    sequence_loss,
)


MODEL_TYPE = "stroke-multimodal-v4.1"
RASTER_CHANNELS = 3


@dataclass
class TrainConfig(V4TrainConfig):
    out_dir: str = "runs/stroke-multimodal-v41-cat"
    batch_size: int = 128
    epochs: int = 12
    raster_size: int = 64
    raster_aux_loss_weight: float = 0.20
    pretrained_lr_scale: float = 0.25
    init_checkpoint: str = ""


def _draw_polyline(canvas: torch.Tensor, points: torch.Tensor, channels: Sequence[int]) -> None:
    """Rasterize a normalized polyline using dense samples and rounded pixels."""
    if points.shape[0] < 2 or not channels:
        return
    size = canvas.shape[-1]
    scale = float(size - 1)
    # V4 points are already sampled at equal arc-length intervals. Eight
    # interpolation positions per segment are enough at 64 px and avoid a
    # Python loop for every segment during large-dataset training.
    steps = torch.linspace(0.0, 1.0, 8, dtype=points.dtype, device=points.device)
    starts = points[:-1].unsqueeze(1)
    deltas = (points[1:] - points[:-1]).unsqueeze(1)
    samples = starts + steps.view(1, -1, 1) * deltas
    indices = (samples.reshape(-1, 2).clamp(0.0, 1.0) * scale).round().long()
    x = indices[:, 0]
    y = indices[:, 1]
    for channel in channels:
        canvas[channel, y, x] = 1.0


def rasterize_context(
    context_points: torch.Tensor,
    context_authors: torch.Tensor,
    context_order: torch.Tensor,
    *,
    raster_size: int,
) -> torch.Tensor:
    """Render visible strokes into all/AI/latest-human semantic channels.

    Channel 0 contains every visible stroke, channel 1 contains AI-authored
    strokes, and channel 2 contains the most recent human/non-AI stroke. The
    result is deliberately constructed only from context tensors.
    """
    if raster_size < 16:
        raise ValueError("raster_size must be at least 16")
    canvas = context_points.new_zeros((RASTER_CHANNELS, raster_size, raster_size))
    stroke_count = context_points.shape[0]
    if stroke_count == 0:
        return canvas

    non_ai = torch.nonzero(context_authors != AUTHOR_AI, as_tuple=False).flatten()
    focus_candidates = non_ai if non_ai.numel() else torch.arange(stroke_count, device=context_points.device)
    focus_index = int(focus_candidates[context_order[focus_candidates].argmax()].item())

    for index in range(stroke_count):
        channels = [0]
        if int(context_authors[index].item()) == AUTHOR_AI:
            channels.append(1)
        if index == focus_index:
            channels.append(2)
        _draw_polyline(canvas, context_points[index], channels)

    # A one-pixel centerline can disappear after strided convolutions. This
    # dilation approximates the visual weight of Quick, Draw! and browser ink.
    return F.max_pool2d(canvas.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)


class RasterStrokeCompletionDataset(StrokeCompletionDataset):
    """Add a target-safe partial-canvas raster to each v4 completion example."""

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        example = super().__getitem__(item)
        example["context_raster"] = rasterize_context(
            example["context_points"],
            example["context_authors"],
            example["context_order"],
            raster_size=self.config.raster_size,
        )
        return example


def collate_batch(examples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = collate_vector_batch(examples)
    batch["context_raster"] = torch.stack([example["context_raster"] for example in examples])
    return batch


class RasterCanvasEncoder(nn.Module):
    """Encode a partial drawing without discarding its coarse spatial layout."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(RASTER_CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        return self.features(raster)


class StrokeMultimodalV41(StrokeRelationalTransformer):
    """Fuse the v4 vector context with a CNN encoding of the partial canvas."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__(config)
        self.raster_encoder = RasterCanvasEncoder(config.d_model, config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.fusion_norm = nn.LayerNorm(config.d_model)
        target_dimensions = (config.points_per_stroke - 1) * 2
        shape_hidden = config.d_model * 2
        self.raster_start_head = nn.Linear(config.d_model, config.mixtures * 6)
        self.raster_stop_head = nn.Linear(config.d_model, 1)
        self.raster_shape_conditioner = nn.Sequential(
            nn.Linear(config.d_model + 2, shape_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(shape_hidden, config.d_model),
            nn.GELU(),
        )
        self.raster_shape_pi_head = nn.Linear(config.d_model, config.mixtures)
        self.raster_shape_mean_head = nn.Linear(
            config.d_model,
            config.mixtures * target_dimensions,
        )
        self.raster_shape_log_sigma_head = nn.Linear(
            config.d_model,
            config.mixtures * target_dimensions,
        )
        # A warm-started v4 model should initially behave like v4.0. The raster
        # branch then earns influence through training instead of perturbing the
        # pretrained vector context with a random residual on step one.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def predict_raster_start(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.raster_start_head(context)
        pi_logits, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat = raw.chunk(6, dim=-1)
        return {
            "pi_logits": pi_logits,
            "mu_x": torch.sigmoid(mu_x),
            "mu_y": torch.sigmoid(mu_y),
            "log_sigma_x": log_sigma_x.clamp(-7.0, 1.0),
            "log_sigma_y": log_sigma_y.clamp(-7.0, 1.0),
            "rho": torch.tanh(rho_hat).clamp(-0.999, 0.999),
        }

    def predict_raster_shape(
        self,
        context: torch.Tensor,
        start: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.raster_shape_conditioner(torch.cat((context, start), dim=-1))
        batch_size = context.shape[0]
        point_count = self.config.points_per_stroke - 1
        return {
            "pi_logits": self.raster_shape_pi_head(hidden),
            "mean": self.raster_shape_mean_head(hidden).view(
                batch_size,
                self.config.mixtures,
                point_count,
                2,
            ),
            "log_sigma": self.raster_shape_log_sigma_head(hidden).view(
                batch_size,
                self.config.mixtures,
                point_count,
                2,
            ).clamp(-7.0, 1.0),
        }

    def encode_multimodal_context(
        self,
        context_points: torch.Tensor,
        context_authors: torch.Tensor,
        context_order: torch.Tensor,
        context_mask: torch.Tensor,
        context_raster: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vector_context = super().encode_context(
            context_points,
            context_authors,
            context_order,
            context_mask,
        )
        raster_context = self.raster_encoder(context_raster)
        fused_update = self.fusion(torch.cat((vector_context, raster_context), dim=-1))
        fused_context = self.fusion_norm(vector_context + fused_update)
        return fused_context, raster_context

    def forward(
        self,
        context_points: torch.Tensor,
        context_authors: torch.Tensor,
        context_order: torch.Tensor,
        context_mask: torch.Tensor,
        context_raster: torch.Tensor,
        target_start: torch.Tensor,
    ) -> tuple[
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
    ]:
        fused, raster = self.encode_multimodal_context(
            context_points,
            context_authors,
            context_order,
            context_mask,
            context_raster,
        )
        main = (
            self.predict_start(fused),
            self.predict_shape(fused, target_start),
            self.stop_head(fused).squeeze(-1),
        )
        # The auxiliary prediction prevents the high-capacity vector branch
        # from learning to ignore the new image representation.
        raster_aux = (
            self.predict_raster_start(raster),
            self.predict_raster_shape(raster, target_start),
            self.raster_stop_head(raster).squeeze(-1),
        )
        return main, raster_aux


@torch.no_grad()
def sample_v41_stroke(
    model: StrokeMultimodalV41,
    context_points: torch.Tensor,
    context_authors: torch.Tensor,
    context_order: torch.Tensor,
    context_mask: torch.Tensor,
    context_raster: torch.Tensor,
    *,
    temperature: float = 0.55,
) -> tuple[torch.Tensor, float]:
    model.eval()
    context, _ = model.encode_multimodal_context(
        context_points,
        context_authors,
        context_order,
        context_mask,
        context_raster,
    )
    start_distribution = model.predict_start(context)
    start, start_log_probability = sample_bivariate_gmm(start_distribution, temperature)
    shape_distribution = model.predict_shape(context, start)
    relative_shape, shape_log_probability = sample_shape_gmm(shape_distribution, temperature)
    origin = torch.zeros((start.shape[0], 1, 2), dtype=start.dtype, device=start.device)
    points = start.unsqueeze(1) + torch.cat((origin, relative_shape), dim=1)
    confidence = (start_log_probability + shape_log_probability).mean()
    return points.clamp(0.01, 0.99), float(confidence.item())


def multimodal_loss(
    outputs: tuple[
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
    ],
    batch: dict[str, torch.Tensor],
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    main, raster_aux = outputs
    main_losses = sequence_loss(
        *main,
        batch["target_start"],
        batch["target_shape"],
        batch["target_stop"],
        config.stop_loss_weight,
    )
    auxiliary_losses = sequence_loss(
        *raster_aux,
        batch["target_start"],
        batch["target_shape"],
        batch["target_stop"],
        config.stop_loss_weight,
    )
    total = main_losses[0] + config.raster_aux_loss_weight * auxiliary_losses[0]
    return total, main_losses[1], main_losses[2], main_losses[3], auxiliary_losses[0]


def run_epoch(
    model: StrokeMultimodalV41,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> tuple[float, float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(5, dtype=torch.float64)
    examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(
                    batch["context_points"],
                    batch["context_authors"],
                    batch["context_order"],
                    batch["context_mask"],
                    batch["context_raster"],
                    batch["target_start"],
                )
                losses = multimodal_loss(outputs, batch, config)
            if optimizer is not None and scaler is not None:
                scaler.scale(losses[0]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            batch_size = batch["target_start"].shape[0]
            totals += torch.tensor([float(loss.item()) for loss in losses]) * batch_size
            examples += batch_size
    return tuple(float(value / max(examples, 1)) for value in totals)  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    model: StrokeMultimodalV41,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": MODEL_TYPE,
        "model": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "val_loss": val_loss,
        "raster_channels": ["all-visible", "ai-visible", "latest-human"],
    }, path)


def initialize_from_v4(model: StrokeMultimodalV41, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "stroke-relational-v4":
        raise ValueError(f"Expected a stroke-relational-v4 checkpoint, got {checkpoint.get('model_type')!r}")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected v4 initialization keys: {incompatible.unexpected_keys}")
    if not incompatible.missing_keys or not all(
        key.startswith((
            "raster_encoder.",
            "fusion.",
            "fusion_norm.",
            "raster_start_head.",
            "raster_stop_head.",
            "raster_shape_conditioner.",
            "raster_shape_pi_head.",
            "raster_shape_mean_head.",
            "raster_shape_log_sigma_head.",
        ))
        for key in incompatible.missing_keys
    ):
        raise ValueError(f"Unexpected missing initialization keys: {incompatible.missing_keys}")
    print(
        f"Warm-started vector branch from {checkpoint_path}; initialized "
        f"{len(incompatible.missing_keys)} raster/fusion tensors",
        flush=True,
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the co-Stroke multimodal v4.1 model.")
    parser.add_argument("--data", required=True, help="Path to a Quick, Draw! simplified .ndjson file.")
    parser.add_argument("--out-dir", default="runs/stroke-multimodal-v41-cat")
    parser.add_argument("--max-drawings", type=int, default=70000)
    parser.add_argument("--include-unrecognized", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.04)
    parser.add_argument("--max-context-strokes", type=int, default=24)
    parser.add_argument("--points-per-stroke", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--mixtures", type=int, default=10)
    parser.add_argument("--prefix-probability", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--stop-loss-weight", type=float, default=0.25)
    parser.add_argument("--raster-size", type=int, default=64)
    parser.add_argument("--raster-aux-loss-weight", type=float, default=0.20)
    parser.add_argument("--pretrained-lr-scale", type=float, default=0.25)
    parser.add_argument("--init-checkpoint", default="")
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
        raster_size=args.raster_size,
        raster_aux_loss_weight=args.raster_aux_loss_weight,
        pretrained_lr_scale=args.pretrained_lr_scale,
        init_checkpoint=args.init_checkpoint,
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
    train_dataset = RasterStrokeCompletionDataset(source.drawings, train_indices, config, training=True)
    val_dataset = RasterStrokeCompletionDataset(source.drawings, val_indices, config, training=False)
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

    model = StrokeMultimodalV41(config)
    if config.init_checkpoint:
        initialize_from_v4(model, Path(config.init_checkpoint))
    model = model.to(device)
    if config.init_checkpoint:
        new_prefixes = (
            "raster_encoder.",
            "fusion.",
            "fusion_norm.",
            "raster_start_head.",
            "raster_stop_head.",
            "raster_shape_conditioner.",
            "raster_shape_pi_head.",
            "raster_shape_mean_head.",
            "raster_shape_log_sigma_head.",
        )
        pretrained_parameters = []
        new_parameters = []
        for name, parameter in model.named_parameters():
            target = new_parameters if name.startswith(new_prefixes) else pretrained_parameters
            target.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": pretrained_parameters, "lr": config.lr * config.pretrained_lr_scale},
                {"params": new_parameters, "lr": config.lr},
            ],
            weight_decay=config.weight_decay,
        )
    else:
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
    print(f"Train/validation drawings: {len(train_dataset)}/{len(val_dataset)}", flush=True)
    print(f"Model parameters: {parameter_count:,}", flush=True)
    print(
        f"Raster: {config.raster_size}x{config.raster_size}, channels=all/AI/latest-human, "
        f"aux-weight={config.raster_aux_loss_weight}, pretrained-lr-scale={config.pretrained_lr_scale}",
        flush=True,
    )
    print(f"Training on {device} (amp={amp_enabled}) -> {out_dir}", flush=True)

    best_val_main = math.inf
    for epoch in range(1, config.epochs + 1):
        train_losses = run_epoch(
            model,
            train_loader,
            device,
            config,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_losses = run_epoch(
            model,
            val_loader,
            device,
            config,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        train_main = train_losses[1] + train_losses[2] + config.stop_loss_weight * train_losses[3]
        val_main = val_losses[1] + val_losses[2] + config.stop_loss_weight * val_losses[3]
        print(
            f"epoch {epoch:03d} | "
            f"train objective {train_losses[0]:.4f}, main {train_main:.4f} "
            f"(start {train_losses[1]:.4f}, shape {train_losses[2]:.4f}, "
            f"stop {train_losses[3]:.4f}, raster-aux {train_losses[4]:.4f}) | "
            f"val objective {val_losses[0]:.4f}, main {val_main:.4f} "
            f"(start {val_losses[1]:.4f}, shape {val_losses[2]:.4f}, "
            f"stop {val_losses[3]:.4f}, raster-aux {val_losses[4]:.4f}) | "
            f"lr {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        # Inference uses the fused main branch. The auxiliary objective exists
        # to train the CNN, but must not select a checkpoint whose actual
        # continuation distribution is worse.
        save_checkpoint(out_dir / "latest.pt", model, config, epoch, val_main)
        if val_main < best_val_main:
            best_val_main = val_main
            save_checkpoint(out_dir / "checkpoint.pt", model, config, epoch, val_main)

    example = val_dataset[0]
    batch = move_batch(collate_batch([example]), device)
    sample, _ = sample_v41_stroke(
        model,
        batch["context_points"],
        batch["context_authors"],
        batch["context_order"],
        batch["context_mask"],
        batch["context_raster"],
    )
    sample_json = make_sample_json(sample[0])
    sample_json["id"] = "sample_stroke_multimodal_v41"
    sample_json["title"] = "stroke-multimodal-v4.1-sample"
    sample_json["strokes"][0]["author"]["model"] = MODEL_TYPE
    (out_dir / "sample.json").write_text(json.dumps(sample_json, indent=2), encoding="utf-8")
    print(f"Saved v4.1 checkpoint and sample to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
