"""Serve the local stroke transformer checkpoint for browser continuation.

The frontend sends the current co-Stroke drawing to /continue. This server
loads a PyTorch checkpoint, samples continuation tokens, and returns co-Stroke
stroke objects that can be appended directly to the timeline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stroke_transformer import (  # noqa: E402
    PEN_CONTINUE,
    PEN_END,
    PEN_LIFT,
    StrokeTransformer,
    TrainConfig,
    clamp,
    dequantize_delta,
    quantize_delta,
)
from train_sketchgpt_primitives import (  # noqa: E402
    PrimitiveTransformer,
    TrainConfig as PrimitiveTrainConfig,
    apply_direction_repetition_penalty,
    direction_token,
    segment_tokens,
    token_to_delta,
)
from train_stroke5_transformer import (  # noqa: E402
    PEN_CONTINUE as V3_PEN_CONTINUE,
    PEN_END as V3_PEN_END,
    PEN_LIFT as V3_PEN_LIFT,
    Stroke5Transformer,
    TrainConfig as Stroke5TrainConfig,
    co_strokes_to_stroke5,
    sample_next_action as sample_next_stroke5_action,
    stroke5_to_polylines,
)
from train_stroke_relational_v4 import (  # noqa: E402
    AUTHOR_AI as V4_AUTHOR_AI,
    AUTHOR_HUMAN as V4_AUTHOR_HUMAN,
    StrokeRelationalTransformer,
    TrainConfig as StrokeRelationalTrainConfig,
    co_stroke_to_v4_tensor,
    sample_v4_stroke,
)


DEFAULT_CHECKPOINT = "runs/stroke-relational-v4-cat/checkpoint.pt"
V3_CONTEXT_POLICY = "human-priority-v3.1"
V3_HUMAN_ACTION_BUDGET = 120
V3_RECENT_AI_STROKES = 6
V3_HUMAN_POINTS_PER_STROKE = 16
V3_AI_POINTS_PER_STROKE = 12
V4_CONTEXT_POLICY = "stroke-relational-human-priority-v4.0.1"
V4_RECENT_AI_STROKES = 6
V4_CANDIDATES = 16


class StrokeTransformerRuntime:
    def __init__(self, checkpoint: dict[str, Any], checkpoint_path: Path, device_name: str) -> None:
        checkpoint_config = checkpoint.get("config", {})
        allowed = {field.name for field in fields(TrainConfig)}
        config_values = {key: value for key, value in checkpoint_config.items() if key in allowed}
        config_values["device"] = device_name
        self.config = TrainConfig(**config_values)
        self.device = torch.device(device_name)
        self.model = StrokeTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.name = "local-stroke-transformer-cat"
        self.version = f"epoch-{checkpoint.get('epoch', 'unknown')}"
        self.model_type = "stroke-transformer"

    @torch.no_grad()
    def continue_drawing(
        self,
        drawing: dict[str, Any],
        current_step: int,
        *,
        steps: int,
        temperature: float,
        max_strokes: int,
        max_points_per_stroke: int,
        style: dict[str, Any] | None,
    ) -> dict[str, Any]:
        visible_strokes = drawing.get("strokes", [])[:current_step]
        context_tokens, start_x, start_y = strokes_to_tokens(visible_strokes, self.config)
        generated_tokens = self.sample_after_context(
            context_tokens,
            steps=steps,
            temperature=temperature,
            max_strokes=max_strokes,
            max_points_per_stroke=max_points_per_stroke,
        )
        strokes = tokens_to_strokes(
            generated_tokens,
            self.config,
            start_x=start_x,
            start_y=start_y,
            inserted_at_step=current_step,
            style=style or {},
        )
        return {
            "model": {"name": self.name, "version": self.version},
            "strokes": strokes,
        }

    def sample_after_context(
        self,
        context_tokens: list[tuple[int, int, int]],
        *,
        steps: int,
        temperature: float,
        max_strokes: int,
        max_points_per_stroke: int,
    ) -> list[tuple[int, int, int]]:
        zero = quantize_delta(0.0, self.config.num_bins)
        tokens = list(context_tokens[-self.config.max_len :]) or [(zero, zero, PEN_LIFT)]
        generated: list[tuple[int, int, int]] = []
        finished_strokes = 0
        current_stroke_points = 0

        for _ in range(max(1, steps)):
            context = tokens[-self.config.max_len :]
            dx = torch.tensor([[item[0] for item in context]], dtype=torch.long, device=self.device)
            dy = torch.tensor([[item[1] for item in context]], dtype=torch.long, device=self.device)
            pen = torch.tensor([[item[2] for item in context]], dtype=torch.long, device=self.device)
            dx_logits, dy_logits, pen_logits = self.model(dx, dy, pen)

            next_token = (
                sample_from_logits(dx_logits[0, -1], temperature, top_k=16),
                sample_from_logits(dy_logits[0, -1], temperature, top_k=16),
                sample_from_logits(pen_logits[0, -1], temperature, top_k=3),
            )
            tokens.append(next_token)
            generated.append(next_token)
            current_stroke_points += 1

            if next_token[2] == PEN_END:
                break
            if next_token[2] == PEN_LIFT:
                finished_strokes += 1
                current_stroke_points = 0
                if finished_strokes >= max_strokes:
                    break
            if current_stroke_points >= max_points_per_stroke:
                generated.append((zero, zero, PEN_LIFT))
                break

        return generated


class Stroke5TransformerRuntime:
    """Runtime for the continuous stroke-5 Transformer v3 checkpoint."""

    def __init__(self, checkpoint: dict[str, Any], checkpoint_path: Path, device_name: str) -> None:
        checkpoint_config = checkpoint.get("config", {})
        allowed = {field.name for field in fields(Stroke5TrainConfig)}
        config_values = {key: value for key, value in checkpoint_config.items() if key in allowed}
        config_values["device"] = device_name
        self.config = Stroke5TrainConfig(**config_values)
        self.device = torch.device(device_name)
        self.model = Stroke5Transformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.name = "local-stroke5-transformer-v3-cat"
        self.version = f"epoch-{checkpoint.get('epoch', 'unknown')}"
        self.model_type = "stroke5-transformer-v3"
        self.context_policy = V3_CONTEXT_POLICY

    @torch.no_grad()
    def continue_drawing(
        self,
        drawing: dict[str, Any],
        current_step: int,
        *,
        steps: int,
        temperature: float,
        max_strokes: int,
        max_points_per_stroke: int,
        style: dict[str, Any] | None,
    ) -> dict[str, Any]:
        visible_strokes = drawing.get("strokes", [])[:current_step]
        context_strokes, context_stats = pack_human_priority_context(
            visible_strokes,
            max_actions=self.config.max_len - 1,
        )
        context_actions = co_strokes_to_stroke5(context_strokes)
        start_x, start_y = stroke5_endpoint(context_actions)
        previous_pen = context_actions[-1][2] if context_actions else V3_PEN_LIFT
        polylines: list[list[dict[str, float]]] = []
        for _ in range(3):
            generated = self.sample_after_context(
                context_actions,
                start_x=start_x,
                start_y=start_y,
                previous_pen=previous_pen,
                steps=steps,
                temperature=temperature,
                max_strokes=max_strokes,
                max_points_per_stroke=max_points_per_stroke,
            )
            polylines = stroke5_to_polylines(
                generated,
                start_x=start_x,
                start_y=start_y,
                previous_pen=previous_pen,
            )
            if polylines:
                break
        color = (style or {}).get("color", "#ff44aa")
        width = (style or {}).get("width", 4)
        strokes = [
            make_stroke(
                points,
                index,
                current_step,
                color,
                width,
                model_name=self.name,
            )
            for index, points in enumerate(polylines[:max_strokes])
        ]
        return {
            "model": {"name": self.name, "version": self.version, "type": self.model_type},
            "strokes": strokes,
            "context": context_stats,
        }

    def sample_after_context(
        self,
        context_actions: list[tuple[float, float, int]],
        *,
        start_x: float,
        start_y: float,
        previous_pen: int,
        steps: int,
        temperature: float,
        max_strokes: int,
        max_points_per_stroke: int,
    ) -> list[tuple[float, float, int]]:
        scale = self.config.coordinate_scale
        if scale <= 1e-6:
            raise ValueError("The v3 checkpoint has an invalid coordinate_scale")

        model_context = [(0.0, 0.0, V3_PEN_LIFT), *context_actions]
        model_context = model_context[-self.config.max_len :]
        input_xy = torch.tensor(
            [[[dx / scale, dy / scale] for dx, dy, _ in model_context]],
            dtype=torch.float32,
            device=self.device,
        )
        input_pen = torch.tensor(
            [[pen for _, _, pen in model_context]],
            dtype=torch.long,
            device=self.device,
        )

        generated: list[tuple[float, float, int]] = []
        x = start_x
        y = start_y
        active_points = 0 if previous_pen == V3_PEN_LIFT else 1
        completed_strokes = 0

        for _ in range(max(1, steps)):
            normalized_dx, normalized_dy, sampled_pen = sample_next_stroke5_action(
                self.model,
                input_xy,
                input_pen,
                temperature,
            )
            raw_dx = clamp(normalized_dx * scale, -0.6, 0.6)
            raw_dy = clamp(normalized_dy * scale, -0.6, 0.6)
            next_x = clamp(x + raw_dx, 0.02, 0.98)
            next_y = clamp(y + raw_dy, 0.02, 0.98)
            raw_dx = next_x - x
            raw_dy = next_y - y

            if sampled_pen == V3_PEN_END:
                generated.append((0.0, 0.0, V3_PEN_END))
                break

            active_points = 1 if previous_pen == V3_PEN_LIFT else active_points + 1
            if active_points >= max_points_per_stroke:
                sampled_pen = V3_PEN_LIFT

            generated.append((raw_dx, raw_dy, sampled_pen))
            x, y = next_x, next_y
            next_xy = torch.tensor(
                [[[raw_dx / scale, raw_dy / scale]]],
                dtype=torch.float32,
                device=self.device,
            )
            next_pen = torch.tensor([[sampled_pen]], dtype=torch.long, device=self.device)
            input_xy = torch.cat((input_xy, next_xy), dim=1)[:, -self.config.max_len :]
            input_pen = torch.cat((input_pen, next_pen), dim=1)[:, -self.config.max_len :]

            if sampled_pen == V3_PEN_LIFT:
                if active_points >= 2:
                    completed_strokes += 1
                active_points = 0
                if completed_strokes >= max(1, max_strokes):
                    break
            previous_pen = sampled_pen

        # A step budget may expire in the middle of a useful stroke. Closing
        # it makes the browser receive a complete polyline instead of nothing.
        if generated and active_points >= 2 and generated[-1][2] == V3_PEN_CONTINUE:
            dx, dy, _ = generated[-1]
            generated[-1] = (dx, dy, V3_PEN_LIFT)
        return generated


class StrokeRelationalV4Runtime:
    """Runtime for the stroke-level relational completion model v4."""

    def __init__(self, checkpoint: dict[str, Any], checkpoint_path: Path, device_name: str) -> None:
        checkpoint_config = checkpoint.get("config", {})
        allowed = {field.name for field in fields(StrokeRelationalTrainConfig)}
        config_values = {key: value for key, value in checkpoint_config.items() if key in allowed}
        config_values["device"] = device_name
        self.config = StrokeRelationalTrainConfig(**config_values)
        self.device = torch.device(device_name)
        self.model = StrokeRelationalTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.name = "local-stroke-relational-v4-cat"
        self.version = f"epoch-{checkpoint.get('epoch', 'unknown')}"
        self.model_type = "stroke-relational-v4"
        self.context_policy = V4_CONTEXT_POLICY

    @torch.no_grad()
    def continue_drawing(
        self,
        drawing: dict[str, Any],
        current_step: int,
        *,
        steps: int,
        temperature: float,
        max_strokes: int,
        max_points_per_stroke: int,
        style: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del max_strokes  # V4 intentionally contributes one complete stroke per turn.
        visible_strokes = drawing.get("strokes", [])[:current_step]
        context_strokes, context_stats = pack_v4_context(
            visible_strokes,
            max_strokes=self.config.max_context_strokes,
        )
        canvas_width, canvas_height = drawing_canvas_size(drawing)
        model_inputs = make_v4_model_inputs(
            context_strokes,
            visible_stroke_count=len(visible_strokes),
            points_per_stroke=self.config.points_per_stroke,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            device=self.device,
        )

        reference_strokes = [
            stroke for stroke in visible_strokes
            if stroke.get("tool", "pen") == "pen" and len(stroke.get("points", [])) >= 2
        ]
        last_human = next(
            (stroke for stroke in reversed(reference_strokes) if stroke_author_type(stroke) != "ai"),
            None,
        )
        candidate_count = max(8, min(V4_CANDIDATES, max(1, steps // 4)))
        candidates: list[dict[str, Any]] = []
        for _ in range(candidate_count):
            tensor, confidence = sample_v4_stroke(
                self.model,
                model_inputs["context_points"],
                model_inputs["context_authors"],
                model_inputs["context_order"],
                model_inputs["context_mask"],
                temperature=temperature,
            )
            points = tensor_to_co_points(
                tensor[0],
                max_points=max_points_per_stroke,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            metrics = score_v4_candidate(
                points,
                reference_strokes,
                last_human,
                confidence=confidence,
            )
            candidates.append({
                "points": points,
                "confidence": confidence,
                **metrics,
            })

        best = max(candidates, key=lambda candidate: candidate["score"])
        color = (style or {}).get("color", "#ff44aa")
        width = (style or {}).get("width", 4)
        stroke = make_stroke(
            best["points"],
            0,
            current_step,
            color,
            width,
            model_name=self.name,
        )
        stroke["metadata"].update({
            "candidateCount": candidate_count,
            "candidateScore": round(float(best["score"]), 6),
            "overlap": round(float(best["overlap"]), 6),
            "humanOverlap": round(float(best["humanOverlap"]), 6),
            "endpointDistance": round(float(best["endpointDistance"]), 6),
            "nearestDistance": round(float(best["nearestDistance"]), 6),
        })
        context_stats.update({
            "candidateCount": candidate_count,
            "selectedOverlap": round(float(best["overlap"]), 6),
            "selectedHumanOverlap": round(float(best["humanOverlap"]), 6),
            "selectedEndpointDistance": round(float(best["endpointDistance"]), 6),
            "selectedNearestDistance": round(float(best["nearestDistance"]), 6),
            "canvasMapping": "letterbox-square-v1",
        })
        return {
            "model": {"name": self.name, "version": self.version, "type": self.model_type},
            "strokes": [stroke],
            "context": context_stats,
        }


class PrimitiveModelRuntime:
    def __init__(self, checkpoint: dict[str, Any], checkpoint_path: Path, device_name: str) -> None:
        checkpoint_config = checkpoint.get("config", {})
        model_type = checkpoint.get("model_type", "sketchgpt-primitives")
        if model_type == "sketchgpt-primitives" and "token_encoding" not in checkpoint_config:
            checkpoint_config = {**checkpoint_config, "token_encoding": "repeated-direction"}
        allowed = {field.name for field in fields(PrimitiveTrainConfig)}
        config_values = {key: value for key, value in checkpoint_config.items() if key in allowed}
        config_values["device"] = device_name
        self.config = PrimitiveTrainConfig(**config_values)
        self.device = torch.device(device_name)
        self.model = PrimitiveTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.name = (
            "local-sketchgpt-segments-v2-cat"
            if self.config.token_encoding == "direction-length"
            else "local-sketchgpt-primitives-cat"
        )
        self.version = f"epoch-{checkpoint.get('epoch', 'unknown')}"
        self.model_type = model_type

    @torch.no_grad()
    def continue_drawing(
        self,
        drawing: dict[str, Any],
        current_step: int,
        *,
        steps: int,
        temperature: float,
        max_strokes: int,
        max_points_per_stroke: int,
        style: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del max_strokes
        visible_strokes = drawing.get("strokes", [])[:current_step]
        context_tokens, start_x, start_y = strokes_to_primitive_tokens(visible_strokes, self.config)
        generated_tokens = self.sample_next_stroke(
            context_tokens,
            steps=steps,
            temperature=temperature,
            max_points_per_stroke=max_points_per_stroke,
        )
        stroke = primitive_tokens_to_stroke(
            generated_tokens,
            self.config,
            start_x=start_x,
            start_y=start_y,
            inserted_at_step=current_step,
            style=style or {},
            max_path_length=estimate_next_stroke_budget(visible_strokes),
        )
        return {
            "model": {"name": self.name, "version": self.version, "type": self.model_type},
            "strokes": [stroke] if stroke else [],
        }

    def sample_next_stroke(
        self,
        context_tokens: list[int],
        *,
        steps: int,
        temperature: float,
        max_points_per_stroke: int,
    ) -> list[int]:
        tokens = list(context_tokens[-self.config.max_len :]) or [self.config.bos_token]
        generated: list[int] = []

        for _ in range(max(1, min(steps, max_points_per_stroke))):
            context = tokens[-self.config.max_len :]
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)[0, -1]
            logits = apply_direction_repetition_penalty(logits, generated, self.config)
            next_token = sample_from_logits(logits, temperature, top_k=24)
            tokens.append(next_token)

            if next_token == self.config.eos_token:
                break
            if next_token == self.config.stroke_end_token:
                break
            if next_token < self.config.movement_vocab_size:
                generated.append(next_token)

        return generated


def pack_human_priority_context(
    strokes: list[dict[str, Any]],
    *,
    max_actions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compact context while keeping human geometry ahead of old AI output.

    Each polyline point becomes one stroke-5 action. Browser pointer events are
    much denser than simplified QuickDraw data, so every selected stroke is
    first resampled by arc length. Human strokes receive a dedicated budget;
    only the most recent AI strokes compete for the remaining actions.
    """
    usable: list[dict[str, Any]] = []
    for index, stroke in enumerate(strokes):
        points = stroke.get("points", [])
        if len(points) < 2:
            continue
        author_type = stroke_author_type(stroke)
        usable.append(
            {
                "index": index,
                "author": author_type,
                "stroke": stroke,
                "raw_points": len(points),
            }
        )

    human_entries = [entry for entry in usable if entry["author"] != "ai"]
    ai_entries = [entry for entry in usable if entry["author"] == "ai"]
    human_budget = min(max(0, max_actions), V3_HUMAN_ACTION_BUDGET)
    selected_human = fit_context_entries(
        human_entries,
        budget=human_budget,
        max_points_per_stroke=V3_HUMAN_POINTS_PER_STROKE,
    )

    human_actions_used = sum(len(entry["points"]) for entry in selected_human)
    ai_budget = max(0, max_actions - human_actions_used)
    recent_ai = ai_entries[-V3_RECENT_AI_STROKES:]
    selected_ai = fit_context_entries(
        recent_ai,
        budget=ai_budget,
        max_points_per_stroke=V3_AI_POINTS_PER_STROKE,
    )

    selected_entries = sorted([*selected_human, *selected_ai], key=lambda entry: entry["index"])
    packed_strokes = []
    for entry in selected_entries:
        packed = dict(entry["stroke"])
        packed["points"] = entry["points"]
        packed_strokes.append(packed)

    human_actions_before = sum(entry["raw_points"] for entry in human_entries)
    ai_actions_before = sum(entry["raw_points"] for entry in ai_entries)
    ai_actions_used = sum(len(entry["points"]) for entry in selected_ai)
    stats = {
        "policy": V3_CONTEXT_POLICY,
        "maxActions": max_actions,
        "visibleStrokes": len(usable),
        "humanStrokes": len(human_entries),
        "aiStrokes": len(ai_entries),
        "humanStrokesUsed": len(selected_human),
        "aiStrokesUsed": len(selected_ai),
        "humanActionsBefore": human_actions_before,
        "aiActionsBefore": ai_actions_before,
        "humanActionsUsed": human_actions_used,
        "aiActionsUsed": ai_actions_used,
        "totalActionsUsed": human_actions_used + ai_actions_used,
        "droppedAIStrokes": len(ai_entries) - len(selected_ai),
        "compacted": human_actions_before + ai_actions_before > human_actions_used + ai_actions_used,
    }
    return packed_strokes, stats


def fit_context_entries(
    entries: list[dict[str, Any]],
    *,
    budget: int,
    max_points_per_stroke: int,
) -> list[dict[str, Any]]:
    if budget < 2 or not entries:
        return []

    # In the extreme case where two endpoints per stroke already exceed the
    # budget, retain evenly distributed strokes so the whole drawing survives
    # better than a simple tail-only crop. First and last are always retained.
    max_strokes = max(1, budget // 2)
    chosen = entries if len(entries) <= max_strokes else evenly_spaced_entries(entries, max_strokes)
    fitted = []
    for entry in chosen:
        copy = dict(entry)
        copy["points"] = resample_polyline(entry["stroke"].get("points", []), max_points_per_stroke)
        fitted.append(copy)

    while sum(len(entry["points"]) for entry in fitted) > budget:
        reducible = [entry for entry in fitted if len(entry["points"]) > 2]
        if not reducible:
            break
        largest = max(reducible, key=lambda entry: len(entry["points"]))
        largest["points"] = resample_polyline(largest["points"], len(largest["points"]) - 1)
    return fitted


def evenly_spaced_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(entries):
        return list(entries)
    if count <= 1:
        return [entries[-1]]
    indices = []
    for position in range(count):
        index = round(position * (len(entries) - 1) / (count - 1))
        if index not in indices:
            indices.append(index)
    return [entries[index] for index in indices]


def resample_polyline(points: list[dict[str, Any]], max_points: int) -> list[dict[str, float]]:
    """Return at most ``max_points`` evenly spaced arc-length samples."""
    clean = [
        {"x": clamp(float(point.get("x", 0.0)), 0.0, 1.0), "y": clamp(float(point.get("y", 0.0)), 0.0, 1.0)}
        for point in points
    ]
    if len(clean) <= max_points:
        return clean
    max_points = max(2, max_points)

    cumulative = [0.0]
    for previous, point in zip(clean, clean[1:]):
        cumulative.append(cumulative[-1] + math.hypot(point["x"] - previous["x"], point["y"] - previous["y"]))
    total_length = cumulative[-1]
    if total_length <= 1e-8:
        return [clean[0], clean[-1]]

    sampled: list[dict[str, float]] = []
    segment = 0
    for sample_index in range(max_points):
        target = total_length * sample_index / (max_points - 1)
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target:
            segment += 1
        start_distance = cumulative[segment]
        end_distance = cumulative[segment + 1]
        fraction = 0.0 if end_distance <= start_distance else (target - start_distance) / (end_distance - start_distance)
        start = clean[segment]
        end = clean[segment + 1]
        sampled.append(
            {
                "x": start["x"] + (end["x"] - start["x"]) * fraction,
                "y": start["y"] + (end["y"] - start["y"]) * fraction,
            }
        )
    return sampled


def pack_v4_context(
    strokes: list[dict[str, Any]],
    *,
    max_strokes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Protect human geometry while packing a stroke-level v4 context."""
    usable: list[dict[str, Any]] = []
    ignored_erasers = 0
    for index, stroke in enumerate(strokes):
        if stroke.get("tool", "pen") != "pen":
            ignored_erasers += 1
            continue
        if len(stroke.get("points", [])) < 2:
            continue
        usable.append({
            "index": index,
            "author": stroke_author_type(stroke),
            "stroke": stroke,
        })

    human_entries = [entry for entry in usable if entry["author"] != "ai"]
    ai_entries = [entry for entry in usable if entry["author"] == "ai"]
    selected_ai = ai_entries[-min(V4_RECENT_AI_STROKES, max_strokes):]
    human_capacity = max(0, max_strokes - len(selected_ai))
    if human_capacity <= 0:
        selected_human = []
    elif len(human_entries) <= human_capacity:
        selected_human = human_entries
    else:
        selected_human = evenly_spaced_entries(human_entries, human_capacity)
    selected = sorted([*selected_human, *selected_ai], key=lambda entry: entry["index"])

    packed: list[dict[str, Any]] = []
    for entry in selected:
        copy = dict(entry["stroke"])
        copy["_contextIndex"] = entry["index"]
        packed.append(copy)

    stats = {
        "policy": V4_CONTEXT_POLICY,
        "unit": "strokes",
        "maxActions": max_strokes,
        "visibleStrokes": len(usable),
        "humanStrokes": len(human_entries),
        "aiStrokes": len(ai_entries),
        "humanStrokesUsed": len(selected_human),
        "aiStrokesUsed": len(selected_ai),
        # Keep the v3-compatible fields so the existing browser summary works.
        "humanActionsBefore": len(human_entries),
        "aiActionsBefore": len(ai_entries),
        "humanActionsUsed": len(selected_human),
        "aiActionsUsed": len(selected_ai),
        "totalActionsUsed": len(selected),
        "droppedAIStrokes": len(ai_entries) - len(selected_ai),
        "ignoredEraserStrokes": ignored_erasers,
        "compacted": len(selected) < len(usable) or ignored_erasers > 0,
    }
    return packed, stats


def drawing_canvas_size(drawing: dict[str, Any]) -> tuple[float, float]:
    canvas = drawing.get("canvas", {})
    width = float(canvas.get("width", 960) or 960)
    height = float(canvas.get("height", 640) or 640)
    if not math.isfinite(width) or width <= 0:
        width = 960.0
    if not math.isfinite(height) or height <= 0:
        height = 640.0
    return width, height


def browser_to_model_point(
    x: float,
    y: float,
    *,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float]:
    """Letterbox browser-normalized coordinates into a square model space."""
    side = max(canvas_width, canvas_height)
    padding_x = (side - canvas_width) * 0.5
    padding_y = (side - canvas_height) * 0.5
    return (
        (clamp(x, 0.0, 1.0) * canvas_width + padding_x) / side,
        (clamp(y, 0.0, 1.0) * canvas_height + padding_y) / side,
    )


def model_to_browser_point(
    x: float,
    y: float,
    *,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float]:
    """Undo square letterboxing and return browser-normalized coordinates."""
    side = max(canvas_width, canvas_height)
    padding_x = (side - canvas_width) * 0.5
    padding_y = (side - canvas_height) * 0.5
    return (
        (x * side - padding_x) / canvas_width,
        (y * side - padding_y) / canvas_height,
    )


def stroke_to_square_model_space(
    stroke: dict[str, Any],
    *,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, Any]:
    converted = dict(stroke)
    converted["points"] = [
        point_to_square_model_space(
            point,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        for point in stroke.get("points", [])
    ]
    return converted


def point_to_square_model_space(
    point: dict[str, Any],
    *,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, Any]:
    x, y = browser_to_model_point(
        float(point.get("x", 0.0)),
        float(point.get("y", 0.0)),
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    return {**point, "x": x, "y": y}


def make_v4_model_inputs(
    context_strokes: list[dict[str, Any]],
    *,
    visible_stroke_count: int,
    points_per_stroke: int,
    canvas_width: float,
    canvas_height: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if context_strokes:
        points = torch.stack([
            co_stroke_to_v4_tensor(
                stroke_to_square_model_space(
                    stroke,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                ),
                points_per_stroke,
            )
            for stroke in context_strokes
        ]).unsqueeze(0).to(device)
        authors = torch.tensor([[
            V4_AUTHOR_AI if stroke_author_type(stroke) == "ai" else V4_AUTHOR_HUMAN
            for stroke in context_strokes
        ]], dtype=torch.long, device=device)
        denominator = max(visible_stroke_count - 1, 1)
        order = torch.tensor([[
            float(stroke.get("_contextIndex", index)) / denominator
            for index, stroke in enumerate(context_strokes)
        ]], dtype=torch.float32, device=device)
        mask = torch.ones((1, len(context_strokes)), dtype=torch.bool, device=device)
    else:
        points = torch.zeros((1, 0, points_per_stroke, 2), dtype=torch.float32, device=device)
        authors = torch.zeros((1, 0), dtype=torch.long, device=device)
        order = torch.zeros((1, 0), dtype=torch.float32, device=device)
        mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
    return {
        "context_points": points,
        "context_authors": authors,
        "context_order": order,
        "context_mask": mask,
    }


def tensor_to_co_points(
    points: torch.Tensor,
    *,
    max_points: int,
    canvas_width: float,
    canvas_height: float,
) -> list[dict[str, float]]:
    values = points.detach().cpu()
    count = max(2, int(max_points))
    if values.shape[0] > count:
        indices = torch.linspace(0, values.shape[0] - 1, count).round().long()
        values = values[indices]
    converted = []
    for index, point in enumerate(values):
        x, y = model_to_browser_point(
            float(point[0].item()),
            float(point[1].item()),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        converted.append({
            "x": clamp(x, 0.0, 1.0),
            "y": clamp(y, 0.0, 1.0),
            "t": index * 40,
            "p": 0.5,
        })
    return converted


def score_v4_candidate(
    candidate: list[dict[str, float]],
    references: list[dict[str, Any]],
    last_human: dict[str, Any] | None,
    *,
    confidence: float,
) -> dict[str, float]:
    overlap = stroke_overlap_fraction(candidate, references, threshold=0.020)
    human_overlap = stroke_overlap_fraction(candidate, [last_human] if last_human else [], threshold=0.022)
    endpoint_distance, nearest_distance, center_distance = stroke_distance_metrics(candidate, references)
    length = sum(
        math.hypot(point["x"] - previous["x"], point["y"] - previous["y"])
        for previous, point in zip(candidate, candidate[1:])
    )
    boundary_fraction = sum(
        point["x"] <= 0.001 or point["x"] >= 0.999 or point["y"] <= 0.001 or point["y"] >= 0.999
        for point in candidate
    ) / max(len(candidate), 1)
    short_penalty = max(0.0, 0.04 - length) * 30.0
    trace_penalty = max(0.0, overlap - 0.22) * 3.2
    human_trace_penalty = max(0.0, human_overlap - 0.25) * 4.0
    isolation_penalty = max(0.0, nearest_distance - 0.12) * 5.0
    composition_penalty = max(0.0, center_distance - 0.32) * 2.0
    endpoint_connection_reward = max(0.0, 1.0 - endpoint_distance / 0.055) * 0.25
    proximity_reward = max(0.0, 1.0 - nearest_distance / 0.08) * 0.20
    score = (
        0.12 * confidence
        + endpoint_connection_reward
        + proximity_reward
        - trace_penalty
        - human_trace_penalty
        - isolation_penalty
        - composition_penalty
        - boundary_fraction
        - short_penalty
    )
    return {
        "score": score,
        "overlap": overlap,
        "humanOverlap": human_overlap,
        "endpointDistance": endpoint_distance,
        "nearestDistance": nearest_distance,
        "centerDistance": center_distance,
    }


def stroke_distance_metrics(
    candidate: list[dict[str, float]],
    references: list[dict[str, Any]],
) -> tuple[float, float, float]:
    reference_points = [
        (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
        for stroke in references
        for point in stroke.get("points", [])
    ]
    if not candidate or not reference_points:
        return 0.0, 0.0, 0.0

    candidate_points = [(float(point["x"]), float(point["y"])) for point in candidate]

    def nearest(point: tuple[float, float]) -> float:
        return math.sqrt(min(
            (point[0] - x) ** 2 + (point[1] - y) ** 2
            for x, y in reference_points
        ))

    endpoint_distance = min(nearest(candidate_points[0]), nearest(candidate_points[-1]))
    nearest_distance = min(nearest(point) for point in candidate_points)
    candidate_center = (
        sum(point[0] for point in candidate_points) / len(candidate_points),
        sum(point[1] for point in candidate_points) / len(candidate_points),
    )
    reference_center = (
        sum(point[0] for point in reference_points) / len(reference_points),
        sum(point[1] for point in reference_points) / len(reference_points),
    )
    center_distance = math.hypot(
        candidate_center[0] - reference_center[0],
        candidate_center[1] - reference_center[1],
    )
    return endpoint_distance, nearest_distance, center_distance


def stroke_overlap_fraction(
    candidate: list[dict[str, float]],
    references: list[dict[str, Any]],
    *,
    threshold: float,
) -> float:
    reference_points = [
        (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
        for stroke in references
        for point in stroke.get("points", [])
    ]
    if not candidate or not reference_points:
        return 0.0
    overlapping = 0
    threshold_squared = threshold * threshold
    for point in candidate:
        if min(
            (point["x"] - x) ** 2 + (point["y"] - y) ** 2
            for x, y in reference_points
        ) <= threshold_squared:
            overlapping += 1
    return overlapping / len(candidate)


def stroke_author_type(stroke: dict[str, Any]) -> str:
    author = stroke.get("author", {})
    if isinstance(author, str):
        return author
    if isinstance(author, dict):
        return str(author.get("type", "human"))
    return "human"


def stroke5_endpoint(actions: list[tuple[float, float, int]]) -> tuple[float, float]:
    x = 0.5
    y = 0.5
    for dx, dy, pen in actions:
        if pen == V3_PEN_END:
            break
        x = clamp(x + dx, 0.0, 1.0)
        y = clamp(y + dy, 0.0, 1.0)
    return x, y


def strokes_to_tokens(strokes: list[dict[str, Any]], config: TrainConfig) -> tuple[list[tuple[int, int, int]], float, float]:
    tokens: list[tuple[int, int, int]] = []
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
            tokens.append(
                (
                    quantize_delta(x - previous_x, config.num_bins),
                    quantize_delta(y - previous_y, config.num_bins),
                    pen,
                )
            )
            previous_x = x
            previous_y = y

    zero = quantize_delta(0.0, config.num_bins)
    return (tokens or [(zero, zero, PEN_LIFT)]), previous_x, previous_y


def tokens_to_strokes(
    tokens: list[tuple[int, int, int]],
    config: TrainConfig,
    *,
    start_x: float,
    start_y: float,
    inserted_at_step: int,
    style: dict[str, Any],
) -> list[dict[str, Any]]:
    strokes: list[dict[str, Any]] = []
    points: list[dict[str, float]] = []
    x = start_x
    y = start_y
    color = style.get("color", "#ff44aa")
    width = style.get("width", 4)

    for dx_bin, dy_bin, pen in tokens:
        if pen == PEN_END:
            break
        x = clamp(x + dequantize_delta(dx_bin, config.num_bins), 0.0, 1.0)
        y = clamp(y + dequantize_delta(dy_bin, config.num_bins), 0.0, 1.0)
        point = {"x": round(x, 4), "y": round(y, 4)}
        if not points or points[-1] != point:
            points.append(point)

        if pen == PEN_LIFT:
            if len(points) >= 2:
                strokes.append(make_stroke(points, len(strokes), inserted_at_step, color, width))
            points = []

    if len(points) >= 2:
        strokes.append(make_stroke(points, len(strokes), inserted_at_step, color, width))

    return strokes


def strokes_to_primitive_tokens(
    strokes: list[dict[str, Any]],
    config: PrimitiveTrainConfig,
) -> tuple[list[int], float, float]:
    tokens = [config.bos_token]
    last_x = 0.5
    last_y = 0.5

    for stroke in strokes:
        points = stroke.get("points", [])
        if len(points) < 2:
            continue

        previous = points[0]
        last_x = clamp(float(previous.get("x", last_x)), 0.0, 1.0)
        last_y = clamp(float(previous.get("y", last_y)), 0.0, 1.0)

        for point in points[1:]:
            x = clamp(float(point.get("x", last_x)), 0.0, 1.0)
            y = clamp(float(point.get("y", last_y)), 0.0, 1.0)
            dx = x - last_x
            dy = y - last_y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > 1e-5:
                tokens.extend(segment_tokens(dx, dy, config))
            last_x = x
            last_y = y

            if len(tokens) >= config.max_len - 2:
                tokens = tokens[-(config.max_len - 2) :]

        tokens.append(config.stroke_end_token)

    return tokens[-config.max_len :], last_x, last_y


def primitive_tokens_to_stroke(
    tokens: list[int],
    config: PrimitiveTrainConfig,
    *,
    start_x: float,
    start_y: float,
    inserted_at_step: int,
    style: dict[str, Any],
    max_path_length: float,
) -> dict[str, Any] | None:
    points: list[dict[str, float]] = [
        {"x": round(start_x, 4), "y": round(start_y, 4), "t": 0, "p": 0.5}
    ]
    x = start_x
    y = start_y
    path_length = 0.0

    for token in tokens:
        if token >= config.movement_vocab_size:
            continue
        dx, dy = token_to_delta(token, config)
        next_x = x + dx
        next_y = y + dy
        if next_x <= 0.02 or next_x >= 0.98 or next_y <= 0.02 or next_y >= 0.98:
            break

        path_length += (dx * dx + dy * dy) ** 0.5
        if path_length > max_path_length:
            break

        x = next_x
        y = next_y
        point = {"x": round(x, 4), "y": round(y, 4), "t": len(points) * 40, "p": 0.5}
        if not points or points[-1] != point:
            points.append(point)

    if len(points) < 2:
        return None

    color = style.get("color", "#ff44aa")
    width = style.get("width", 4)
    model_name = (
        "local-sketchgpt-segments-v2-cat"
        if config.token_encoding == "direction-length"
        else "local-sketchgpt-primitives-cat"
    )
    return make_stroke(points, 0, inserted_at_step, color, width, model_name=model_name)


def estimate_next_stroke_budget(strokes: list[dict[str, Any]]) -> float:
    lengths = []
    for stroke in strokes[-5:]:
        points = stroke.get("points", [])
        if len(points) < 2:
            continue
        length = 0.0
        previous = points[0]
        for point in points[1:]:
            dx = float(point.get("x", 0.0)) - float(previous.get("x", 0.0))
            dy = float(point.get("y", 0.0)) - float(previous.get("y", 0.0))
            length += (dx * dx + dy * dy) ** 0.5
            previous = point
        if length > 0:
            lengths.append(length)

    if not lengths:
        return 0.25

    average = sum(lengths) / len(lengths)
    return max(0.08, min(0.35, average * 1.4))


def make_stroke(
    points: list[dict[str, float]],
    index: int,
    inserted_at_step: int,
    color: str,
    width: float,
    model_name: str = "local-stroke-transformer-cat",
) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return {
        "id": f"stroke_ai_model_{now}_{index + 1:03d}",
        "author": {
            "type": "ai",
            "id": "local-model",
            "model": model_name,
        },
        "tool": "pen",
        "style": {
            "color": color,
            "width": width,
            "opacity": 1,
            "lineCap": "round",
            "lineJoin": "round",
        },
        "timing": {"startMs": 0, "durationMs": max(120, len(points) * 40)},
        "points": points,
        "metadata": {"insertedAtStep": inserted_at_step, "source": "local-model-server"},
    }


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int | None = None) -> int:
    if top_k is not None and 0 < top_k < logits.numel():
        values, indices = torch.topk(logits, k=top_k)
        probs = F.softmax(values / max(float(temperature), 1e-5), dim=-1)
        sampled = int(torch.multinomial(probs, num_samples=1).item())
        return int(indices[sampled].item())
    probs = F.softmax(logits / max(float(temperature), 1e-5), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def load_runtime(
    checkpoint_path: Path,
    device_name: str,
) -> StrokeTransformerRuntime | Stroke5TransformerRuntime | StrokeRelationalV4Runtime | PrimitiveModelRuntime:
    checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    model_type = checkpoint.get("model_type", "stroke-transformer")
    if model_type == "stroke-relational-v4":
        return StrokeRelationalV4Runtime(checkpoint, checkpoint_path, device_name)
    if model_type == "stroke5-transformer-v3":
        return Stroke5TransformerRuntime(checkpoint, checkpoint_path, device_name)
    if model_type in {"sketchgpt-primitives", "sketchgpt-segments-v2"}:
        return PrimitiveModelRuntime(checkpoint, checkpoint_path, device_name)
    return StrokeTransformerRuntime(checkpoint, checkpoint_path, device_name)


def make_handler(
    runtime: StrokeTransformerRuntime | Stroke5TransformerRuntime | StrokeRelationalV4Runtime | PrimitiveModelRuntime,
) -> type[BaseHTTPRequestHandler]:
    class StrokeModelHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_empty(204)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(
                    {
                        "ok": True,
                        "model": runtime.name,
                        "version": runtime.version,
                        "model_type": runtime.model_type,
                        "checkpoint": str(runtime.checkpoint_path),
                        "device": str(runtime.device),
                        "context_policy": getattr(runtime, "context_policy", None),
                    }
                )
                return
            self.send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/continue":
                self.send_json({"error": "Not found"}, status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                options = payload.get("options", {})
                seed = int(options.get("seed", random.randrange(1, 2**31 - 1)))
                random.seed(seed)
                torch.manual_seed(seed)
                response = runtime.continue_drawing(
                    payload.get("drawing", {}),
                    int(payload.get("currentStep", 0)),
                    steps=int(options.get("steps", 48)),
                    temperature=float(options.get("temperature", 0.25)),
                    max_strokes=int(options.get("maxStrokes", 1)),
                    max_points_per_stroke=int(options.get("maxPointsPerStroke", 28)),
                    style=options.get("style"),
                )
                self.send_json(response)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def send_empty(self, status: int) -> None:
            self.send_response(status)
            self.send_cors_headers()
            self.end_headers()

        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    return StrokeModelHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local co-Stroke continuation model.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_runtime(Path(args.checkpoint), args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"Serving {runtime.name} from {args.checkpoint}")
    print(f"Listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
