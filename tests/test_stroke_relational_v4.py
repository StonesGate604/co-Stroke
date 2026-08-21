from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from serve_stroke_model import (  # noqa: E402
    browser_to_model_point,
    model_to_browser_point,
    pack_v4_context,
    score_v4_candidate,
)
from train_stroke_relational_v4 import (  # noqa: E402
    StrokeRelationalTransformer,
    TrainConfig,
    resample_stroke,
    sample_v4_stroke,
    sequence_loss,
)


def make_stroke(author: str, offset: float) -> dict:
    return {
        "author": {"type": author},
        "tool": "pen",
        "points": [
            {"x": offset, "y": 0.2},
            {"x": offset + 0.05, "y": 0.25},
            {"x": offset + 0.10, "y": 0.2},
        ],
    }


class StrokeRelationalV4Tests(unittest.TestCase):
    def test_letterbox_mapping_preserves_screen_space_shape_and_round_trip(self) -> None:
        center = browser_to_model_point(
            0.5, 0.5, canvas_width=960, canvas_height=640
        )
        horizontal = browser_to_model_point(
            0.5 + 64 / 960, 0.5, canvas_width=960, canvas_height=640
        )
        vertical = browser_to_model_point(
            0.5, 0.5 + 64 / 640, canvas_width=960, canvas_height=640
        )
        self.assertAlmostEqual(horizontal[0] - center[0], vertical[1] - center[1])

        browser = model_to_browser_point(
            *browser_to_model_point(0.23, 0.71, canvas_width=960, canvas_height=640),
            canvas_width=960,
            canvas_height=640,
        )
        self.assertAlmostEqual(browser[0], 0.23)
        self.assertAlmostEqual(browser[1], 0.71)

    def test_resampling_preserves_endpoints_and_fixed_width(self) -> None:
        sampled = resample_stroke([(0.1, 0.2), (0.4, 0.2), (0.9, 0.8)], 8)
        self.assertEqual(tuple(sampled.shape), (8, 2))
        self.assertTrue(torch.allclose(sampled[0], torch.tensor([0.1, 0.2])))
        self.assertTrue(torch.allclose(sampled[-1], torch.tensor([0.9, 0.8])))

    def test_forward_loss_and_sampling_are_finite(self) -> None:
        config = TrainConfig(
            data="unused.ndjson",
            max_context_strokes=5,
            points_per_stroke=4,
            d_model=24,
            layers=1,
            heads=3,
            mixtures=3,
            dropout=0.0,
        )
        model = StrokeRelationalTransformer(config)
        context_points = torch.rand(2, 3, 4, 2)
        context_authors = torch.tensor([[1, 1, 2], [1, 2, 0]])
        context_order = torch.tensor([[0.0, 0.3, 0.7], [0.0, 0.5, 0.9]])
        context_mask = torch.tensor([[True, True, True], [True, True, False]])
        target_start = torch.rand(2, 2)
        target_shape = torch.rand(2, 3, 2) * 0.2
        target_stop = torch.tensor([0.0, 1.0])

        outputs = model(
            context_points,
            context_authors,
            context_order,
            context_mask,
            target_start,
        )
        self.assertEqual(outputs[0]["pi_logits"].shape, (2, 3))
        self.assertEqual(outputs[1]["mean"].shape, (2, 3, 3, 2))
        losses = sequence_loss(*outputs, target_start, target_shape, target_stop, 0.25)
        self.assertTrue(all(math.isfinite(float(loss.item())) for loss in losses))

        sampled, confidence = sample_v4_stroke(
            model,
            context_points,
            context_authors,
            context_order,
            context_mask,
        )
        self.assertEqual(sampled.shape, (2, 4, 2))
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertTrue(math.isfinite(confidence))

    def test_v4_context_retains_humans_and_recent_ai(self) -> None:
        strokes = [make_stroke("human", 0.02 * index) for index in range(8)]
        strokes.extend(make_stroke("ai", 0.35 + 0.02 * index) for index in range(10))
        packed, stats = pack_v4_context(strokes, max_strokes=10)

        self.assertEqual(stats["humanStrokesUsed"], 4)
        self.assertEqual(stats["aiStrokesUsed"], 6)
        self.assertEqual(stats["droppedAIStrokes"], 4)
        self.assertEqual(len(packed), 10)
        self.assertEqual(packed[-1]["author"]["type"], "ai")

    def test_score_prefers_connection_over_tracing_or_isolation(self) -> None:
        human = make_stroke("human", 0.2)
        traced = [{**point, "t": index * 40, "p": 0.5} for index, point in enumerate(human["points"])]
        connected = [
            {"x": 0.30, "y": 0.20, "t": 0, "p": 0.5},
            {"x": 0.35, "y": 0.15, "t": 40, "p": 0.5},
            {"x": 0.40, "y": 0.10, "t": 80, "p": 0.5},
            {"x": 0.45, "y": 0.12, "t": 120, "p": 0.5},
            {"x": 0.48, "y": 0.16, "t": 160, "p": 0.5},
        ]
        distant = [
            {"x": 0.7, "y": 0.7, "t": 0, "p": 0.5},
            {"x": 0.8, "y": 0.75, "t": 40, "p": 0.5},
            {"x": 0.9, "y": 0.7, "t": 80, "p": 0.5},
        ]
        traced_metrics = score_v4_candidate(
            traced, [human], human, confidence=0.0
        )
        connected_metrics = score_v4_candidate(
            connected, [human], human, confidence=0.0
        )
        distant_metrics = score_v4_candidate(
            distant, [human], human, confidence=0.0
        )
        self.assertGreater(traced_metrics["humanOverlap"], connected_metrics["humanOverlap"])
        self.assertGreater(connected_metrics["score"], traced_metrics["score"])
        self.assertGreater(connected_metrics["score"], distant_metrics["score"])
        self.assertLess(connected_metrics["endpointDistance"], distant_metrics["endpointDistance"])


if __name__ == "__main__":
    unittest.main()
