from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_stroke5_transformer import (  # noqa: E402
    PEN_CONTINUE,
    PEN_END,
    PEN_LIFT,
    Stroke5Transformer,
    TrainConfig,
    drawing_to_stroke5,
    sequence_loss,
    stroke5_to_polylines,
)


class Stroke5TokenizerTests(unittest.TestCase):
    def test_round_trip_preserves_separate_stroke_anchors(self) -> None:
        drawing = [
            [[25, 50, 75], [40, 45, 50]],
            [[180, 200, 220], [160, 150, 140]],
        ]

        actions = drawing_to_stroke5(drawing)
        restored = stroke5_to_polylines(actions)

        self.assertEqual(len(restored), 2)
        for source, decoded in zip(drawing, restored):
            xs, ys = source
            self.assertEqual(len(decoded), len(xs))
            for raw_x, raw_y, point in zip(xs, ys, decoded):
                self.assertAlmostEqual(point["x"], raw_x / 255.0, places=7)
                self.assertAlmostEqual(point["y"], raw_y / 255.0, places=7)

        # The first action of the second stroke contains the long invisible
        # move from the previous endpoint to the new stroke anchor.
        second_start = actions[3]
        self.assertAlmostEqual(second_start[0], (180 - 75) / 255.0, places=7)
        self.assertAlmostEqual(second_start[1], (160 - 50) / 255.0, places=7)

    def test_pen_states_mark_stroke_ends_and_drawing_end(self) -> None:
        actions = drawing_to_stroke5([[[10, 20], [30, 40]], [[80, 90], [100, 110]]])
        self.assertEqual([action[2] for action in actions], [
            PEN_CONTINUE,
            PEN_LIFT,
            PEN_CONTINUE,
            PEN_LIFT,
            PEN_END,
        ])


class Stroke5ModelTests(unittest.TestCase):
    def test_forward_and_loss_are_finite(self) -> None:
        config = TrainConfig(
            data="unused.ndjson",
            max_len=12,
            d_model=24,
            layers=1,
            heads=3,
            mixtures=4,
            dropout=0.0,
        )
        model = Stroke5Transformer(config)
        xy = torch.randn(2, 5, 2)
        pen = torch.tensor([[1, 0, 0, 1, 2], [1, 0, 1, 2, 2]])
        mask = torch.tensor([[True] * 5, [True, True, True, True, False]])
        mixture, pen_logits = model(xy, pen, padding_mask=~mask)

        self.assertEqual(mixture["pi_logits"].shape, (2, 5, 4))
        self.assertEqual(pen_logits.shape, (2, 5, 3))
        losses = sequence_loss(mixture, pen_logits, xy, pen, mask)
        self.assertTrue(all(math.isfinite(float(loss.item())) for loss in losses))


if __name__ == "__main__":
    unittest.main()
