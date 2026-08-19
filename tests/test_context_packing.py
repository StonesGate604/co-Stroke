from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from serve_stroke_model import pack_human_priority_context, resample_polyline  # noqa: E402


def make_points(count: int, x_offset: float = 0.0) -> list[dict[str, float]]:
    return [
        {
            "x": min(0.98, x_offset + index / max(count - 1, 1) * 0.2),
            "y": 0.2 + (index % 5) * 0.01,
        }
        for index in range(count)
    ]


def make_stroke(author: str, count: int, x_offset: float) -> dict:
    return {
        "author": {"type": author},
        "points": make_points(count, x_offset),
    }


class ContextPackingTests(unittest.TestCase):
    def test_resampling_preserves_endpoints_and_cap(self) -> None:
        points = make_points(80, 0.1)
        sampled = resample_polyline(points, 16)
        self.assertEqual(len(sampled), 16)
        self.assertEqual(sampled[0], points[0])
        self.assertAlmostEqual(sampled[-1]["x"], points[-1]["x"])
        self.assertAlmostEqual(sampled[-1]["y"], points[-1]["y"])

    def test_human_strokes_survive_while_old_ai_is_dropped(self) -> None:
        strokes = [make_stroke("human", 40, 0.05 + index * 0.03) for index in range(3)]
        strokes.extend(make_stroke("ai", 20, 0.3 + index * 0.02) for index in range(10))

        packed, stats = pack_human_priority_context(strokes, max_actions=191)

        self.assertEqual(stats["humanStrokesUsed"], 3)
        self.assertEqual(stats["aiStrokesUsed"], 6)
        self.assertEqual(stats["droppedAIStrokes"], 4)
        self.assertLessEqual(stats["totalActionsUsed"], 191)
        self.assertGreater(stats["humanActionsUsed"], 0)
        self.assertTrue(stats["compacted"])
        self.assertEqual(packed[-1]["author"]["type"], "ai")
        self.assertAlmostEqual(packed[-1]["points"][-1]["x"], strokes[-1]["points"][-1]["x"])

    def test_large_human_prompt_gets_reserved_budget(self) -> None:
        strokes = [make_stroke("human", 50, index * 0.03) for index in range(10)]
        strokes.extend(make_stroke("ai", 30, 0.5 + index * 0.02) for index in range(8))

        _, stats = pack_human_priority_context(strokes, max_actions=191)

        self.assertEqual(stats["humanStrokesUsed"], 10)
        self.assertLessEqual(stats["humanActionsUsed"], 120)
        self.assertLessEqual(stats["totalActionsUsed"], 191)
        self.assertEqual(stats["aiStrokesUsed"], 6)


if __name__ == "__main__":
    unittest.main()
