from __future__ import annotations

import unittest

from scripts.smoke_gate_median import _enforce_budgets, _metric_median
from snake_frame.smoke_runner import SmokeBudgets


class TestSmokeGateMedian(unittest.TestCase):
    def test_metric_median_returns_middle_value(self) -> None:
        rows = [
            {"training_steps_per_sec": 260.0},
            {"training_steps_per_sec": 300.0},
            {"training_steps_per_sec": 280.0},
        ]
        self.assertEqual(float(_metric_median(rows, "training_steps_per_sec")), 280.0)

    def test_enforce_raises_when_median_throughput_below_budget(self) -> None:
        budgets = SmokeBudgets(
            max_frame_p95_ms=40.0,
            max_frame_avg_ms=34.0,
            max_frame_jitter_ms=8.0,
            max_inference_p95_ms=12.0,
            min_training_steps_per_sec=250.0,
        )
        medians = {
            "training_steps_per_sec": 220.0,
            "frame_ms_p95": 25.0,
            "frame_ms_avg": 24.0,
            "frame_ms_jitter": 2.0,
            "inference_step_ms_p95": 7.0,
        }
        with self.assertRaisesRegex(RuntimeError, "Median training throughput"):
            _enforce_budgets(medians, budgets)


if __name__ == "__main__":
    unittest.main()
