from __future__ import annotations

import unittest

from scripts.worst_seed_gate import _enforce, compute_worst_seed_report


class TestWorstSeedGate(unittest.TestCase):
    def test_compute_worst_seed_report_orders_by_delta(self) -> None:
        suite = {
            "ppo_only": {"rows": [{"seed": 1, "score": 100}, {"seed": 2, "score": 100}, {"seed": 3, "score": 100}]},
            "controller_on": {"rows": [{"seed": 1, "score": 80}, {"seed": 2, "score": 95}, {"seed": 3, "score": 110}]},
        }
        report = compute_worst_seed_report(suite, top_n=2)
        rows = list(report["worst_rows"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(rows[0]["seed"]), 1)
        self.assertEqual(int(rows[0]["delta_controller_minus_ppo"]), -20)
        self.assertEqual(int(rows[1]["seed"]), 2)
        self.assertEqual(int(rows[1]["delta_controller_minus_ppo"]), -5)

    def test_enforce_fails_when_worse_count_exceeds_max(self) -> None:
        report = {
            "worst_worse_count": 9,
            "worst_mean_delta_controller_minus_ppo": -10.0,
        }
        with self.assertRaisesRegex(RuntimeError, "worse_count"):
            _enforce(report, max_worse_count=8, min_mean_delta=-25.0)


if __name__ == "__main__":
    unittest.main()
