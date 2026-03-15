from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.blind_spot_replay import annotate_steps_until_death, build_blind_spot_report


class TestBlindSpotReplay(unittest.TestCase):
    def test_annotate_steps_until_death_uses_terminal_row(self) -> None:
        rows = [
            {"step": 0, "game_over": False},
            {"step": 1, "game_over": False},
            {"step": 2, "game_over": True},
        ]
        out = annotate_steps_until_death(rows)
        self.assertEqual([int(r["steps_until_death"]) for r in out], [2, 1, 0])

    def test_build_report_extracts_high_conf_near_terminal_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "seed_17001.jsonl"
            rows = [
                {"step": 0, "predicted_confidence": 0.40, "override_used": False, "game_over": False},
                {"step": 1, "predicted_confidence": 0.92, "override_used": False, "game_over": False},
                {"step": 2, "predicted_confidence": 0.88, "override_used": True, "game_over": False},
                {"step": 3, "predicted_confidence": 0.95, "override_used": False, "game_over": True},
            ]
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            report = build_blind_spot_report(
                trace_files=[p],
                min_confidence=0.9,
                max_steps_to_death=2,
                replay_window=2,
                max_spots=10,
                only_no_override=True,
            )
            summary = dict(report["summary"])
            self.assertEqual(int(summary["blind_spot_count"]), 2)
            spots = list(report["blind_spots"])
            self.assertEqual(int(spots[0]["seed"]), 17001)
            self.assertLessEqual(int(spots[0]["steps_until_death"]), 2)
            self.assertTrue(all(not bool(s["override_used"]) for s in spots))
            self.assertTrue(all(len(list(s["window_rows"])) <= 2 for s in spots))


if __name__ == "__main__":
    unittest.main()
