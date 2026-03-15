from __future__ import annotations

import unittest

from scripts.blind_spot_replay_view import build_payload, normalize_window_rows


class TestBlindSpotReplayView(unittest.TestCase):
    def test_normalize_window_rows_falls_back_to_head_when_snake_missing(self) -> None:
        rows = normalize_window_rows(
            [
                {
                    "step": 10,
                    "head_before": [3, 4],
                    "head_after": [4, 4],
                    "food_before": [7, 7],
                    "food_after": [7, 6],
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["snake_before"], [[3, 4]])
        self.assertEqual(row["snake_after"], [[4, 4]])
        self.assertEqual(row["food_before"], [[7, 7]])
        self.assertEqual(row["food_after"], [[7, 6]])

    def test_build_payload_infers_board_and_spot_fields(self) -> None:
        report = {
            "summary": {"rows_scanned": 12},
            "blind_spots": [
                {
                    "seed": 17001,
                    "trace_file": "x/seed_17001.jsonl",
                    "predicted_confidence": 0.9,
                    "steps_until_death": 3,
                    "window_rows": [
                        {"snake_before": [[0, 0], [1, 0]], "food_before": [5, 5]},
                        {"snake_after": [[2, 0], [1, 0]], "food_after": [6, 5]},
                    ],
                }
            ],
        }
        payload = build_payload(report)
        self.assertIn("summary", payload)
        self.assertEqual(len(payload["spots"]), 1)
        spot = payload["spots"][0]
        self.assertEqual(int(spot["seed"]), 17001)
        self.assertGreaterEqual(int(spot["board_cells"]), 20)
        self.assertEqual(len(spot["window_rows"]), 2)


if __name__ == "__main__":
    unittest.main()
