from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from snake_frame.app import SnakeFrameApp


class TestAppKpis(unittest.TestCase):
    def test_format_age_short(self) -> None:
        self.assertEqual(SnakeFrameApp._format_age_short(None), "n/a")
        self.assertEqual(SnakeFrameApp._format_age_short(0), "<1m")
        self.assertEqual(SnakeFrameApp._format_age_short(95), "1m")
        self.assertEqual(SnakeFrameApp._format_age_short(3600), "1h")
        self.assertEqual(SnakeFrameApp._format_age_short(90000), "1d")

    def test_training_episode_total_prefers_death_count_total(self) -> None:
        total = SnakeFrameApp._training_episode_total(
            scores=[1] * 240,
            death_counts={"wall": 300, "body": 40, "starvation": 10, "fill": 0, "other": 5},
        )
        self.assertEqual(total, 355)

    def test_training_episode_total_falls_back_to_window_size(self) -> None:
        total = SnakeFrameApp._training_episode_total(
            scores=[1] * 12,
            death_counts={"wall": 0, "body": 0, "starvation": 0, "fill": 0, "other": 0},
        )
        self.assertEqual(total, 12)

    def test_run_graph_scores_exclude_live_in_progress_score(self) -> None:
        app = SnakeFrameApp.__new__(SnakeFrameApp)
        app.game = SimpleNamespace(episode_scores=[4, 9], score=39)
        scores = SnakeFrameApp._run_graph_scores(app)
        self.assertEqual(scores, [4, 9])

    def test_training_episode_info_appends_score_when_score_callback_missing(self) -> None:
        app = SnakeFrameApp.__new__(SnakeFrameApp)
        app.app_state = SimpleNamespace(
            training_episode_scores=[],
            training_episode_steps=[],
            training_death_counts={"wall": 0, "body": 0, "starvation": 0, "fill": 0, "other": 0},
        )
        app.game = SimpleNamespace(EPISODE_HISTORY_LIMIT=240)
        app._append_episode_score = lambda score: app.app_state.training_episode_scores.append(int(score))
        app._normalize_death_reason = SnakeFrameApp._normalize_death_reason
        SnakeFrameApp._append_training_episode_info(app, {"score": 7, "death_reason": "body"})
        self.assertEqual(app.app_state.training_episode_scores, [7])
        self.assertEqual(app.app_state.training_death_counts["body"], 1)

    def test_training_episode_info_captures_steps(self) -> None:
        app = SnakeFrameApp.__new__(SnakeFrameApp)
        app.app_state = SimpleNamespace(
            training_episode_scores=[],
            training_episode_steps=[],
            training_death_counts={"wall": 0, "body": 0, "starvation": 0, "fill": 0, "other": 0},
        )
        app.game = SimpleNamespace(EPISODE_HISTORY_LIMIT=240)
        app._append_episode_score = lambda score: app.app_state.training_episode_scores.append(int(score))
        app._normalize_death_reason = SnakeFrameApp._normalize_death_reason
        SnakeFrameApp._append_training_episode_info(app, {"score": 7, "steps": 150, "death_reason": "body"})
        self.assertEqual(app.app_state.training_episode_steps, [150])
        SnakeFrameApp._append_training_episode_info(app, {"score": 12, "steps": 200, "death_reason": "wall"})
        self.assertEqual(app.app_state.training_episode_steps, [150, 200])

    def test_training_episode_info_trims_steps_history(self) -> None:
        app = SnakeFrameApp.__new__(SnakeFrameApp)
        app.app_state = SimpleNamespace(
            training_episode_scores=[1] * 240,
            training_episode_steps=list(range(240)),
            training_death_counts={"wall": 300, "body": 40, "starvation": 10, "fill": 0, "other": 5},
        )
        app.game = SimpleNamespace(EPISODE_HISTORY_LIMIT=240)
        app._append_episode_score = lambda score: app.app_state.training_episode_scores.append(int(score))
        app._normalize_death_reason = SnakeFrameApp._normalize_death_reason
        SnakeFrameApp._append_training_episode_info(app, {"score": 9, "steps": 999, "death_reason": "body"})
        self.assertEqual(len(app.app_state.training_episode_steps), 240)
        self.assertEqual(app.app_state.training_episode_steps[-1], 999)

    def test_training_episode_info_does_not_duplicate_scores_when_window_is_capped(self) -> None:
        app = SnakeFrameApp.__new__(SnakeFrameApp)
        app.app_state = SimpleNamespace(
            training_episode_scores=[1] * 240,
            training_death_counts={"wall": 300, "body": 40, "starvation": 10, "fill": 0, "other": 5},
        )
        app.game = SimpleNamespace(EPISODE_HISTORY_LIMIT=240)
        app._append_episode_score = lambda score: app.app_state.training_episode_scores.append(int(score))
        app._normalize_death_reason = SnakeFrameApp._normalize_death_reason
        SnakeFrameApp._append_training_episode_info(app, {"score": 9, "death_reason": "body"})
        self.assertEqual(len(app.app_state.training_episode_scores), 240)
        self.assertEqual(app.app_state.training_death_counts["body"], 41)

    def test_runtime_health_lines_include_operational_kpis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SnakeFrameApp.__new__(SnakeFrameApp)
            root = Path(tmpdir)
            app.state_file = root / "ui_state.json"
            app.state_file.write_text("{}", encoding="utf-8")
            metadata = app.state_file.parent / "ppo" / "baseline" / "metadata.json"
            app.experiment_name = "baseline"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text("{}", encoding="utf-8")
            app._train_rate_last_time_s = 0.0
            app._train_rate_last_done_steps = 0
            app._train_rate_ema_sps = 0.0
            app._runtime_health_next_refresh_s = 0.0
            app._runtime_health_cached_lines = []
            app.agent = SimpleNamespace(
                get_model_selector=lambda: "last",
                is_inference_available=True,
                is_ready=True,
                is_sync_pending=False,
            )
            snap = SimpleNamespace(active=True, target_steps=10_000, done_steps=2_000, current_steps=2_000, start_steps=0)
            telemetry = SimpleNamespace(interventions_total=15, decisions_total=100)
            lines = SnakeFrameApp._runtime_health_lines(app, snap, telemetry)
            joined = " | ".join(lines)
            self.assertIn("Train SPS:", joined)
            self.assertIn("Train ETA:", joined)
            self.assertIn("Model src: last", joined)
            self.assertIn("IntvN: 15/100", joined)
            self.assertIn("Freshness eval/chk/ui:", joined)


if __name__ == "__main__":
    unittest.main()
