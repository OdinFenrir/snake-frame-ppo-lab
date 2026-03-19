from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from snake_frame.app import SnakeFrameApp
from snake_frame.holdout_eval import HoldoutEvalController
from snake_frame.settings import ObsConfig, RewardConfig, Settings


class _FakeHoldoutAgent:
    def __init__(self, *, score_by_seed: dict[int, int] | None = None) -> None:
        self.is_ready = True
        self.is_inference_available = True
        self.is_sync_pending = False
        self._train_vecnormalize = None
        self._score_by_seed = dict(score_by_seed or {})

    def evaluate_holdout(self, *, seeds, max_steps: int = 5000, model_selector: str | None = None):
        _ = (max_steps, model_selector)
        out: list[int] = []
        for seed in seeds:
            s = int(seed)
            out.append(int(self._score_by_seed.get(s, (s % 13) + 20)))
        return out

    def load_if_exists_detailed(self, selector: str | None = None):
        _ = selector
        return type("R", (), {"ok": True})()

    def get_model_selector(self) -> str:
        return "last"


class TestLiveArtifacts(unittest.TestCase):
    def _make_app_for_suite(self, out_dir: Path) -> SnakeFrameApp:
        app = SnakeFrameApp.__new__(SnakeFrameApp)
        app._eval_suite_dir = Path(out_dir)
        app._eval_suite_started_at_unix_s = 1234.5
        app._eval_suite_max_steps = 5000
        app.agent = SimpleNamespace(get_model_selector=lambda: "last")
        return app

    def _last_jsonl_row(self, path: Path) -> dict:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(rows)
        return json.loads(rows[-1])

    def test_eval_suite_contract_and_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = self._make_app_for_suite(Path(tmpdir) / "suites")
            app._eval_suite_ppo_summary = {
                "scores": {"count": 3, "mean": 110.0, "median": 100.0, "p90": 150.0},
                "rows": [{"seed": 1, "score": 100}, {"seed": 2, "score": 120}, {"seed": 3, "score": 110}],
            }
            app._eval_suite_controller_summary = {
                "scores": {"count": 3, "mean": 95.0, "median": 96.0, "p90": 120.0},
                "rows": [{"seed": 1, "score": 90}, {"seed": 2, "score": 96}, {"seed": 3, "score": 99}],
                "mean_interventions_pct": 4.5,
            }

            stamped, message = SnakeFrameApp._persist_eval_suite_bundle(app)
            self.assertIsNotNone(stamped)
            assert stamped is not None
            self.assertTrue(stamped.exists())
            self.assertIn("Suite done:", message)

            latest = app._eval_suite_dir / "latest_suite.json"
            self.assertTrue(latest.exists())

            stamped_payload = json.loads(stamped.read_text(encoding="utf-8"))
            latest_payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(stamped_payload, latest_payload)

            self.assertIn("generated_at_utc", latest_payload)
            self.assertEqual(int(latest_payload["max_steps"]), 5000)
            self.assertEqual(str(latest_payload["model_selector"]), "last")
            self.assertIn("comparison", latest_payload)

            cmp = dict(latest_payload["comparison"])
            self.assertAlmostEqual(float(cmp["mean_delta_controller_minus_ppo"]), -15.0)
            self.assertAlmostEqual(float(cmp["median_delta_controller_minus_ppo"]), -4.0)
            self.assertAlmostEqual(float(cmp["p90_delta_controller_minus_ppo"]), -30.0)
            self.assertEqual(int(cmp["paired_seed_count"]), 3)
            self.assertEqual(int(cmp["paired_worse_count"]), 3)
            self.assertEqual(int(cmp["paired_improved_count"]), 0)
            self.assertEqual(int(cmp["paired_equal_count"]), 0)
            self.assertAlmostEqual(float(cmp["paired_mean_delta_controller_minus_ppo"]), -15.0)
            self.assertAlmostEqual(float(cmp["paired_median_delta_controller_minus_ppo"]), -11.0)
            self.assertAlmostEqual(float(cmp["mean_interventions_pct"]), 4.5)

    def test_eval_suite_prunes_old_stamped_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir) / "suites"
            suite_dir.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                p = suite_dir / f"suite_20000101_00000{i}.json"
                p.write_text("{}", encoding="utf-8")
                p.touch()
            app = self._make_app_for_suite(suite_dir)
            app._eval_suite_ppo_summary = {"scores": {"count": 1, "mean": 1.0, "median": 1.0, "p90": 1.0}, "rows": []}
            app._eval_suite_controller_summary = {
                "scores": {"count": 1, "mean": 1.0, "median": 1.0, "p90": 1.0},
                "rows": [],
            }
            SnakeFrameApp._persist_eval_suite_bundle(app)
            stamped = sorted([p for p in suite_dir.glob("suite_*.json") if p.is_file()])
            self.assertLessEqual(len(stamped), 2)
            self.assertTrue((suite_dir / "latest_suite.json").exists())

    def test_paired_seed_delta_stats_known_fixture(self) -> None:
        ppo = [
            {"seed": 101, "score": 100},
            {"seed": 102, "score": 100},
            {"seed": 103, "score": 100},
            {"seed": 104, "score": 100},
        ]
        ctrl = [
            {"seed": 101, "score": 95},   # -5
            {"seed": 102, "score": 110},  # +10
            {"seed": 103, "score": 100},  # 0
            {"seed": 104, "score": 90},   # -10
        ]
        stats = SnakeFrameApp._paired_seed_delta_stats(ppo, ctrl)
        self.assertEqual(int(stats["paired_seed_count"]), 4)
        self.assertEqual(int(stats["paired_worse_count"]), 2)
        self.assertEqual(int(stats["paired_improved_count"]), 1)
        self.assertEqual(int(stats["paired_equal_count"]), 1)
        self.assertAlmostEqual(float(stats["paired_mean_delta_controller_minus_ppo"]), -1.25)
        self.assertAlmostEqual(float(stats["paired_median_delta_controller_minus_ppo"]), -2.5)

    def test_artifact_consistency_metadata_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            art = root / "state" / "ppo" / "baseline"
            eval_logs = art / "eval_logs"
            eval_logs.mkdir(parents=True, exist_ok=True)

            run_id = "r1700000000_0_1024"
            metadata = {
                "latest_run_id": run_id,
                "actual_total_timesteps": 1024,
                "last_eval_score": 22.5,
                "eval_runs_completed": 2,
            }
            (art / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            train_rows = [
                {"run_id": "old", "actual_total_timesteps": 256},
                {"run_id": run_id, "actual_total_timesteps": 1024},
            ]
            (art / "training_trace.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )

            eval_rows = [
                {"run_id": "old", "mean_reward": 10.0, "eval_run_index": 1},
                {"run_id": run_id, "mean_reward": 22.5, "eval_run_index": 2},
            ]
            (eval_logs / "evaluations_trace.jsonl").write_text(
                "\n".join(json.dumps(row) for row in eval_rows) + "\n",
                encoding="utf-8",
            )

            latest_summary = {
                "scores": {"count": 3, "mean": 20.0},
                "rows": [{"seed": 1, "score": 10}, {"seed": 2, "score": 20}, {"seed": 3, "score": 30}],
            }
            summary_path = root / "artifacts" / "live_eval" / "latest_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(latest_summary), encoding="utf-8")

            meta = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
            train_last = self._last_jsonl_row(art / "training_trace.jsonl")
            eval_last = self._last_jsonl_row(eval_logs / "evaluations_trace.jsonl")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(str(meta["latest_run_id"]), str(train_last["run_id"]))
            self.assertEqual(str(meta["latest_run_id"]), str(eval_last["run_id"]))
            self.assertEqual(int(meta["actual_total_timesteps"]), int(train_last["actual_total_timesteps"]))
            self.assertAlmostEqual(float(meta["last_eval_score"]), float(eval_last["mean_reward"]))
            self.assertEqual(int(summary["scores"]["count"]), len(summary["rows"]))
            mean = sum(int(r["score"]) for r in summary["rows"]) / float(len(summary["rows"]))
            self.assertAlmostEqual(float(summary["scores"]["mean"]), float(mean))

    def test_run_session_log_rows_include_required_sane_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SnakeFrameApp.__new__(SnakeFrameApp)
            app._run_session_log_path = Path(tmpdir) / "run_session_log.jsonl"
            app._last_logged_run_episodes = 0
            app._runlog_prev_decisions = 10
            app._runlog_prev_interventions = 4
            app._runlog_prev_stuck_episodes = 1
            app.game = SimpleNamespace(episode_scores=[25])
            app.training = SimpleNamespace(snapshot=lambda: SimpleNamespace(current_steps=1024))
            app.gameplay = SimpleNamespace(
                telemetry_snapshot=lambda: SimpleNamespace(
                    decisions_total=30,
                    interventions_total=7,
                    stuck_episodes_total=2,
                    last_death_reason="body",
                    current_mode="ppo",
                    pocket_risk_total=123,
                    loop_escape_activations_total=1,
                )
            )

            rows: list[dict] = []
            app._append_jsonl = lambda _path, payload: rows.append(dict(payload))

            SnakeFrameApp._append_run_session_log_if_needed(app)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            required = {
                "generated_at_unix_s",
                "episode_index",
                "score",
                "death_reason",
                "mode",
                "train_total_steps",
                "interventions_pct",
                "interventions_delta",
                "decisions_delta",
                "risk_total",
                "stuck_episode_delta",
                "loop_escape_activations_total",
            }
            self.assertTrue(required.issubset(set(row.keys())))
            self.assertGreaterEqual(int(row["decisions_delta"]), 0)
            self.assertGreaterEqual(int(row["interventions_delta"]), 0)
            self.assertGreaterEqual(float(row["interventions_pct"]), 0.0)
            self.assertGreaterEqual(int(row["risk_total"]), 0)
            self.assertGreaterEqual(int(row["stuck_episode_delta"]), 0)

            # No new episodes => no additional rows.
            SnakeFrameApp._append_run_session_log_if_needed(app)
            self.assertEqual(len(rows), 1)

    def test_holdout_floor_gate_on_fixed_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            score_by_seed = {17001: 40, 17002: 35, 17003: 45, 17004: 50, 17005: 30}
            agent = _FakeHoldoutAgent(score_by_seed=score_by_seed)
            ctl = HoldoutEvalController(
                agent=agent,
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            started = ctl.start(
                mode=HoldoutEvalController.MODE_PPO_ONLY,
                model_selector="last",
                seeds=[17001, 17002, 17003, 17004, 17005],
                max_steps=500,
            )
            self.assertTrue(started)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                msg = ctl.poll_completion()
                if msg is not None:
                    break
                time.sleep(0.01)
            latest = json.loads((Path(tmpdir) / "latest_summary.json").read_text(encoding="utf-8"))
            floor = 20.0
            self.assertGreaterEqual(float(latest["scores"]["mean"]), floor)


if __name__ == "__main__":
    unittest.main()
