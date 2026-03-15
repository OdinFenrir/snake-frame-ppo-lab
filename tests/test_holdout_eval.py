from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from snake_frame.holdout_eval import HoldoutEvalController
from snake_frame.settings import ObsConfig, RewardConfig, Settings


class _FakeAgent:
    def __init__(self) -> None:
        self.is_ready = True
        self.is_inference_available = True
        self.is_sync_pending = False
        self._train_vecnormalize = None
        self.load_calls = 0

    def evaluate_holdout(self, *, seeds, max_steps: int = 5000, model_selector: str | None = None):
        _ = (max_steps, model_selector)
        return [int((s % 17) + 10) for s in seeds]

    def load_if_exists_detailed(self, selector: str | None = None):
        _ = selector
        self.load_calls += 1
        return type("R", (), {"ok": True})()

    def predict_action(self, _obs, action_masks=None) -> int:
        _ = action_masks
        return 0

    def predict_action_with_probs(self, _obs, action_masks=None):
        _ = action_masks
        return 0, (0.7, 0.2, 0.1)


class TestHoldoutEval(unittest.TestCase):
    def _wait(self, controller: HoldoutEvalController, timeout_s: float = 3.0) -> str | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            msg = controller.poll_completion()
            if msg is not None:
                return msg
            time.sleep(0.01)
        return None

    def test_ppo_only_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctl = HoldoutEvalController(
                agent=_FakeAgent(),
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            self.assertTrue(
                ctl.start(
                    mode=HoldoutEvalController.MODE_PPO_ONLY,
                    seeds=[17001, 17002, 17003],
                    max_steps=500,
                    model_selector="best",
                )
            )
            msg = self._wait(ctl)
            self.assertIsNotNone(msg)
            self.assertIn("Holdout eval done", str(msg))
            latest = Path(tmpdir) / "latest_summary.json"
            self.assertTrue(latest.exists())

    def test_controller_mode_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctl = HoldoutEvalController(
                agent=_FakeAgent(),
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            self.assertTrue(
                ctl.start(
                    mode=HoldoutEvalController.MODE_CONTROLLER_ON,
                    seeds=[17001],
                    max_steps=300,
                    model_selector="best",
                )
            )
            msg = self._wait(ctl)
            self.assertIsNotNone(msg)
            self.assertNotIn("failed", str(msg).lower())
            latest = json.loads((Path(tmpdir) / "latest_summary.json").read_text(encoding="utf-8"))
            self.assertIn("mean_interventions_pct", latest)
            self.assertIn("controller_telemetry_rows", latest)
            rows = list(latest["controller_telemetry_rows"])
            self.assertEqual(len(rows), 1)
            self.assertGreaterEqual(float(latest["mean_interventions_pct"]), 0.0)
            self.assertGreaterEqual(float(rows[0]["interventions_pct"]), 0.0)

    def test_controller_mode_trace_writes_seed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctl = HoldoutEvalController(
                agent=_FakeAgent(),
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            self.assertTrue(
                ctl.start(
                    mode=HoldoutEvalController.MODE_CONTROLLER_ON,
                    seeds=[17001],
                    max_steps=220,
                    model_selector="best",
                    trace_enabled=True,
                    trace_tag="worst10",
                )
            )
            msg = self._wait(ctl, timeout_s=6.0)
            self.assertIsNotNone(msg)
            trace_root = Path(tmpdir) / "focused_traces"
            traces = list(trace_root.glob("**/seed_17001.jsonl"))
            self.assertTrue(traces)
            rows = [line.strip() for line in traces[0].read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(rows)
            sample = json.loads(rows[0])
            self.assertIn("decision_index", sample)
            self.assertIn("mode", sample)
            self.assertIn("switch_reason", sample)
            self.assertIn("score_before", sample)
            self.assertIn("score_after", sample)

    def test_controller_eval_disables_controller_learning(self) -> None:
        class _FakeGameplayController:
            instances: list["_FakeGameplayController"] = []

            def __init__(self, *args, **kwargs) -> None:
                _ = (args, kwargs)
                self.learning_flags: list[bool] = []
                _FakeGameplayController.instances.append(self)

            def set_learning_enabled(self, enabled: bool) -> None:
                self.learning_flags.append(bool(enabled))

            def set_debug_options(self, *, debug_overlay: bool, reachable_overlay: bool) -> None:
                _ = (debug_overlay, reachable_overlay)

            def _apply_agent_control(self) -> None:
                return

            def decision_trace_snapshot(self):
                return {"decision_index": 1, "mode": "ppo", "switch_reason": "none"}

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "snake_frame.holdout_eval.GameplayController",
            _FakeGameplayController,
        ):
            ctl = HoldoutEvalController(
                agent=_FakeAgent(),
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            self.assertTrue(
                ctl.start(
                    mode=HoldoutEvalController.MODE_CONTROLLER_ON,
                    seeds=[17001],
                    max_steps=1,
                    model_selector="best",
                )
            )
            msg = self._wait(ctl)
            self.assertIsNotNone(msg)
            self.assertTrue(_FakeGameplayController.instances)
            self.assertIn(False, _FakeGameplayController.instances[0].learning_flags)

    def test_training_active_does_not_start_or_reload_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _FakeAgent()
            agent._train_vecnormalize = object()
            ctl = HoldoutEvalController(
                agent=agent,
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            started = ctl.start(
                mode=HoldoutEvalController.MODE_PPO_ONLY,
                seeds=[17001, 17002],
                max_steps=300,
                model_selector="best",
            )
            self.assertFalse(started)
            self.assertEqual(int(agent.load_calls), 0)

    def test_start_rejected_when_training_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _FakeAgent()
            agent._train_vecnormalize = object()
            ctl = HoldoutEvalController(
                agent=agent,
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            started = ctl.start(
                mode=HoldoutEvalController.MODE_PPO_ONLY,
                seeds=[17001, 17002],
                max_steps=300,
                model_selector="best",
            )
            self.assertFalse(started)
            snap = ctl.snapshot()
            self.assertFalse(bool(snap.active))
            self.assertEqual(str(snap.last_error), "training_active")

    def test_start_fails_when_model_load_fails(self) -> None:
        class _FailAgent(_FakeAgent):
            def load_if_exists_detailed(self, selector: str | None = None):
                _ = selector
                self.load_calls += 1
                return type("R", (), {"ok": False, "code": "missing", "detail": "selector artifact not found"})()

        with tempfile.TemporaryDirectory() as tmpdir:
            ctl = HoldoutEvalController(
                agent=_FailAgent(),
                settings=Settings(),
                obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
                reward_config=RewardConfig(),
                out_dir=Path(tmpdir),
            )
            self.assertTrue(
                ctl.start(
                    mode=HoldoutEvalController.MODE_PPO_ONLY,
                    seeds=[17001],
                    max_steps=300,
                    model_selector="best",
                )
            )
            msg = self._wait(ctl)
            self.assertIsNotNone(msg)
            self.assertIn("failed", str(msg).lower())
            self.assertIn("eval model load failed", str(msg).lower())


if __name__ == "__main__":
    unittest.main()
