from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from snake_frame.state_io import delete_ui_state, load_ui_state, save_ui_state

try:
    from snake_frame.ppo_agent import ModelOpCode, PpoSnakeAgent
    from snake_frame.settings import ObsConfig, PpoConfig, RewardConfig, Settings
    _AGENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - env-dependent import gate
    _AGENT_IMPORT_ERROR = exc

_SKIP_REASON = (
    "ML dependency setup required for PPO persistence tests. "
    "Install with `pip install -r requirements.txt`. "
    f"Import error: {_AGENT_IMPORT_ERROR}"
)


class _FakeModel:
    def __init__(self, num_timesteps: int = 0) -> None:
        self.num_timesteps = int(num_timesteps)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake-model", encoding="utf-8")


class TestPersistence(unittest.TestCase):
    def test_state_io_save_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state" / "ui_state.json"
            payload = {"episodeScores": [1, 2, 3], "savedAt": 123.0}
            save_ui_state(state_file, payload)
            self.assertTrue(state_file.exists())
            deleted = delete_ui_state(state_file)
            self.assertTrue(deleted)
            self.assertFalse(state_file.exists())

    def test_load_ui_state_invalid_json_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state" / "ui_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text("{invalid-json", encoding="utf-8")
            self.assertIsNone(load_ui_state(state_file))

    def test_load_ui_state_invalid_encoding_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state" / "ui_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_bytes(b"\xff\xfe\xfa\x00")
            self.assertIsNone(load_ui_state(state_file))

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_save_load_delete_cycle_v2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "state" / "ppo" / "baseline"
            settings = Settings()
            agent = PpoSnakeAgent(
                settings=settings,
                artifact_dir=artifact_dir,
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
                legacy_model_path=Path(tmpdir) / "state" / "ppo_snake_model.zip",
            )
            agent.model = _FakeModel(num_timesteps=123)
            fake_vec = MagicMock()
            fake_vec.save.side_effect = lambda path: Path(path).write_text("fake-vec", encoding="utf-8")
            agent._train_vecnormalize = fake_vec

            with patch("snake_frame.ppo_agent.PPO.load", return_value=_FakeModel(num_timesteps=123)):
                saved = agent.save()
                self.assertTrue(saved)

            self.assertTrue((artifact_dir / "last_model.zip").exists())
            self.assertTrue((artifact_dir / "best_model.zip").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())
            self.assertEqual(fake_vec.save.call_count, 2)
            metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("provenance", metadata)
            self.assertIn("runtime_controls", metadata)
            self.assertIn("settings", dict(metadata["runtime_controls"]))
            self.assertIn("dependencies", dict(metadata["provenance"]))
            self.assertIn("git", dict(metadata["provenance"]))

            with patch("snake_frame.ppo_agent.PPO.load", return_value=_FakeModel(num_timesteps=999)):
                loaded = agent.load_if_exists()
                self.assertTrue(loaded)
                self.assertIsNotNone(agent.model)

            removed = agent.delete()
            self.assertTrue(removed)
            self.assertFalse((artifact_dir / "last_model.zip").exists())

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_delete_cleans_trace_and_log_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "state" / "ppo" / "baseline"
            run_logs = artifact_dir / "run_logs"
            eval_logs = artifact_dir / "eval_logs"
            run_logs.mkdir(parents=True, exist_ok=True)
            eval_logs.mkdir(parents=True, exist_ok=True)

            (artifact_dir / "last_model.zip").write_text("model", encoding="utf-8")
            (artifact_dir / "resume_model.zip").write_text("resume", encoding="utf-8")
            (artifact_dir / "metadata.json").write_text("{}", encoding="utf-8")
            (artifact_dir / "training_trace.jsonl").write_text('{"run_id":"r1"}\n', encoding="utf-8")
            (eval_logs / "evaluations_trace.jsonl").write_text('{"run_id":"r1"}\n', encoding="utf-8")
            (run_logs / "train_trace_r1.jsonl").write_text('{"run_id":"r1"}\n', encoding="utf-8")

            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=artifact_dir,
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
                legacy_model_path=Path(tmpdir) / "state" / "ppo_snake_model.zip",
            )
            removed = agent.delete()
            self.assertTrue(removed)
            self.assertFalse((artifact_dir / "training_trace.jsonl").exists())
            self.assertFalse((artifact_dir / "run_logs").exists())
            self.assertFalse((artifact_dir / "eval_logs").exists())

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_legacy_model_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "state" / "ppo_snake_model.zip"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text("legacy", encoding="utf-8")
            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
                legacy_model_path=legacy_path,
            )
            result = agent.load_if_exists_detailed()
            self.assertFalse(result.ok)
            self.assertEqual(result.code, ModelOpCode.LEGACY_FORMAT_UNSUPPORTED)

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_evaluate_uses_training_model_for_last_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )

            class _PredictModel:
                def __init__(self, action: int) -> None:
                    self.action = int(action)
                    self.observation_space = SimpleNamespace(shape=(11,))

                def predict(self, _obs, deterministic: bool = True):
                    return np.array([self.action], dtype=np.int64), None

            class _FakeEvalEnv:
                def __init__(self, *args, **kwargs) -> None:
                    self.score = 0

                def reset(self, **kwargs):
                    return np.zeros((11,), dtype=np.float32), {}

                def step(self, action: int):
                    self.score = int(action)
                    info = {"score": int(action), "steps": 1}
                    return np.zeros((11,), dtype=np.float32), 0.0, True, False, info

                def close(self) -> None:
                    return None

            agent.model = _PredictModel(action=2)
            agent.inference_model = _PredictModel(action=1)
            with patch("snake_frame.ppo_agent.SnakePPOEnv", _FakeEvalEnv):
                score = agent.evaluate(episodes=3, max_steps=5, model_selector="last")
            self.assertEqual(score, 2)

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_evaluate_uses_inference_model_for_last_selector_while_training_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )

            class _PredictModel:
                def __init__(self, action: int) -> None:
                    self.action = int(action)
                    self.observation_space = SimpleNamespace(shape=(11,))

                def predict(self, _obs, deterministic: bool = True):
                    return np.array([self.action], dtype=np.int64), None

            class _FakeEvalEnv:
                def __init__(self, *args, **kwargs) -> None:
                    self.score = 0

                def reset(self, **kwargs):
                    return np.zeros((11,), dtype=np.float32), {}

                def step(self, action: int):
                    self.score = int(action)
                    info = {"score": int(action), "steps": 1}
                    return np.zeros((11,), dtype=np.float32), 0.0, True, False, info

                def close(self) -> None:
                    return None

            agent.model = _PredictModel(action=2)
            agent.inference_model = _PredictModel(action=1)
            agent._train_vecnormalize = object()
            with patch("snake_frame.ppo_agent.SnakePPOEnv", _FakeEvalEnv):
                score = agent.evaluate(episodes=3, max_steps=5, model_selector="last")
            self.assertEqual(score, 1)

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_load_if_exists_detailed_reports_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "state" / "ppo" / "baseline"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "best_model.zip").write_text("broken", encoding="utf-8")
            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=artifact_dir,
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )
            with patch("snake_frame.ppo_agent.PPO.load", side_effect=RuntimeError("bad zip")):
                result = agent.load_if_exists_detailed(selector="best")
            self.assertFalse(result.ok)
            self.assertEqual(result.code, ModelOpCode.CORRUPT)

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_save_returns_false_on_io_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )
            agent.model = _FakeModel(num_timesteps=1)
            with patch.object(_FakeModel, "save", side_effect=OSError("disk full")):
                saved = agent.save()
            self.assertFalse(saved)
            leftovers = list(agent.artifact_dir.glob("*.tmp"))
            self.assertEqual(leftovers, [])

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_save_uses_atomic_replace_without_temp_leftovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "state" / "ppo" / "baseline"
            settings = Settings()
            agent = PpoSnakeAgent(
                settings=settings,
                artifact_dir=artifact_dir,
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
                legacy_model_path=Path(tmpdir) / "state" / "ppo_snake_model.zip",
            )
            agent.model = _FakeModel(num_timesteps=55)
            fake_vec = MagicMock()
            fake_vec.save.side_effect = lambda path: Path(path).write_text("fake-vec", encoding="utf-8")
            agent._train_vecnormalize = fake_vec

            replace_calls: list[tuple[str, str]] = []
            real_replace = os.replace

            def _spy_replace(src: str, dst: str) -> None:
                replace_calls.append((str(src), str(dst)))
                real_replace(src, dst)

            with (
                patch("snake_frame.ppo_agent.os.replace", side_effect=_spy_replace),
                patch("snake_frame.ppo_agent.PPO.load", return_value=_FakeModel(num_timesteps=55)),
            ):
                ok = agent.save()
            self.assertTrue(ok)
            self.assertGreaterEqual(len(replace_calls), 5)  # model x2 + vec x2 + metadata
            self.assertTrue((artifact_dir / "last_model.zip").exists())
            self.assertTrue((artifact_dir / "resume_model.zip").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())
            tmp_files = list(artifact_dir.glob("*.tmp"))
            self.assertEqual(tmp_files, [])

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_init_rejects_invalid_rollout_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                PpoSnakeAgent(
                    settings=Settings(),
                    artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                    config=PpoConfig(env_count=1, n_steps=1),
                    reward_config=RewardConfig(),
                    obs_config=ObsConfig(),
                    autoload=False,
                )

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_init_rejects_batch_size_above_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                PpoSnakeAgent(
                    settings=Settings(),
                    artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                    config=PpoConfig(env_count=1, n_steps=16, batch_size=64),
                    reward_config=RewardConfig(),
                    obs_config=ObsConfig(),
                    autoload=False,
                )

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_init_rejects_truncated_minibatch_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                PpoSnakeAgent(
                    settings=Settings(),
                    artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                    config=PpoConfig(env_count=3, n_steps=128, batch_size=100),
                    reward_config=RewardConfig(),
                    obs_config=ObsConfig(),
                    autoload=False,
                )

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_init_rejects_negative_eval_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                PpoSnakeAgent(
                    settings=Settings(),
                    artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                    config=PpoConfig(env_count=1, n_steps=16, batch_size=16, eval_freq_steps=-1),
                    reward_config=RewardConfig(),
                    obs_config=ObsConfig(),
                    autoload=False,
                )

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_agent_init_rejects_non_positive_target_kl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                PpoSnakeAgent(
                    settings=Settings(),
                    artifact_dir=Path(tmpdir) / "state" / "ppo" / "baseline",
                    config=PpoConfig(env_count=1, n_steps=16, batch_size=16, target_kl=0.0),
                    reward_config=RewardConfig(),
                    obs_config=ObsConfig(),
                    autoload=False,
                )

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_load_prefers_resume_model_over_checkpoint_when_both_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "state" / "ppo" / "baseline"
            checkpoints_dir = artifact_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            last_model = artifact_dir / "last_model.zip"
            resume_model = artifact_dir / "resume_model.zip"
            ckpt_model = checkpoints_dir / "step_100_steps.zip"
            last_model.write_text("last", encoding="utf-8")
            resume_model.write_text("resume", encoding="utf-8")
            ckpt_model.write_text("checkpoint", encoding="utf-8")

            class _Loaded:
                def __init__(self) -> None:
                    self.observation_space = SimpleNamespace(shape=(11,))
                    self.num_timesteps = 100

            load_calls: list[str] = []

            def _load(path: str, device: str = "cpu", **kwargs):
                _ = device
                _ = kwargs
                load_calls.append(str(path))
                return _Loaded()

            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=artifact_dir,
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )
            with patch("snake_frame.ppo_agent.PPO.load", side_effect=_load):
                result = agent.load_if_exists_detailed(selector="last")
            self.assertTrue(result.ok)
            self.assertEqual(Path(load_calls[0]).name, "last_model.zip")
            self.assertEqual(Path(load_calls[1]).name, "resume_model.zip")
            self.assertNotIn("step_100_steps.zip", [Path(v).name for v in load_calls])

    @unittest.skipIf(_AGENT_IMPORT_ERROR is not None, _SKIP_REASON)
    def test_load_latest_checkpoint_reports_incompatible_observation_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "state" / "ppo" / "baseline"
            checkpoints_dir = artifact_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            (checkpoints_dir / "step_100_steps.zip").write_text("checkpoint", encoding="utf-8")

            class _Loaded:
                def __init__(self) -> None:
                    self.observation_space = SimpleNamespace(shape=(99,))
                    self.num_timesteps = 100

            agent = PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=artifact_dir,
                config=PpoConfig(env_count=1),
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )
            with patch("snake_frame.ppo_agent.PPO.load", return_value=_Loaded()):
                result = agent.load_latest_checkpoint_detailed()
            self.assertFalse(result.ok)
            self.assertEqual(result.code, ModelOpCode.INCOMPATIBLE)


if __name__ == "__main__":
    unittest.main()
