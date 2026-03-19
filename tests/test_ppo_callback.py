from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from snake_frame.ppo_agent import PpoSnakeAgent, _StopAndProgressCallback
except Exception as exc:  # pragma: no cover - env-dependent import gate
    raise RuntimeError(
        "ML dependency setup is required for PPO callback tests. "
        "Install with `pip install -r requirements.txt`."
    ) from exc

from snake_frame.settings import ObsConfig, PpoConfig, RewardConfig, Settings


class TestPpoCallback(unittest.TestCase):
    def test_episode_scores_are_emitted_on_done_infos(self) -> None:
        scores: list[int] = []
        callback = _StopAndProgressCallback(
            stop_flag=lambda: False,
            on_episode_score=scores.append,
        )
        callback.locals = {
            "dones": [False, True, True],
            "infos": [
                {"score": 1},
                {"score": 4},
                {"steps": 99},
            ],
        }
        callback.num_timesteps = 16
        self.assertTrue(callback._on_step())
        self.assertEqual(scores, [4])

    def test_episode_scores_are_emitted_from_gymnasium_final_info_dict(self) -> None:
        scores: list[int] = []
        callback = _StopAndProgressCallback(
            stop_flag=lambda: False,
            on_episode_score=scores.append,
        )
        callback.locals = {
            "dones": [False, False],
            "infos": {
                "final_info": [
                    {"score": 8, "death_reason": "body"},
                    None,
                ],
                "_final_info": [True, False],
            },
        }
        callback.num_timesteps = 32
        self.assertTrue(callback._on_step())
        self.assertEqual(scores, [8])

    def test_episode_scores_are_emitted_from_nested_final_info_in_list(self) -> None:
        scores: list[int] = []
        callback = _StopAndProgressCallback(
            stop_flag=lambda: False,
            on_episode_score=scores.append,
        )
        callback.locals = {
            "dones": [False],
            "infos": [
                {"final_info": {"score": 6, "death_reason": "wall"}},
            ],
        }
        callback.num_timesteps = 48
        self.assertTrue(callback._on_step())
        self.assertEqual(scores, [6])

    def test_missing_inference_model_is_bootstrapped_during_sync(self) -> None:
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=PpoConfig(env_count=1, n_steps=8, batch_size=4, n_epochs=2),
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )
        train_model = MagicMock()
        train_model.num_timesteps = 4096

        def _save_to_buffer(buffer) -> None:
            buffer.write(b"model-bytes")

        train_model.save.side_effect = _save_to_buffer
        agent.model = train_model
        agent.inference_model = None
        agent.request_inference_sync()

        loaded_model = MagicMock()
        with patch("snake_frame.ppo_agent.PPO.load", return_value=loaded_model) as mock_load:
            agent._maybe_refresh_inference(steps_done=4096)

        self.assertIs(agent.inference_model, loaded_model)
        self.assertFalse(agent.is_sync_pending)
        self.assertGreaterEqual(agent._last_inference_sync_steps, 4096)
        self.assertEqual(mock_load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
