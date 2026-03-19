from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from snake_frame.ppo_agent import ModelOpCode, ModelOpResult, PpoSnakeAgent
except Exception as exc:  # pragma: no cover - env-dependent import gate
    raise RuntimeError(
        "ML dependency setup is required for PPO agent tests. "
        "Install with `pip install -r requirements.txt`."
    ) from exc

from snake_frame.settings import ObsConfig, PpoConfig, RewardConfig, Settings


class _FakeVecEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestPpoAgentSeed(unittest.TestCase):
    def test_config_rejects_partial_split_policy_architecture(self) -> None:
        bad_config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            policy_net_arch_pi=(128, 64),
            policy_net_arch_vf=None,
        )
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            PpoSnakeAgent(
                settings=Settings(),
                artifact_dir=Path("state/ppo/baseline_seed_test"),
                config=bad_config,
                reward_config=RewardConfig(),
                obs_config=ObsConfig(),
                autoload=False,
            )

    def test_train_passes_seed_to_ppo_constructor(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            seed=12345,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )

        fake_model = MagicMock()
        fake_model.num_timesteps = 0

        def _fake_learn(*, total_timesteps: int, callback, reset_num_timesteps: bool) -> None:
            _ = (callback, reset_num_timesteps)
            fake_model.num_timesteps += int(total_timesteps)

        fake_model.learn.side_effect = _fake_learn

        with (
            patch.object(agent, "_save_last_and_stats", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch.object(agent, "load_if_exists_detailed"),
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model) as mock_ppo,
        ):
            agent.train(total_timesteps=32, stop_flag=lambda: False)

        kwargs = mock_ppo.call_args.kwargs
        self.assertEqual(kwargs.get("seed"), 12345)

    def test_train_passes_target_kl_to_ppo_constructor(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            target_kl=0.03,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )

        fake_model = MagicMock()
        fake_model.num_timesteps = 0
        fake_model.learn.side_effect = lambda *, total_timesteps, callback, reset_num_timesteps: setattr(
            fake_model, "num_timesteps", int(fake_model.num_timesteps) + int(total_timesteps)
        )

        with (
            patch.object(agent, "_save_last_and_stats", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch.object(agent, "load_if_exists_detailed", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model) as mock_ppo,
        ):
            agent.train(total_timesteps=16, stop_flag=lambda: False)

        kwargs = mock_ppo.call_args.kwargs
        self.assertEqual(float(kwargs.get("target_kl")), 0.03)

    def test_train_supports_split_policy_architecture(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            policy_net_arch=(64, 64),
            policy_net_arch_pi=(128, 64),
            policy_net_arch_vf=(64, 32),
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )

        fake_model = MagicMock()
        fake_model.num_timesteps = 0
        fake_model.learn.side_effect = lambda *, total_timesteps, callback, reset_num_timesteps: setattr(
            fake_model, "num_timesteps", int(fake_model.num_timesteps) + int(total_timesteps)
        )

        with (
            patch.object(agent, "_save_last_and_stats", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch.object(agent, "load_if_exists_detailed", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model) as mock_ppo,
        ):
            agent.train(total_timesteps=16, stop_flag=lambda: False)

        kwargs = mock_ppo.call_args.kwargs
        policy_kwargs = kwargs.get("policy_kwargs", {})
        net_arch = policy_kwargs.get("net_arch", {})
        self.assertEqual(net_arch.get("pi"), [128, 64])
        self.assertEqual(net_arch.get("vf"), [64, 32])

    def test_train_raises_when_post_train_save_fails(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            seed=12345,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )

        fake_model = MagicMock()
        fake_model.num_timesteps = 0
        fake_model.learn.side_effect = lambda *, total_timesteps, callback, reset_num_timesteps: setattr(
            fake_model, "num_timesteps", int(fake_model.num_timesteps) + int(total_timesteps)
        )

        with (
            patch.object(
                agent,
                "_save_last_and_stats",
                return_value=ModelOpResult(
                    ok=False,
                    code=ModelOpCode.FILESYSTEM_ERROR,
                    detail="disk full",
                ),
            ),
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model),
        ):
            with self.assertRaises(RuntimeError):
                agent.train(total_timesteps=32, stop_flag=lambda: False)

    def test_train_disables_eval_and_checkpoint_callbacks_when_freq_is_zero(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            eval_freq_steps=0,
            checkpoint_freq_steps=0,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )

        fake_model = MagicMock()
        fake_model.num_timesteps = 0
        fake_model.learn.side_effect = lambda *, total_timesteps, callback, reset_num_timesteps: setattr(
            fake_model, "num_timesteps", int(fake_model.num_timesteps) + int(total_timesteps)
        )

        with (
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model),
            patch("snake_frame.ppo_agent._SyncEvalCallback") as mock_eval_cb,
            patch("snake_frame.ppo_agent.CheckpointCallback") as mock_ckpt_cb,
            patch.object(
                agent,
                "_save_last_and_stats",
                return_value=ModelOpResult(ok=True, code=ModelOpCode.OK),
            ),
            patch.object(agent, "load_if_exists_detailed", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch.object(agent, "_make_eval_vec_env", side_effect=AssertionError("eval env should not be built")),
        ):
            steps = agent.train(total_timesteps=16, stop_flag=lambda: False)
        self.assertEqual(int(steps), 16)
        self.assertEqual(mock_eval_cb.call_count, 0)
        self.assertEqual(mock_ckpt_cb.call_count, 0)

    def test_save_detailed_propagates_reload_failure(self) -> None:
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=PpoConfig(env_count=1, n_steps=8, batch_size=4, n_epochs=2),
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )
        with (
            patch.object(agent, "_save_last_and_stats", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
            patch.object(agent, "_ensure_best_model_artifact"),
            patch.object(
                agent,
                "load_if_exists_detailed",
                return_value=ModelOpResult(ok=False, code=ModelOpCode.CORRUPT, detail="bad artifact"),
            ),
        ):
            result = agent.save_detailed()
        self.assertFalse(result.ok)
        self.assertEqual(result.code, ModelOpCode.CORRUPT)

    def test_train_preserves_original_learn_exception_and_cleans_env(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            eval_freq_steps=0,
            checkpoint_freq_steps=0,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )
        fake_model = MagicMock()
        fake_model.num_timesteps = 0
        fake_model.learn.side_effect = RuntimeError("learn failed")
        fake_env = _FakeVecEnv()
        with (
            patch.object(agent, "_make_train_vec_env", return_value=fake_env),
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model),
        ):
            with self.assertRaisesRegex(RuntimeError, "learn failed"):
                agent.train(total_timesteps=16, stop_flag=lambda: False)
        self.assertTrue(fake_env.closed)
        self.assertIsNone(agent._train_vecnormalize)

    def test_train_closes_env_when_model_constructor_fails(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            eval_freq_steps=0,
            checkpoint_freq_steps=0,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )
        fake_env = _FakeVecEnv()
        with (
            patch.object(agent, "_make_train_vec_env", return_value=fake_env),
            patch("snake_frame.ppo_agent.PPO", side_effect=RuntimeError("constructor failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                agent.train(total_timesteps=16, stop_flag=lambda: False)
        self.assertTrue(fake_env.closed)
        self.assertIsNone(agent._train_vecnormalize)

    def test_train_with_eval_disabled_keeps_prior_eval_metadata(self) -> None:
        config = PpoConfig(
            env_count=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            eval_freq_steps=0,
            checkpoint_freq_steps=0,
        )
        agent = PpoSnakeAgent(
            settings=Settings(),
            artifact_dir=Path("state/ppo/baseline_seed_test"),
            config=config,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(),
            autoload=False,
        )
        agent._best_eval_score = 12.5
        agent._best_eval_step = 1024
        agent._last_eval_score = 11.0
        agent._eval_runs_completed = 7

        fake_model = MagicMock()
        fake_model.num_timesteps = 100
        fake_model.learn.side_effect = lambda *, total_timesteps, callback, reset_num_timesteps: setattr(
            fake_model, "num_timesteps", int(fake_model.num_timesteps) + int(total_timesteps)
        )

        with (
            patch("snake_frame.ppo_agent.PPO", return_value=fake_model),
            patch.object(
                agent,
                "_save_last_and_stats",
                return_value=ModelOpResult(ok=True, code=ModelOpCode.OK),
            ),
            patch.object(agent, "load_if_exists_detailed", return_value=ModelOpResult(ok=True, code=ModelOpCode.OK)),
        ):
            agent.train(total_timesteps=16, stop_flag=lambda: False)

        self.assertEqual(agent.best_eval_score, 12.5)
        self.assertEqual(agent.best_eval_step, 1024)
        self.assertEqual(agent.last_eval_score, 11.0)
        self.assertEqual(agent.eval_runs_completed, 7)


if __name__ == "__main__":
    unittest.main()
