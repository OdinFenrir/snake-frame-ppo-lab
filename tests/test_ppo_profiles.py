from __future__ import annotations

import unittest

from snake_frame.settings import ppo_profile_config


class TestPpoProfiles(unittest.TestCase):
    def test_research_long_profile_matches_expected_defaults(self) -> None:
        cfg = ppo_profile_config("research_long", seed=1337)
        self.assertEqual(cfg.env_count, 16)
        self.assertEqual(cfg.n_steps, 2048)
        self.assertEqual(cfg.batch_size, 1024)
        self.assertEqual(cfg.n_epochs, 10)
        self.assertAlmostEqual(cfg.gamma, 0.99)
        self.assertAlmostEqual(cfg.gae_lambda, 0.95)
        self.assertAlmostEqual(cfg.learning_rate_start, 3e-4)
        self.assertAlmostEqual(cfg.learning_rate_end, 1e-5)
        self.assertAlmostEqual(cfg.ent_coef_start, 0.02)
        self.assertAlmostEqual(cfg.ent_coef_end, 5e-4)
        self.assertEqual(cfg.eval_freq_steps, 50_000)
        self.assertEqual(cfg.eval_episodes, 20)
        self.assertEqual(cfg.seed, 1337)

    def test_fast_profile_is_small(self) -> None:
        cfg = ppo_profile_config("fast", seed=7)
        self.assertEqual(cfg.env_count, 8)
        self.assertFalse(cfg.use_subproc_env)
        self.assertEqual(cfg.n_steps, 256)
        self.assertEqual(cfg.batch_size, 256)
        self.assertEqual(cfg.n_epochs, 2)
        self.assertEqual(cfg.seed, 7)
        self.assertFalse(cfg.use_stop_on_no_improvement)

    def test_app_alias_maps_to_default_profile(self) -> None:
        cfg = ppo_profile_config("app", seed=11)
        self.assertEqual(cfg.seed, 11)
        self.assertFalse(cfg.use_subproc_env)
        self.assertEqual(cfg.env_count, 8)
        self.assertEqual(cfg.n_steps, 1024)
        self.assertEqual(cfg.batch_size, 256)
        self.assertIsNone(cfg.policy_net_arch_pi)
        self.assertIsNone(cfg.policy_net_arch_vf)


if __name__ == "__main__":
    unittest.main()
