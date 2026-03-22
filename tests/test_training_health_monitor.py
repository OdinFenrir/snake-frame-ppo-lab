"""One-off test for TrainingHealthMonitor"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock


class TestTrainingHealthMonitor(unittest.TestCase):
    def test_import(self) -> None:
        """Verify class can be imported."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        self.assertIsNotNone(TrainingHealthMonitor)
    
    def test_nan_loss_detected(self) -> None:
        """NaN in loss should trigger failure."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor()
        monitor.model = MagicMock()
        
        # Mock logger with NaN value
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/policy_loss": float('nan'),
            "train/value_loss": 0.5,
            "train/entropy_loss": -0.1,
            "train/approx_kl": 0.01,
            "train/clip_fraction": 0.1,
            "rollout/ep_rew_mean": 10.0,
        }
        monitor.model.logger = mock_logger
        
        # Run health check
        status = monitor._check_health(monitor._extract_rollout_metrics())
        
        self.assertFalse(status.healthy)
        self.assertEqual(status.failure_reason, "nan_loss")
    
    def test_inf_loss_detected(self) -> None:
        """Inf in loss should trigger failure."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor()
        monitor.model = MagicMock()
        
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/policy_loss": float('inf'),
            "train/value_loss": 0.5,
        }
        monitor.model.logger = mock_logger
        
        status = monitor._check_health(monitor._extract_rollout_metrics())
        
        self.assertFalse(status.healthy)
        self.assertEqual(status.failure_reason, "nan_loss")
    
    def test_kl_divergence_persistent_fail(self) -> None:
        """KL > 0.2 for 2+ rollouts should trigger failure."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor(
            kl_fail_threshold=0.2,
            kl_persistence_required=2,
        )
        monitor.model = MagicMock()
        monitor._rollout_count = 2
        
        # First rollout - KL elevated (first call increments counter)
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/approx_kl": 0.25,
            "train/value_loss": 0.5,
        }
        monitor.model.logger = mock_logger
        
        # First call - increments _kl_fail_count
        status1 = monitor._check_health(monitor._extract_rollout_metrics())
        self.assertTrue(status1.healthy)  # Not yet persistent
        
        # Second call - now persistent, should fail
        status2 = monitor._check_health(monitor._extract_rollout_metrics())
        
        self.assertFalse(status2.healthy)
        self.assertEqual(status2.failure_reason, "kl_divergence_persistent")
    
    def test_kl_warning_under_threshold(self) -> None:
        """KL > 0.1 but < 0.2 should warn but not fail."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor(
            kl_fail_threshold=0.2,
            kl_warn_threshold=0.1,
        )
        monitor.model = MagicMock()
        monitor._rollout_count = 5  # After warmup
        
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/approx_kl": 0.15,
            "train/value_loss": 0.5,
        }
        monitor.model.logger = mock_logger
        
        status = monitor._check_health(monitor._extract_rollout_metrics())
        
        self.assertTrue(status.healthy)
        # Warning includes value: "KL divergence elevated: 0.1500"
        self.assertTrue(any("KL divergence" in w for w in status.warnings))
    
    def test_explained_variance_fail(self) -> None:
        """Explained variance < -5.0 after warmup should trigger failure."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor(
            explained_variance_fail=-5.0,
        )
        monitor.model = MagicMock()
        
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/explained_variance": -6.0,  # Below -5.0 threshold
            "train/value_loss": 0.5,
        }
        monitor.model.logger = mock_logger
        monitor._rollout_count = 5  # After warmup
        
        status = monitor._check_health(monitor._extract_rollout_metrics())
        
        self.assertFalse(status.healthy)
        self.assertEqual(status.failure_reason, "explained_variance_negative")
    
    def test_explained_variance_ok_during_warmup(self) -> None:
        """Explained variance < -5.0 during warmup should NOT trigger failure."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor(
            explained_variance_fail=-5.0,
        )
        monitor.model = MagicMock()
        
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/explained_variance": -10.0,  # Way below threshold
            "train/value_loss": 0.5,
        }
        monitor.model.logger = mock_logger
        monitor._rollout_count = 2  # During warmup
        
        status = monitor._check_health(monitor._extract_rollout_metrics())
        
        # Should be healthy during warmup
        self.assertTrue(status.healthy)
    
    def test_healthy_training(self) -> None:
        """Healthy training should pass all checks."""
        from snake_frame.ppo_agent import TrainingHealthMonitor
        
        monitor = TrainingHealthMonitor()
        monitor.model = MagicMock()
        monitor._rollout_count = 5
        
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/policy_loss": 0.5,
            "train/value_loss": 0.3,
            "train/entropy_loss": -1.5,
            "train/approx_kl": 0.02,
            "train/clip_fraction": 0.15,
            "train/explained_variance": 0.8,
            "rollout/ep_rew_mean": 25.0,
        }
        monitor.model.logger = mock_logger
        
        status = monitor._check_health(monitor._extract_rollout_metrics())
        
        self.assertTrue(status.healthy)
        self.assertEqual(status.failure_reason, "")
    
    def test_missing_metrics_after_warmup(self) -> None:
        """Missing metrics after warmup should warn but not fail (SB3 dump_logs timing artifact)."""
        from snake_frame.ppo_agent import TrainingHealthMonitor

        monitor = TrainingHealthMonitor(warmup_rollouts=3)
        monitor.model = MagicMock()
        monitor._rollout_count = 5
        monitor._train_has_completed = True

        mock_logger = MagicMock()
        mock_logger.name_to_value = {}
        monitor.model.logger = mock_logger

        status = monitor._check_health({})

        self.assertTrue(status.healthy)
        self.assertEqual(status.failure_reason, "")
        self.assertTrue(any("dump_logs" in w for w in status.warnings))

    def test_missing_metrics_before_train_completed(self) -> None:
        """Missing metrics before train() has completed should warn but not fail."""
        from snake_frame.ppo_agent import TrainingHealthMonitor

        monitor = TrainingHealthMonitor(warmup_rollouts=3)
        monitor.model = MagicMock()
        monitor._rollout_count = 2
        monitor._train_has_completed = False

        status = monitor._check_health({})

        self.assertTrue(status.healthy)
        self.assertIn("train() not completed", status.warnings[0])

    def test_missing_metrics_after_train_completed(self) -> None:
        """Missing metrics after train() has completed should warn (SB3 timing), not fail."""
        from snake_frame.ppo_agent import TrainingHealthMonitor

        monitor = TrainingHealthMonitor(warmup_rollouts=3)
        monitor.model = MagicMock()
        monitor._rollout_count = 5
        monitor._train_has_completed = True

        status = monitor._check_health({})

        self.assertTrue(status.healthy)
        self.assertEqual(status.failure_reason, "")
        self.assertTrue(any("dump_logs" in w for w in status.warnings))


if __name__ == "__main__":
    unittest.main()
