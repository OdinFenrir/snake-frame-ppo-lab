from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from snake_frame.app_actions import AppActions
from snake_frame.app_state import AppState
from snake_frame.diagnostics import DiagnosticsBundleResult
from snake_frame.state_io import UiStateErrorCode
from snake_frame.ui_state_model import ModelState, TrainingState, UIStateModel


@dataclass
class _FakeSnapshot:
    active: bool = False
    target_steps: int = 0
    start_steps: int = 0
    current_steps: int = 0
    last_error: str | None = None
    stop_requested: bool = False
    best_eval_score: float | None = None
    best_eval_step: int = 0
    last_eval_score: float | None = None
    eval_runs_completed: int = 0

    @property
    def done_steps(self) -> int:
        return max(0, int(self.current_steps) - int(self.start_steps))


class _FakeTraining:
    def __init__(self) -> None:
        self._snapshot = _FakeSnapshot()
        self.start_result = True
        self.poll_message: str | None = None
        self.stop_called = False
        self.reset_called = False

    def snapshot(self) -> _FakeSnapshot:
        return self._snapshot

    def start(self, target_steps: int) -> bool:
        self._snapshot.active = bool(self.start_result)
        self._snapshot.target_steps = int(target_steps)
        self._snapshot.start_steps = int(self._snapshot.current_steps)
        return bool(self.start_result)

    def stop(self) -> None:
        self.stop_called = True
        self._snapshot.active = False

    def reset_tracking_from_agent(self) -> None:
        self.reset_called = True
        self._snapshot.active = False
        self._snapshot.current_steps = 0
        self._snapshot.start_steps = 0
        self._snapshot.target_steps = 0

    def poll_completion(self) -> str | None:
        value = self.poll_message
        self.poll_message = None
        return value


class _FakeAgent:
    def __init__(self) -> None:
        self.device = "cpu"
        self.is_ready = False
        self.is_inference_available = False
        self.is_sync_pending = False
        self.best_eval_score = None
        self.best_eval_step = 0
        self.last_eval_score = None
        self.eval_runs_completed = 0
        self.sync_requested = False
        self.adaptive_reward_enabled = True
        self.save_called = False
        self.save_result = True
        self.load_called = False
        self.load_checkpoint_called = False
        self.delete_called = False
        self.load_result = False
        self.load_checkpoint_result = False
        self.delete_result = False
        self.save_code = "ok"
        self.load_code = "ok"
        self.load_checkpoint_code = "ok"
        self.delete_code = "ok"

    def request_inference_sync(self) -> None:
        self.sync_requested = True

    def save(self) -> bool:
        self.save_called = True
        return bool(self.save_result)

    def save_detailed(self):
        self.save_called = True
        return type("R", (), {"ok": bool(self.save_result), "code": self.save_code})()

    def load_if_exists(self) -> bool:
        self.load_called = True
        self.is_ready = bool(self.load_result)
        return bool(self.load_result)

    def load_if_exists_detailed(self):
        self.load_called = True
        self.is_ready = bool(self.load_result)
        return type("R", (), {"ok": bool(self.load_result), "code": self.load_code})()

    def load_latest_checkpoint_detailed(self):
        self.load_checkpoint_called = True
        self.is_ready = bool(self.load_checkpoint_result)
        detail = "loaded checkpoint step 50000" if self.load_checkpoint_result else ""
        return type(
            "R",
            (),
            {"ok": bool(self.load_checkpoint_result), "code": self.load_checkpoint_code, "detail": detail},
        )()

    def delete(self) -> bool:
        self.delete_called = True
        self.is_ready = False
        return bool(self.delete_result)

    def delete_detailed(self):
        self.delete_called = True
        self.is_ready = False
        return type("R", (), {"ok": bool(self.delete_result), "code": self.delete_code})()

    def is_adaptive_reward_enabled(self) -> bool:
        return bool(self.adaptive_reward_enabled)

    def set_adaptive_reward_enabled(self, enabled: bool) -> None:
        self.adaptive_reward_enabled = bool(enabled)


class _FakeGame:
    def __init__(self) -> None:
        self.episode_scores: list[int] = []
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True


class _FakeNumericInput:
    def __init__(self, value: str = "500000") -> None:
        self.value = str(value)

    def as_int(self, minimum: int = 1, maximum: int = 100000) -> int:
        try:
            parsed = int(str(self.value).strip() or "0")
        except Exception:
            parsed = minimum
        return max(minimum, min(maximum, parsed))


class TestAppActions(unittest.TestCase):
    def test_train_start_and_stop_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AppState()
            actions = AppActions(
                app_state=state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput("1234"),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            actions.on_train_start_clicked()
            self.assertIn("PPO training started", state.status_text)
            actions.on_train_stop_clicked()
            self.assertIn("stop requested", state.status_text.lower())

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            app_state = AppState(
                game_running=True,
                space_strategy_enabled=False,
                debug_overlay=True,
                debug_reachable_overlay=True,
                training_episode_scores=[3, 7, 9],
            )
            app_state.training_death_counts = {"wall": 2, "body": 3, "starvation": 1, "fill": 0, "none": 0, "other": 4}
            game = _FakeGame()
            game.episode_scores = [3, 7, 9]
            agent = _FakeAgent()
            agent.adaptive_reward_enabled = False
            training = _FakeTraining()
            training._snapshot.current_steps = 321
            training._snapshot.target_steps = 999
            generations = _FakeNumericInput("100")
            theme_name = {"value": "retro_forest_noir"}

            actions = AppActions(
                app_state=app_state,
                game=game,
                agent=agent,
                training=training,
                generations_input=generations,
                state_file=state_file,
                get_theme_name=lambda: str(theme_name["value"]),
                set_theme_name=lambda value: theme_name.__setitem__("value", str(value)),
            )
            actions.handle_save_clicked()
            self.assertTrue(state_file.exists())
            self.assertTrue(agent.save_called)
            self.assertIn('"themeName"', state_file.read_text(encoding="utf-8"))

            game.episode_scores = []
            app_state.game_running = False
            generations.value = "1"
            app_state.space_strategy_enabled = True
            app_state.debug_overlay = False
            app_state.debug_reachable_overlay = False
            app_state.training_episode_scores = []
            app_state.training_death_counts = {"wall": 0, "body": 0, "starvation": 0, "fill": 0, "none": 0, "other": 0}
            agent.adaptive_reward_enabled = True
            agent.load_result = True
            theme_name["value"] = "terminal_sunset"
            actions.handle_load_clicked()
            self.assertTrue(agent.load_called)
            self.assertTrue(training.reset_called)
            self.assertEqual(game.episode_scores, [3, 7, 9])
            self.assertEqual(app_state.training_episode_scores, [3, 7, 9])
            self.assertEqual(app_state.training_death_counts, {"wall": 2, "body": 3, "starvation": 1, "fill": 0, "none": 0, "other": 4})
            self.assertEqual(generations.value, "999")
            self.assertTrue(app_state.game_running)
            self.assertFalse(app_state.space_strategy_enabled)
            self.assertTrue(app_state.debug_overlay)
            self.assertTrue(app_state.debug_reachable_overlay)
            self.assertFalse(agent.adaptive_reward_enabled)
            self.assertEqual(theme_name["value"], "retro_forest_noir")

    def test_mutation_guard_while_training_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            training = _FakeTraining()
            training._snapshot.active = True
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=training,
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            actions.handle_delete_clicked()
            self.assertIn("Cannot delete while training is active", app_state.status_text)

    def test_mutation_guard_uses_ui_state_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            training = _FakeTraining()
            training._snapshot.active = False
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=training,
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
                ui_state_provider=lambda: UIStateModel(
                    model_state=ModelState.READY,
                    training_state=TrainingState.RUNNING,
                    game_running=True,
                ),
            )
            actions.handle_save_clicked()
            self.assertIn("Cannot save while training is active", app_state.status_text)

    def test_save_status_when_model_save_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            agent = _FakeAgent()
            agent.save_result = False
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=agent,
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            actions.handle_save_clicked()
            self.assertIn("model save failed", app_state.status_text)

    def test_poll_completion_updates_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            training = _FakeTraining()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=training,
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            training.poll_message = "Training complete"
            actions.poll_training_state()
            self.assertEqual(app_state.last_train_message, "Training complete")
            self.assertEqual(app_state.status_text, "Training complete")

    def test_load_corrupted_ui_state_shows_status_and_still_loads_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            state_file.write_text("{bad-json", encoding="utf-8")
            app_state = AppState()
            game = _FakeGame()
            agent = _FakeAgent()
            agent.load_result = True
            training = _FakeTraining()
            actions = AppActions(
                app_state=app_state,
                game=game,
                agent=agent,
                training=training,
                generations_input=_FakeNumericInput(),
                state_file=state_file,
            )
            actions.handle_load_clicked()
            self.assertTrue(agent.load_called)
            self.assertTrue(training.reset_called)
            self.assertIn("saved UI is invalid/corrupted", app_state.status_text)

    def test_load_ignores_invalid_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            state_file.write_text(
                '{"episodeScores":["10","bad",null,5],"trainingTarget":"oops","gameRunning":1}',
                encoding="utf-8",
            )
            app_state = AppState()
            game = _FakeGame()
            training = _FakeTraining()
            actions = AppActions(
                app_state=app_state,
                game=game,
                agent=_FakeAgent(),
                training=training,
                generations_input=_FakeNumericInput("1234"),
                state_file=state_file,
            )
            actions.handle_load_clicked()
            self.assertTrue(training.reset_called)
            self.assertEqual(game.episode_scores, [10, 5])
            self.assertEqual(actions.generations_input.value, "1234")
            self.assertTrue(app_state.game_running)

    def test_load_filesystem_error_sets_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=state_file,
            )
            with patch("snake_frame.app_actions.load_ui_state_result", side_effect=OSError("io")):
                actions.handle_load_clicked()
            self.assertIn("filesystem error", app_state.status_text.lower())

    def test_delete_filesystem_error_sets_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=state_file,
            )
            with patch("snake_frame.app_actions.delete_ui_state_result", side_effect=OSError("io")):
                actions.handle_delete_clicked()
            self.assertIn("filesystem error", app_state.status_text.lower())

    def test_restart_requests_sync_resets_game_and_starts_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState(game_running=False)
            game = _FakeGame()
            agent = _FakeAgent()
            actions = AppActions(
                app_state=app_state,
                game=game,
                agent=agent,
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            actions.on_restart_clicked()
            self.assertTrue(agent.sync_requested)
            self.assertTrue(game.reset_called)
            self.assertTrue(app_state.game_running)
            self.assertIn("restarted", app_state.status_text.lower())

    def test_adaptive_toggle_flips_agent_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            game = _FakeGame()
            agent = _FakeAgent()
            actions = AppActions(
                app_state=app_state,
                game=game,
                agent=agent,
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            self.assertTrue(agent.is_adaptive_reward_enabled())
            actions.on_adaptive_toggle_clicked()
            self.assertFalse(agent.is_adaptive_reward_enabled())
            self.assertIn("adaptive reward off", app_state.status_text.lower())

    def test_debug_and_reachable_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            actions.on_debug_toggle_clicked()
            self.assertTrue(app_state.debug_overlay)
            self.assertIn("debug overlay on", app_state.status_text.lower())
            actions.on_reachable_toggle_clicked()
            self.assertTrue(app_state.debug_reachable_overlay)
            self.assertIn("reachable overlay on", app_state.status_text.lower())

    def test_status_lines_show_paused_loading_snapshot_when_model_loaded_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState(game_running=True)
            training = _FakeTraining()
            agent = _FakeAgent()
            agent.is_ready = True
            agent.is_inference_available = False
            agent.is_sync_pending = True
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=agent,
                training=training,
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            lines = actions.build_status_lines()
            self.assertTrue(any("Control: paused (loading snapshot)" in line for line in lines))

    def test_load_v1_payload_migrates_theme_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            state_file.write_text('{"uiStateVersion":1,"episodeScores":[1,2]}', encoding="utf-8")
            app_state = AppState()
            theme_name = {"value": "terminal_sunset"}
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=state_file,
                set_theme_name=lambda value: theme_name.__setitem__("value", str(value)),
            )
            actions.handle_load_clicked()
            self.assertEqual(theme_name["value"], "retro_forest_noir")
            self.assertEqual(app_state.ui_state_version, 2)

    def test_load_unsupported_schema_sets_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            state_file.write_text('{"uiStateVersion":99}', encoding="utf-8")
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=state_file,
            )
            actions.handle_load_clicked()
            self.assertIn("unsupported", app_state.status_text.lower())
            self.assertEqual(app_state.last_error_code, "io_schema_unsupported")

    def test_save_partial_write_recovery_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            with patch(
                "snake_frame.app_actions.save_ui_state_result",
                return_value=type(
                    "R",
                    (),
                    {"ok": False, "error_code": UiStateErrorCode.PARTIAL_WRITE},
                )(),
            ):
                actions.handle_save_clicked()
            self.assertIn("partial write recovered", app_state.status_text.lower())

    def test_diagnostics_bundle_action_sets_success_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=state_file,
            )
            with patch(
                "snake_frame.app_actions.create_diagnostics_bundle",
                return_value=Path(tmpdir) / "diag.zip",
            ):
                actions.handle_diagnostics_clicked()
            self.assertIn("diagnostics bundle created", app_state.status_text.lower())

    def test_diagnostics_bundle_cleanup_warning_sets_last_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "ui_state.json"
            app_state = AppState()
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=_FakeAgent(),
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=state_file,
            )
            with patch(
                "snake_frame.app_actions.create_diagnostics_bundle",
                return_value=DiagnosticsBundleResult(
                    bundle_path=Path(tmpdir) / "diag.zip",
                    cleanup_warnings=("stale cleanup failed",),
                    error_codes=("diagnostics_cleanup_failed",),
                ),
            ):
                actions.handle_diagnostics_clicked()
            self.assertEqual(app_state.last_error_code, "diagnostics_cleanup_failed")

    def test_status_lines_include_runtime_health_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState(game_running=True)
            agent = _FakeAgent()
            agent.is_ready = True
            agent.is_inference_available = True
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=agent,
                training=_FakeTraining(),
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            lines = actions.build_status_lines()
            self.assertTrue(any(line.startswith("Health: ") for line in lines))

    def test_load_latest_checkpoint_action_updates_status_and_resets_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_state = AppState()
            training = _FakeTraining()
            agent = _FakeAgent()
            agent.load_checkpoint_result = True
            actions = AppActions(
                app_state=app_state,
                game=_FakeGame(),
                agent=agent,
                training=training,
                generations_input=_FakeNumericInput(),
                state_file=Path(tmpdir) / "ui_state.json",
            )
            actions.handle_load_latest_checkpoint_clicked()
            self.assertTrue(agent.load_checkpoint_called)
            self.assertTrue(training.reset_called)
            self.assertIn("loaded latest checkpoint", app_state.status_text.lower())


if __name__ == "__main__":
    unittest.main()
