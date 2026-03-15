from __future__ import annotations

import unittest
from unittest.mock import patch

from snake_frame.gameplay_controller import ControlMode, GameplayController
from snake_frame.settings import DynamicControlConfig, ObsConfig, Settings


class _FakeGame:
    def __init__(self) -> None:
        self.game_over = False
        self.snake = [(10, 10), (9, 10), (8, 10)]
        self.direction = (1, 0)
        self.food = (14, 10)
        self.death_reason = "none"
        self.steps_without_food = 0
        self.queued: list[tuple[int, int]] = []
        self.reset_called = False
        self.update_called = False
        self.advance_next = True

    def queue_direction(self, dx: int, dy: int) -> None:
        self.queued.append((int(dx), int(dy)))

    def reset(self) -> None:
        self.reset_called = True

    def update(self) -> None:
        self.update_called = True

    def starvation_limit(self) -> int:
        return 800

    def will_advance_on_next_update(self) -> bool:
        return bool(self.advance_next)


class _FakeAgent:
    def __init__(self) -> None:
        self.is_ready = False
        self.is_inference_available = False
        self.is_sync_pending = False
        self.sync_requested = False
        self.predicted_action = 0
        self.predict_calls = 0
        self.predict_with_probs_calls = 0

    def request_inference_sync(self) -> None:
        self.sync_requested = True

    def predict_action(self, _obs, action_masks=None) -> int:
        _ = action_masks
        self.predict_calls += 1
        return int(self.predicted_action)

    def predict_action_with_probs(self, _obs, action_masks=None):
        _ = action_masks
        self.predict_with_probs_calls += 1
        return int(self.predicted_action), (0.34, 0.33, 0.33)


class TestGameplayController(unittest.TestCase):
    def test_step_ignores_when_not_running(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.step(False)
        self.assertFalse(game.update_called)
        self.assertFalse(game.reset_called)

    def test_step_resets_when_game_over(self) -> None:
        game = _FakeGame()
        game.game_over = True
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.step(True)
        self.assertTrue(agent.sync_requested)
        self.assertTrue(game.reset_called)
        self.assertFalse(game.update_called)

    def test_step_updates_and_controls_when_ready(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 2  # turn right
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.step(True)
        self.assertTrue(game.update_called)
        self.assertTrue(len(game.queued) >= 1)

    def test_safety_override_avoids_immediate_collision(self) -> None:
        game = _FakeGame()
        game.snake = [(19, 10), (18, 10), (17, 10)]
        game.direction = (1, 0)
        game.food = (19, 9)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0  # straight would hit wall
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl.step(True)
        self.assertTrue(game.update_called)
        self.assertTrue(game.queued)
        self.assertNotEqual(game.queued[-1], (1, 0))

    def test_safety_override_does_not_change_safe_agent_action(self) -> None:
        game = _FakeGame()
        game.snake = [(10, 10), (9, 10), (8, 10)]
        game.direction = (1, 0)
        game.food = (14, 10)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl.step(True)
        self.assertTrue(game.queued)
        self.assertEqual(game.queued[-1], (1, 0))

    def test_safety_override_prioritizes_tail_reachability(self) -> None:
        game = _FakeGame()
        game.snake = [(10, 10), (9, 10), (8, 10), (7, 10)]
        game.direction = (1, 0)
        game.food = (14, 10)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )

        # Force all options to appear equally reachable, but mark straight as tail-unreachable.
        with (
            patch.object(ctrl, "_reachable_space", return_value=40),
            patch.object(
                ctrl,
                "_tail_is_reachable",
                side_effect=lambda _cells, snake_after_move: snake_after_move[0] != (11, 10),
            ),
        ):
            ctrl.step(True)
        self.assertTrue(game.queued)
        self.assertNotEqual(game.queued[-1], (1, 0))

    def test_telemetry_tracks_interventions_and_deaths(self) -> None:
        game = _FakeGame()
        game.snake = [(19, 10), (18, 10), (17, 10)]
        game.direction = (1, 0)
        game.food = (10, 10)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl.step(True)
        snap = ctrl.telemetry_snapshot()
        self.assertGreaterEqual(snap.decisions_total, 1)
        self.assertGreaterEqual(snap.interventions_total, 1)

    def test_decision_trace_snapshot_contains_core_fields(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl.set_debug_options(debug_overlay=True, reachable_overlay=False)
        ctrl.step(True)
        row = ctrl.decision_trace_snapshot()
        required = {
            "decision_index",
            "predicted_action",
            "chosen_action",
            "override_used",
            "mode",
            "switch_reason",
            "free_ratio",
            "food_pressure",
            "proposed_viable",
            "chosen_tail_reachable",
            "chosen_capacity_shortfall",
            "risk_guard_candidate",
            "risk_guard_eligible",
            "risk_guard_blockers",
            "interventions_total",
            "pocket_risk_total",
        }
        self.assertTrue(required.issubset(set(row.keys())))
        self.assertIsInstance(row.get("risk_guard_blockers"), list)

    def test_telemetry_tracks_starvation_death_reason(self) -> None:
        game = _FakeGame()
        game.game_over = True
        game.death_reason = "starvation"
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl.step(True)
        snap = ctrl.telemetry_snapshot()
        self.assertGreaterEqual(snap.deaths_starvation, 1)
        self.assertEqual(str(snap.last_death_reason), "starvation")

    def test_starvation_death_does_not_increment_stuck_episode_counter(self) -> None:
        game = _FakeGame()
        game.game_over = True
        game.death_reason = "starvation"
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._episode_stuck = True
        ctrl.step(True)
        snap = ctrl.telemetry_snapshot()
        self.assertEqual(int(snap.stuck_episodes_total), 0)

    def test_space_strategy_toggle_setter_and_getter(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
            space_strategy_enabled=True,
        )
        self.assertTrue(ctrl.is_space_strategy_enabled())
        ctrl.set_space_strategy_enabled(False)
        self.assertFalse(ctrl.is_space_strategy_enabled())

    def test_escape_controller_used_when_triggered(self) -> None:
        game = _FakeGame()
        # Dense fake body to ensure crowded/endgame trigger is plausible.
        game.snake = [(x % 6, x // 6) for x in range(20)]
        game.direction = (1, 0)
        game.food = (5, 5)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(board_cells=6, agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )

        with (
            patch.object(ctrl, "_select_mode", return_value=ControlMode.ESCAPE),
            patch.object(ctrl._escape_controller, "choose_action", return_value=2) as choose_mock,
            patch.object(ctrl, "_evaluate_action", return_value=(0.0, True, 0)),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(action, 2)
        choose_mock.assert_called_once()

    def test_step_does_not_queue_agent_action_when_inference_unavailable(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = False
        agent.predicted_action = 2
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.step(True)
        self.assertTrue(game.update_called)
        self.assertEqual(game.queued, [])

    def test_step_skips_agent_control_when_update_will_not_advance_move(self) -> None:
        game = _FakeGame()
        game.advance_next = False
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.step(True)
        self.assertTrue(game.update_called)
        self.assertEqual(agent.predict_calls, 0)
        self.assertEqual(agent.predict_with_probs_calls, 0)

    def test_step_uses_probability_call_for_confidence_even_when_debug_overlays_disabled(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.set_debug_options(debug_overlay=False, reachable_overlay=False)
        ctrl.step(True)
        self.assertEqual(agent.predict_calls, 0)
        self.assertGreaterEqual(agent.predict_with_probs_calls, 1)

    def test_step_uses_probability_call_when_debug_overlay_enabled(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True),
        )
        ctrl.set_debug_options(debug_overlay=True, reachable_overlay=False)
        ctrl.step(True)
        self.assertEqual(agent.predict_calls, 0)
        self.assertGreaterEqual(agent.predict_with_probs_calls, 1)

    def test_cycle_break_prefers_unvisited_candidate_head(self) -> None:
        game = _FakeGame()
        game.snake = [(10, 10), (9, 10), (8, 10)]
        game.direction = (1, 0)
        game.food = (14, 10)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._recent_heads.extend([(11, 10)] * 8)

        def _eval(*, action: int, **_kwargs):
            if int(action) == 0:
                return (1000.0, True, 0)
            if int(action) == 1:
                return (900.0, True, 0)
            return (800.0, True, 0)

        with (
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.SPACE_FILL),
            patch.object(ctrl, "_evaluate_action", side_effect=_eval),
            patch.object(ctrl._space_fill_controller, "choose_action", return_value=0) as space_fill_mock,
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(action, 1)
        space_fill_mock.assert_not_called()

    def test_reset_episode_tracking_resets_dynamic_mode(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._dynamic.current_mode = ControlMode.SPACE_FILL
        ctrl._dynamic.last_switch_reason = "cycle_repeat"
        ctrl._recent_heads.extend([(1, 1), (1, 2), (1, 3)])
        ctrl.reset_episode_tracking()
        snap = ctrl.telemetry_snapshot()
        self.assertEqual(snap.current_mode, ControlMode.PPO.value)
        self.assertEqual(snap.last_switch_reason, "init")

    def test_reset_episode_tracking_clears_stuck_flag(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._episode_stuck = True
        ctrl.reset_episode_tracking()
        self.assertFalse(bool(ctrl._episode_stuck))

    def test_cycle_repeat_increments_telemetry_counter(self) -> None:
        game = _FakeGame()
        game.snake = [(10, 10), (9, 10), (8, 10)]
        game.direction = (1, 0)
        game.food = (14, 10)
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        agent.predicted_action = 0
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        with (
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(0.0, True, 0)),
        ):
            ctrl.step(True)
        snap = ctrl.telemetry_snapshot()
        self.assertGreaterEqual(snap.cycle_repeats_total, 1)

    def test_evaluate_action_uses_configured_lookahead_depth(self) -> None:
        game = _FakeGame()
        ctrl = GameplayController(
            game=game,
            agent=_FakeAgent(),
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        with patch.object(ctrl, "_lookahead_viability", return_value=1.0) as lookahead_mock:
            _ = ctrl._evaluate_action(
                board_cells=20,
                snake=list(game.snake),
                direction=tuple(game.direction),
                food=tuple(game.food),
                action=0,
                food_weight=0.02,
                capacity_penalty_scale=1.0,
            )
        self.assertTrue(lookahead_mock.called)
        self.assertEqual(int(lookahead_mock.call_args.kwargs.get("depth", 0)), 3)

    def test_lookahead_bonus_increases_action_score(self) -> None:
        game = _FakeGame()
        ctrl = GameplayController(
            game=game,
            agent=_FakeAgent(),
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        with (
            patch.object(ctrl, "_reachable_space", return_value=40),
            patch.object(ctrl, "_tail_is_reachable", return_value=True),
            patch.object(ctrl, "_lookahead_viability", side_effect=[0.0, 1.0]),
        ):
            score_straight, _tail_ok_0, _shortfall_0 = ctrl._evaluate_action(
                board_cells=20,
                snake=list(game.snake),
                direction=tuple(game.direction),
                food=tuple(game.food),
                action=0,
                food_weight=0.02,
                capacity_penalty_scale=1.0,
            )
            score_left, _tail_ok_1, _shortfall_1 = ctrl._evaluate_action(
                board_cells=20,
                snake=list(game.snake),
                direction=tuple(game.direction),
                food=tuple(game.food),
                action=1,
                food_weight=0.02,
                capacity_penalty_scale=1.0,
            )
        self.assertGreater(float(score_left), float(score_straight))

    def test_loop_escape_burst_activates_under_cycle_and_starvation_pressure(self) -> None:
        game = _FakeGame()
        game.steps_without_food = 500
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 120
        with (
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.SPACE_FILL),
            patch.object(ctrl, "_evaluate_action", return_value=(10.0, True, 0)),
            patch.object(ctrl, "_choose_loop_escape_action", return_value=2) as loop_mock,
            patch.object(ctrl._space_fill_controller, "choose_action", return_value=0),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(action, 2)
        self.assertTrue(loop_mock.called)
        snap = ctrl.telemetry_snapshot()
        self.assertGreaterEqual(snap.loop_escape_activations_total, 1)

    def test_loop_escape_can_override_safe_ppo_when_pressure_is_high(self) -> None:
        game = _FakeGame()
        game.steps_without_food = 520
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        with (
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(10.0, True, 0)),
            patch.object(ctrl, "_choose_loop_escape_action", return_value=2) as loop_mock,
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(action, 2)
        self.assertTrue(loop_mock.called)

    def test_food_pressure_bypasses_space_fill_controller(self) -> None:
        game = _FakeGame()
        game.steps_without_food = 720
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 160
        ctrl._dynamic.last_food_step = 0
        with (
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.SPACE_FILL),
            patch.object(ctrl, "_evaluate_action", return_value=(10.0, True, 0)),
            patch.object(ctrl, "_best_safe_action", return_value=2) as safe_mock,
            patch.object(ctrl._space_fill_controller, "choose_action", return_value=1) as space_fill_mock,
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(action, 2)
        self.assertTrue(safe_mock.called)
        space_fill_mock.assert_not_called()

    def test_best_safe_action_prioritizes_food_progress_under_pressure(self) -> None:
        game = _FakeGame()
        game.steps_without_food = 720
        ctrl = GameplayController(
            game=game,
            agent=_FakeAgent(),
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._recent_heads.extend([(11, 10)] * 4)
        with patch.object(ctrl, "_evaluate_action", return_value=(100.0, True, 0)):
            action = ctrl._best_safe_action(
                proposed_action=1,
                board_cells=20,
                snake=list(game.snake),
                direction=tuple(game.direction),
                food=tuple(game.food),
                food_weight=0.02,
                capacity_penalty_scale=1.0,
            )
        self.assertEqual(action, 0)

    def test_loop_escape_burst_skips_when_signals_are_weak(self) -> None:
        game = _FakeGame()
        game.steps_without_food = 10
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 60
        with (
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.SPACE_FILL),
            patch.object(ctrl, "_evaluate_action", return_value=(10.0, True, 0)),
            patch.object(ctrl, "_food_distance_stalled", return_value=False),
            patch.object(ctrl, "_choose_loop_escape_action", return_value=2) as loop_mock,
            patch.object(ctrl._space_fill_controller, "choose_action", return_value=1),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertNotEqual(action, 2)
        self.assertFalse(loop_mock.called)

    def test_choose_safe_action_resets_risk_flags_when_eval_is_unavailable(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._last_chosen_tail_reachable = False
        ctrl._last_capacity_shortfall = 2
        with (
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=None),
        ):
            _ = ctrl._choose_safe_action(0)
        self.assertTrue(bool(ctrl._last_chosen_tail_reachable))
        self.assertEqual(int(ctrl._last_capacity_shortfall), 0)

    def test_choose_safe_action_uses_warmup_ppo_when_viable(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 40
        with (
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(1000.0, True, 0)),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 0)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "warmup_ppo")

    def test_choose_safe_action_does_not_force_conf_trust_in_sustained_narrow_corridor(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    ppo_confidence_trust_threshold=0.8,
                    ppo_confidence_trust_min_safe_options=2,
                    narrow_corridor_trigger_steps=2,
                    enable_learned_arbiter=False,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._last_predicted_confidence = 0.99
        ctrl._narrow_corridor_streak = 2
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, True, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.ESCAPE),
            patch.object(ctrl, "_evaluate_action", return_value=(100.0, True, 0)),
            patch.object(ctrl._escape_controller, "choose_action", return_value=1),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 1)
        self.assertNotEqual(str(ctrl.last_mode_switch_reason()), "ppo_conf_trust")

    def test_choose_safe_action_reverts_override_in_open_field_low_pressure(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_open_field_trust_food_pressure_max=0.35,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 300
        ctrl._loop_escape_steps_left = 1
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, False, False]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(100.0, True, 0)),
            patch.object(ctrl, "_best_safe_action", return_value=2),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 0)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "ppo_open_field_trust")

    def test_choose_safe_action_nonviable_single_safe_option_can_still_tolerate_low_risk(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 300
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, True, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(100.0, True, 5)),
            patch.object(ctrl, "_best_safe_action", return_value=2),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 0)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "ppo_tolerate_low_risk")

    def test_pocket_exit_guard_disabled_keeps_tolerate_low_risk(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    enable_pocket_exit_guard=False,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 300
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, True, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(100.0, True, 5)),
            patch.object(ctrl, "_best_safe_action", return_value=2),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 0)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "ppo_tolerate_low_risk")

    def test_pocket_exit_guard_enabled_overrides_when_safer_alt_measured(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    enable_pocket_exit_guard=True,
                    pocket_exit_guard_max_safe_options=3,
                    pocket_exit_guard_min_no_progress_steps=0,
                    pocket_exit_guard_min_food_pressure=0.0,
                    pocket_exit_guard_min_shortfall_gain=1,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 300
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, True, True]),
            patch.object(ctrl, "_is_food_reachable_after_action", side_effect=[False, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_best_safe_action", return_value=2),
            patch.object(ctrl, "_evaluate_action", side_effect=[(100.0, True, 5), (120.0, True, 2)]),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 2)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "pocket_exit_guard")

    def test_choose_safe_action_viable_ppo_path_overwrites_stale_reason(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(agent_safety_override=True),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._dynamic.last_switch_reason = "ppo_conf_trust"
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 300
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, False, False]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.PPO),
            patch.object(ctrl, "_evaluate_action", return_value=(100.0, True, 0)),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 0)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "ppo_mode_viable")

    def test_choose_safe_action_high_conf_override_guard_prefers_ppo_without_safety_gain(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_confidence_trust_threshold=1.1,
                    ppo_high_conf_override_guard_threshold=0.97,
                    ppo_high_conf_override_guard_min_shortfall_gain=2,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 300
        ctrl._last_predicted_confidence = 0.99
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, False, False]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.ESCAPE),
            patch.object(ctrl._escape_controller, "choose_action", return_value=2),
            patch.object(ctrl, "_evaluate_action", side_effect=[(90.0, True, 6), (91.0, True, 5)]),
        ):
            action = ctrl._choose_safe_action(0)
        self.assertEqual(int(action), 0)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "ppo_high_conf_guard")

    def test_risk_switch_guard_downgrades_no_progress_risk_when_no_safety_gain(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_confidence_trust_threshold=1.1,
                    ppo_high_conf_override_guard_threshold=1.1,
                    enable_risk_switch_guard=True,
                    risk_switch_guard_confidence_min=0.9,
                    risk_switch_guard_min_safe_options=2,
                    risk_switch_guard_min_shortfall_gain=2,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 220
        ctrl._last_predicted_confidence = 0.99

        captured: dict[str, object] = {}

        def _capture_mode(**kwargs):
            captured.update(kwargs)
            return ControlMode.PPO

        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, False, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_is_eval_viable", return_value=False),
            patch.object(ctrl, "_best_safe_action", return_value=1),
            patch.object(ctrl, "_evaluate_action", side_effect=[(100.0, True, 6), (99.0, True, 5), (99.0, True, 5)]),
            patch.object(ctrl, "_select_mode", side_effect=_capture_mode),
        ):
            _ = ctrl._choose_safe_action(0)
        self.assertFalse(bool(captured.get("significant_risk", True)))
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "risk_guard_hold")

    def test_risk_switch_guard_keeps_risk_when_alt_has_measured_safety_gain(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_confidence_trust_threshold=1.1,
                    ppo_high_conf_override_guard_threshold=1.1,
                    enable_risk_switch_guard=True,
                    risk_switch_guard_confidence_min=0.9,
                    risk_switch_guard_min_safe_options=2,
                    risk_switch_guard_min_shortfall_gain=2,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 220
        ctrl._last_predicted_confidence = 0.99

        captured: dict[str, object] = {}

        def _capture_mode(**kwargs):
            captured.update(kwargs)
            return ControlMode.ESCAPE

        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, False, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_is_eval_viable", return_value=False),
            patch.object(ctrl, "_best_safe_action", return_value=1),
            patch.object(ctrl, "_evaluate_action", side_effect=[(100.0, True, 6), (101.0, True, 3), (101.0, True, 3)]),
            patch.object(ctrl, "_select_mode", side_effect=_capture_mode),
            patch.object(ctrl._escape_controller, "choose_action", return_value=1),
        ):
            _ = ctrl._choose_safe_action(0)
        self.assertTrue(bool(captured.get("significant_risk", False)))

    def test_risk_guard_diagnostics_reports_disabled_blocker_for_significant_risk(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_confidence_trust_threshold=1.1,
                    ppo_high_conf_override_guard_threshold=1.1,
                    enable_risk_switch_guard=False,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 220
        ctrl._last_predicted_confidence = 0.99
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, False, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=True),
            patch.object(ctrl, "_is_eval_viable", return_value=False),
            patch.object(ctrl, "_evaluate_action", side_effect=[(100.0, True, 6), (99.0, True, 5)]),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.ESCAPE),
            patch.object(ctrl._escape_controller, "choose_action", return_value=1),
        ):
            _ = ctrl._choose_safe_action(0)
        row = ctrl.decision_trace_snapshot()
        self.assertTrue(bool(row.get("risk_guard_candidate")))
        self.assertFalse(bool(row.get("risk_guard_eligible")))
        self.assertIn("guard_disabled", list(row.get("risk_guard_blockers", [])))

    def test_risk_guard_diagnostics_reports_narrow_corridor_blocker_when_not_allowed(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_confidence_trust_threshold=1.1,
                    ppo_high_conf_override_guard_threshold=1.1,
                    enable_risk_switch_guard=True,
                    risk_switch_guard_allow_narrow_corridor=False,
                    narrow_corridor_trigger_steps=1,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 280
        ctrl._last_predicted_confidence = 0.99
        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, True, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_is_eval_viable", return_value=False),
            patch.object(ctrl, "_evaluate_action", side_effect=[(100.0, True, 6), (99.0, True, 5)]),
            patch.object(ctrl, "_select_mode", return_value=ControlMode.ESCAPE),
            patch.object(ctrl._escape_controller, "choose_action", return_value=0),
        ):
            _ = ctrl._choose_safe_action(0)
        row = ctrl.decision_trace_snapshot()
        self.assertTrue(bool(row.get("risk_guard_candidate")))
        self.assertFalse(bool(row.get("risk_guard_eligible")))
        self.assertIn("narrow_corridor", list(row.get("risk_guard_blockers", [])))

    def test_risk_switch_guard_allows_narrow_corridor_when_flagged(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        agent.is_ready = True
        agent.is_inference_available = True
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(
                agent_safety_override=True,
                dynamic_control=DynamicControlConfig(
                    enable_learned_arbiter=False,
                    ppo_confidence_trust_threshold=1.1,
                    ppo_high_conf_override_guard_threshold=1.1,
                    enable_risk_switch_guard=True,
                    risk_switch_guard_allow_narrow_corridor=True,
                    risk_switch_guard_narrow_confidence_min=0.95,
                    risk_switch_guard_narrow_min_no_progress_steps=12,
                    risk_switch_guard_narrow_no_progress_margin=30,
                    narrow_corridor_trigger_steps=1,
                    risk_switch_guard_min_safe_options=1,
                ),
            ),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._decisions_total = 300
        ctrl._dynamic.last_food_step = 280
        ctrl._last_predicted_confidence = 0.99
        captured: dict[str, object] = {}

        def _capture_mode(**kwargs):
            captured.update(kwargs)
            return ControlMode.PPO

        with (
            patch("snake_frame.gameplay_controller.is_danger", side_effect=[False, True, True]),
            patch.object(ctrl, "_register_cycle_state", return_value=False),
            patch.object(ctrl, "_is_eval_viable", return_value=False),
            patch.object(ctrl, "_best_safe_action", return_value=0),
            patch.object(ctrl, "_evaluate_action", side_effect=[(100.0, True, 6), (100.0, True, 6)]),
            patch.object(ctrl, "_select_mode", side_effect=_capture_mode),
        ):
            _ = ctrl._choose_safe_action(0)
        self.assertFalse(bool(captured.get("significant_risk", True)))
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "risk_guard_hold")
        row = ctrl.decision_trace_snapshot()
        self.assertTrue(bool(row.get("risk_guard_eligible")))
        self.assertNotIn("narrow_corridor", list(row.get("risk_guard_blockers", [])))

    def test_select_mode_holds_escape_on_no_progress_in_narrow_corridor_when_tail_reachable(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._dynamic.current_mode = ControlMode.ESCAPE
        ctrl._decisions_total = 300
        mode = ctrl._select_mode(
            significant_risk=True,
            imminent_danger=False,
            cycle_repeat=False,
            no_progress_steps=int(ctrl._dynamic_cfg.no_progress_steps_escape),
            safe_option_count=1,
            proposed_tail_reachable=True,
            proposed_capacity_shortfall=0,
        )
        self.assertEqual(mode, ControlMode.ESCAPE)

    def test_select_mode_uses_space_fill_on_no_progress_when_not_narrow_corridor(self) -> None:
        game = _FakeGame()
        agent = _FakeAgent()
        ctrl = GameplayController(
            game=game,
            agent=agent,
            settings=Settings(),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True),
        )
        ctrl._dynamic.current_mode = ControlMode.ESCAPE
        ctrl._decisions_total = 300
        mode = ctrl._select_mode(
            significant_risk=True,
            imminent_danger=False,
            cycle_repeat=False,
            no_progress_steps=int(ctrl._dynamic_cfg.no_progress_steps_escape),
            safe_option_count=2,
            proposed_tail_reachable=True,
            proposed_capacity_shortfall=0,
        )
        self.assertEqual(mode, ControlMode.SPACE_FILL)
        self.assertEqual(str(ctrl.last_mode_switch_reason()), "no_progress_escape")


if __name__ == "__main__":
    unittest.main()
