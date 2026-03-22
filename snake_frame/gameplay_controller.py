from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

import pygame

from .arbiter_model import LearnedArbiterModel
from .board_analysis import (
    reachable_cell_count,
    reachable_cells as board_reachable_cells,
    simulate_next_snake,
    tail_is_reachable as board_tail_is_reachable,
)
from .escape_controller import EscapeController
from .observation import action_to_direction, build_observation, is_danger, next_head, valid_action_mask
from .protocols import AgentLike, GameLike
from .settings import DynamicControlConfig, ObsConfig, Settings
from .space_fill_controller import SpaceFillController
from .tactic_memory import TacticMemoryBank

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    PPO = "ppo"
    ESCAPE = "escape"
    SPACE_FILL = "space_fill"


@dataclass
class DynamicControllerState:
    current_mode: ControlMode = ControlMode.PPO
    mode_enter_step: int = 0
    last_food_step: int = 0
    cycle_window: deque[int] = field(default_factory=deque)
    cycle_hash_counts: dict[int, int] = field(default_factory=dict)
    cooldown_until_step: int = 0
    last_switch_reason: str = "init"


@dataclass(frozen=True)
class CandidateDebug:
    action: int
    cell: tuple[int, int]
    danger: bool
    reachable_ratio: float
    reachable_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AgentDebugSnapshot:
    head: tuple[int, int]
    predicted_action: int
    chosen_action: int
    candidates: tuple[CandidateDebug, CandidateDebug, CandidateDebug]
    action_probs: tuple[float, float, float] | None


@dataclass(frozen=True)
class GameplayTelemetrySnapshot:
    decisions_total: int
    interventions_total: int
    pocket_risk_total: int
    tail_unreachable_total: int
    deaths_wall: int
    deaths_body: int
    deaths_starvation: int
    deaths_fill: int
    deaths_other: int
    deaths_early: int
    deaths_mid: int
    deaths_late: int
    avg_death_confidence: float
    decisions_mode_ppo: int
    decisions_mode_escape: int
    decisions_mode_space_fill: int
    mode_switches_total: int
    cycle_breaks_total: int
    stuck_episodes_total: int
    cycle_repeats_total: int
    no_progress_steps: int
    starvation_steps: int
    starvation_limit: int
    loop_escape_activations_total: int
    loop_escape_steps_left: int
    current_mode: str
    last_switch_reason: str
    last_death_reason: str


@dataclass(frozen=True)
class _DecisionContext:
    features: tuple[float, ...]
    proposed_action: int
    chosen_action: int
    override_used: bool


@dataclass(frozen=True)
class _CandidateAnalysis:
    action: int
    candidate_head: tuple[int, int]
    danger: bool
    simulated_snake: tuple[tuple[int, int], ...]
    reachable_count: int
    tail_reachable: bool
    capacity_shortfall: int
    next_food_dist: int
    food_progress: int
    revisit_count: int
    score: float | None
    eval_result: tuple[float, bool, int] | None
    viable: bool


@dataclass(frozen=True)
class _DecisionTraceInputs:
    proposed_action: int
    board_cells: int
    snake: tuple[tuple[int, int], ...]
    direction: tuple[int, int]
    food: tuple[int, int]
    food_pressure: float
    no_progress_steps: int
    candidate_analyses: tuple[tuple[int, _CandidateAnalysis], ...]
    action_eval_tuples: tuple[tuple[str, dict[str, object]], ...]


class GameplayController:
    _TAIL_REACHABLE_BONUS = 1200.0
    _TAIL_UNREACHABLE_PENALTY = 1600.0
    _CAPACITY_SHORTFALL_PENALTY = 60.0
    _FOOD_DIST_WEIGHT = 0.02
    _FOOD_DIST_WEIGHT_OPEN = 0.04
    _FOOD_DIST_WEIGHT_CROWDED = 0.008
    _CROWDED_FREE_RATIO_THRESHOLD = 0.35
    _CROWDED_CAPACITY_PENALTY_SCALE = 1.4
    _ESCAPE_FREE_RATIO_THRESHOLD = 0.50
    _ESCAPE_LENGTH_RATIO_THRESHOLD = 0.16
    _FOOD_PRESSURE_TRIGGER = 0.62
    _FOOD_PROGRESS_SCORE_WEIGHT = 140.0
    _FOOD_REVISIT_PENALTY = 24.0
    _SHORTFALL_TOLERANCE_MAX = 3

    def __init__(
        self,
        *,
        game: GameLike,
        agent: AgentLike,
        settings: Settings,
        obs_config: ObsConfig,
        space_strategy_enabled: bool = True,
        artifact_dir: Path | None = None,
    ) -> None:
        self.game = game
        self.agent = agent
        self.settings = settings
        self.obs_config = obs_config
        self._debug_snapshot: AgentDebugSnapshot | None = None
        self._decisions_total = 0
        self._interventions_total = 0
        self._pocket_risk_total = 0
        self._tail_unreachable_total = 0
        self._deaths_wall = 0
        self._deaths_body = 0
        self._deaths_starvation = 0
        self._deaths_fill = 0
        self._deaths_other = 0
        self._deaths_early = 0
        self._deaths_mid = 0
        self._deaths_late = 0
        self._death_confidences: deque[float] = deque(maxlen=120)
        self._last_chosen_confidence: float | None = None
        self._last_predicted_confidence: float | None = None
        self._last_action_probs: tuple[float, float, float] | None = None
        self._last_predicted_action: int | None = None
        self._last_chosen_action: int | None = None
        self._last_chosen_tail_reachable = True
        self._tail_reachable_streak = 0
        self._tail_unreachable_streak = 0
        self._last_capacity_shortfall = 0
        self._last_proposed_tail_reachable = True
        self._last_proposed_capacity_shortfall = 0
        self._last_food_pressure = 0.0
        self._last_free_ratio = 0.0
        self._last_safe_option_count = 3
        self._narrow_corridor_streak = 0
        self._last_cycle_repeat = False
        self._last_imminent_danger = False
        self._last_proposed_viable = False
        self._last_risk_guard_candidate = False
        self._last_risk_guard_eligible = False
        self._last_risk_guard_blockers: tuple[str, ...] = ()
        self._last_pre_no_exit_guard_candidate = False
        self._last_pre_no_exit_guard_applied = False
        self._last_pre_no_exit_guard_blocker = "disabled"
        self._last_pre_no_exit_guard_alt_action: int | None = None
        self._last_pre_no_exit_guard_safe_collapsing = False
        self._last_pre_no_exit_guard_near_no_exit_signal = False
        self._last_no_exit_state = False
        self._last_entered_no_exit_this_step = False
        self._last_action_eval_tuples: dict[str, dict[str, object]] = {}
        self._last_candidate_analyses: dict[int, _CandidateAnalysis] = {}
        self._candidate_eval_cache: dict[tuple[object, ...], _CandidateAnalysis] = {}
        self._last_trace_inputs: _DecisionTraceInputs | None = None
        self._space_strategy_enabled = bool(space_strategy_enabled)
        self._escape_controller = EscapeController()
        self._space_fill_controller = SpaceFillController()
        self._dynamic_cfg: DynamicControlConfig = getattr(settings, "dynamic_control", DynamicControlConfig())
        self._arbiter_feature_dim = 8
        self._decision_contexts: deque[_DecisionContext] = deque(maxlen=192)
        self._last_decision_context: _DecisionContext | None = None
        self._learning_enabled = True
        self._persist_learning = artifact_dir is not None
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else (Path(__file__).resolve().parents[1] / "state" / "ppo" / "baseline")
        self._arbiter_path = self._artifact_dir / "arbiter_model.json"
        self._tactic_memory_path = self._artifact_dir / "tactic_memory.json"
        self._arbiter_dirty = False
        self._tactic_dirty = False
        self._arbiter_model = (
            LearnedArbiterModel.load(self._arbiter_path, fallback_dim=self._arbiter_feature_dim)
            if bool(self._persist_learning)
            else LearnedArbiterModel(dim=self._arbiter_feature_dim)
        )
        self._arbiter_model.learning_rate = float(getattr(self._dynamic_cfg, "arbiter_learning_rate", self._arbiter_model.learning_rate))
        self._arbiter_model.l2 = float(getattr(self._dynamic_cfg, "arbiter_l2", self._arbiter_model.l2))
        self._tactic_memory = (
            TacticMemoryBank.load(self._tactic_memory_path, fallback_dim=self._arbiter_feature_dim)
            if bool(self._persist_learning)
            else TacticMemoryBank(
                dim=self._arbiter_feature_dim,
                max_clusters=self._dynamic_cfg.tactic_memory_max_clusters,
                merge_radius=self._dynamic_cfg.tactic_memory_merge_radius,
                memory_weight=self._dynamic_cfg.tactic_memory_weight,
                adaptive_merge=self._dynamic_cfg.tactic_memory_adaptive_merge,
                crowded_radius=self._dynamic_cfg.tactic_memory_merge_radius_crowded,
                open_radius=self._dynamic_cfg.tactic_memory_merge_radius_open,
                low_threshold=self._dynamic_cfg.tactic_memory_merge_ratio_low,
                high_threshold=self._dynamic_cfg.tactic_memory_merge_ratio_high,
            )
        )
        self._tactic_memory.max_clusters = int(getattr(self._dynamic_cfg, "tactic_memory_max_clusters", self._tactic_memory.max_clusters))
        self._tactic_memory.merge_radius = float(getattr(self._dynamic_cfg, "tactic_memory_merge_radius", self._tactic_memory.merge_radius))
        self._tactic_memory.memory_weight = float(getattr(self._dynamic_cfg, "tactic_memory_weight", self._tactic_memory.memory_weight))
        self._tactic_memory._adaptive_merge = bool(getattr(self._dynamic_cfg, "tactic_memory_adaptive_merge", self._tactic_memory._adaptive_merge))
        self._tactic_memory._crowded_radius = float(getattr(self._dynamic_cfg, "tactic_memory_merge_radius_crowded", self._tactic_memory._crowded_radius))
        self._tactic_memory._open_radius = float(getattr(self._dynamic_cfg, "tactic_memory_merge_radius_open", self._tactic_memory._open_radius))
        self._tactic_memory._low_threshold = float(getattr(self._dynamic_cfg, "tactic_memory_merge_ratio_low", self._tactic_memory._low_threshold))
        self._tactic_memory._high_threshold = float(getattr(self._dynamic_cfg, "tactic_memory_merge_ratio_high", self._tactic_memory._high_threshold))
        self._dynamic = DynamicControllerState()
        self._last_score_seen = int(getattr(self.game, "score", 0))
        self._last_action: int | None = None
        self._recent_heads: deque[tuple[int, int]] = deque(maxlen=64)
        self._mode_switches_total = 0
        self._cycle_breaks_total = 0
        self._stuck_episodes_total = 0
        self._cycle_repeats_total = 0
        self._loop_escape_activations_total = 0
        self._loop_escape_steps_left = 0
        self._loop_escape_cooldown_until = 0
        self._recent_food_distances: deque[int] = deque(maxlen=96)
        self._episode_stuck = False
        self._decisions_mode_ppo = 0
        self._decisions_mode_escape = 0
        self._decisions_mode_space_fill = 0
        self._decision_mode_now = ControlMode.PPO
        self._debug_overlay_enabled = False
        self._reachable_overlay_enabled = False
        self._last_death_reason = "none"
        self._tail_trend_enabled = bool(getattr(self.obs_config, "use_tail_trend_features", True))

    def set_debug_options(self, *, debug_overlay: bool, reachable_overlay: bool) -> None:
        self._debug_overlay_enabled = bool(debug_overlay)
        self._reachable_overlay_enabled = bool(reachable_overlay)

    def set_learning_enabled(self, enabled: bool) -> None:
        self._learning_enabled = bool(enabled)

    def set_artifact_dir(self, artifact_dir: Path | None) -> None:
        self._persist_learning_state()
        self._persist_learning = artifact_dir is not None
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else (Path(__file__).resolve().parents[1] / "state" / "ppo" / "baseline")
        self._arbiter_path = self._artifact_dir / "arbiter_model.json"
        self._tactic_memory_path = self._artifact_dir / "tactic_memory.json"
        self._arbiter_dirty = False
        self._tactic_dirty = False
        if bool(self._persist_learning):
            self._arbiter_model = LearnedArbiterModel.load(self._arbiter_path, fallback_dim=self._arbiter_feature_dim)
            self._tactic_memory = TacticMemoryBank.load(self._tactic_memory_path, fallback_dim=self._arbiter_feature_dim)
        else:
            self._arbiter_model = LearnedArbiterModel(dim=self._arbiter_feature_dim)
            self._tactic_memory = TacticMemoryBank(
                dim=self._arbiter_feature_dim,
                max_clusters=self._dynamic_cfg.tactic_memory_max_clusters,
                merge_radius=self._dynamic_cfg.tactic_memory_merge_radius,
                memory_weight=self._dynamic_cfg.tactic_memory_weight,
                adaptive_merge=self._dynamic_cfg.tactic_memory_adaptive_merge,
                crowded_radius=self._dynamic_cfg.tactic_memory_merge_radius_crowded,
                open_radius=self._dynamic_cfg.tactic_memory_merge_radius_open,
                low_threshold=self._dynamic_cfg.tactic_memory_merge_ratio_low,
                high_threshold=self._dynamic_cfg.tactic_memory_merge_ratio_high,
            )
        self._arbiter_model.learning_rate = float(getattr(self._dynamic_cfg, "arbiter_learning_rate", self._arbiter_model.learning_rate))
        self._arbiter_model.l2 = float(getattr(self._dynamic_cfg, "arbiter_l2", self._arbiter_model.l2))
        self._tactic_memory.max_clusters = int(getattr(self._dynamic_cfg, "tactic_memory_max_clusters", self._tactic_memory.max_clusters))
        self._tactic_memory.merge_radius = float(getattr(self._dynamic_cfg, "tactic_memory_merge_radius", self._tactic_memory.merge_radius))
        self._tactic_memory.memory_weight = float(getattr(self._dynamic_cfg, "tactic_memory_weight", self._tactic_memory.memory_weight))
        self._tactic_memory._adaptive_merge = bool(getattr(self._dynamic_cfg, "tactic_memory_adaptive_merge", self._tactic_memory._adaptive_merge))
        self._tactic_memory._crowded_radius = float(getattr(self._dynamic_cfg, "tactic_memory_merge_radius_crowded", self._tactic_memory._crowded_radius))
        self._tactic_memory._open_radius = float(getattr(self._dynamic_cfg, "tactic_memory_merge_radius_open", self._tactic_memory._open_radius))
        self._tactic_memory._low_threshold = float(getattr(self._dynamic_cfg, "tactic_memory_merge_ratio_low", self._tactic_memory._low_threshold))
        self._tactic_memory._high_threshold = float(getattr(self._dynamic_cfg, "tactic_memory_merge_ratio_high", self._tactic_memory._high_threshold))

    def set_tail_trend_enabled(self, enabled: bool) -> None:
        self._tail_trend_enabled = bool(enabled)

    def step(self, game_running: bool) -> None:
        if not bool(game_running):
            return
        if self.game.game_over:
            self._record_episode_end()
            self._reset_dynamic_state()
            self.agent.request_inference_sync()
            self.game.reset()
            self._last_score_seen = int(getattr(self.game, "score", 0))
            return
        if self._should_compute_agent_action():
            self._apply_agent_control()
        self.game.update()

    def _should_compute_agent_action(self) -> bool:
        will_advance = getattr(self.game, "will_advance_on_next_update", None)
        if callable(will_advance):
            try:
                return bool(will_advance())
            except Exception:
                logger.debug("will_advance_on_next_update failed; defaulting to per-frame control", exc_info=True)
        return True

    def _decision_features(
        self,
        *,
        free_ratio: float,
        food_pressure: float,
        no_progress_steps: int,
        cycle_repeat: bool,
        imminent_danger: bool,
        proposed_viable: bool,
        proposed_eval: tuple[float, bool, int] | None,
        chosen_eval: tuple[float, bool, int] | None,
    ) -> list[float]:
        soft = max(1, int(getattr(self._dynamic_cfg, "no_progress_steps_escape", 64)))
        hard = max(soft + 1, int(getattr(self._dynamic_cfg, "no_progress_steps_space_fill", 128)))
        no_progress_norm = float(max(0.0, min(1.0, float(no_progress_steps - soft) / float(max(1, hard - soft)))))
        pred_conf = float(self._last_predicted_confidence if self._last_predicted_confidence is not None else 0.0)
        shortfall = 0.0 if proposed_eval is None else float(max(0, int(proposed_eval[2])))
        delta = 0.0
        if proposed_eval is not None and chosen_eval is not None:
            delta = float(chosen_eval[0]) - float(proposed_eval[0])
        return [
            float(max(0.0, min(1.0, free_ratio))),
            float(max(0.0, min(1.0, food_pressure))),
            float(max(0.0, min(1.0, no_progress_norm))),
            float(max(0.0, min(1.0, pred_conf))),
            1.0 if bool(cycle_repeat) else 0.0,
            1.0 if bool(imminent_danger) else 0.0,
            1.0 if bool(proposed_viable) else 0.0,
            float(max(-2.0, min(2.0, (delta / 250.0) - (shortfall / 12.0)))),
        ]

    def _reinforce_recent_contexts(self, *, success: bool, weight: float) -> None:
        if not bool(self._learning_enabled):
            return
        if not self._decision_contexts:
            return
        horizon = min(8, len(self._decision_contexts))
        samples = list(self._decision_contexts)[-horizon:]
        for ctx in samples:
            if bool(getattr(self._dynamic_cfg, "enable_learned_arbiter", False)) and bool(ctx.override_used):
                self._arbiter_model.update(list(ctx.features), label=1 if bool(success) else 0, weight=float(weight))
                self._arbiter_dirty = True
            if bool(getattr(self._dynamic_cfg, "enable_tactic_memory", False)):
                self._tactic_memory.record(
                    features=list(ctx.features),
                    action=int(ctx.chosen_action),
                    success=bool(success),
                    weight=float(weight),
                    free_ratio=float(ctx.features[0]) if ctx.features else None,
                )
                self._tactic_dirty = True

    def _maybe_persist_learning_state(self) -> None:
        if not bool(self._learning_enabled):
            return
        # Avoid synchronous writes each frame; persist every 128 decisions and at episode end.
        if int(self._decisions_total) <= 0 or (int(self._decisions_total) % 128) != 0:
            return
        self._persist_learning_state()

    def _persist_learning_state(self) -> None:
        if not bool(self._learning_enabled):
            return
        if not bool(self._persist_learning):
            return
        try:
            if self._arbiter_dirty:
                self._arbiter_model.save(self._arbiter_path)
                self._arbiter_dirty = False
            if self._tactic_dirty:
                self._tactic_memory.save(self._tactic_memory_path)
                self._tactic_dirty = False
        except Exception:
            logger.debug("Failed persisting controller learning state", exc_info=True)

    def _apply_agent_control(self) -> None:
        inference_available = getattr(self.agent, "is_inference_available", getattr(self.agent, "is_ready", False))
        if callable(inference_available):
            inference_available = inference_available()
        if not bool(inference_available):
            self._debug_snapshot = None
            return
        score_now = int(getattr(self.game, "score", 0))
        if score_now > self._last_score_seen:
            self._reinforce_recent_contexts(success=True, weight=float(max(1, score_now - self._last_score_seen)))
            self._dynamic.last_food_step = int(self._decisions_total)
            self._last_score_seen = score_now
        if self.game.snake:
            self._recent_heads.append((int(self.game.snake[0][0]), int(self.game.snake[0][1])))
            food = tuple(self.game.food)
            head = tuple(self.game.snake[0])
            self._recent_food_distances.append(int(abs(head[0] - food[0]) + abs(head[1] - food[1])))
        obs = build_observation(
            board_cells=self.settings.board_cells,
            snake=list(self.game.snake),
            direction=self.game.direction,
            food=self.game.food,
            obs_config=self.obs_config,
            tail_reachable_streak=self._tail_reachable_streak if bool(self._tail_trend_enabled) else 0,
            tail_unreachable_streak=self._tail_unreachable_streak if bool(self._tail_trend_enabled) else 0,
        )
        action_probs: tuple[float, float, float] | None = None
        debug_needed = bool(self._debug_overlay_enabled or self._reachable_overlay_enabled)
        predict_with_probs = getattr(self.agent, "predict_action_with_probs", None)
        action_masks = valid_action_mask(
            self.settings.board_cells,
            list(self.game.snake),
            tuple(self.game.direction),
        )
        if callable(predict_with_probs):
            try:
                predicted_action, action_probs = predict_with_probs(obs, action_masks=action_masks)
                predicted_action = int(predicted_action)
            except Exception:
                logger.exception("predict_action_with_probs failed; falling back to predict_action")
                predicted_action = int(self.agent.predict_action(obs, action_masks=action_masks))
                action_probs = None
        else:
            predicted_action = int(self.agent.predict_action(obs, action_masks=action_masks))
        predicted_confidence = None
        if action_probs is not None:
            try:
                idx = int(predicted_action)
                if 0 <= idx < len(action_probs):
                    predicted_confidence = float(action_probs[idx])
            except Exception:
                predicted_confidence = None
        self._last_predicted_confidence = predicted_confidence
        self._last_action_probs = action_probs
        self._last_predicted_action = int(predicted_action)
        action = self._choose_safe_action(predicted_action)
        self._last_chosen_action = int(action)
        self._record_decision(
            predicted_action=predicted_action,
            chosen_action=action,
            action_probs=action_probs,
        )
        if self._last_decision_context is not None:
            self._decision_contexts.append(self._last_decision_context)
            self._last_decision_context = None
        self._maybe_persist_learning_state()
        if debug_needed:
            self._update_debug_snapshot(
                predicted_action=predicted_action,
                chosen_action=action,
                action_probs=action_probs,
                include_reachable_cells=bool(self._reachable_overlay_enabled),
                candidate_analyses=self._last_candidate_analyses,
            )
        else:
            self._debug_snapshot = None
        next_direction = action_to_direction(self.game.direction, action)
        self.game.queue_direction(next_direction[0], next_direction[1])
        self._last_action = int(action)

    def draw_debug_overlay(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        snapshot = self._debug_snapshot
        if snapshot is None:
            return
        cp = int(self.settings.cell_px)
        ox = int(self.settings.board_offset_x)
        oy = int(self.settings.board_offset_y)
        head_cx = int(ox + (snapshot.head[0] * cp) + (cp // 2))
        head_cy = int(oy + (snapshot.head[1] * cp) + (cp // 2))

        panel_rect = pygame.Rect(ox + 8, oy + 74, 430, 124)
        pygame.draw.rect(surface, (12, 12, 18), panel_rect, border_radius=6)
        pygame.draw.rect(surface, (52, 90, 128), panel_rect, width=1, border_radius=6)

        labels = {0: "S", 1: "L", 2: "R"}
        for candidate in snapshot.candidates:
            tx = int(ox + (candidate.cell[0] * cp) + (cp // 2))
            ty = int(oy + (candidate.cell[1] * cp) + (cp // 2))
            base_color = (245, 90, 90) if candidate.danger else (85, 190, 255)
            width = 2
            if candidate.action == snapshot.predicted_action:
                base_color = (255, 210, 60)
                width = 3
            if candidate.action == snapshot.chosen_action:
                base_color = (60, 235, 130)
                width = 4
            pygame.draw.line(surface, base_color, (head_cx, head_cy), (tx, ty), width=width)
            if candidate.danger:
                cell_rect = pygame.Rect(ox + (candidate.cell[0] * cp), oy + (candidate.cell[1] * cp), cp, cp)
                pygame.draw.rect(surface, (245, 90, 90), cell_rect, width=2, border_radius=4)
            tag = f"{labels.get(candidate.action, '?')} r={candidate.reachable_ratio:.2f}"
            surface.blit(font.render(tag, True, base_color), (tx + 6, ty - 10))

        caption = f"AI debug pred={labels.get(snapshot.predicted_action, '?')} used={labels.get(snapshot.chosen_action, '?')}"
        surface.blit(font.render(caption, True, (220, 230, 255)), (panel_rect.x + 8, panel_rect.y + 8))
        mode_line = f"Mode: {self.current_control_mode()}  reason: {self.last_mode_switch_reason()}"
        surface.blit(font.render(mode_line, True, (190, 212, 240)), (panel_rect.x + 8, panel_rect.y + 58))
        if snapshot.action_probs is not None:
            p0, p1, p2 = snapshot.action_probs
            probs_text = f"pi(S/L/R)=({p0:.2f}, {p1:.2f}, {p2:.2f})"
            surface.blit(font.render(probs_text, True, (180, 210, 245)), (panel_rect.x + 8, panel_rect.y + 34))

    def draw_reachable_overlay(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        snapshot = self._debug_snapshot
        if snapshot is None:
            return
        cp = int(self.settings.cell_px)
        ox = int(self.settings.board_offset_x)
        oy = int(self.settings.board_offset_y)
        alpha_surface = pygame.Surface((self.settings.window_px, self.settings.window_px), pygame.SRCALPHA)
        colors = {0: (55, 215, 255, 56), 1: (255, 205, 65, 56), 2: (170, 100, 255, 56)}
        highlight = {0: (55, 215, 255), 1: (255, 205, 65), 2: (170, 100, 255)}
        for candidate in snapshot.candidates:
            fill = colors.get(candidate.action, (120, 120, 120, 48))
            stroke = highlight.get(candidate.action, (150, 150, 150))
            for x, y in candidate.reachable_cells:
                rect = pygame.Rect(int(x * cp), int(y * cp), cp, cp)
                alpha_surface.fill(fill, rect)
            cx = int(ox + (candidate.cell[0] * cp) + (cp // 2))
            cy = int(oy + (candidate.cell[1] * cp) + (cp // 2))
            label = f"{('S','L','R')[candidate.action]} area={len(candidate.reachable_cells)}"
            surface.blit(font.render(label, True, stroke), (cx + 8, cy + 10))
        surface.blit(alpha_surface, (ox, oy))
        legend = "F4 reachable overlay  S=cyan  L=yellow  R=purple"
        surface.blit(font.render(legend, True, (210, 220, 240)), (ox + 10, oy + 180))

    def _choose_safe_action(self, proposed_action: int) -> int:
        self._candidate_eval_cache = {}
        self._last_candidate_analyses = {}
        board_cells = int(self.settings.board_cells)
        snake = list(self.game.snake)
        direction = tuple(self.game.direction)
        food = tuple(self.game.food)
        safe_option_count = 0
        head = snake[0]
        danger_by_action: dict[int, bool] = {}
        for action in (0, 1, 2):
            cand_dir = action_to_direction(direction, action)
            cand_head = next_head(head, cand_dir)
            danger = bool(is_danger(board_cells, snake, cand_head))
            danger_by_action[int(action)] = bool(danger)
            if not danger:
                safe_option_count += 1
        prev_safe_option_count = int(self._last_safe_option_count)
        self._last_safe_option_count = int(safe_option_count)
        if int(safe_option_count) <= 1:
            self._narrow_corridor_streak += 1
        else:
            self._narrow_corridor_streak = 0
        board_total = max(1, int(board_cells * board_cells))
        free_ratio = float(max(0, board_total - len(snake))) / float(board_total)
        self._last_free_ratio = float(max(0.0, min(1.0, free_ratio)))
        crowded = bool(self._space_strategy_enabled and free_ratio <= float(self._CROWDED_FREE_RATIO_THRESHOLD))
        food_weight = float(self._FOOD_DIST_WEIGHT_CROWDED if crowded else (self._FOOD_DIST_WEIGHT_OPEN if self._space_strategy_enabled else self._FOOD_DIST_WEIGHT))
        capacity_penalty_scale = float(self._CROWDED_CAPACITY_PENALTY_SCALE) if crowded else 1.0

        proposed_eval = self._evaluate_action(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            action=int(proposed_action),
            food_weight=food_weight,
            capacity_penalty_scale=capacity_penalty_scale,
        )
        if not self.settings.agent_safety_override:
            self._decision_mode_now = ControlMode.PPO
            return int(proposed_action)
        self._last_risk_guard_candidate = False
        self._last_risk_guard_eligible = False
        self._last_risk_guard_blockers = ()
        self._last_pre_no_exit_guard_candidate = False
        self._last_pre_no_exit_guard_applied = False
        self._last_pre_no_exit_guard_blocker = "disabled"
        self._last_pre_no_exit_guard_alt_action = None
        self._last_pre_no_exit_guard_safe_collapsing = False
        self._last_pre_no_exit_guard_near_no_exit_signal = False
        self._last_entered_no_exit_this_step = False
        self._last_action_eval_tuples = {}
        self._last_trace_inputs = None

        if not bool(self._dynamic_cfg.enable_dynamic_control):
            action = self._legacy_safe_action(
                proposed_action=int(proposed_action),
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                food_weight=food_weight,
                capacity_penalty_scale=capacity_penalty_scale,
                proposed_eval=proposed_eval,
            )
            self._decision_mode_now = ControlMode.ESCAPE if action != int(proposed_action) else ControlMode.PPO
            return int(action)

        no_progress_steps = int(self._decisions_total - self._dynamic.last_food_step)
        food_pressure = self._food_pressure(no_progress_steps=no_progress_steps)
        self._last_food_pressure = float(max(0.0, min(1.0, food_pressure)))
        cycle_repeat = self._register_cycle_state(snake=snake, direction=direction, board_cells=board_cells, free_ratio=free_ratio, proposed_eval=proposed_eval)
        self._last_cycle_repeat = bool(cycle_repeat)
        if cycle_repeat:
            self._cycle_repeats_total += 1
        starvation_ratio = self._starvation_progress_ratio()
        if self._should_start_loop_escape(
            cycle_repeat=cycle_repeat,
            no_progress_steps=no_progress_steps,
            starvation_ratio=starvation_ratio,
        ):
            self._start_loop_escape_burst(no_progress_steps=no_progress_steps, starvation_ratio=starvation_ratio)
        proposed_viable = bool(proposed_eval is not None) and self._is_eval_viable(
            board_cells=board_cells,
            snake_len=len(snake),
            tail_reachable=bool(proposed_eval[1]),
            capacity_shortfall=int(proposed_eval[2]),
            food_pressure=food_pressure,
        )
        relaxed_open_viable = bool(
            proposed_eval is not None
            and int(safe_option_count) >= 3
            and float(free_ratio) >= 0.80
            and bool(proposed_eval[1])
            and float(food_pressure) <= 0.70
        )
        effective_proposed_viable = bool(proposed_viable) or bool(relaxed_open_viable)
        candidate_analyses: dict[int, _CandidateAnalysis] = {
            int(proposed_action): self._synthesize_candidate_analysis(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                action=int(proposed_action),
                eval_result=proposed_eval,
                food_pressure=float(food_pressure),
                known_danger=bool(danger_by_action.get(int(proposed_action), proposed_eval is None)),
            )
        }
        for action, is_dangerous in danger_by_action.items():
            if bool(is_dangerous):
                candidate_analyses[int(action)] = self._synthesize_candidate_analysis(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    action=int(action),
                    eval_result=None,
                    food_pressure=float(food_pressure),
                    known_danger=True,
                )
        if int(safe_option_count) <= 1:
            self._ensure_safe_candidate_analyses(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                food_weight=float(food_weight),
                capacity_penalty_scale=float(capacity_penalty_scale),
                food_pressure=float(food_pressure),
                danger_by_action=danger_by_action,
                candidate_analyses=candidate_analyses,
            )
        self._last_candidate_analyses = dict(candidate_analyses)
        self._last_action_eval_tuples = self._build_runtime_action_eval_tuples(
            proposed_action=int(proposed_action),
            board_cells=board_cells,
            snake=snake,
            food=food,
            food_pressure=float(food_pressure),
            no_progress_steps=int(no_progress_steps),
            candidate_analyses=self._last_candidate_analyses,
        )
        self._last_trace_inputs = _DecisionTraceInputs(
            proposed_action=int(proposed_action),
            board_cells=int(board_cells),
            snake=tuple((int(x), int(y)) for x, y in snake),
            direction=(int(direction[0]), int(direction[1])),
            food=(int(food[0]), int(food[1])),
            food_pressure=float(food_pressure),
            no_progress_steps=int(no_progress_steps),
            candidate_analyses=tuple(
                (int(action_key), analysis) for action_key, analysis in sorted(self._last_candidate_analyses.items())
            ),
            action_eval_tuples=tuple(
                (str(action_key), dict(row)) for action_key, row in sorted(self._last_action_eval_tuples.items())
            ),
        )
        prev_no_exit_state = bool(self._last_no_exit_state)
        no_exit_state = self._compute_no_exit_state(
            safe_option_count=int(safe_option_count),
            candidate_analyses=self._last_candidate_analyses,
        )
        self._last_entered_no_exit_this_step = bool(no_exit_state and not bool(self._last_no_exit_state))
        self._last_no_exit_state = bool(no_exit_state)
        no_exit_trend_active = bool(no_exit_state or prev_no_exit_state)
        self._last_proposed_viable = bool(effective_proposed_viable)
        if proposed_eval is not None:
            self._last_proposed_tail_reachable = bool(proposed_eval[1])
            self._last_proposed_capacity_shortfall = int(proposed_eval[2])
        else:
            self._last_proposed_tail_reachable = False
            self._last_proposed_capacity_shortfall = 0
        # Trust high-confidence PPO decisions when they are viable and not near starvation pressure.
        # This reduces controller over-intervention that can underperform PPO-only evaluation.
        conf_threshold = float(getattr(self._dynamic_cfg, "ppo_confidence_trust_threshold", 1.0))
        conf_pressure_max = float(getattr(self._dynamic_cfg, "ppo_confidence_trust_food_pressure_max", 0.0))
        conf_min_free_ratio = float(getattr(self._dynamic_cfg, "ppo_confidence_trust_min_free_ratio", 0.0))
        conf_min_safe_options = max(1, int(getattr(self._dynamic_cfg, "ppo_confidence_trust_min_safe_options", 2)))
        narrow_corridor_trigger = max(1, int(getattr(self._dynamic_cfg, "narrow_corridor_trigger_steps", 6)))
        high_conf_ppo = bool(
            self._last_predicted_confidence is not None
            and float(self._last_predicted_confidence) >= conf_threshold
        )
        narrow_corridor_risk = bool(int(safe_option_count) <= 1 and int(self._narrow_corridor_streak) >= int(narrow_corridor_trigger))
        if (
            effective_proposed_viable
            and high_conf_ppo
            and float(food_pressure) <= conf_pressure_max
            and float(free_ratio) >= conf_min_free_ratio
            and int(safe_option_count) >= int(conf_min_safe_options)
            and not bool(narrow_corridor_risk)
        ):
            self._decision_mode_now = ControlMode.PPO
            self._dynamic.last_switch_reason = "ppo_conf_trust"
            self._last_chosen_tail_reachable = True
            self._last_capacity_shortfall = int(proposed_eval[2]) if proposed_eval is not None else 0
            return int(proposed_action)
        imminent_danger = bool(proposed_eval is None)
        self._last_imminent_danger = bool(imminent_danger)
        risk_override_trigger = bool(
            imminent_danger
            or cycle_repeat
            or int(no_progress_steps) >= int(self._dynamic_cfg.no_progress_steps_space_fill)
            or float(food_pressure) >= float(self._FOOD_PRESSURE_TRIGGER)
            or bool(narrow_corridor_risk)
        )
        # Avoid broad over-intervention: if PPO action is not strictly "viable" but
        # risk pressure is still low, keep PPO action and monitor progression.
        tail_unreachable = bool(proposed_eval is not None and not bool(proposed_eval[1]))
        if (not bool(effective_proposed_viable)) and (not risk_override_trigger) and (not tail_unreachable):
            pre_no_exit_guard_enabled = bool(getattr(self._dynamic_cfg, "enable_pre_no_exit_guard", False))
            if pre_no_exit_guard_enabled and proposed_eval is not None:
                self._last_pre_no_exit_guard_candidate = True
                guard_blockers: list[str] = []
                max_safe_opts = max(1, int(getattr(self._dynamic_cfg, "pre_no_exit_guard_max_safe_options", 2)))
                min_no_progress = max(0, int(getattr(self._dynamic_cfg, "pre_no_exit_guard_min_no_progress_steps", 24)))
                no_exit_safe_opts = max(1, int(getattr(self._dynamic_cfg, "pre_no_exit_guard_no_exit_safe_options", 1)))
                min_shortfall_gain = max(1, int(getattr(self._dynamic_cfg, "pre_no_exit_guard_min_shortfall_gain", 1)))
                require_collapsing = bool(getattr(self._dynamic_cfg, "pre_no_exit_guard_require_collapsing_safe_options", True))
                require_no_exit_signal = bool(getattr(self._dynamic_cfg, "pre_no_exit_guard_require_no_exit_signal", True))
                safe_collapsing = bool(int(safe_option_count) < int(prev_safe_option_count))
                near_no_exit_signal = bool(
                    bool(no_exit_state)
                    or bool(self._last_entered_no_exit_this_step)
                    or int(safe_option_count) <= int(no_exit_safe_opts)
                )
                self._last_pre_no_exit_guard_safe_collapsing = bool(safe_collapsing)
                self._last_pre_no_exit_guard_near_no_exit_signal = bool(near_no_exit_signal)
                if int(safe_option_count) > int(max_safe_opts):
                    guard_blockers.append("safe_options_not_low")
                if int(no_progress_steps) < int(min_no_progress):
                    guard_blockers.append("no_progress_below_floor")
                if require_collapsing and not bool(safe_collapsing):
                    guard_blockers.append("safe_options_not_collapsing")
                if require_no_exit_signal and not bool(near_no_exit_signal):
                    guard_blockers.append("no_exit_signal_missing")
                alt_action = self._best_safe_action(
                    proposed_action=int(proposed_action),
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    food_weight=food_weight,
                    capacity_penalty_scale=capacity_penalty_scale,
                    candidate_analyses=candidate_analyses,
                    food_pressure=food_pressure,
                    no_progress_steps=no_progress_steps,
                    free_ratio=free_ratio,
                )
                self._last_pre_no_exit_guard_alt_action = int(alt_action)
                if int(alt_action) == int(proposed_action):
                    guard_blockers.append("no_alternative_action")
                else:
                    alt_analysis = candidate_analyses.get(int(alt_action))
                    if alt_analysis is None or alt_analysis.eval_result is None:
                        alt_analysis = self._analysis_for_action(
                            board_cells=board_cells,
                            snake=snake,
                            direction=direction,
                            food=food,
                            action=int(alt_action),
                            food_weight=float(food_weight),
                            capacity_penalty_scale=float(capacity_penalty_scale),
                            food_pressure=float(food_pressure),
                        )
                        candidate_analyses[int(alt_action)] = alt_analysis
                    alt_eval = None if alt_analysis is None else alt_analysis.eval_result
                    if alt_eval is None:
                        guard_blockers.append("alt_eval_unavailable")
                    else:
                        alt_viable = self._is_eval_viable(
                            board_cells=board_cells,
                            snake_len=len(snake),
                            tail_reachable=bool(alt_eval[1]),
                            capacity_shortfall=int(alt_eval[2]),
                            food_pressure=food_pressure,
                        )
                        if not bool(alt_viable):
                            guard_blockers.append("alt_not_viable")
                        if not bool(alt_eval[1]):
                            guard_blockers.append("alt_tail_unreachable")
                        proposed_shortfall = int(proposed_eval[2])
                        alt_shortfall = int(alt_eval[2])
                        if int(proposed_shortfall - alt_shortfall) < int(min_shortfall_gain):
                            guard_blockers.append("shortfall_gain_too_small")
                        if not guard_blockers:
                            self._decision_mode_now = ControlMode.ESCAPE
                            self._dynamic.last_switch_reason = "pre_no_exit_guard"
                            self._last_chosen_tail_reachable = bool(alt_eval[1])
                            self._last_capacity_shortfall = int(alt_eval[2])
                            self._last_pre_no_exit_guard_applied = True
                            self._last_pre_no_exit_guard_blocker = "applied"
                            return int(alt_action)
                self._last_pre_no_exit_guard_blocker = (
                    str(guard_blockers[0]) if guard_blockers else "conditions_not_met"
                )
            if bool(getattr(self._dynamic_cfg, "enable_pocket_exit_guard", False)) and proposed_eval is not None:
                max_safe_opts = max(1, int(getattr(self._dynamic_cfg, "pocket_exit_guard_max_safe_options", 2)))
                min_no_progress = max(0, int(getattr(self._dynamic_cfg, "pocket_exit_guard_min_no_progress_steps", 32)))
                min_food_pressure = float(getattr(self._dynamic_cfg, "pocket_exit_guard_min_food_pressure", 0.25))
                min_shortfall_gain = max(1, int(getattr(self._dynamic_cfg, "pocket_exit_guard_min_shortfall_gain", 2)))
                proposed_food_reachable = self._is_food_reachable_after_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    action=int(proposed_action),
                )
                pocket_exit_risk = bool(
                    (not bool(proposed_food_reachable))
                    and int(safe_option_count) <= int(max_safe_opts)
                    and (
                        int(no_progress_steps) >= int(min_no_progress)
                        or float(food_pressure) >= float(min_food_pressure)
                    )
                )
                if pocket_exit_risk:
                    alt_action = self._best_safe_action(
                        proposed_action=int(proposed_action),
                        board_cells=board_cells,
                        snake=snake,
                        direction=direction,
                        food=food,
                        food_weight=food_weight,
                        capacity_penalty_scale=capacity_penalty_scale,
                        candidate_analyses=candidate_analyses,
                        food_pressure=food_pressure,
                        no_progress_steps=no_progress_steps,
                        free_ratio=free_ratio,
                    )
                    if int(alt_action) != int(proposed_action):
                        alt_analysis = candidate_analyses.get(int(alt_action))
                        if alt_analysis is None or alt_analysis.eval_result is None:
                            alt_analysis = self._analysis_for_action(
                                board_cells=board_cells,
                                snake=snake,
                                direction=direction,
                                food=food,
                                action=int(alt_action),
                                food_weight=float(food_weight),
                                capacity_penalty_scale=float(capacity_penalty_scale),
                                food_pressure=float(food_pressure),
                            )
                            candidate_analyses[int(alt_action)] = alt_analysis
                        alt_eval = None if alt_analysis is None else alt_analysis.eval_result
                        if alt_eval is not None:
                            alt_viable = self._is_eval_viable(
                                board_cells=board_cells,
                                snake_len=len(snake),
                                tail_reachable=bool(alt_eval[1]),
                                capacity_shortfall=int(alt_eval[2]),
                                food_pressure=food_pressure,
                            )
                            alt_food_reachable = self._is_food_reachable_after_action(
                                board_cells=board_cells,
                                snake=snake,
                                direction=direction,
                                food=food,
                                action=int(alt_action),
                            )
                            proposed_shortfall = int(proposed_eval[2])
                            alt_shortfall = int(alt_eval[2])
                            measured_gain = bool(
                                bool(alt_viable)
                                or (
                                    bool(alt_food_reachable)
                                    and int(proposed_shortfall - alt_shortfall) >= int(min_shortfall_gain)
                                )
                            )
                            if measured_gain:
                                self._decision_mode_now = ControlMode.ESCAPE
                                self._dynamic.last_switch_reason = "pocket_exit_guard"
                                self._last_chosen_tail_reachable = bool(alt_eval[1])
                                self._last_capacity_shortfall = int(alt_eval[2])
                                return int(alt_action)
            self._decision_mode_now = ControlMode.PPO
            self._dynamic.last_switch_reason = "ppo_tolerate_low_risk"
            if proposed_eval is not None:
                self._last_chosen_tail_reachable = bool(proposed_eval[1])
                self._last_capacity_shortfall = int(proposed_eval[2])
            else:
                self._last_chosen_tail_reachable = True
                self._last_capacity_shortfall = 0
            return int(proposed_action)
        significant_risk = bool((not bool(effective_proposed_viable)) and risk_override_trigger)
        if bool(narrow_corridor_risk):
            significant_risk = True
        self._last_risk_guard_candidate = bool(significant_risk)
        risk_guard_blockers: list[str] = []
        risk_guard_eligible = False
        risk_guard_enabled = bool(getattr(self._dynamic_cfg, "enable_risk_switch_guard", False))
        allow_narrow_corridor_guard = bool(getattr(self._dynamic_cfg, "risk_switch_guard_allow_narrow_corridor", False))
        narrow_guard_active = bool(narrow_corridor_risk) and bool(allow_narrow_corridor_guard)
        risk_guard_no_progress_floor = int(self._dynamic_cfg.no_progress_steps_escape)
        if bool(narrow_guard_active):
            risk_guard_no_progress_floor = max(
                0,
                int(getattr(self._dynamic_cfg, "risk_switch_guard_narrow_min_no_progress_steps", 16)),
            )
        if not bool(significant_risk):
            risk_guard_blockers.append("not_significant_risk")
        else:
            if bool(imminent_danger):
                risk_guard_blockers.append("imminent_danger")
            if proposed_eval is None:
                risk_guard_blockers.append("proposed_eval_unavailable")
            if int(no_progress_steps) < int(risk_guard_no_progress_floor):
                if bool(narrow_guard_active):
                    risk_guard_blockers.append("no_progress_below_narrow_floor")
                else:
                    risk_guard_blockers.append("no_progress_below_escape")
            if float(food_pressure) >= float(self._FOOD_PRESSURE_TRIGGER):
                risk_guard_blockers.append("food_pressure_high")
            if bool(narrow_corridor_risk) and not bool(allow_narrow_corridor_guard):
                risk_guard_blockers.append("narrow_corridor")
            if self._last_predicted_confidence is None:
                risk_guard_blockers.append("confidence_missing")
            if not bool(risk_guard_enabled):
                risk_guard_blockers.append("guard_disabled")
            if not risk_guard_blockers:
                conf_min = float(getattr(self._dynamic_cfg, "risk_switch_guard_confidence_min", 0.95))
                if bool(narrow_guard_active):
                    conf_min = max(
                        float(conf_min),
                        float(getattr(self._dynamic_cfg, "risk_switch_guard_narrow_confidence_min", conf_min)),
                    )
                min_safe_options = max(1, int(getattr(self._dynamic_cfg, "risk_switch_guard_min_safe_options", 2)))
                no_progress_margin = max(1, int(getattr(self._dynamic_cfg, "risk_switch_guard_no_progress_margin", 20)))
                if bool(narrow_guard_active):
                    no_progress_margin = max(
                        0,
                        int(getattr(self._dynamic_cfg, "risk_switch_guard_narrow_no_progress_margin", 0)),
                    )
                if float(self._last_predicted_confidence) < float(conf_min):
                    risk_guard_blockers.append("confidence_too_low")
                if int(safe_option_count) < int(min_safe_options):
                    risk_guard_blockers.append("safe_options_below_threshold")
                if int(no_progress_steps) > int(risk_guard_no_progress_floor) + int(no_progress_margin):
                    risk_guard_blockers.append("no_progress_margin_failed")
                if not risk_guard_blockers:
                    risk_guard_eligible = True
        self._last_risk_guard_eligible = bool(risk_guard_eligible)
        self._last_risk_guard_blockers = tuple(risk_guard_blockers)
        # Narrow risk-switch guard (feature-flagged): in cycle-only risk transitions,
        # keep PPO mode when no measured safety gain is available from a safe fallback.
        if (
            bool(significant_risk)
            and bool(risk_guard_enabled)
            and not bool(imminent_danger)
            and proposed_eval is not None
            and int(no_progress_steps) >= int(risk_guard_no_progress_floor)
            and float(food_pressure) < float(self._FOOD_PRESSURE_TRIGGER)
            and (not bool(narrow_corridor_risk) or bool(allow_narrow_corridor_guard))
            and self._last_predicted_confidence is not None
        ):
            conf_min = float(getattr(self._dynamic_cfg, "risk_switch_guard_confidence_min", 0.95))
            if bool(narrow_guard_active):
                conf_min = max(
                    float(conf_min),
                    float(getattr(self._dynamic_cfg, "risk_switch_guard_narrow_confidence_min", conf_min)),
                )
            min_safe_options = max(1, int(getattr(self._dynamic_cfg, "risk_switch_guard_min_safe_options", 2)))
            min_shortfall_gain = max(1, int(getattr(self._dynamic_cfg, "risk_switch_guard_min_shortfall_gain", 2)))
            no_progress_margin = max(1, int(getattr(self._dynamic_cfg, "risk_switch_guard_no_progress_margin", 20)))
            if bool(narrow_guard_active):
                no_progress_margin = max(
                    0,
                    int(getattr(self._dynamic_cfg, "risk_switch_guard_narrow_no_progress_margin", 0)),
                )
            if (
                float(self._last_predicted_confidence) >= float(conf_min)
                and int(safe_option_count) >= int(min_safe_options)
                and int(no_progress_steps) <= int(risk_guard_no_progress_floor) + int(no_progress_margin)
            ):
                alt_action = self._best_safe_action(
                    proposed_action=int(proposed_action),
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    food_weight=food_weight,
                    capacity_penalty_scale=capacity_penalty_scale,
                    candidate_analyses=candidate_analyses,
                    food_pressure=food_pressure,
                    no_progress_steps=no_progress_steps,
                    free_ratio=free_ratio,
                )
                if int(alt_action) == int(proposed_action):
                    significant_risk = False
                    self._dynamic.last_switch_reason = "risk_guard_hold"
                else:
                    alt_analysis = candidate_analyses.get(int(alt_action))
                    if alt_analysis is None or alt_analysis.eval_result is None:
                        alt_analysis = self._analysis_for_action(
                            board_cells=board_cells,
                            snake=snake,
                            direction=direction,
                            food=food,
                            action=int(alt_action),
                            food_weight=float(food_weight),
                            capacity_penalty_scale=float(capacity_penalty_scale),
                            food_pressure=float(food_pressure),
                        )
                        candidate_analyses[int(alt_action)] = alt_analysis
                    alt_eval = None if alt_analysis is None else alt_analysis.eval_result
                    if alt_eval is not None:
                        proposed_tail_ok = bool(proposed_eval[1])
                        proposed_shortfall = int(proposed_eval[2])
                        alt_tail_ok = bool(alt_eval[1])
                        alt_shortfall = int(alt_eval[2])
                        safety_gain = bool(
                            (not proposed_tail_ok and alt_tail_ok)
                            or (int(proposed_shortfall - alt_shortfall) >= int(min_shortfall_gain))
                        )
                        if not bool(safety_gain):
                            significant_risk = False
                            self._dynamic.last_switch_reason = "risk_guard_hold"
        mode = self._select_mode(
            significant_risk=significant_risk,
            imminent_danger=imminent_danger,
            cycle_repeat=cycle_repeat,
            no_progress_steps=no_progress_steps,
            safe_option_count=int(safe_option_count),
            proposed_tail_reachable=bool(proposed_eval[1]) if proposed_eval is not None else False,
            proposed_capacity_shortfall=int(proposed_eval[2]) if proposed_eval is not None else 0,
        )
        self._decision_mode_now = mode
        warmup_steps = max(0, int(getattr(self._dynamic_cfg, "dynamic_warmup_steps", 0)))
        if (
            mode == ControlMode.PPO
            and effective_proposed_viable
            and int(self._decisions_total) < warmup_steps
            and not bool(narrow_corridor_risk)
        ):
            self._dynamic.last_switch_reason = "warmup_ppo"
            self._last_chosen_tail_reachable = True
            self._last_capacity_shortfall = int(proposed_eval[2]) if proposed_eval is not None else 0
            return int(proposed_action)

        if (
            mode == ControlMode.PPO
            and effective_proposed_viable
            and int(self._loop_escape_steps_left) <= 0
            and not bool(narrow_corridor_risk)
            and not (
                float(food_pressure) > conf_pressure_max
                and int(safe_option_count) >= conf_min_safe_options
            )
        ):
            self._dynamic.last_switch_reason = "ppo_mode_viable"
            self._last_chosen_tail_reachable = True
            self._last_capacity_shortfall = int(proposed_eval[2]) if proposed_eval is not None else 0
            return int(proposed_action)

        action = None
        if self._loop_escape_steps_left > 0:
            allow_loop_escape = bool(mode != ControlMode.PPO) or int(no_progress_steps) >= int(self._loop_escape_hard_trigger())
            if allow_loop_escape:
                action = self._choose_loop_escape_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    food_weight=food_weight,
                    capacity_penalty_scale=capacity_penalty_scale,
                    candidate_analyses=candidate_analyses,
                    food_pressure=food_pressure,
                    no_progress_steps=no_progress_steps,
                    free_ratio=free_ratio,
                )
                if action is not None:
                    self._loop_escape_steps_left = max(0, int(self._loop_escape_steps_left) - 1)
                    self._dynamic.last_switch_reason = "loop_escape_active"
        food_pressure_override_allowed = bool(
            int(safe_option_count) > 1 and not bool(no_exit_trend_active)
        )
        if (
            action is None
            and not imminent_danger
            and bool(food_pressure_override_allowed)
            and float(food_pressure) >= float(self._FOOD_PRESSURE_TRIGGER)
        ):
            action = self._best_safe_action(
                proposed_action=int(proposed_action),
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                food_weight=food_weight,
                capacity_penalty_scale=capacity_penalty_scale,
                candidate_analyses=candidate_analyses,
                food_pressure=food_pressure,
                no_progress_steps=no_progress_steps,
                free_ratio=free_ratio,
            )
            if mode != ControlMode.PPO:
                self._dynamic.last_switch_reason = "food_pressure"
        if action is None and mode == ControlMode.ESCAPE:
            action = self._escape_controller.choose_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
            )
        # Low-confidence fallback in controller-managed modes
        low_conf_trigger_reasons = {"food_pressure", "risk", "no_progress_escape"}
        if (
            action is None
            and mode in (ControlMode.SPACE_FILL, ControlMode.ESCAPE)
            and self._last_predicted_confidence is not None
            and float(self._last_predicted_confidence) < 0.60
            and str(self._dynamic.last_switch_reason) in low_conf_trigger_reasons
        ):
            fallback_action = self._best_safe_action(
                proposed_action=int(proposed_action),
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                food_weight=food_weight,
                capacity_penalty_scale=capacity_penalty_scale,
                candidate_analyses=candidate_analyses,
                food_pressure=food_pressure,
                no_progress_steps=no_progress_steps,
                free_ratio=free_ratio,
            )
            if fallback_action is not None:
                action = fallback_action
                self._dynamic.last_switch_reason = "low_conf_fallback"
        elif action is None and mode == ControlMode.SPACE_FILL:
            if cycle_repeat:
                action = self._choose_cycle_break_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    food_weight=food_weight,
                    capacity_penalty_scale=capacity_penalty_scale,
                    candidate_analyses=candidate_analyses,
                    food_pressure=food_pressure,
                    no_progress_steps=no_progress_steps,
                    free_ratio=free_ratio,
                )
            action = self._space_fill_controller.choose_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                prev_action=self._last_action,
                config=self._dynamic_cfg,
            ) if action is None else action
            if action is None:
                action = self._escape_controller.choose_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                )
        if action is None:
            action = self._best_safe_action(
                proposed_action=int(proposed_action),
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                food_weight=food_weight,
                capacity_penalty_scale=capacity_penalty_scale,
                candidate_analyses=candidate_analyses,
                food_pressure=food_pressure,
                no_progress_steps=no_progress_steps,
                free_ratio=free_ratio,
            )
        open_field_pressure_max = float(getattr(self._dynamic_cfg, "ppo_open_field_trust_food_pressure_max", 0.35))
        open_field_trust = bool(
            mode == ControlMode.PPO
            and int(action) != int(proposed_action)
            and bool(proposed_viable)
            and int(safe_option_count) >= 3
            and float(food_pressure) <= float(open_field_pressure_max)
            and int(no_progress_steps) < int(self._dynamic_cfg.no_progress_steps_escape)
            and proposed_eval is not None
        )
        if open_field_trust:
            action = int(proposed_action)
            self._dynamic.last_switch_reason = "ppo_open_field_trust"
        chosen_analysis = candidate_analyses.get(int(action))
        if chosen_analysis is None:
            chosen_analysis = self._analysis_for_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                action=int(action),
                food_weight=float(food_weight),
                capacity_penalty_scale=float(capacity_penalty_scale),
                food_pressure=float(food_pressure),
            )
            candidate_analyses[int(action)] = chosen_analysis
            self._last_candidate_analyses = dict(candidate_analyses)
        chosen_eval = None if chosen_analysis is None else chosen_analysis.eval_result
        if (
            int(action) != int(proposed_action)
            and not imminent_danger
            and proposed_eval is not None
            and chosen_eval is not None
            and bool(self._learning_enabled)
            and bool(getattr(self._dynamic_cfg, "enable_learned_arbiter", False))
            and mode == ControlMode.PPO
            and effective_proposed_viable
            and int(self._loop_escape_steps_left) <= 0
        ):
            features = self._decision_features(
                free_ratio=free_ratio,
                food_pressure=food_pressure,
                no_progress_steps=no_progress_steps,
                cycle_repeat=cycle_repeat,
                imminent_danger=imminent_danger,
                proposed_viable=effective_proposed_viable,
                proposed_eval=proposed_eval,
                chosen_eval=chosen_eval,
            )
            proba = float(self._arbiter_model.predict_proba(features))
            threshold = float(getattr(self._dynamic_cfg, "arbiter_threshold", 0.56))
            if proba < threshold:
                action = int(proposed_action)
                chosen_eval = proposed_eval
                self._dynamic.last_switch_reason = "arbiter_veto"
                self._arbiter_model.update(features, label=0, weight=1.0)
            else:
                delta_ok = float(chosen_eval[0]) > float(proposed_eval[0])
                self._arbiter_model.update(features, label=1 if delta_ok else 0, weight=1.0)
            self._arbiter_dirty = True
        if chosen_eval is not None:
            _score, tail_reachable, capacity_shortfall = chosen_eval
            self._last_chosen_tail_reachable = bool(tail_reachable)
            if self._last_chosen_tail_reachable:
                self._tail_reachable_streak += 1
                self._tail_unreachable_streak = 0
            else:
                self._tail_unreachable_streak += 1
                self._tail_reachable_streak = 0
            self._last_capacity_shortfall = int(capacity_shortfall)
        else:
            self._last_chosen_tail_reachable = True
            self._tail_reachable_streak = 0
            self._tail_unreachable_streak = 0
            self._last_capacity_shortfall = 0
        high_conf_guard_threshold = float(getattr(self._dynamic_cfg, "ppo_high_conf_override_guard_threshold", 0.97))
        high_conf_guard_pressure_max = float(getattr(self._dynamic_cfg, "ppo_high_conf_override_guard_food_pressure_max", 0.6))
        high_conf_guard_min_safe_options = max(1, int(getattr(self._dynamic_cfg, "ppo_high_conf_override_guard_min_safe_options", 2)))
        high_conf_guard_min_shortfall_gain = max(0, int(getattr(self._dynamic_cfg, "ppo_high_conf_override_guard_min_shortfall_gain", 2)))
        high_conf_override_guard = bool(
            int(action) != int(proposed_action)
            and not bool(imminent_danger)
            and proposed_eval is not None
            and chosen_eval is not None
            and self._last_predicted_confidence is not None
            and float(self._last_predicted_confidence) >= float(high_conf_guard_threshold)
            and float(food_pressure) <= float(high_conf_guard_pressure_max)
            and int(safe_option_count) >= int(high_conf_guard_min_safe_options)
        )
        if high_conf_override_guard:
            chosen_tail_ok = bool(chosen_eval[1])
            chosen_shortfall = int(chosen_eval[2])
            proposed_tail_ok = bool(proposed_eval[1])
            proposed_shortfall = int(proposed_eval[2])
            safety_gain = bool(
                (not proposed_tail_ok and chosen_tail_ok)
                or (int(proposed_shortfall - chosen_shortfall) >= int(high_conf_guard_min_shortfall_gain))
            )
            if not bool(safety_gain):
                action = int(proposed_action)
                chosen_eval = proposed_eval
                self._last_chosen_tail_reachable = bool(proposed_tail_ok)
                self._last_capacity_shortfall = int(proposed_shortfall)
                self._dynamic.last_switch_reason = "ppo_high_conf_guard"
        decision_features = self._decision_features(
            free_ratio=free_ratio,
            food_pressure=food_pressure,
            no_progress_steps=no_progress_steps,
            cycle_repeat=cycle_repeat,
            imminent_danger=imminent_danger,
            proposed_viable=effective_proposed_viable,
            proposed_eval=proposed_eval,
            chosen_eval=chosen_eval,
        )
        self._last_decision_context = _DecisionContext(
            features=tuple(float(v) for v in decision_features),
            proposed_action=int(proposed_action),
            chosen_action=int(action),
            override_used=bool(int(action) != int(proposed_action)),
        )
        return int(action)

    def _compute_no_exit_state(
        self,
        *,
        safe_option_count: int,
        candidate_analyses: dict[int, _CandidateAnalysis],
    ) -> bool:
        if int(safe_option_count) > 1:
            return False
        if not candidate_analyses:
            return False
        for action in (0, 1, 2):
            analysis = candidate_analyses.get(int(action))
            if analysis is None or analysis.eval_result is None:
                continue
            if bool(analysis.viable) and bool(analysis.tail_reachable):
                return False
        return True

    def _choose_loop_escape_action(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        food_weight: float,
        capacity_penalty_scale: float,
        candidate_analyses: dict[int, _CandidateAnalysis] | None = None,
        food_pressure: float | None = None,
        no_progress_steps: int | None = None,
        free_ratio: float | None = None,
    ) -> int | None:
        if not snake:
            return None
        head = snake[0]
        current_food_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        current_food_pressure = self._food_pressure() if food_pressure is None else float(food_pressure)
        current_no_progress = int(self._decisions_total - self._dynamic.last_food_step) if no_progress_steps is None else int(no_progress_steps)
        current_free_ratio = (
            float(max(0, board_cells * board_cells - len(snake))) / float(max(1, board_cells * board_cells))
            if free_ratio is None
            else float(free_ratio)
        )
        analyses = candidate_analyses or self._collect_candidate_analyses(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            food_weight=float(food_weight),
            capacity_penalty_scale=float(capacity_penalty_scale),
            food_pressure=float(current_food_pressure),
        )

        best_action: int | None = None
        best_key: tuple[float, ...] | None = None
        for action in (0, 1, 2):
            analysis = analyses.get(int(action))
            if analysis is None:
                analysis = self._analysis_for_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    action=int(action),
                    food_weight=float(food_weight),
                    capacity_penalty_scale=float(capacity_penalty_scale),
                    food_pressure=float(current_food_pressure),
                )
                analyses[int(action)] = analysis
            eval_result = analysis.eval_result
            if eval_result is None:
                continue
            score, tail_ok, shortfall = eval_result
            revisit_count = int(analysis.revisit_count)
            next_food_dist = int(analysis.next_food_dist)
            food_progress = int(current_food_dist - next_food_dist)
            turn_change = 1 if self._last_action is not None and int(action) != int(self._last_action) else 0
            key = (
                float(food_progress),
                float(-revisit_count),
                float(1 if tail_ok else 0),
                float(-int(shortfall)),
                float(turn_change),
                float(score)
                + float(
                    self._tactic_memory.action_bias(
                        features=self._decision_features(
                            free_ratio=float(current_free_ratio),
                            food_pressure=float(current_food_pressure),
                            no_progress_steps=int(current_no_progress),
                            cycle_repeat=False,
                            imminent_danger=False,
                            proposed_viable=True,
                            proposed_eval=eval_result,
                            chosen_eval=eval_result,
                        ),
                        action=int(action),
                    )
                ) if bool(getattr(self._dynamic_cfg, "enable_tactic_memory", False)) else float(score),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_action = int(action)
        return best_action

    def _choose_cycle_break_action(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        food_weight: float,
        capacity_penalty_scale: float,
        candidate_analyses: dict[int, _CandidateAnalysis] | None = None,
        food_pressure: float | None = None,
        no_progress_steps: int | None = None,
        free_ratio: float | None = None,
    ) -> int | None:
        if not snake:
            return None
        head = snake[0]
        current_food_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        current_food_pressure = self._food_pressure() if food_pressure is None else float(food_pressure)
        current_no_progress = int(self._decisions_total - self._dynamic.last_food_step) if no_progress_steps is None else int(no_progress_steps)
        current_free_ratio = (
            float(max(0, board_cells * board_cells - len(snake))) / float(max(1, board_cells * board_cells))
            if free_ratio is None
            else float(free_ratio)
        )
        analyses = candidate_analyses or self._collect_candidate_analyses(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            food_weight=float(food_weight),
            capacity_penalty_scale=float(capacity_penalty_scale),
            food_pressure=float(current_food_pressure),
        )

        best_action: int | None = None
        best_key: tuple[float, ...] | None = None
        for action in (0, 1, 2):
            analysis = analyses.get(int(action))
            if analysis is None:
                analysis = self._analysis_for_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    action=int(action),
                    food_weight=float(food_weight),
                    capacity_penalty_scale=float(capacity_penalty_scale),
                    food_pressure=float(current_food_pressure),
                )
                analyses[int(action)] = analysis
            eval_result = analysis.eval_result
            if eval_result is None:
                continue
            score, tail_ok, shortfall = eval_result
            revisit_count = int(analysis.revisit_count)
            next_food_dist = int(analysis.next_food_dist)
            food_progress = int(current_food_dist - next_food_dist)
            turn_change = 1 if self._last_action is not None and int(action) != int(self._last_action) else 0
            key = (
                float(-revisit_count),
                float(1 if tail_ok else 0),
                float(-int(shortfall)),
                float(food_progress),
                float(score)
                + float(
                    self._tactic_memory.action_bias(
                        features=self._decision_features(
                            free_ratio=float(current_free_ratio),
                            food_pressure=float(current_food_pressure),
                            no_progress_steps=int(current_no_progress),
                            cycle_repeat=True,
                            imminent_danger=False,
                            proposed_viable=True,
                            proposed_eval=eval_result,
                            chosen_eval=eval_result,
                        ),
                        action=int(action),
                    )
                ) if bool(getattr(self._dynamic_cfg, "enable_tactic_memory", False)) else float(score),
                float(turn_change),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_action = int(action)
        return best_action

    def _legacy_safe_action(
        self,
        *,
        proposed_action: int,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        food_weight: float,
        capacity_penalty_scale: float,
        proposed_eval: tuple[float, bool, int] | None,
    ) -> int:
        if self._should_use_escape_controller(
            snake=snake,
            board_cells=board_cells,
            free_ratio=float(max(0, board_cells * board_cells - len(snake))) / float(max(1, board_cells * board_cells)),
            proposed_eval=proposed_eval,
        ):
            action = self._escape_controller.choose_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
            )
            if action is not None:
                return int(action)
        if proposed_eval is not None:
            _s, tail_ok, shortfall = proposed_eval
            if self._is_eval_viable(
                board_cells=board_cells,
                snake_len=len(snake),
                tail_reachable=bool(tail_ok),
                capacity_shortfall=int(shortfall),
                food_pressure=self._food_pressure(),
            ):
                return int(proposed_action)
        return self._best_safe_action(
            proposed_action=proposed_action,
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            food_weight=food_weight,
            capacity_penalty_scale=capacity_penalty_scale,
        )

    def _best_safe_action(
        self,
        *,
        proposed_action: int,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        food_weight: float,
        capacity_penalty_scale: float,
        candidate_analyses: dict[int, _CandidateAnalysis] | None = None,
        food_pressure: float | None = None,
        no_progress_steps: int | None = None,
        free_ratio: float | None = None,
    ) -> int:
        current_food_dist = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])
        current_food_pressure = self._food_pressure() if food_pressure is None else float(food_pressure)
        current_no_progress = int(self._decisions_total - self._dynamic.last_food_step) if no_progress_steps is None else int(no_progress_steps)
        current_free_ratio = (
            float(max(0, board_cells * board_cells - len(snake))) / float(max(1, board_cells * board_cells))
            if free_ratio is None
            else float(free_ratio)
        )
        analyses = candidate_analyses or self._collect_candidate_analyses(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            food_weight=float(food_weight),
            capacity_penalty_scale=float(capacity_penalty_scale),
            food_pressure=float(current_food_pressure),
        )
        best_action = int(proposed_action)
        best_key: tuple[float, ...] | None = None
        for action in (0, 1, 2):
            analysis = analyses.get(int(action))
            if analysis is None:
                analysis = self._analysis_for_action(
                    board_cells=board_cells,
                    snake=snake,
                    direction=direction,
                    food=food,
                    action=int(action),
                    food_weight=float(food_weight),
                    capacity_penalty_scale=float(capacity_penalty_scale),
                    food_pressure=float(current_food_pressure),
                )
                analyses[int(action)] = analysis
            eval_result = analysis.eval_result
            if eval_result is None:
                continue
            score, tail_ok, shortfall = eval_result
            candidate_head = analysis.candidate_head
            revisit_count = int(analysis.revisit_count)
            next_food_dist = int(analysis.next_food_dist)
            food_progress = int(current_food_dist - next_food_dist)
            viable = bool(analysis.viable)
            key = (
                float(1 if viable else 0),
                float(1 if candidate_head == food else 0),
                float(food_progress if current_food_pressure >= float(self._FOOD_PRESSURE_TRIGGER) else 0.0),
                float(score) + (float(food_progress) * float(current_food_pressure) * float(self._FOOD_PROGRESS_SCORE_WEIGHT)),
                float(-revisit_count) * float(current_food_pressure) * float(self._FOOD_REVISIT_PENALTY),
                float(
                    self._tactic_memory.action_bias(
                        features=self._decision_features(
                            free_ratio=float(current_free_ratio),
                            food_pressure=float(current_food_pressure),
                            no_progress_steps=int(current_no_progress),
                            cycle_repeat=False,
                            imminent_danger=False,
                            proposed_viable=bool(viable),
                            proposed_eval=eval_result,
                            chosen_eval=eval_result,
                        ),
                        action=int(action),
                    )
                ) if bool(getattr(self._dynamic_cfg, "enable_tactic_memory", False)) else 0.0,
                float(-next_food_dist),
                float(1 if int(action) == int(proposed_action) else 0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_action = int(action)
        return int(best_action)

    def _build_runtime_action_eval_tuples(
        self,
        *,
        proposed_action: int,
        board_cells: int,
        snake: list[tuple[int, int]],
        food: tuple[int, int],
        food_pressure: float,
        no_progress_steps: int,
        candidate_analyses: dict[int, _CandidateAnalysis],
    ) -> dict[str, dict[str, object]]:
        board_total = max(1, int(board_cells * board_cells))
        free_ratio = float(max(0, board_cells * board_cells - len(snake))) / float(max(1, board_cells * board_cells))
        out: dict[str, dict[str, object]] = {}
        for action in (0, 1, 2):
            analysis = candidate_analyses.get(int(action))
            if analysis is None:
                continue
            eval_result = analysis.eval_result
            row: dict[str, object] = {
                "danger": bool(analysis.danger),
                "reachable_cells": int(analysis.reachable_count),
                "reachable_ratio": float(float(analysis.reachable_count) / float(board_total)),
                "next_food_dist": int(analysis.next_food_dist),
                "food_progress": int(analysis.food_progress),
                "revisit_count": int(analysis.revisit_count),
                "is_proposed_action": bool(int(action) == int(proposed_action)),
                "food_pressure": float(food_pressure),
                "no_progress_steps": int(no_progress_steps),
                "eval_available": bool(eval_result is not None),
                "score": None,
                "tail_reachable": None,
                "capacity_shortfall": None,
                "viable": False,
                "rank_inputs": None,
            }
            if eval_result is not None:
                score, tail_ok, shortfall = eval_result
                tactic_bias = float(
                    self._tactic_memory.action_bias(
                        features=self._decision_features(
                            free_ratio=float(free_ratio),
                            food_pressure=float(food_pressure),
                            no_progress_steps=int(no_progress_steps),
                            cycle_repeat=False,
                            imminent_danger=bool(analysis.danger),
                            proposed_viable=bool(analysis.viable),
                            proposed_eval=eval_result,
                            chosen_eval=eval_result,
                        ),
                        action=int(action),
                    )
                ) if bool(getattr(self._dynamic_cfg, "enable_tactic_memory", False)) else 0.0
                rank_inputs = {
                    "viable_bit": int(1 if bool(analysis.viable) else 0),
                    "food_hit_bit": int(1 if analysis.candidate_head == food else 0),
                    "food_progress_pressure": float(
                        analysis.food_progress if float(food_pressure) >= float(self._FOOD_PRESSURE_TRIGGER) else 0.0
                    ),
                    "score_blend": float(score) + (float(analysis.food_progress) * float(food_pressure) * float(self._FOOD_PROGRESS_SCORE_WEIGHT)),
                    "revisit_penalty": float(-analysis.revisit_count) * float(food_pressure) * float(self._FOOD_REVISIT_PENALTY),
                    "tactic_bias": float(tactic_bias),
                    "neg_next_food_dist": float(-analysis.next_food_dist),
                    "same_as_proposed_bit": int(1 if int(action) == int(proposed_action) else 0),
                }
                row.update(
                    {
                        "score": float(score),
                        "tail_reachable": bool(tail_ok),
                        "capacity_shortfall": int(shortfall),
                        "viable": bool(analysis.viable),
                        "rank_inputs": rank_inputs,
                    }
                )
            out[str(int(action))] = row
        return out

    @staticmethod
    def _candidate_cache_key(
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        action: int,
        food_weight: float,
        capacity_penalty_scale: float,
    ) -> tuple[object, ...]:
        return (
            int(board_cells),
            tuple(snake),
            tuple(direction),
            tuple(food),
            int(action),
            float(food_weight),
            float(capacity_penalty_scale),
        )

    def _collect_candidate_analyses(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        food_weight: float,
        capacity_penalty_scale: float,
        food_pressure: float,
        known_danger_by_action: dict[int, bool] | None = None,
    ) -> dict[int, _CandidateAnalysis]:
        analyses: dict[int, _CandidateAnalysis] = {}
        for action in (0, 1, 2):
            analyses[int(action)] = self._analysis_for_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                action=int(action),
                food_weight=float(food_weight),
                capacity_penalty_scale=float(capacity_penalty_scale),
                food_pressure=float(food_pressure),
                known_danger=None if known_danger_by_action is None else known_danger_by_action.get(int(action)),
            )
        return analyses

    def _ensure_safe_candidate_analyses(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        food_weight: float,
        capacity_penalty_scale: float,
        food_pressure: float,
        danger_by_action: dict[int, bool],
        candidate_analyses: dict[int, _CandidateAnalysis],
    ) -> None:
        for action in (0, 1, 2):
            if bool(danger_by_action.get(int(action), False)):
                continue
            analysis = candidate_analyses.get(int(action))
            if analysis is not None and analysis.eval_result is not None:
                continue
            candidate_analyses[int(action)] = self._analysis_for_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                action=int(action),
                food_weight=float(food_weight),
                capacity_penalty_scale=float(capacity_penalty_scale),
                food_pressure=float(food_pressure),
            )

    def _analysis_for_action(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        action: int,
        food_weight: float,
        capacity_penalty_scale: float,
        food_pressure: float,
        known_danger: bool | None = None,
    ) -> _CandidateAnalysis:
        key = self._candidate_cache_key(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            action=action,
            food_weight=food_weight,
            capacity_penalty_scale=capacity_penalty_scale,
        )
        cached = self._candidate_eval_cache.get(key)
        if cached is not None:
            return cached
        if bool(known_danger):
            analysis = self._synthesize_candidate_analysis(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                action=int(action),
                eval_result=None,
                food_pressure=float(food_pressure),
                known_danger=True,
            )
            self._candidate_eval_cache[key] = analysis
            return analysis
        eval_result = self._evaluate_action(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            action=int(action),
            food_weight=float(food_weight),
            capacity_penalty_scale=float(capacity_penalty_scale),
        )
        cached = self._candidate_eval_cache.get(key)
        if cached is not None:
            return cached
        return self._synthesize_candidate_analysis(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            action=int(action),
            eval_result=eval_result,
            food_pressure=float(food_pressure),
            known_danger=bool(eval_result is None),
        )

    def _synthesize_candidate_analysis(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        action: int,
        eval_result: tuple[float, bool, int] | None,
        food_pressure: float,
        known_danger: bool,
    ) -> _CandidateAnalysis:
        head = snake[0]
        candidate_direction = action_to_direction(direction, int(action))
        candidate_head = next_head(head, candidate_direction)
        next_food_dist = int(abs(candidate_head[0] - food[0]) + abs(candidate_head[1] - food[1]))
        current_food_dist = int(abs(head[0] - food[0]) + abs(head[1] - food[1]))
        revisit_count = int(sum(1 for point in self._recent_heads if point == candidate_head))
        tail_reachable = bool(eval_result[1]) if eval_result is not None else False
        capacity_shortfall = int(eval_result[2]) if eval_result is not None else 0
        viable = bool(
            eval_result is not None
            and self._is_eval_viable(
                board_cells=board_cells,
                snake_len=len(snake),
                tail_reachable=tail_reachable,
                capacity_shortfall=capacity_shortfall,
                food_pressure=float(food_pressure),
            )
        )
        return _CandidateAnalysis(
            action=int(action),
            candidate_head=(int(candidate_head[0]), int(candidate_head[1])),
            danger=bool(known_danger),
            simulated_snake=(),
            reachable_count=0,
            tail_reachable=bool(tail_reachable),
            capacity_shortfall=int(capacity_shortfall),
            next_food_dist=int(next_food_dist),
            food_progress=int(current_food_dist - next_food_dist),
            revisit_count=int(revisit_count),
            score=None if eval_result is None else float(eval_result[0]),
            eval_result=eval_result,
            viable=bool(viable),
        )

    def _register_cycle_state(
        self,
        *,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        board_cells: int,
        free_ratio: float,
        proposed_eval: tuple[float, bool, int] | None,
    ) -> bool:
        if not snake:
            return False
        head = snake[0]
        tail = snake[-1]
        safe_sig = []
        for action in (0, 1, 2):
            cand_dir = action_to_direction(direction, action)
            cand_head = next_head(head, cand_dir)
            safe_sig.append(0 if is_danger(board_cells, snake, cand_head) else 1)
        tail_rel = (
            -1 if tail[0] < head[0] else (1 if tail[0] > head[0] else 0),
            -1 if tail[1] < head[1] else (1 if tail[1] > head[1] else 0),
        )
        risk_bit = 1
        if proposed_eval is not None:
            risk_bit = 0 if self._is_eval_viable(
                board_cells=board_cells,
                snake_len=len(snake),
                tail_reachable=bool(proposed_eval[1]),
                capacity_shortfall=int(proposed_eval[2]),
                food_pressure=self._food_pressure(),
            ) else 1
        bucket = int(free_ratio * 10.0)
        cycle_hash = hash((head, direction, tuple(safe_sig), tail_rel, risk_bit, bucket))
        window = self._dynamic.cycle_window
        counts = self._dynamic.cycle_hash_counts
        maxlen = max(6, int(self._dynamic_cfg.cycle_window_steps))
        if len(window) >= maxlen:
            old = window.popleft()
            old_count = int(counts.get(old, 0))
            if old_count <= 1:
                counts.pop(old, None)
            else:
                counts[old] = int(old_count - 1)
        window.append(cycle_hash)
        counts[cycle_hash] = int(counts.get(cycle_hash, 0) + 1)
        return int(counts[cycle_hash]) >= int(self._dynamic_cfg.cycle_repeat_threshold)

    def _select_mode(
        self,
        *,
        significant_risk: bool,
        imminent_danger: bool,
        cycle_repeat: bool,
        no_progress_steps: int,
        safe_option_count: int,
        proposed_tail_reachable: bool,
        proposed_capacity_shortfall: int,
    ) -> ControlMode:
        current = self._dynamic.current_mode
        desired = current
        reason = "hold"
        cycle_break_counted = False
        if current == ControlMode.PPO:
            if significant_risk:
                desired = ControlMode.ESCAPE
                reason = "risk"
        elif current == ControlMode.ESCAPE:
            if imminent_danger:
                desired = ControlMode.SPACE_FILL
                reason = "imminent_danger"
            elif cycle_repeat:
                desired = ControlMode.SPACE_FILL
                reason = "cycle_repeat"
            elif no_progress_steps >= int(self._dynamic_cfg.no_progress_steps_escape):
                # In narrow corridors with a tail-reachable path and no pocket shortfall,
                # keep ESCAPE instead of forcing an early SPACE_FILL transition.
                if (
                    int(safe_option_count) <= 1
                    and bool(proposed_tail_reachable)
                    and int(proposed_capacity_shortfall) <= 0
                ):
                    desired = ControlMode.ESCAPE
                    reason = "escape_hold_narrow_corridor"
                else:
                    desired = ControlMode.SPACE_FILL
                    reason = "no_progress_escape"
            elif not significant_risk and no_progress_steps <= int(self._dynamic_cfg.risk_recovery_window):
                desired = ControlMode.PPO
                reason = "risk_cleared"
        else:  # SPACE_FILL
            if imminent_danger:
                desired = ControlMode.ESCAPE
                reason = "imminent_danger"
            elif not significant_risk and no_progress_steps <= int(self._dynamic_cfg.risk_recovery_window):
                desired = ControlMode.PPO
                reason = "space_fill_recovered"
            elif cycle_repeat and no_progress_steps >= int(self._dynamic_cfg.no_progress_steps_space_fill):
                self._cycle_breaks_total += 1
                cycle_break_counted = True
                self._episode_stuck = True
                reason = "space_fill_cycle_break"

        step_now = int(self._decisions_total)
        if desired != current and step_now < int(self._dynamic.cooldown_until_step) and not imminent_danger:
            desired = current
        if desired != current:
            self._mode_switches_total += 1
            self._dynamic.current_mode = desired
            self._dynamic.mode_enter_step = step_now
            self._dynamic.last_switch_reason = str(reason)
            self._dynamic.cooldown_until_step = int(step_now + int(self._dynamic_cfg.mode_switch_cooldown_steps))
            if ("cycle" in reason or "no_progress" in reason) and not bool(cycle_break_counted):
                self._cycle_breaks_total += 1
                self._episode_stuck = True
        return self._dynamic.current_mode

    def _should_use_escape_controller(
        self,
        *,
        snake: list[tuple[int, int]],
        board_cells: int,
        free_ratio: float,
        proposed_eval: tuple[float, bool, int] | None,
    ) -> bool:
        if not bool(self._space_strategy_enabled):
            return False
        if proposed_eval is None:
            return True
        _score, tail_reachable, capacity_shortfall = proposed_eval
        board_total = max(1, int(board_cells * board_cells))
        length_ratio = float(len(snake)) / float(board_total)
        crowded_or_long = (
            float(free_ratio) <= float(self._ESCAPE_FREE_RATIO_THRESHOLD)
            or float(length_ratio) >= float(self._ESCAPE_LENGTH_RATIO_THRESHOLD)
        )
        return bool(
            crowded_or_long
            and not self._is_eval_viable(
                board_cells=board_cells,
                snake_len=len(snake),
                tail_reachable=bool(tail_reachable),
                capacity_shortfall=int(capacity_shortfall),
                food_pressure=self._food_pressure(),
            )
        )

    def _evaluate_action(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        action: int,
        food_weight: float,
        capacity_penalty_scale: float,
    ) -> tuple[float, bool, int] | None:
        key = self._candidate_cache_key(
            board_cells=board_cells,
            snake=snake,
            direction=direction,
            food=food,
            action=action,
            food_weight=food_weight,
            capacity_penalty_scale=capacity_penalty_scale,
        )
        cached = self._candidate_eval_cache.get(key)
        if cached is not None:
            return cached.eval_result
        head = snake[0]
        candidate_direction = action_to_direction(direction, int(action))
        candidate_head = next_head(head, candidate_direction)
        next_food_dist = int(abs(candidate_head[0] - food[0]) + abs(candidate_head[1] - food[1]))
        current_food_dist = int(abs(head[0] - food[0]) + abs(head[1] - food[1]))
        revisit_count = int(sum(1 for point in self._recent_heads if point == candidate_head))
        if is_danger(board_cells, snake, candidate_head):
            self._candidate_eval_cache[key] = _CandidateAnalysis(
                action=int(action),
                candidate_head=(int(candidate_head[0]), int(candidate_head[1])),
                danger=True,
                simulated_snake=(),
                reachable_count=0,
                tail_reachable=False,
                capacity_shortfall=0,
                next_food_dist=int(next_food_dist),
                food_progress=int(current_food_dist - next_food_dist),
                revisit_count=int(revisit_count),
                score=None,
                eval_result=None,
                viable=False,
            )
            return None
        simulated_snake = self._simulate_next_snake(snake, candidate_head, food)
        reachable = self._reachable_space(board_cells, simulated_snake, candidate_head)
        tail_reachable = self._tail_is_reachable(board_cells, simulated_snake)
        capacity_shortfall = max(0, len(simulated_snake) - reachable)
        score = float(reachable) - (float(food_weight) * float(next_food_dist))
        board_total = max(1, int(board_cells * board_cells))
        length_ratio = float(len(simulated_snake)) / float(board_total)
        food_pressure = self._food_pressure()
        tail_penalty_scale = max(
            0.45,
            min(
                1.25,
                (0.45 + (0.9 * float(length_ratio)) + (0.25 * max(0.0, float(food_pressure) - 0.5))),
            ),
        )
        if tail_reachable:
            score += self._TAIL_REACHABLE_BONUS
        else:
            score -= self._TAIL_UNREACHABLE_PENALTY * float(tail_penalty_scale)
        if capacity_shortfall > 0:
            score -= self._CAPACITY_SHORTFALL_PENALTY * float(capacity_penalty_scale) * float(capacity_shortfall)
        depth = max(1, int(getattr(self._dynamic_cfg, "lookahead_depth", 3)))
        lookahead = self._lookahead_viability(
            board_cells=board_cells,
            snake=simulated_snake,
            direction=candidate_direction,
            food=food,
            depth=depth,
        )
        score += float(getattr(self._dynamic_cfg, "lookahead_weight", 0.0)) * float(lookahead)
        eval_result = (float(score), bool(tail_reachable), int(capacity_shortfall))
        self._candidate_eval_cache[key] = _CandidateAnalysis(
            action=int(action),
            candidate_head=(int(candidate_head[0]), int(candidate_head[1])),
            danger=False,
            simulated_snake=tuple(simulated_snake),
            reachable_count=int(reachable),
            tail_reachable=bool(tail_reachable),
            capacity_shortfall=int(capacity_shortfall),
            next_food_dist=int(next_food_dist),
            food_progress=int(current_food_dist - next_food_dist),
            revisit_count=int(revisit_count),
            score=float(score),
            eval_result=eval_result,
            viable=bool(
                self._is_eval_viable(
                    board_cells=board_cells,
                    snake_len=len(snake),
                    tail_reachable=bool(tail_reachable),
                    capacity_shortfall=int(capacity_shortfall),
                    food_pressure=float(food_pressure),
                )
            ),
        )
        return eval_result

    def _food_pressure(self, *, no_progress_steps: int | None = None) -> float:
        starvation_pressure = self._starvation_progress_ratio()
        steps_since_food = (
            max(0, int(no_progress_steps))
            if no_progress_steps is not None
            else int(max(0, self._decisions_total - self._dynamic.last_food_step))
        )
        soft_trigger = max(1, int(self._dynamic_cfg.no_progress_steps_escape))
        hard_trigger = max(soft_trigger + 1, int(self._dynamic_cfg.no_progress_steps_space_fill))
        if steps_since_food <= soft_trigger:
            no_progress_pressure = 0.0
        else:
            no_progress_pressure = min(
                1.0,
                float(steps_since_food - soft_trigger) / float(max(1, hard_trigger - soft_trigger)),
            )
        return float(max(0.0, min(1.0, max(float(starvation_pressure), float(no_progress_pressure)))))

    def _capacity_shortfall_limit(self, *, board_cells: int, snake_len: int, food_pressure: float) -> int:
        board_total = max(1, int(board_cells * board_cells))
        length_ratio = float(snake_len) / float(board_total)
        limit = 0
        if length_ratio >= 0.08:
            limit = 1
        if length_ratio >= 0.20:
            limit = 2
        if float(food_pressure) >= float(self._FOOD_PRESSURE_TRIGGER):
            limit += 1
        return int(min(int(self._SHORTFALL_TOLERANCE_MAX), int(limit)))

    def _is_eval_viable(
        self,
        *,
        board_cells: int,
        snake_len: int,
        tail_reachable: bool,
        capacity_shortfall: int,
        food_pressure: float,
    ) -> bool:
        if not bool(tail_reachable):
            return False
        limit = self._capacity_shortfall_limit(
            board_cells=board_cells,
            snake_len=snake_len,
            food_pressure=food_pressure,
        )
        return int(capacity_shortfall) <= int(limit)

    def _lookahead_viability(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        depth: int,
    ) -> float:
        if depth <= 0 or not snake:
            return 1.0
        head = snake[0]
        children: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        for action in (0, 1, 2):
            next_direction = action_to_direction(direction, action)
            candidate_head = next_head(head, next_direction)
            if is_danger(board_cells, snake, candidate_head):
                continue
            children.append(
                (
                    self._simulate_next_snake(snake, candidate_head, food),
                    next_direction,
                )
            )
        if not children:
            return 0.0
        if depth <= 1:
            return float(len(children)) / 3.0
        branch_score = float(len(children)) / 3.0
        child_scores = [
            self._lookahead_viability(
                board_cells=board_cells,
                snake=next_snake,
                direction=next_direction,
                food=food,
                depth=int(depth - 1),
            )
            for next_snake, next_direction in children
        ]
        best_future = max(child_scores) if child_scores else 0.0
        return float(max(0.0, min(1.0, (0.45 * branch_score) + (0.55 * float(best_future)))))

    def set_space_strategy_enabled(self, enabled: bool) -> None:
        self._space_strategy_enabled = bool(enabled)

    def is_space_strategy_enabled(self) -> bool:
        return bool(self._space_strategy_enabled)

    def current_control_mode(self) -> str:
        return str(self._dynamic.current_mode.value)

    def last_mode_switch_reason(self) -> str:
        return str(self._dynamic.last_switch_reason)

    def _update_debug_snapshot(
        self,
        *,
        predicted_action: int,
        chosen_action: int,
        action_probs: tuple[float, float, float] | None,
        include_reachable_cells: bool,
        candidate_analyses: dict[int, _CandidateAnalysis] | None = None,
    ) -> None:
        board_cells = int(self.settings.board_cells)
        snake = list(self.game.snake)
        direction = tuple(self.game.direction)
        food = tuple(self.game.food)
        head = snake[0]
        total = float(board_cells * board_cells)
        analyses = candidate_analyses or {}
        candidates: list[CandidateDebug] = []
        for action in (0, 1, 2):
            analysis = analyses.get(int(action))
            candidate_direction = action_to_direction(direction, action)
            candidate_head = analysis.candidate_head if analysis is not None else next_head(head, candidate_direction)
            danger = bool(analysis.danger) if analysis is not None else bool(is_danger(board_cells, snake, candidate_head))
            reachable_ratio = 0.0
            reachable_cells: tuple[tuple[int, int], ...] = ()
            if not danger:
                simulated = list(analysis.simulated_snake) if analysis is not None and analysis.simulated_snake else self._simulate_next_snake(snake, candidate_head, food)
                if include_reachable_cells:
                    reachable = self._reachable_cells(board_cells, simulated, candidate_head)
                    reachable_ratio = float(len(reachable)) / total
                    reachable_cells = tuple(sorted(reachable))
                else:
                    reachable_count = (
                        int(analysis.reachable_count)
                        if analysis is not None and int(analysis.reachable_count) > 0
                        else self._reachable_space(board_cells, simulated, candidate_head)
                    )
                    reachable_ratio = float(reachable_count) / total
            candidates.append(
                CandidateDebug(
                    action=int(action),
                    cell=(int(candidate_head[0]), int(candidate_head[1])),
                    danger=bool(danger),
                    reachable_ratio=float(max(0.0, min(1.0, reachable_ratio))),
                    reachable_cells=reachable_cells,
                )
            )
        self._debug_snapshot = AgentDebugSnapshot(
            head=(int(head[0]), int(head[1])),
            predicted_action=int(predicted_action),
            chosen_action=int(chosen_action),
            candidates=(candidates[0], candidates[1], candidates[2]),
            action_probs=action_probs,
        )

    @staticmethod
    def _simulate_next_snake(
        snake: list[tuple[int, int]],
        new_head: tuple[int, int],
        food: tuple[int, int],
    ) -> list[tuple[int, int]]:
        return simulate_next_snake(snake, new_head, food)

    @staticmethod
    def _reachable_space(board_cells: int, snake_after_move: list[tuple[int, int]], start: tuple[int, int]) -> int:
        return int(reachable_cell_count(board_cells, snake_after_move, start))

    @staticmethod
    def _reachable_cells(board_cells: int, snake_after_move: list[tuple[int, int]], start: tuple[int, int]) -> set[tuple[int, int]]:
        return board_reachable_cells(board_cells, snake_after_move, start)

    @staticmethod
    def _tail_is_reachable(board_cells: int, snake_after_move: list[tuple[int, int]]) -> bool:
        return bool(board_tail_is_reachable(board_cells, snake_after_move))

    def _is_food_reachable_after_action(
        self,
        *,
        board_cells: int,
        snake: list[tuple[int, int]],
        direction: tuple[int, int],
        food: tuple[int, int],
        action: int,
    ) -> bool:
        if not snake:
            return False
        candidate_direction = action_to_direction(direction, int(action))
        candidate_head = next_head(snake[0], candidate_direction)
        if is_danger(board_cells, snake, candidate_head):
            return False
        simulated = self._simulate_next_snake(snake, candidate_head, food)
        reachable = self._reachable_cells(board_cells, simulated, candidate_head)
        return tuple(food) in reachable

    def telemetry_snapshot(self) -> GameplayTelemetrySnapshot:
        avg_conf = float(sum(self._death_confidences)) / float(len(self._death_confidences)) if self._death_confidences else 0.0
        starvation_steps = int(getattr(self.game, "steps_without_food", 0))
        starvation_limit = int(getattr(self.game, "starvation_limit", lambda: 0)())
        no_progress_steps = int(max(0, self._decisions_total - self._dynamic.last_food_step))
        return GameplayTelemetrySnapshot(
            decisions_total=int(self._decisions_total),
            interventions_total=int(self._interventions_total),
            pocket_risk_total=int(self._pocket_risk_total),
            tail_unreachable_total=int(self._tail_unreachable_total),
            deaths_wall=int(self._deaths_wall),
            deaths_body=int(self._deaths_body),
            deaths_starvation=int(self._deaths_starvation),
            deaths_fill=int(self._deaths_fill),
            deaths_other=int(self._deaths_other),
            deaths_early=int(self._deaths_early),
            deaths_mid=int(self._deaths_mid),
            deaths_late=int(self._deaths_late),
            avg_death_confidence=float(avg_conf),
            decisions_mode_ppo=int(self._decisions_mode_ppo),
            decisions_mode_escape=int(self._decisions_mode_escape),
            decisions_mode_space_fill=int(self._decisions_mode_space_fill),
            mode_switches_total=int(self._mode_switches_total),
            cycle_breaks_total=int(self._cycle_breaks_total),
            stuck_episodes_total=int(self._stuck_episodes_total),
            cycle_repeats_total=int(self._cycle_repeats_total),
            no_progress_steps=int(no_progress_steps),
            starvation_steps=int(starvation_steps),
            starvation_limit=int(starvation_limit),
            loop_escape_activations_total=int(self._loop_escape_activations_total),
            loop_escape_steps_left=int(self._loop_escape_steps_left),
            current_mode=str(self.current_control_mode()),
            last_switch_reason=str(self.last_mode_switch_reason()),
            last_death_reason=str(self._last_death_reason),
        )

    def decision_trace_snapshot(self) -> dict[str, object]:
        action_eval_tuples = {str(k): dict(v) for k, v in self._last_action_eval_tuples.items()}
        if self._last_trace_inputs is not None:
            trace_inputs = self._last_trace_inputs
            candidate_analyses = {int(action): analysis for action, analysis in trace_inputs.candidate_analyses}
            if len(candidate_analyses) < 3:
                board_cells = int(trace_inputs.board_cells)
                snake = list(trace_inputs.snake)
                direction = tuple(trace_inputs.direction)
                food = tuple(trace_inputs.food)
                free_ratio = float(max(0, board_cells * board_cells - len(snake))) / float(max(1, board_cells * board_cells))
                crowded = bool(self._space_strategy_enabled and free_ratio <= float(self._CROWDED_FREE_RATIO_THRESHOLD))
                food_weight = float(
                    self._FOOD_DIST_WEIGHT_CROWDED
                    if crowded
                    else (self._FOOD_DIST_WEIGHT_OPEN if self._space_strategy_enabled else self._FOOD_DIST_WEIGHT)
                )
                capacity_penalty_scale = float(self._CROWDED_CAPACITY_PENALTY_SCALE) if crowded else 1.0
                for action in (0, 1, 2):
                    if int(action) not in candidate_analyses:
                        candidate_analyses[int(action)] = self._analysis_for_action(
                            board_cells=board_cells,
                            snake=snake,
                            direction=direction,
                            food=food,
                            action=int(action),
                            food_weight=float(food_weight),
                            capacity_penalty_scale=float(capacity_penalty_scale),
                            food_pressure=float(trace_inputs.food_pressure),
                        )
            action_eval_tuples = self._build_runtime_action_eval_tuples(
                proposed_action=int(trace_inputs.proposed_action),
                board_cells=int(trace_inputs.board_cells),
                snake=list(trace_inputs.snake),
                food=tuple(trace_inputs.food),
                food_pressure=float(trace_inputs.food_pressure),
                no_progress_steps=int(trace_inputs.no_progress_steps),
                candidate_analyses=candidate_analyses,
            )
        no_progress_steps = int(max(0, self._decisions_total - self._dynamic.last_food_step))
        starvation_steps = int(getattr(self.game, "steps_without_food", 0))
        starvation_limit = int(getattr(self.game, "starvation_limit", lambda: 0)())
        row: dict[str, object] = {
            "decision_index": int(self._decisions_total),
            "predicted_action": None if self._last_predicted_action is None else int(self._last_predicted_action),
            "chosen_action": None if self._last_chosen_action is None else int(self._last_chosen_action),
            "override_used": bool(
                self._last_predicted_action is not None
                and self._last_chosen_action is not None
                and int(self._last_predicted_action) != int(self._last_chosen_action)
            ),
            "action_probs": None if self._last_action_probs is None else [float(v) for v in self._last_action_probs],
            "predicted_confidence": None if self._last_predicted_confidence is None else float(self._last_predicted_confidence),
            "chosen_confidence": None if self._last_chosen_confidence is None else float(self._last_chosen_confidence),
            "mode": str(self.current_control_mode()),
            "switch_reason": str(self.last_mode_switch_reason()),
            "free_ratio": float(self._last_free_ratio),
            "food_pressure": float(self._last_food_pressure),
            "safe_option_count": int(self._last_safe_option_count),
            "narrow_corridor_streak": int(self._narrow_corridor_streak),
            "cycle_repeat": bool(self._last_cycle_repeat),
            "imminent_danger": bool(self._last_imminent_danger),
            "proposed_viable": bool(self._last_proposed_viable),
            "proposed_tail_reachable": bool(self._last_proposed_tail_reachable),
            "proposed_capacity_shortfall": int(self._last_proposed_capacity_shortfall),
            "chosen_tail_reachable": bool(self._last_chosen_tail_reachable),
            "chosen_capacity_shortfall": int(self._last_capacity_shortfall),
            "risk_guard_candidate": bool(self._last_risk_guard_candidate),
            "risk_guard_eligible": bool(self._last_risk_guard_eligible),
            "risk_guard_blockers": [str(v) for v in self._last_risk_guard_blockers],
            "pre_no_exit_guard_candidate": bool(self._last_pre_no_exit_guard_candidate),
            "pre_no_exit_guard_applied": bool(self._last_pre_no_exit_guard_applied),
            "pre_no_exit_guard_blocker": str(self._last_pre_no_exit_guard_blocker),
            "pre_no_exit_guard_alt_action": (
                None if self._last_pre_no_exit_guard_alt_action is None else int(self._last_pre_no_exit_guard_alt_action)
            ),
            "pre_no_exit_guard_safe_collapsing": bool(self._last_pre_no_exit_guard_safe_collapsing),
            "pre_no_exit_guard_near_no_exit_signal": bool(self._last_pre_no_exit_guard_near_no_exit_signal),
            "no_exit_state": bool(self._last_no_exit_state),
            "entered_no_exit_this_step": bool(self._last_entered_no_exit_this_step),
            "action_eval_tuples": action_eval_tuples,
            "no_progress_steps": int(no_progress_steps),
            "starvation_steps": int(starvation_steps),
            "starvation_limit": int(starvation_limit),
            "loop_escape_steps_left": int(self._loop_escape_steps_left),
            "interventions_total": int(self._interventions_total),
            "pocket_risk_total": int(self._pocket_risk_total),
            "cycle_repeats_total": int(self._cycle_repeats_total),
        }
        snapshot = self._debug_snapshot
        if snapshot is not None:
            row["candidate_reachable_ratio"] = {
                str(int(candidate.action)): float(candidate.reachable_ratio) for candidate in snapshot.candidates
            }
            row["candidate_danger"] = {
                str(int(candidate.action)): bool(candidate.danger) for candidate in snapshot.candidates
            }
        return row

    def reset_episode_tracking(self) -> None:
        self._reset_dynamic_state()
        self._last_score_seen = int(getattr(self.game, "score", 0))
        self._debug_snapshot = None

    def _should_start_loop_escape(
        self,
        *,
        cycle_repeat: bool,
        no_progress_steps: int,
        starvation_ratio: float,
    ) -> bool:
        if not bool(cycle_repeat):
            return False
        if self._loop_escape_steps_left > 0:
            return False
        if int(self._decisions_total) < int(self._loop_escape_cooldown_until):
            return False
        no_progress_floor = int(self._dynamic_cfg.no_progress_steps_escape)
        no_progress_soft = max(
            no_progress_floor,
            int(0.75 * float(self._dynamic_cfg.no_progress_steps_space_fill)),
        )
        if int(no_progress_steps) < no_progress_soft:
            return False
        hard_no_progress_trigger = self._loop_escape_hard_trigger()
        if int(no_progress_steps) >= int(hard_no_progress_trigger):
            return True
        starvation_trigger = float(self._dynamic_cfg.loop_escape_starvation_trigger_ratio)
        if float(starvation_ratio) >= starvation_trigger:
            return True
        return bool(self._food_distance_stalled())

    def _loop_escape_hard_trigger(self) -> int:
        return max(
            int(self._dynamic_cfg.no_progress_steps_space_fill) * 2,
            int(self._dynamic_cfg.no_progress_steps_escape) * 3,
        )

    def _start_loop_escape_burst(self, *, no_progress_steps: int, starvation_ratio: float) -> None:
        base = max(4, int(self._dynamic_cfg.loop_escape_base_steps))
        max_steps = max(base, int(self._dynamic_cfg.loop_escape_max_steps))
        floor = max(1, int(self._dynamic_cfg.no_progress_steps_escape))
        no_progress_extra = max(0, int(no_progress_steps - floor)) // max(1, floor // 2)
        starvation_extra = int(max(0.0, float(starvation_ratio)) * 12.0)
        steps = min(max_steps, max(base, int(base + no_progress_extra + starvation_extra)))
        self._loop_escape_steps_left = int(steps)
        self._loop_escape_activations_total += 1
        cooldown = max(1, int(self._dynamic_cfg.loop_escape_cooldown_steps))
        self._loop_escape_cooldown_until = int(self._decisions_total + steps + cooldown)
        self._dynamic.last_switch_reason = "loop_escape_start"

    def _food_distance_stalled(self) -> bool:
        window = max(6, int(self._dynamic_cfg.loop_escape_stall_window))
        if len(self._recent_food_distances) < window:
            return False
        recent = list(self._recent_food_distances)[-window:]
        spread = int(max(recent) - min(recent))
        return spread <= 1

    def _starvation_progress_ratio(self) -> float:
        limit_fn = getattr(self.game, "starvation_limit", None)
        if not callable(limit_fn):
            return 0.0
        limit = max(1, int(limit_fn()))
        steps = max(0, int(getattr(self.game, "steps_without_food", 0)))
        return float(steps) / float(limit)

    def _record_decision(
        self,
        *,
        predicted_action: int,
        chosen_action: int,
        action_probs: tuple[float, float, float] | None,
    ) -> None:
        self._decisions_total += 1
        if self._decision_mode_now == ControlMode.PPO:
            self._decisions_mode_ppo += 1
        elif self._decision_mode_now == ControlMode.ESCAPE:
            self._decisions_mode_escape += 1
        else:
            self._decisions_mode_space_fill += 1
        if int(predicted_action) != int(chosen_action):
            self._interventions_total += 1
        if not bool(self._last_chosen_tail_reachable):
            self._tail_unreachable_total += 1
        if int(self._last_capacity_shortfall) > 0:
            self._pocket_risk_total += 1
        self._last_chosen_confidence = None
        if action_probs is None:
            return
        idx = int(chosen_action)
        if 0 <= idx <= 2:
            self._last_chosen_confidence = float(action_probs[idx])

    def _record_episode_end(self) -> None:
        reason = str(self.game.death_reason or "other")
        self._last_death_reason = reason
        success = bool(reason in ("fill",)) or bool(int(getattr(self.game, "score", 0)) >= 100)
        self._reinforce_recent_contexts(success=success, weight=2.0 if success else 1.0)
        if reason == "wall":
            self._deaths_wall += 1
        elif reason == "body":
            self._deaths_body += 1
        elif reason == "starvation":
            self._deaths_starvation += 1
        elif reason == "fill":
            self._deaths_fill += 1
        else:
            self._deaths_other += 1
        board_total = max(1, int(self.settings.board_cells * self.settings.board_cells))
        length_ratio = float(len(self.game.snake)) / float(board_total)
        if length_ratio < 0.33:
            self._deaths_early += 1
        elif length_ratio < 0.66:
            self._deaths_mid += 1
        else:
            self._deaths_late += 1
        count_as_stuck = bool(self._episode_stuck) and reason not in ("starvation", "fill")
        if count_as_stuck:
            self._stuck_episodes_total += 1
        self._episode_stuck = False
        if self._last_chosen_confidence is not None:
            self._death_confidences.append(float(self._last_chosen_confidence))
        self._persist_learning_state()

    def _reset_dynamic_state(self) -> None:
        self._dynamic = DynamicControllerState()
        self._last_action = None
        self._last_predicted_confidence = None
        self._recent_heads.clear()
        self._recent_food_distances.clear()
        self._decision_contexts.clear()
        self._last_decision_context = None
        self._loop_escape_steps_left = 0
        self._loop_escape_cooldown_until = 0
        self._decision_mode_now = ControlMode.PPO
        self._episode_stuck = False
        self._narrow_corridor_streak = 0
        self._last_pre_no_exit_guard_candidate = False
        self._last_pre_no_exit_guard_applied = False
        self._last_pre_no_exit_guard_blocker = "disabled"
        self._last_pre_no_exit_guard_alt_action = None
        self._last_pre_no_exit_guard_safe_collapsing = False
        self._last_pre_no_exit_guard_near_no_exit_signal = False
        self._last_no_exit_state = False
        self._last_entered_no_exit_this_step = False
