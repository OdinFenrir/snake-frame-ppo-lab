from __future__ import annotations

from dataclasses import dataclass

from .board_analysis import is_point_danger, reachable_cell_count, simulate_next_snake, tail_is_reachable
from .observation import action_to_direction, is_danger, next_head
from .settings import DynamicControlConfig

Point = tuple[int, int]
Direction = tuple[int, int]


@dataclass(frozen=True)
class SpaceFillDecision:
    action: int
    score: float


class SpaceFillController:
    def choose_action(
        self,
        *,
        board_cells: int,
        snake: list[Point],
        direction: Direction,
        food: Point,
        prev_action: int | None,
        config: DynamicControlConfig,
    ) -> int | None:
        best: SpaceFillDecision | None = None
        for action in (0, 1, 2):
            score = self._evaluate_action(
                board_cells=board_cells,
                snake=snake,
                direction=direction,
                food=food,
                action=action,
                prev_action=prev_action,
                config=config,
            )
            if score is None:
                continue
            candidate = SpaceFillDecision(action=int(action), score=float(score))
            if best is None or float(candidate.score) > float(best.score):
                best = candidate
        return None if best is None else int(best.action)

    def _evaluate_action(
        self,
        *,
        board_cells: int,
        snake: list[Point],
        direction: Direction,
        food: Point,
        action: int,
        prev_action: int | None,
        config: DynamicControlConfig,
    ) -> float | None:
        head = snake[0]
        candidate_direction = action_to_direction(direction, int(action))
        candidate_head = next_head(head, candidate_direction)
        if is_danger(board_cells, snake, candidate_head):
            return None
        simulated = simulate_next_snake(snake, candidate_head, food)
        reachable = int(reachable_cell_count(board_cells, simulated, candidate_head))
        margin = int(reachable - len(simulated))
        shortfall = max(0, int(len(simulated) - reachable))
        tail_ok = bool(tail_is_reachable(board_cells, simulated))
        wall_distance = int(
            min(
                candidate_head[0],
                candidate_head[1],
                (board_cells - 1) - candidate_head[0],
                (board_cells - 1) - candidate_head[1],
            )
        )
        food_dist = abs(candidate_head[0] - food[0]) + abs(candidate_head[1] - food[1])
        zigzag = bool(prev_action is not None and int(action) in (1, 2) and int(prev_action) in (1, 2) and int(action) != int(prev_action))

        score = 0.0
        score += float(config.space_fill_tail_reachable_bonus if tail_ok else -config.space_fill_tail_unreachable_penalty)
        score += float(config.space_fill_reachable_margin_weight) * float(margin)
        score -= float(config.space_fill_capacity_shortfall_penalty) * float(shortfall)
        score += float(config.space_fill_wall_distance_weight) * float(wall_distance)
        score -= float(config.space_fill_food_distance_weight) * float(food_dist)
        if zigzag:
            score -= float(config.space_fill_zigzag_penalty)
        next_safe_options = 0
        for next_action in (0, 1, 2):
            next_dir = action_to_direction(candidate_direction, next_action)
            next_pos = next_head(candidate_head, next_dir)
            if not is_point_danger(board_cells, simulated, next_pos):
                next_safe_options += 1
        if next_safe_options <= 1:
            penalty = getattr(config, "space_fill_low_liberty_penalty", 500.0)
            score -= float(penalty)
        elif next_safe_options == 2:
            penalty = getattr(config, "space_fill_low_liberty_penalty", 500.0) * 0.3
            score -= float(penalty)
        return float(score)
