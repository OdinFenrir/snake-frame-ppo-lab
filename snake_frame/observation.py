from __future__ import annotations

import numpy as np

from .board_analysis import is_point_danger, reachable_cell_count, reachable_space_ratio as board_reachable_space_ratio, simulate_next_snake, tail_path_length as board_tail_path_length
from .settings import ObsConfig

Direction = tuple[int, int]
Point = tuple[int, int]

UP: Direction = (0, -1)
RIGHT: Direction = (1, 0)
DOWN: Direction = (0, 1)
LEFT: Direction = (-1, 0)
TURN_LEFT_MAP: dict[Direction, Direction] = {
    UP: LEFT,
    LEFT: DOWN,
    DOWN: RIGHT,
    RIGHT: UP,
}
TURN_RIGHT_MAP: dict[Direction, Direction] = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}


def turn_left(direction: Direction) -> Direction:
    return TURN_LEFT_MAP.get(tuple(direction), tuple(direction))


def turn_right(direction: Direction) -> Direction:
    return TURN_RIGHT_MAP.get(tuple(direction), tuple(direction))


def action_to_direction(direction: Direction, action: int) -> Direction:
    if action == 1:
        return turn_left(direction)
    if action == 2:
        return turn_right(direction)
    return direction


def next_head(head: Point, direction: Direction) -> Point:
    return head[0] + direction[0], head[1] + direction[1]


def is_danger(board_cells: int, snake: list[Point], point: Point) -> bool:
    return bool(is_point_danger(board_cells, snake, point))


def observation_size(obs_config: ObsConfig) -> int:
    base = 11
    if obs_config.use_extended_features:
        base += 6
    if obs_config.use_path_features:
        base += 3
    if obs_config.use_tail_path_features:
        base += 8
    if obs_config.use_free_space_features:
        base += 1
    if obs_config.use_tail_trend_features:
        base += 2
    return base


def _tail_direction_features(snake: list[Point]) -> tuple[float, float, float, float]:
    if len(snake) < 2:
        return 0.0, 0.0, 0.0, 0.0
    tail = snake[-1]
    before_tail = snake[-2]
    direction = (before_tail[0] - tail[0], before_tail[1] - tail[1])
    return (
        float(direction == UP),
        float(direction == RIGHT),
        float(direction == DOWN),
        float(direction == LEFT),
    )


def _simulate_next_snake(snake: list[Point], new_head: Point, food: Point) -> list[Point]:
    return simulate_next_snake(snake, new_head, food)


def _reachable_ratio(board_cells: int, snake_after_move: list[Point], start: Point) -> float:
    return float(board_reachable_space_ratio(board_cells, snake_after_move, start))


def _tail_path_features(board_cells: int, snake_after_move: list[Point]) -> tuple[float, float]:
    path_len = board_tail_path_length(board_cells, snake_after_move)
    reachable = 1.0 if path_len is not None else 0.0
    max_path = max(1, (int(board_cells) - 1) * 2)
    path_norm = 1.0 if path_len is None else min(1.0, float(path_len) / float(max_path))
    return reachable, float(path_norm)


def valid_action_mask(
    board_cells: int,
    snake: list[Point],
    direction: Direction,
) -> tuple[bool, bool, bool]:
    head = snake[0]
    mask = []
    for action in (0, 1, 2):
        candidate_direction = action_to_direction(direction, action)
        candidate_head = next_head(head, candidate_direction)
        mask.append(not is_danger(board_cells, snake, candidate_head))
    if any(mask):
        return bool(mask[0]), bool(mask[1]), bool(mask[2])
    return True, True, True


def build_observation(
    board_cells: int,
    snake: list[Point],
    direction: Direction,
    food: Point,
    obs_config: ObsConfig | None = None,
    tail_reachable_streak: int = 0,
    tail_unreachable_streak: int = 0,
) -> np.ndarray:
    config = obs_config or ObsConfig()
    head = snake[0]
    dir_left = turn_left(direction)
    dir_right = turn_right(direction)

    danger_straight = float(is_danger(board_cells, snake, next_head(head, direction)))
    danger_left = float(is_danger(board_cells, snake, next_head(head, dir_left)))
    danger_right = float(is_danger(board_cells, snake, next_head(head, dir_right)))

    dir_up = float(direction == UP)
    dir_right_f = float(direction == RIGHT)
    dir_down = float(direction == DOWN)
    dir_left_f = float(direction == LEFT)

    food_left = float(food[0] < head[0])
    food_right = float(food[0] > head[0])
    food_up = float(food[1] < head[1])
    food_down = float(food[1] > head[1])

    values: list[float] = [
        danger_straight,
        danger_left,
        danger_right,
        dir_up,
        dir_right_f,
        dir_down,
        dir_left_f,
        food_left,
        food_right,
        food_up,
        food_down,
    ]

    if config.use_extended_features:
        length_norm = float(len(snake)) / float(board_cells * board_cells)
        tail_up, tail_right, tail_down, tail_left = _tail_direction_features(snake)
        food_dist_norm = float(abs(food[0] - head[0]) + abs(food[1] - head[1])) / float(
            (board_cells - 1) * 2
        )
        values.extend([length_norm, tail_up, tail_right, tail_down, tail_left, food_dist_norm])

    if config.use_path_features:
        path_ratios: list[float] = []
        for candidate in (direction, dir_left, dir_right):
            new_head = next_head(head, candidate)
            if is_danger(board_cells, snake, new_head):
                path_ratios.append(0.0)
                continue
            simulated = _simulate_next_snake(snake, new_head, food)
            path_ratios.append(_reachable_ratio(board_cells, simulated, new_head))
        values.extend(path_ratios)

    if config.use_tail_path_features:
        current_tail_reachable, current_tail_path_norm = _tail_path_features(board_cells, snake)
        values.extend([current_tail_reachable, current_tail_path_norm])
        for candidate in (direction, dir_left, dir_right):
            new_head = next_head(head, candidate)
            if is_danger(board_cells, snake, new_head):
                values.extend([0.0, 1.0])
                continue
            simulated = _simulate_next_snake(snake, new_head, food)
            cand_tail_reachable, cand_tail_path_norm = _tail_path_features(board_cells, simulated)
            values.extend([cand_tail_reachable, cand_tail_path_norm])

    if config.use_free_space_features:
        free_cells = (board_cells * board_cells) - len(snake)
        reachable = max(0, reachable_cell_count(board_cells, snake, head) - 1)
        free_ratio = float(reachable) / float(max(1, free_cells))
        values.append(float(free_ratio))

    if config.use_tail_trend_features:
        tail_trend_streak_max = 20
        values.append(float(min(tail_reachable_streak, tail_trend_streak_max)) / float(tail_trend_streak_max))
        values.append(float(min(tail_unreachable_streak, tail_trend_streak_max)) / float(tail_trend_streak_max))

    return np.array(values, dtype=np.float32)
