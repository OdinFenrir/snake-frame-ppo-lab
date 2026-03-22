from __future__ import annotations

import gymnasium as gym
import numpy as np

from .board_analysis import is_point_danger, reachable_space_ratio as board_reachable_space_ratio
from .observation import (
    RIGHT,
    action_to_direction,
    build_observation,
    is_danger,
    next_head,
    observation_size,
    valid_action_mask,
)
from .settings import ObsConfig, RewardConfig
Direction = tuple[int, int]
Point = tuple[int, int]


def safe_option_count(board_cells: int, snake: list[Point], direction: Direction) -> int:
    head = snake[0]
    direction_left = action_to_direction(direction, 1)
    direction_right = action_to_direction(direction, 2)
    directions = (direction, direction_left, direction_right)
    safe = 0
    for candidate in directions:
        point = next_head(head, candidate)
        if not is_point_danger(board_cells, snake, point):
            safe += 1
    return int(safe)


def reachable_space_ratio(board_cells: int, snake_after_move: list[Point], start: Point) -> float:
    # Compatibility wrapper kept for tests/patching.
    return float(board_reachable_space_ratio(board_cells, snake_after_move, start))


class SnakePPOEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        board_cells: int = 20,
        seed: int | None = None,
        reward_config: RewardConfig | None = None,
        obs_config: ObsConfig | None = None,
    ) -> None:
        super().__init__()
        self.board_cells = int(board_cells)
        self.reward_config = reward_config or RewardConfig()
        self.obs_config = obs_config or ObsConfig()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size(self.obs_config),),
            dtype=np.float32,
        )

        self.snake: list[Point] = []
        self.direction: Direction = RIGHT
        self.food: Point = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.prev_food_dist = 0
        self._tail_reachable_streak: int = 0
        self._tail_unreachable_streak: int = 0
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        c = self.board_cells // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self._tail_reachable_streak = 0
        self._tail_unreachable_streak = 0
        self._spawn_food()
        self.prev_food_dist = self._food_distance()
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_i = int(action)
        if not self.action_space.contains(action_i):
            raise ValueError(f"Invalid action: {action_i}")
        self.steps += 1
        self.steps_without_food += 1
        reward = -float(self.reward_config.living_penalty)
        terminated = False
        truncated = False

        self.direction = action_to_direction(self.direction, action_i)
        new_head = next_head(self.snake[0], self.direction)

        if is_danger(self.board_cells, self.snake, new_head):
            reward -= float(self.reward_config.death_penalty)
            terminated = True
            death_reason = "wall"
            nx, ny = new_head
            if not (nx < 0 or ny < 0 or nx >= self.board_cells or ny >= self.board_cells):
                death_reason = "body"
            info = {"score": int(self.score), "steps": int(self.steps), "death_reason": str(death_reason)}
            return self._obs(), float(reward), terminated, truncated, info

        self.snake.insert(0, new_head)
        ate = new_head == self.food
        if ate:
            self.score += 1
            reward += float(self.reward_config.eat_reward)
            self.steps_without_food = 0
            self._spawn_food()
        else:
            self.snake.pop()

        dist = self._food_distance()
        if dist < self.prev_food_dist:
            reward += float(self.reward_config.approach_food_reward)
        elif dist > self.prev_food_dist:
            reward -= float(self.reward_config.approach_food_reward)  # symmetric shaping
        self.prev_food_dist = dist

        safe_options = safe_option_count(self.board_cells, self.snake, self.direction)
        if safe_options <= 1:
            reward -= float(self.reward_config.low_safe_options_penalty)  # 0.05 (was 0.07)
        elif safe_options >= 3:
            reward += float(self.reward_config.high_safe_options_bonus)  # 0.03 (was 0.015)

        if bool(self.reward_config.use_reachable_space_penalty):
            reach_ratio = reachable_space_ratio(self.board_cells, self.snake, self.snake[0])
            threshold = max(0.0, float(self.reward_config.trap_penalty_threshold))
            if threshold > 0.0 and reach_ratio < threshold:
                depth = (threshold - reach_ratio) / threshold
                length_ratio = float(len(self.snake)) / float(self.board_cells * self.board_cells)
                start_ratio = float(self.reward_config.endgame_length_ratio_start)
                endgame_scale = 1.0 + max(0.0, length_ratio - start_ratio) * float(
                    self.reward_config.endgame_trap_penalty_scale
                )
                reward -= float(self.reward_config.trap_penalty) * depth * endgame_scale  # reduced from 0.68/2.4

        starvation_limit = self.board_cells * self.board_cells * int(self.reward_config.board_starvation_factor)
        if self.steps_without_food > starvation_limit:
            reward -= float(self.reward_config.starvation_penalty)
            truncated = True

        if len(self.snake) >= self.board_cells * self.board_cells:
            reward += float(self.reward_config.fill_board_bonus)
            terminated = True

        self._update_tail_streak()
        death_reason = "none"
        if terminated:
            death_reason = "fill"
        elif truncated:
            death_reason = "starvation"
        info = {"score": int(self.score), "steps": int(self.steps), "death_reason": str(death_reason)}
        return self._obs(), float(reward), terminated, truncated, info

    def _update_tail_streak(self) -> None:
        from .board_analysis import tail_path_length
        current_tail_reachable = tail_path_length(self.board_cells, self.snake) is not None
        if current_tail_reachable:
            self._tail_reachable_streak += 1
            self._tail_unreachable_streak = 0
        else:
            self._tail_unreachable_streak += 1
            self._tail_reachable_streak = 0

    def _obs(self) -> np.ndarray:
        return build_observation(
            self.board_cells,
            self.snake,
            self.direction,
            self.food,
            obs_config=self.obs_config,
            tail_reachable_streak=self._tail_reachable_streak,
            tail_unreachable_streak=self._tail_unreachable_streak,
        )

    def _food_distance(self) -> int:
        hx, hy = self.snake[0]
        fx, fy = self.food
        return abs(hx - fx) + abs(hy - fy)

    def _spawn_food(self) -> None:
        occupied = set(self.snake)
        free_cells = [
            (x, y)
            for y in range(self.board_cells)
            for x in range(self.board_cells)
            if (x, y) not in occupied
        ]
        if not free_cells:
            self.food = self.snake[0]
            return
        idx = int(self.np_random.integers(0, len(free_cells)))
        self.food = free_cells[idx]

    def close(self) -> None:
        return None

    def action_masks(self) -> np.ndarray:
        mask = valid_action_mask(self.board_cells, self.snake, self.direction)
        return np.asarray(mask, dtype=bool)
