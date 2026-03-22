from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

from .ppo_agent import PpoSnakeAgent

ScoreAppendFn = Callable[[int], None]
EpisodeInfoFn = Callable[[dict], None]
logger = logging.getLogger(__name__)


@dataclass
class TrainingSnapshot:
    active: bool
    target_steps: int
    start_steps: int
    current_steps: int
    last_error: str | None
    stop_requested: bool
    best_eval_score: float | None
    best_eval_step: int
    last_eval_score: float | None
    eval_runs_completed: int

    @property
    def done_steps(self) -> int:
        return max(0, self.current_steps - self.start_steps)


class PpoTrainingController:
    _CLOSE_JOIN_TIMEOUT_S = 3.0

    def __init__(
        self,
        agent: PpoSnakeAgent,
        on_score: ScoreAppendFn,
        on_episode_info: EpisodeInfoFn | None = None,
    ) -> None:
        self.agent = agent
        self.on_score = on_score
        self.on_episode_info = on_episode_info
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._active = False
        self._target_steps = 0
        self._start_steps = int(self.agent.model.num_timesteps) if self.agent.model is not None else 0
        self._current_steps = self._start_steps
        self._last_error: str | None = None
        self._best_eval_score: float | None = self._agent_best_eval_score()
        self._best_eval_step: int = self._agent_best_eval_step()
        self._last_eval_score: float | None = self._agent_last_eval_score()
        self._eval_runs_completed: int = self._agent_eval_runs_completed()

    def snapshot(self) -> TrainingSnapshot:
        with self._lock:
            current_steps = int(self._current_steps)
            best_eval_score = self._best_eval_score
            best_eval_step = int(self._best_eval_step)
            last_eval_score = self._last_eval_score
            eval_runs_completed = int(self._eval_runs_completed)
            if self._active:
                model = getattr(self.agent, "model", None)
                if model is not None:
                    try:
                        live_steps = int(getattr(model, "num_timesteps", current_steps))
                        if live_steps > current_steps:
                            current_steps = live_steps
                            self._current_steps = live_steps
                    except Exception:
                        pass
                best_eval_score = self._agent_best_eval_score()
                best_eval_step = self._agent_best_eval_step()
                last_eval_score = self._agent_last_eval_score()
                eval_runs_completed = self._agent_eval_runs_completed()
                self._best_eval_score = best_eval_score
                self._best_eval_step = best_eval_step
                self._last_eval_score = last_eval_score
                self._eval_runs_completed = eval_runs_completed
            return TrainingSnapshot(
                active=bool(self._active),
                target_steps=int(self._target_steps),
                start_steps=int(self._start_steps),
                current_steps=int(current_steps),
                last_error=self._last_error,
                stop_requested=self._stop_event.is_set(),
                best_eval_score=best_eval_score,
                best_eval_step=int(best_eval_step),
                last_eval_score=last_eval_score,
                eval_runs_completed=int(eval_runs_completed),
            )

    def start(self, target_steps: int) -> bool:
        target_steps_i = max(1, int(target_steps))
        with self._lock:
            if self._active:
                return False
            self._active = True
            setattr(self.agent, "_external_training_active", True)
            self._target_steps = target_steps_i
            self._start_steps = int(self.agent.model.num_timesteps) if self.agent.model is not None else 0
            self._current_steps = self._start_steps
            self._last_error = None
            self._best_eval_score = self._agent_best_eval_score()
            self._best_eval_step = self._agent_best_eval_step()
            self._last_eval_score = self._agent_last_eval_score()
            self._eval_runs_completed = self._agent_eval_runs_completed()
            self._stop_event.clear()

        thread = threading.Thread(target=self._worker, daemon=True, name="ppo-training-worker")
        try:
            thread.start()
        except Exception as exc:
            logger.exception("Failed to start training worker thread")
            with self._lock:
                self._active = False
                setattr(self.agent, "_external_training_active", False)
                self._target_steps = 0
                self._last_error = str(exc)
                self._thread = None
            return False
        self._thread = thread
        return True

    def stop(self) -> None:
        import traceback
        logger.warning("Training stop() called from:\n%s", "".join(traceback.format_stack()))
        self._stop_event.set()

    def poll_completion(self) -> str | None:
        thread = self._thread
        if thread is None or thread.is_alive():
            return None
        thread.join(timeout=0.1)
        self._thread = None
        snap = self.snapshot()
        if snap.last_error:
            return f"Training error: {snap.last_error}"
        if snap.stop_requested:
            return "Training stopped"
        return "Training complete"

    def close(self) -> None:
        self.stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._CLOSE_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning("Training worker did not stop within %.1fs", self._CLOSE_JOIN_TIMEOUT_S)
            else:
                self._thread = None

    def reset_tracking_from_agent(self) -> None:
        with self._lock:
            self._active = False
            setattr(self.agent, "_external_training_active", False)
            self._target_steps = 0
            self._start_steps = int(self.agent.model.num_timesteps) if self.agent.model is not None else 0
            self._current_steps = int(self._start_steps)
            self._last_error = None
            self._best_eval_score = self._agent_best_eval_score()
            self._best_eval_step = self._agent_best_eval_step()
            self._last_eval_score = self._agent_last_eval_score()
            self._eval_runs_completed = self._agent_eval_runs_completed()
            self._stop_event.clear()

    def _worker(self) -> None:
        def is_stopped() -> bool:
            return self._stop_event.is_set()

        def on_progress(steps_done: int) -> None:
            with self._lock:
                self._current_steps = int(steps_done)

        def on_score(score: int) -> None:
            self.on_score(int(score))

        def on_episode_info(info: dict) -> None:
            if self.on_episode_info is None:
                return
            self.on_episode_info(dict(info))

        snap = self.snapshot()
        try:
            final_steps = self.agent.train(
                total_timesteps=snap.target_steps,
                stop_flag=is_stopped,
                on_progress=on_progress,
                on_score=on_score,
                on_episode_info=on_episode_info,
            )
            with self._lock:
                self._current_steps = int(final_steps)
                self._best_eval_score = self._agent_best_eval_score()
                self._best_eval_step = self._agent_best_eval_step()
                self._last_eval_score = self._agent_last_eval_score()
                self._eval_runs_completed = self._agent_eval_runs_completed()
        except Exception as exc:
            logger.exception("Training worker failed")
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._active = False
                setattr(self.agent, "_external_training_active", False)

    def _agent_best_eval_score(self) -> float | None:
        try:
            value = getattr(self.agent, "best_eval_score", None)
            return None if value is None else float(value)
        except Exception:
            return None

    def _agent_best_eval_step(self) -> int:
        try:
            return int(getattr(self.agent, "best_eval_step", 0))
        except Exception:
            return 0

    def _agent_last_eval_score(self) -> float | None:
        try:
            value = getattr(self.agent, "last_eval_score", None)
            return None if value is None else float(value)
        except Exception:
            return None

    def _agent_eval_runs_completed(self) -> int:
        try:
            return int(getattr(self.agent, "eval_runs_completed", 0))
        except Exception:
            return 0
