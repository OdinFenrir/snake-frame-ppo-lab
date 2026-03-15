from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Iterable

import numpy as np

from .game import SnakeGame
from .gameplay_controller import GameplayController
from .settings import ObsConfig, RewardConfig, Settings


def _default_holdout_seeds() -> list[int]:
    return [int(17_001 + i) for i in range(30)]


def _summary(scores: list[int]) -> dict[str, float | int]:
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "best": 0,
            "min": 0,
        }
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "best": int(arr.max()),
        "min": int(arr.min()),
    }


def _mean_interventions_pct(rows: list[dict[str, float | int]]) -> float:
    if not rows:
        return 0.0
    vals = [float(r.get("interventions_pct", 0.0)) for r in rows]
    return float(sum(vals) / float(len(vals)))


@dataclass(frozen=True)
class HoldoutEvalSnapshot:
    active: bool
    mode: str
    completed: int
    total: int
    last_error: str | None
    latest_summary_path: str


class HoldoutEvalController:
    MODE_PPO_ONLY = "ppo_only"
    MODE_CONTROLLER_ON = "controller_on"

    def __init__(
        self,
        *,
        agent,
        settings: Settings,
        obs_config: ObsConfig,
        reward_config: RewardConfig,
        out_dir: Path,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.obs_config = obs_config
        self.reward_config = reward_config
        self.out_dir = Path(out_dir)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._done_event = threading.Event()
        self._active = False
        self._mode = self.MODE_PPO_ONLY
        self._completed = 0
        self._total = 0
        self._last_error: str | None = None
        self._latest_summary_path = ""
        self._completion_message: str | None = None

    def _is_agent_training_active(self) -> bool:
        # The agent owns the training lifecycle; this lightweight probe prevents
        # holdout eval from mutating model artifacts mid-training.
        if bool(getattr(self.agent, "_external_training_active", False)):
            return True
        return bool(getattr(self.agent, "_train_vecnormalize", None) is not None)

    def snapshot(self) -> HoldoutEvalSnapshot:
        with self._lock:
            return HoldoutEvalSnapshot(
                active=bool(self._active),
                mode=str(self._mode),
                completed=int(self._completed),
                total=int(self._total),
                last_error=self._last_error,
                latest_summary_path=str(self._latest_summary_path),
            )

    def start(
        self,
        *,
        mode: str,
        model_selector: str = "best",
        seeds: Iterable[int] | None = None,
        max_steps: int = 5000,
        trace_enabled: bool = False,
        trace_tag: str = "",
    ) -> bool:
        if self._is_agent_training_active():
            with self._lock:
                self._last_error = "training_active"
                self._completion_message = None
            return False
        mode_n = str(mode or self.MODE_PPO_ONLY).strip().lower()
        if mode_n not in (self.MODE_PPO_ONLY, self.MODE_CONTROLLER_ON):
            mode_n = self.MODE_PPO_ONLY
        seeds_list = [int(s) for s in (list(seeds) if seeds is not None else _default_holdout_seeds())]
        if not seeds_list:
            seeds_list = _default_holdout_seeds()
        max_steps_i = max(200, int(max_steps))
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._mode = mode_n
            self._completed = 0
            self._total = int(len(seeds_list))
            self._last_error = None
            self._completion_message = None
            self._done_event.clear()
        t = threading.Thread(
            target=self._worker,
            kwargs={
                "mode": mode_n,
                "model_selector": str(model_selector),
                "seeds": seeds_list,
                "max_steps": max_steps_i,
                "trace_enabled": bool(trace_enabled),
                "trace_tag": str(trace_tag),
            },
            daemon=True,
            name="holdout-eval-worker",
        )
        self._thread = t
        t.start()
        return True

    def poll_completion(self) -> str | None:
        if not self._done_event.is_set():
            return None
        with self._lock:
            msg = self._completion_message
            self._completion_message = None
            self._done_event.clear()
            return msg

    def close(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)

    def _worker(
        self,
        *,
        mode: str,
        model_selector: str,
        seeds: list[int],
        max_steps: int,
        trace_enabled: bool,
        trace_tag: str,
    ) -> None:
        rows: list[dict[str, int]] = []
        prev_selector: str | None = None
        try:
            get_selector = getattr(self.agent, "get_model_selector", None)
            if callable(get_selector):
                try:
                    prev_selector = str(get_selector() or "").strip().lower() or None
                except Exception:
                    prev_selector = None
            self._prepare_model_selector_for_eval(model_selector=str(model_selector))
            if mode == self.MODE_PPO_ONLY:
                rows = []
                for idx, seed in enumerate(seeds):
                    score_list = list(
                        int(v)
                        for v in self.agent.evaluate_holdout(
                            seeds=[int(seed)],
                            max_steps=int(max_steps),
                            model_selector=str(model_selector),
                        )
                    )
                    score = int(score_list[0]) if score_list else 0
                    rows.append({"seed": int(seed), "score": int(score)})
                    with self._lock:
                        self._completed = int(idx + 1)
            else:
                rows, telemetry_rows = self._eval_with_controller(
                    seeds=seeds,
                    max_steps=int(max_steps),
                    model_selector=str(model_selector),
                    trace_enabled=bool(trace_enabled),
                    trace_tag=str(trace_tag),
                )
            scores = [int(r["score"]) for r in rows]
            summary = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": str(mode),
                "model_selector": str(model_selector),
                "max_steps": int(max_steps),
                "trace_enabled": bool(trace_enabled),
                "trace_tag": str(trace_tag),
                "scores": _summary(scores),
                "rows": rows,
            }
            if mode == self.MODE_CONTROLLER_ON:
                summary["controller_telemetry_rows"] = telemetry_rows
                summary["mean_interventions_pct"] = _mean_interventions_pct(telemetry_rows)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            latest = self.out_dir / "latest_summary.json"
            stamped = self.out_dir / f"summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            payload = json.dumps(summary, indent=2, allow_nan=False)
            latest.write_text(payload, encoding="utf-8")
            stamped.write_text(payload, encoding="utf-8")
            with self._lock:
                self._latest_summary_path = str(latest)
                self._completion_message = (
                    f"Holdout eval done ({mode}): mean={summary['scores']['mean']:.1f} "
                    f"best={summary['scores']['best']} n={summary['scores']['count']}"
                )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._completion_message = f"Holdout eval failed: {exc}"
        finally:
            if prev_selector is not None:
                try:
                    self._prepare_model_selector_for_eval(model_selector=str(prev_selector))
                except Exception:
                    pass
            with self._lock:
                self._active = False
            self._done_event.set()

    def _prepare_model_selector_for_eval(self, *, model_selector: str) -> None:
        selector = str(model_selector or "best").strip().lower()
        set_selector = getattr(self.agent, "set_model_selector", None)
        if callable(set_selector):
            try:
                set_selector(selector)
            except Exception:
                pass
        if self._is_agent_training_active():
            return
        loader = getattr(self.agent, "load_if_exists_detailed", None)
        if callable(loader):
            result = loader(selector=selector)
            ok = bool(getattr(result, "ok", False))
            if not ok:
                code = str(getattr(result, "code", "unknown"))
                detail = str(getattr(result, "detail", "")).strip()
                suffix = f" ({detail})" if detail else ""
                raise RuntimeError(f"eval model load failed [{code}]{suffix}")
            return
        basic_loader = getattr(self.agent, "load_if_exists", None)
        if callable(basic_loader) and not bool(basic_loader()):
            raise RuntimeError("eval model load failed [missing]")

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False))
            f.write("\n")

    def _eval_with_controller(
        self,
        *,
        seeds: list[int],
        max_steps: int,
        model_selector: str,
        trace_enabled: bool,
        trace_tag: str,
    ) -> tuple[list[dict[str, int]], list[dict[str, float | int]]]:
        _ = model_selector
        rows: list[dict[str, int]] = []
        telemetry_rows: list[dict[str, float | int]] = []
        trace_root: Path | None = None
        if bool(trace_enabled):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_tag = "".join(ch for ch in str(trace_tag) if ch.isalnum() or ch in ("-", "_")).strip("_")
            suffix = f"_{safe_tag}" if safe_tag else ""
            trace_root = self.out_dir / "focused_traces" / f"{stamp}{suffix}"
            trace_root.mkdir(parents=True, exist_ok=True)
        base = Settings(
            board_cells=int(self.settings.board_cells),
            cell_px=int(self.settings.cell_px),
            fps=int(self.settings.fps),
            ticks_per_move=1,
            left_panel_px=int(self.settings.left_panel_px),
            right_panel_px=int(self.settings.right_panel_px),
            agent_safety_override=bool(self.settings.agent_safety_override),
            window_height_px=int(self.settings.window_height_px or self.settings.window_px),
            window_borderless=bool(self.settings.window_borderless),
            layout_preset=str(self.settings.layout_preset),
            theme_name=str(self.settings.theme_name),
            ui_scale=float(self.settings.ui_scale),
            min_cell_px=int(self.settings.min_cell_px),
            max_cell_px=int(self.settings.max_cell_px),
            min_left_panel_px=int(self.settings.min_left_panel_px),
            min_right_panel_px=int(self.settings.min_right_panel_px),
            left_panel_ratio=float(self.settings.left_panel_ratio),
            dynamic_control=self.settings.dynamic_control,
        )
        for idx, seed in enumerate(seeds):
            game = SnakeGame(base, starvation_factor=int(self.reward_config.board_starvation_factor))
            game.rng.seed(int(seed))
            game.reset()
            gameplay = GameplayController(
                game=game,
                agent=self.agent,
                settings=base,
                obs_config=self.obs_config,
                space_strategy_enabled=True,
                artifact_dir=Path(getattr(self.agent, "artifact_dir", self.out_dir)),
            )
            gameplay.set_learning_enabled(False)
            if bool(trace_enabled):
                gameplay.set_debug_options(debug_overlay=True, reachable_overlay=False)
            seed_trace_path = (trace_root / f"seed_{int(seed)}.jsonl") if trace_root is not None else None
            for step_idx in range(int(max_steps)):
                if bool(game.game_over):
                    break
                score_before = int(getattr(game, "score", 0))
                head_before = tuple(game.snake[0]) if game.snake else None
                gameplay._apply_agent_control()  # keep controller logic in the eval loop
                trace_row = gameplay.decision_trace_snapshot() if bool(trace_enabled) else None
                game.update()
                if trace_row is not None and seed_trace_path is not None:
                    row = dict(trace_row)
                    row.update(
                        {
                            "seed": int(seed),
                            "step": int(step_idx),
                            "score_before": int(score_before),
                            "score_after": int(getattr(game, "score", 0)),
                            "ate_food": bool(int(getattr(game, "score", 0)) > int(score_before)),
                            "head_before": None if head_before is None else [int(head_before[0]), int(head_before[1])],
                            "head_after": None if not game.snake else [int(game.snake[0][0]), int(game.snake[0][1])],
                            "game_over": bool(game.game_over),
                            "death_reason": str(getattr(game, "death_reason", "none")),
                        }
                    )
                    self._append_jsonl(seed_trace_path, row)
            rows.append({"seed": int(seed), "score": int(game.score)})
            get_snap = getattr(gameplay, "telemetry_snapshot", None)
            snap = get_snap() if callable(get_snap) else None
            decisions = int(getattr(snap, "decisions_total", 0))
            interventions = int(getattr(snap, "interventions_total", 0))
            intervention_pct = 100.0 * float(interventions) / float(max(1, decisions))
            telemetry_rows.append(
                {
                    "seed": int(seed),
                    "score": int(game.score),
                    "decisions": int(decisions),
                    "interventions": int(interventions),
                    "interventions_pct": float(intervention_pct),
                }
            )
            with self._lock:
                self._completed = int(idx + 1)
        return rows, telemetry_rows
