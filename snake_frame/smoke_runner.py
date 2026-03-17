from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pygame

from .app_factory import build_runtime
from .app_state import AppState
from .controls_builder import build_controls
from .gameplay_controller import GameplayController
from .layout_engine import LayoutEngine
from .panel_ui import PanelRenderData
from .settings import ObsConfig, RewardConfig, Settings, ppo_profile_config
from .theme import get_theme
from .training_metrics import avg_last


@dataclass(frozen=True)
class SmokeBudgets:
    max_frame_p95_ms: float
    max_frame_avg_ms: float | None
    max_frame_jitter_ms: float | None
    max_inference_p95_ms: float
    min_training_steps_per_sec: float


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(float(v) for v in samples)
    idx = int(0.95 * (len(ordered) - 1))
    return float(ordered[idx])


def _p50(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(float(v) for v in samples)
    idx = int(0.50 * (len(ordered) - 1))
    return float(ordered[idx])


def _should_record_frame_sample(*, frame_index: int, total_frames: int) -> bool:
    total_i = max(1, int(total_frames))
    # Keep warmup protection for normal runs while preserving metrics for tiny runs.
    return bool(total_i <= 1 or int(frame_index) > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless training/inference smoke with optional perf budgets.")
    parser.add_argument("--train-steps", type=int, default=2048)
    parser.add_argument("--game-steps", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--space-strategy", choices=("on", "off"), default="on")
    parser.add_argument("--ppo-profile", choices=("fast", "app", "research_long"), default="fast")
    parser.add_argument("--metrics-out", type=str, default="")
    parser.add_argument("--enforce-budgets", action="store_true")
    parser.add_argument("--max-frame-p95-ms", type=float, default=40.0)
    parser.add_argument("--max-frame-avg-ms", type=float, default=None)
    parser.add_argument("--max-frame-jitter-ms", type=float, default=None)
    parser.add_argument("--max-inference-p95-ms", type=float, default=12.0)
    parser.add_argument("--min-training-steps-per-sec", type=float, default=250.0)
    return parser.parse_args(argv)


def build_budgets_from_args(args: argparse.Namespace) -> SmokeBudgets | None:
    if not bool(getattr(args, "enforce_budgets", False)):
        return None
    max_frame_avg_ms = getattr(args, "max_frame_avg_ms", None)
    max_frame_jitter_ms = getattr(args, "max_frame_jitter_ms", None)
    return SmokeBudgets(
        max_frame_p95_ms=float(args.max_frame_p95_ms),
        max_frame_avg_ms=None if max_frame_avg_ms is None else float(max_frame_avg_ms),
        max_frame_jitter_ms=None if max_frame_jitter_ms is None else float(max_frame_jitter_ms),
        max_inference_p95_ms=float(args.max_inference_p95_ms),
        min_training_steps_per_sec=float(args.min_training_steps_per_sec),
    )


def run_headless_smoke(
    *,
    train_steps: int = 2048,
    game_steps: int = 300,
    timeout_seconds: float = 120.0,
    seed: int = 1337,
    space_strategy_enabled: bool = True,
    ppo_profile: str = "fast",
    budgets: SmokeBudgets | None = None,
) -> dict:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = Path(temp_dir_obj.name)
    frame_ms_samples: list[float] = []
    inference_ms_samples: list[float] = []
    training_scores: list[int] = []
    training_start = time.perf_counter()
    training_end = training_start
    training_done_steps = 0
    app_state = AppState(game_running=True)

    try:
        settings = Settings()
        layout = LayoutEngine(settings).update(settings.window_width_px, int(settings.window_height_px or settings.window_px))
        settings.apply_window_size(layout.window.width, layout.window.height)
        pygame.display.set_mode((layout.window.width, layout.window.height))
        theme = get_theme(settings.theme_name)
        font = pygame.font.SysFont("Arial", 20, bold=True)
        small_font = pygame.font.SysFont("Arial", 16)
        profile = str(ppo_profile or "fast").strip().lower()
        if profile in ("app", "fast", "research_long"):
            ppo_cfg = ppo_profile_config(profile if profile != "app" else "", seed=int(seed))
        else:
            profile = "fast"
            ppo_cfg = ppo_profile_config("fast", seed=int(seed))
        train_steps_requested = max(1, int(train_steps))
        train_step_granularity = max(1, int(ppo_cfg.env_count) * int(ppo_cfg.n_steps))
        train_steps_target_effective = max(
            1000,
            int(((train_steps_requested + train_step_granularity - 1) // train_step_granularity) * train_step_granularity),
        )

        runtime = build_runtime(
            settings=settings,
            font=font,
            small_font=small_font,
            on_score=lambda s: training_scores.append(int(s)),
            ppo_config=ppo_cfg,
            reward_config=RewardConfig(),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True, use_free_space_features=True, use_tail_trend_features=False),
            state_dir=temp_dir / "state",
        )
        gameplay = GameplayController(
            game=runtime.game,
            agent=runtime.agent,
            settings=settings,
            obs_config=runtime.obs_config,
            space_strategy_enabled=bool(space_strategy_enabled),
        )
        controls = build_controls(
            settings=settings,
            min_graph_height=layout.graph.min_graph_height,
            max_graph_height=layout.graph.max_graph_height,
            graph_margin=layout.graph.graph_margin,
            graph_top=layout.graph.graph_top,
            control_row_height=layout.graph.control_row_height,
            control_gap=layout.graph.control_gap,
            status_line_height=layout.graph.status_line_height,
            status_line_count=layout.graph.status_line_count,
        )
        surface = pygame.display.get_surface()
        if surface is None:
            raise RuntimeError("Failed to create headless surface")
        clock = pygame.time.Clock()

        if not runtime.training.start(target_steps=int(train_steps_target_effective)):
            raise RuntimeError("Training did not start in smoke run")

        deadline = time.perf_counter() + float(timeout_seconds)
        training_start_steps = int(runtime.training.snapshot().start_steps)
        while time.perf_counter() < deadline:
            message = runtime.training.poll_completion()
            snap = runtime.training.snapshot()
            training_done_steps = int(snap.current_steps) - int(training_start_steps)
            if message is not None:
                if message.startswith("Training error:"):
                    raise RuntimeError(message)
                training_end = time.perf_counter()
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("Timed out waiting for training completion")

        if not bool(getattr(runtime.agent, "is_inference_available", False)):
            runtime.agent.request_inference_sync()
            wait_deadline = time.perf_counter() + 5.0
            while time.perf_counter() < wait_deadline and not bool(getattr(runtime.agent, "is_inference_available", False)):
                gameplay.step(False)
                time.sleep(0.01)
        if not bool(getattr(runtime.agent, "is_inference_available", False)):
            raise RuntimeError("Inference snapshot did not become active")

        # Reset the frame clock after the long training wait so frame timing
        # samples are not polluted by pre-loop idle time.
        clock.tick(0)
        game_steps_i = max(1, int(game_steps))
        for frame_idx in range(game_steps_i):
            t0 = time.perf_counter()
            gameplay.step(app_state.game_running)
            t1 = time.perf_counter()
            inference_ms_samples.append(float((t1 - t0) * 1000.0))

            surface.fill(theme.surface_bg)
            runtime.panel_renderer.draw(
                surface=surface,
                controls=controls.panel_controls,
                data=PanelRenderData(
                    training_episode_scores=list(training_scores),
                    run_episode_scores=[int(v) for v in runtime.game.episode_scores],
                    training_graph_rect=pygame.Rect(controls.training_graph_rect),
                    run_graph_rect=pygame.Rect(controls.run_graph_rect),
                    training_graph_badges=[
                        f"Train {training_done_steps}/{max(1, int(train_steps))}",
                        f"Eps {len(training_scores)}",
                        f"Avg20 {avg_last(training_scores, 20):.1f}",
                    ],
                    run_graph_badges=[
                        f"RunEps {len(runtime.game.episode_scores)}",
                        f"Avg20 {avg_last(runtime.game.episode_scores, 20):.1f}",
                    ],
                    run_status_lines=["Smoke run active"],
                    settings_lines=["Headless CI mode"],
                ),
            )
            runtime.game.draw(surface, font)
            pygame.display.flip()
            frame_ms = float(clock.tick(settings.fps))
            # Skip the first sample as additional warmup protection against
            # occasional one-off scheduler spikes.
            if _should_record_frame_sample(frame_index=frame_idx, total_frames=game_steps_i):
                frame_ms_samples.append(frame_ms)

        duration_s = max(1e-6, float(training_end - training_start))
        training_sps = float(max(0, training_done_steps) / duration_s)
        telemetry = gameplay.telemetry_snapshot()
        metrics = {
            "train_steps_target": int(train_steps),
            "train_steps_target_effective": int(train_steps_target_effective),
            "train_step_granularity": int(train_step_granularity),
            "train_steps_done": int(training_done_steps),
            "training_duration_s": float(duration_s),
            "training_steps_per_sec": float(training_sps),
            "frame_ms_p95": float(_p95(frame_ms_samples)),
            "frame_ms_avg": float(sum(frame_ms_samples) / max(1, len(frame_ms_samples))),
            "frame_ms_p50": float(_p50(frame_ms_samples)),
            "frame_ms_jitter": float(_p95(frame_ms_samples) - _p50(frame_ms_samples)),
            "inference_step_ms_p95": float(_p95(inference_ms_samples)),
            "inference_step_ms_avg": float(sum(inference_ms_samples) / max(1, len(inference_ms_samples))),
            "training_episode_scores": [int(v) for v in training_scores],
            "run_episode_scores": [int(v) for v in runtime.game.episode_scores],
            "training_avg20": float(avg_last(training_scores, 20)),
            "training_best": int(max(training_scores)) if training_scores else 0,
            "run_avg20": float(avg_last(runtime.game.episode_scores, 20)),
            "run_best": int(max(runtime.game.episode_scores)) if runtime.game.episode_scores else 0,
            "seed": int(seed),
            "space_strategy_enabled": bool(space_strategy_enabled),
            "ppo_profile": str(profile),
            "ppo_n_steps": int(ppo_cfg.n_steps),
            "ppo_batch_size": int(ppo_cfg.batch_size),
            "ppo_n_epochs": int(ppo_cfg.n_epochs),
            "ppo_env_count": int(ppo_cfg.env_count),
            "control_mode": str(telemetry.current_mode),
            "switch_reason": str(telemetry.last_switch_reason),
            "decisions_total": int(telemetry.decisions_total),
            "interventions_total": int(telemetry.interventions_total),
            "cycle_repeats_total": int(telemetry.cycle_repeats_total),
            "cycle_breaks_total": int(telemetry.cycle_breaks_total),
            "stuck_episodes_total": int(telemetry.stuck_episodes_total),
            "deaths_wall": int(telemetry.deaths_wall),
            "deaths_body": int(telemetry.deaths_body),
            "deaths_starvation": int(telemetry.deaths_starvation),
            "deaths_fill": int(telemetry.deaths_fill),
            "deaths_other": int(telemetry.deaths_other),
            "no_progress_steps": int(telemetry.no_progress_steps),
            "starvation_steps": int(telemetry.starvation_steps),
            "starvation_limit": int(telemetry.starvation_limit),
            "loop_escape_activations_total": int(telemetry.loop_escape_activations_total),
            "loop_escape_steps_left": int(telemetry.loop_escape_steps_left),
        }
        if budgets is not None:
            if float(metrics["frame_ms_p95"]) > float(budgets.max_frame_p95_ms):
                raise RuntimeError(
                    f"Frame p95 {metrics['frame_ms_p95']:.2f}ms exceeded budget {budgets.max_frame_p95_ms:.2f}ms"
                )
            if budgets.max_frame_avg_ms is not None and float(metrics["frame_ms_avg"]) > float(budgets.max_frame_avg_ms):
                raise RuntimeError(
                    f"Frame avg {metrics['frame_ms_avg']:.2f}ms exceeded budget {budgets.max_frame_avg_ms:.2f}ms"
                )
            if budgets.max_frame_jitter_ms is not None and float(metrics["frame_ms_jitter"]) > float(budgets.max_frame_jitter_ms):
                raise RuntimeError(
                    f"Frame jitter {metrics['frame_ms_jitter']:.2f}ms exceeded budget {budgets.max_frame_jitter_ms:.2f}ms"
                )
            if float(metrics["inference_step_ms_p95"]) > float(budgets.max_inference_p95_ms):
                raise RuntimeError(
                    f"Inference p95 {metrics['inference_step_ms_p95']:.2f}ms exceeded budget {budgets.max_inference_p95_ms:.2f}ms"
                )
            if float(metrics["training_steps_per_sec"]) < float(budgets.min_training_steps_per_sec):
                raise RuntimeError(
                    f"Training throughput {metrics['training_steps_per_sec']:.1f} below budget {budgets.min_training_steps_per_sec:.1f}"
                )
        return metrics
    finally:
        try:
            pygame.quit()
        finally:
            temp_dir_obj.cleanup()


def main() -> None:
    args = parse_args()
    budgets = build_budgets_from_args(args)
    metrics = run_headless_smoke(
        train_steps=int(args.train_steps),
        game_steps=int(args.game_steps),
        timeout_seconds=float(args.timeout_seconds),
        seed=int(args.seed),
        space_strategy_enabled=(str(args.space_strategy).lower() == "on"),
        ppo_profile=str(args.ppo_profile),
        budgets=budgets,
    )
    payload = {"metrics": metrics, "budgets": asdict(budgets) if budgets else None}
    print(json.dumps(payload, indent=2))
    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
