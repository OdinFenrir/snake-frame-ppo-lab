from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_frame.smoke_runner import SmokeBudgets, run_headless_smoke  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run headless smoke multiple times and enforce perf budgets on median metrics."
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1337)
    parser.add_argument("--train-steps", type=int, default=2048)
    parser.add_argument("--game-steps", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--space-strategy", choices=("on", "off"), default="on")
    parser.add_argument("--ppo-profile", choices=("fast", "app", "research_long"), default="fast")
    parser.add_argument("--metrics-out", type=str, default="")
    parser.add_argument("--max-frame-p95-ms", type=float, default=40.0)
    parser.add_argument("--max-frame-avg-ms", type=float, default=None)
    parser.add_argument("--max-frame-jitter-ms", type=float, default=None)
    parser.add_argument("--max-inference-p95-ms", type=float, default=12.0)
    parser.add_argument("--min-training-steps-per-sec", type=float, default=250.0)
    return parser.parse_args(argv)


def _metric_median(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if key in r]
    return float(median(values)) if values else 0.0


def _enforce_budgets(medians: dict, budgets: SmokeBudgets) -> None:
    if float(medians["frame_ms_p95"]) > float(budgets.max_frame_p95_ms):
        raise RuntimeError(
            f"Median frame p95 {medians['frame_ms_p95']:.2f}ms exceeded budget {budgets.max_frame_p95_ms:.2f}ms"
        )
    if budgets.max_frame_avg_ms is not None and float(medians["frame_ms_avg"]) > float(budgets.max_frame_avg_ms):
        raise RuntimeError(
            f"Median frame avg {medians['frame_ms_avg']:.2f}ms exceeded budget {budgets.max_frame_avg_ms:.2f}ms"
        )
    if budgets.max_frame_jitter_ms is not None and float(medians["frame_ms_jitter"]) > float(budgets.max_frame_jitter_ms):
        raise RuntimeError(
            f"Median frame jitter {medians['frame_ms_jitter']:.2f}ms exceeded budget {budgets.max_frame_jitter_ms:.2f}ms"
        )
    if float(medians["inference_step_ms_p95"]) > float(budgets.max_inference_p95_ms):
        raise RuntimeError(
            f"Median inference p95 {medians['inference_step_ms_p95']:.2f}ms exceeded budget {budgets.max_inference_p95_ms:.2f}ms"
        )
    if float(medians["training_steps_per_sec"]) < float(budgets.min_training_steps_per_sec):
        raise RuntimeError(
            "Median training throughput "
            f"{medians['training_steps_per_sec']:.1f} below budget {budgets.min_training_steps_per_sec:.1f}"
        )


def main() -> None:
    args = parse_args()
    runs = max(1, int(args.runs))
    budgets = SmokeBudgets(
        max_frame_p95_ms=float(args.max_frame_p95_ms),
        max_frame_avg_ms=None if args.max_frame_avg_ms is None else float(args.max_frame_avg_ms),
        max_frame_jitter_ms=None if args.max_frame_jitter_ms is None else float(args.max_frame_jitter_ms),
        max_inference_p95_ms=float(args.max_inference_p95_ms),
        min_training_steps_per_sec=float(args.min_training_steps_per_sec),
    )

    metrics_rows: list[dict] = []
    for idx in range(runs):
        metrics = run_headless_smoke(
            train_steps=int(args.train_steps),
            game_steps=int(args.game_steps),
            timeout_seconds=float(args.timeout_seconds),
            seed=int(args.seed_base) + int(idx),
            space_strategy_enabled=(str(args.space_strategy).lower() == "on"),
            ppo_profile=str(args.ppo_profile),
            budgets=None,
        )
        metrics_rows.append(metrics)

    medians = {
        "training_steps_per_sec": _metric_median(metrics_rows, "training_steps_per_sec"),
        "frame_ms_p95": _metric_median(metrics_rows, "frame_ms_p95"),
        "frame_ms_avg": _metric_median(metrics_rows, "frame_ms_avg"),
        "frame_ms_jitter": _metric_median(metrics_rows, "frame_ms_jitter"),
        "inference_step_ms_p95": _metric_median(metrics_rows, "inference_step_ms_p95"),
    }
    _enforce_budgets(medians, budgets)
    payload = {
        "runs": int(runs),
        "seed_base": int(args.seed_base),
        "metrics": metrics_rows,
        "median_metrics": medians,
        "budgets": asdict(budgets),
    }
    print(json.dumps(payload, indent=2))
    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
