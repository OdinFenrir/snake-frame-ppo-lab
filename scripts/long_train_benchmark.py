from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.env_checker import check_env  # noqa: E402

from snake_frame.eval_stats import (  # noqa: E402
    bootstrap_ci_iqm,
    bootstrap_ci_mean,
    probability_of_improvement,
    summary,
)
from snake_frame.ppo_agent import ModelSelector, PpoSnakeAgent  # noqa: E402
from snake_frame.ppo_env import SnakePPOEnv  # noqa: E402
from snake_frame.settings import ObsConfig, RewardConfig, Settings, ppo_profile_config  # noqa: E402
from snake_frame.training_metrics import avg_last  # noqa: E402

TRAIN_SEEDS_DEFAULT = "1337,2026,4242,5151,9001"
HOLDOUT_SEEDS_DEFAULT = "17001-17030"


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_seed_spec(raw: str) -> list[int]:
    token = str(raw).strip()
    if "-" in token and "," not in token:
        lo, hi = token.split("-", 1)
        lo_i = int(lo.strip())
        hi_i = int(hi.strip())
        if hi_i < lo_i:
            raise ValueError("Invalid range: high < low")
        return [int(v) for v in range(lo_i, hi_i + 1)]
    return _parse_int_list(token)


def _build_agent(*, seed: int, state_dir: Path, profile: str) -> PpoSnakeAgent:
    settings = Settings()
    obs_cfg = ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True, use_free_space_features=True)
    reward_cfg = RewardConfig()
    ppo_cfg = ppo_profile_config(profile, seed=int(seed))
    return PpoSnakeAgent(
        settings=settings,
        artifact_dir=state_dir / "ppo" / "baseline",
        config=ppo_cfg,
        reward_config=reward_cfg,
        obs_config=obs_cfg,
        autoload=False,
        legacy_model_path=state_dir / "ppo_snake_model.zip",
    )


def _validate_env_api(*, board_cells: int, seed: int) -> None:
    env = SnakePPOEnv(board_cells=int(board_cells), seed=int(seed))
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


def run_seed_benchmark(
    *,
    seed: int,
    checkpoints: list[int],
    holdout_seeds: list[int],
    eval_max_steps: int,
    state_root: Path,
    profile: str,
    selector: ModelSelector,
) -> dict:
    seed_dir = state_root / f"seed_{int(seed)}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir, ignore_errors=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    agent = _build_agent(seed=int(seed), state_dir=seed_dir, profile=profile)
    agent.set_model_selector(selector)
    _validate_env_api(board_cells=agent.settings.board_cells, seed=int(seed))

    training_scores: list[int] = []
    completed_steps = 0
    rows: list[dict] = []
    holdout_seed_values = [int(v) for v in holdout_seeds]
    try:
        for checkpoint_requested in checkpoints:
            checkpoint_requested_i = int(checkpoint_requested)
            delta_requested = int(checkpoint_requested_i - completed_steps)
            if delta_requested <= 0:
                continue
            steps_before = int(completed_steps)
            done = agent.train(
                total_timesteps=delta_requested,
                stop_flag=lambda: False,
                on_progress=None,
                on_score=lambda s: training_scores.append(int(s)),
            )
            completed_steps = int(done)
            actual_delta = int(completed_steps - steps_before)
            eval_scores = agent.evaluate_holdout(
                seeds=holdout_seed_values,
                max_steps=int(eval_max_steps),
                model_selector=selector,
            )
            row_summary = summary(eval_scores)
            rows.append(
                {
                    "checkpoint_requested_total_steps": int(checkpoint_requested_i),
                    "checkpoint_actual_total_steps": int(completed_steps),
                    "checkpoint_requested_delta_steps": int(delta_requested),
                    "checkpoint_actual_delta_steps": int(actual_delta),
                    "train_avg20": float(avg_last(training_scores, 20)),
                    "train_best": int(max(training_scores)) if training_scores else 0,
                    "eval_model_selector": str(selector.value),
                    "eval_holdout_seeds_count": int(len(holdout_seed_values)),
                    "eval_scores": [int(v) for v in eval_scores],
                    "eval_summary": row_summary,
                    "best_eval_score": None if agent.best_eval_score is None else float(agent.best_eval_score),
                    "best_eval_step": int(agent.best_eval_step),
                    "last_eval_score": None if agent.last_eval_score is None else float(agent.last_eval_score),
                    "eval_runs_completed": int(agent.eval_runs_completed),
                    "sampled_timesteps": int(completed_steps),
                }
            )
    finally:
        try:
            if agent.model is not None and getattr(agent.model, "env", None) is not None:
                agent.model.env.close()
        except Exception:
            pass
    return {
        "seed": int(seed),
        "checkpoints": rows,
    }


def _aggregate_by_checkpoint(
    runs: list[dict],
    checkpoints: list[int],
    *,
    bootstrap_samples: int,
) -> list[dict]:
    out: list[dict] = []
    baseline_means: list[float] = []
    for idx, _checkpoint in enumerate(checkpoints):
        combined_eval_scores: list[int] = []
        per_seed_means: list[float] = []
        actual_steps: list[int] = []
        requested_steps: list[int] = []
        for run in runs:
            rows = run.get("checkpoints", [])
            if idx >= len(rows):
                continue
            row = rows[idx]
            eval_scores = [int(v) for v in row.get("eval_scores", [])]
            combined_eval_scores.extend(eval_scores)
            per_seed_means.append(float(row.get("eval_summary", {}).get("mean", 0.0)))
            actual_steps.append(int(row.get("checkpoint_actual_total_steps", 0)))
            requested_steps.append(int(row.get("checkpoint_requested_total_steps", 0)))
        checkpoint_summary = summary(combined_eval_scores)
        record = {
            "checkpoint_index": int(idx),
            "requested_total_steps_mean": float(sum(requested_steps) / max(1, len(requested_steps))),
            "actual_total_steps_mean": float(sum(actual_steps) / max(1, len(actual_steps))),
            "eval_summary_all_scores": checkpoint_summary,
            "eval_summary_per_seed_mean": summary(per_seed_means),
            "bootstrap_ci_mean": bootstrap_ci_mean(combined_eval_scores, samples=int(bootstrap_samples), seed=idx + 11),
            "bootstrap_ci_iqm": bootstrap_ci_iqm(combined_eval_scores, samples=int(bootstrap_samples), seed=idx + 17),
            "probability_of_improvement_vs_baseline": 0.0,
        }
        if idx == 0:
            baseline_means = list(per_seed_means)
            record["probability_of_improvement_vs_baseline"] = 0.5
        else:
            record["probability_of_improvement_vs_baseline"] = probability_of_improvement(
                per_seed_means, baseline_means
            )
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-horizon PPO benchmark with holdout eval + confidence stats.")
    parser.add_argument("--seeds", type=str, default=TRAIN_SEEDS_DEFAULT)
    parser.add_argument("--checkpoints", type=str, default="500000,5000000")
    parser.add_argument("--holdout-seeds", type=str, default=HOLDOUT_SEEDS_DEFAULT)
    parser.add_argument("--eval-max-steps", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--model-selector", choices=("best", "last"), default="best")
    parser.add_argument("--ppo-profile", choices=("app", "fast", "research_long"), default="research_long")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--out", type=str, default="artifacts/long_eval/benchmark_summary.json")
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    checkpoints = sorted(set(_parse_int_list(args.checkpoints)))
    holdout_seeds = _parse_seed_spec(args.holdout_seeds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selector = ModelSelector(str(args.model_selector))

    if args.artifact_dir:
        state_root = Path(args.artifact_dir)
        state_root.mkdir(parents=True, exist_ok=True)
        runs = [
            run_seed_benchmark(
                seed=int(seed),
                checkpoints=checkpoints,
                holdout_seeds=holdout_seeds,
                eval_max_steps=int(args.eval_max_steps),
                state_root=state_root,
                profile=str(args.ppo_profile),
                selector=selector,
            )
            for seed in seeds
        ]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            runs = [
                run_seed_benchmark(
                    seed=int(seed),
                    checkpoints=checkpoints,
                    holdout_seeds=holdout_seeds,
                    eval_max_steps=int(args.eval_max_steps),
                    state_root=state_root,
                    profile=str(args.ppo_profile),
                    selector=selector,
                )
                for seed in seeds
            ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_seeds": seeds,
            "checkpoints_requested_total_steps": checkpoints,
            "holdout_eval_seeds": holdout_seeds,
            "eval_max_steps": int(args.eval_max_steps),
            "bootstrap_samples": int(args.bootstrap_samples),
            "model_selector": str(selector.value),
            "ppo_profile": str(args.ppo_profile),
        },
        "runs": runs,
        "aggregate_by_checkpoint": _aggregate_by_checkpoint(
            runs, checkpoints, bootstrap_samples=int(args.bootstrap_samples)
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Benchmark written: {out_path}")
    for run in runs:
        seed = int(run["seed"])
        for row in run["checkpoints"]:
            print(
                "seed={seed} req={req} actual={actual} eval_mean={eval_mean:.2f} eval_iqm={eval_iqm:.2f}".format(
                    seed=seed,
                    req=int(row["checkpoint_requested_total_steps"]),
                    actual=int(row["checkpoint_actual_total_steps"]),
                    eval_mean=float(row["eval_summary"]["mean"]),
                    eval_iqm=float(row["eval_summary"]["iqm"]),
                )
            )


if __name__ == "__main__":
    main()
