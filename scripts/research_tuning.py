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

from snake_frame.eval_stats import bootstrap_ci_iqm, bootstrap_ci_mean, summary  # noqa: E402
from snake_frame.ppo_agent import ModelSelector, PpoSnakeAgent  # noqa: E402
from snake_frame.settings import ObsConfig, RewardConfig, Settings, ppo_profile_config  # noqa: E402

TRAIN_SEEDS_DEFAULT = [1337, 2026, 4242, 5151, 9001]
HOLDOUT_SEEDS_DEFAULT = list(range(17001, 17031))


def _parse_seed_list(raw: str) -> list[int]:
    values: list[int] = []
    token = str(raw).strip()
    if "-" in token and "," not in token:
        lo, hi = token.split("-", 1)
        lo_i = int(lo.strip())
        hi_i = int(hi.strip())
        if hi_i < lo_i:
            raise ValueError("Invalid seed range")
        return [int(v) for v in range(lo_i, hi_i + 1)]
    for part in token.split(","):
        p = part.strip()
        if not p:
            continue
        values.append(int(p))
    if not values:
        raise ValueError("No seeds provided")
    return values


def _candidate_configs() -> list[dict]:
    return [
        {
            "name": "base_research_long",
            "ppo_overrides": {},
            "reward_overrides": {},
        },
        {
            "name": "trap_stronger",
            "ppo_overrides": {},
            "reward_overrides": {
                "trap_penalty": 0.8,
                "trap_penalty_threshold": 0.18,
            },
        },
        {
            "name": "entropy_decay_slow",
            "ppo_overrides": {
                "ent_coef_end": 0.001,
            },
            "reward_overrides": {},
        },
        {
            "name": "food_focus",
            "ppo_overrides": {},
            "reward_overrides": {
                "approach_food_reward": 0.3,
                "retreat_food_penalty": 0.15,
            },
        },
    ]


def _build_agent(
    *,
    run_dir: Path,
    seed: int,
    ppo_profile: str,
    ppo_overrides: dict,
    reward_overrides: dict,
    selector: ModelSelector,
) -> PpoSnakeAgent:
    settings = Settings()
    obs_cfg = ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True, use_free_space_features=True)
    base_cfg = ppo_profile_config(ppo_profile, seed=int(seed))
    cfg = base_cfg.__class__(**(base_cfg.__dict__ | dict(ppo_overrides)))
    reward_cfg = RewardConfig(**(RewardConfig().__dict__ | dict(reward_overrides)))
    agent = PpoSnakeAgent(
        settings=settings,
        artifact_dir=run_dir / "ppo" / "baseline",
        config=cfg,
        reward_config=reward_cfg,
        obs_config=obs_cfg,
        autoload=False,
        legacy_model_path=run_dir / "ppo_snake_model.zip",
    )
    agent.set_model_selector(selector)
    return agent


def _evaluate_candidate(
    *,
    candidate: dict,
    train_steps: int,
    train_seeds: list[int],
    holdout_seeds: list[int],
    eval_max_steps: int,
    ppo_profile: str,
    selector: ModelSelector,
    stage_name: str,
    state_root: Path,
) -> dict:
    seed_rows: list[dict] = []
    all_scores: list[int] = []
    for seed in train_seeds:
        run_dir = state_root / stage_name / str(candidate["name"]) / f"seed_{int(seed)}"
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        agent = _build_agent(
            run_dir=run_dir,
            seed=int(seed),
            ppo_profile=ppo_profile,
            ppo_overrides=dict(candidate.get("ppo_overrides", {})),
            reward_overrides=dict(candidate.get("reward_overrides", {})),
            selector=selector,
        )
        try:
            actual_steps = int(
                agent.train(
                    total_timesteps=int(train_steps),
                    stop_flag=lambda: False,
                    on_progress=None,
                    on_score=None,
                    on_episode_info=None,
                )
            )
            eval_scores = agent.evaluate_holdout(
                seeds=holdout_seeds,
                max_steps=int(eval_max_steps),
                model_selector=selector,
            )
            all_scores.extend(int(v) for v in eval_scores)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "requested_train_steps": int(train_steps),
                    "actual_train_steps": int(actual_steps),
                    "eval_scores": [int(v) for v in eval_scores],
                    "eval_summary": summary(eval_scores),
                    "best_eval_score": agent.best_eval_score,
                    "best_eval_step": int(agent.best_eval_step),
                    "last_eval_score": agent.last_eval_score,
                    "eval_runs_completed": int(agent.eval_runs_completed),
                }
            )
        finally:
            try:
                if agent.model is not None and getattr(agent.model, "env", None) is not None:
                    agent.model.env.close()
            except Exception:
                pass
    candidate_summary = summary(all_scores)
    return {
        "candidate": candidate["name"],
        "ppo_overrides": candidate.get("ppo_overrides", {}),
        "reward_overrides": candidate.get("reward_overrides", {}),
        "seed_runs": seed_rows,
        "eval_summary_all_scores": candidate_summary,
        "bootstrap_ci_mean": bootstrap_ci_mean(all_scores, samples=1000, seed=41),
        "bootstrap_ci_iqm": bootstrap_ci_iqm(all_scores, samples=1000, seed=73),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage tuning workflow for research_long PPO profile.")
    parser.add_argument("--train-seeds", type=str, default="1337,2026,4242,5151,9001")
    parser.add_argument("--holdout-seeds", type=str, default="17001-17030")
    parser.add_argument("--stage1-steps", type=int, default=500_000)
    parser.add_argument("--stage2-steps", type=int, default=5_000_000)
    parser.add_argument("--eval-max-steps", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ppo-profile", choices=("research_long",), default="research_long")
    parser.add_argument("--model-selector", choices=("best", "last"), default="best")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--out", type=str, default="artifacts/long_eval/tuning_summary.json")
    args = parser.parse_args()

    train_seeds = _parse_seed_list(args.train_seeds)
    holdout_seeds = _parse_seed_list(args.holdout_seeds)
    selector = ModelSelector(str(args.model_selector))
    candidates = _candidate_configs()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.artifact_dir:
        state_root = Path(args.artifact_dir)
        state_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory()
        state_root = Path(temp_ctx.name)

    try:
        stage1 = [
            _evaluate_candidate(
                candidate=c,
                train_steps=int(args.stage1_steps),
                train_seeds=train_seeds,
                holdout_seeds=holdout_seeds,
                eval_max_steps=int(args.eval_max_steps),
                ppo_profile=str(args.ppo_profile),
                selector=selector,
                stage_name="stage1",
                state_root=state_root,
            )
            for c in candidates
        ]
        stage1_sorted = sorted(
            stage1,
            key=lambda row: float(row["eval_summary_all_scores"]["iqm"]),
            reverse=True,
        )
        top_k = max(1, int(args.top_k))
        finalists = stage1_sorted[:top_k]

        finalists_by_name = {str(row["candidate"]): row for row in finalists}
        stage2_candidates = [c for c in candidates if str(c["name"]) in finalists_by_name]
        stage2 = [
            _evaluate_candidate(
                candidate=c,
                train_steps=int(args.stage2_steps),
                train_seeds=train_seeds,
                holdout_seeds=holdout_seeds,
                eval_max_steps=int(args.eval_max_steps),
                ppo_profile=str(args.ppo_profile),
                selector=selector,
                stage_name="stage2",
                state_root=state_root,
            )
            for c in stage2_candidates
        ]
        stage2_sorted = sorted(
            stage2,
            key=lambda row: float(row["eval_summary_all_scores"]["iqm"]),
            reverse=True,
        )
        winner = stage2_sorted[0] if stage2_sorted else None
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "train_seeds": train_seeds,
                "holdout_seeds": holdout_seeds,
                "stage1_steps": int(args.stage1_steps),
                "stage2_steps": int(args.stage2_steps),
                "eval_max_steps": int(args.eval_max_steps),
                "top_k": int(top_k),
                "ppo_profile": str(args.ppo_profile),
                "model_selector": str(selector.value),
            },
            "stage1": stage1_sorted,
            "stage2": stage2_sorted,
            "winner": winner,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Tuning summary written: {out_path}")
        if winner is not None:
            print(
                "winner={name} iqm={iqm:.3f} mean={mean:.3f}".format(
                    name=str(winner["candidate"]),
                    iqm=float(winner["eval_summary_all_scores"]["iqm"]),
                    mean=float(winner["eval_summary_all_scores"]["mean"]),
                )
            )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
