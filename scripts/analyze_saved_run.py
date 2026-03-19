from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _format_pct(value: float) -> str:
    return f"{(100.0 * value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved PPO run diagnostics.")
    parser.add_argument("--artifact-dir", type=str, default="state/ppo/baseline")
    parser.add_argument("--requested-steps", type=int, default=0)
    parser.add_argument("--run-id", type=str, default="")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    eval_trace_path = artifact_dir / "eval_logs" / "evaluations_trace.jsonl"
    train_trace_path = artifact_dir / "training_trace.jsonl"

    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata: {metadata_path}")

    meta = _read_json(metadata_path)
    eval_rows = _read_jsonl(eval_trace_path)
    train_rows = _read_jsonl(train_trace_path)
    run_id = str(args.run_id or meta.get("latest_run_id") or "").strip()
    requested_steps = int(args.requested_steps or 0)
    if run_id:
        eval_rows = [row for row in eval_rows if str(row.get("run_id", "")).strip() == run_id]
        train_rows = [row for row in train_rows if str(row.get("run_id", "")).strip() == run_id]
    elif requested_steps > 0:
        eval_rows = [row for row in eval_rows if int(row.get("requested_total_timesteps", 0) or 0) == requested_steps]
        train_rows = [row for row in train_rows if int(row.get("requested_total_timesteps", 0) or 0) == requested_steps]

    print(f"Artifact: {artifact_dir}")
    if run_id:
        print(f"Filter run_id: {run_id}")
    elif requested_steps > 0:
        print(f"Filter requested_steps: {requested_steps}")
    print(f"Requested steps: {meta.get('requested_total_timesteps')}")
    print(f"Actual steps: {meta.get('actual_total_timesteps')}")
    print(
        "Eval summary: best={best} at step={step}, last={last}, runs={runs}".format(
            best=meta.get("best_eval_score"),
            step=meta.get("best_eval_step"),
            last=meta.get("last_eval_score"),
            runs=meta.get("eval_runs_completed"),
        )
    )
    prov = dict(meta.get("provenance", {})) if isinstance(meta.get("provenance"), dict) else {}
    if prov:
        git = dict(prov.get("git", {})) if isinstance(prov.get("git"), dict) else {}
        deps = dict(prov.get("dependencies", {})) if isinstance(prov.get("dependencies"), dict) else {}
        py = dict(prov.get("environment", {})) if isinstance(prov.get("environment"), dict) else {}
        print(
            "Provenance: commit={commit} branch={branch} dirty={dirty} py={pyv}".format(
                commit=git.get("git_commit"),
                branch=git.get("git_branch"),
                dirty=git.get("git_dirty"),
                pyv=py.get("python_version"),
            )
        )
        print(
            "Deps: torch={torch} sb3={sb3} gym={gym} pygame={pygame}".format(
                torch=deps.get("torch"),
                sb3=deps.get("stable_baselines3"),
                gym=deps.get("gymnasium"),
                pygame=deps.get("pygame"),
            )
        )

    if eval_rows:
        means = [float(row.get("mean_reward", 0.0)) for row in eval_rows]
        steps = [int(row.get("step", 0)) for row in eval_rows]
        best_idx = max(range(len(means)), key=lambda i: means[i])
        print(f"Eval trace points: {len(eval_rows)}")
        print(f"Peak eval mean: {means[best_idx]:.2f} at step {steps[best_idx]}")
        if len(means) > best_idx + 1:
            tail = means[best_idx + 1 :]
            tail_mean = statistics.fmean(tail) if tail else means[best_idx]
            decay = means[best_idx] - float(tail_mean)
            print(f"Post-peak decay (vs. tail mean): {decay:.2f}")

        for row in eval_rows:
            ep = [float(v) for v in row.get("episode_rewards", [])]
            if not ep:
                continue
            low = min(ep)
            high = max(ep)
            span = high - low
            print(
                " step={step} mean={mean:.2f} min={low:.2f} max={high:.2f} span={span:.2f}".format(
                    step=int(row.get("step", 0)),
                    mean=float(row.get("mean_reward", 0.0)),
                    low=low,
                    high=high,
                    span=span,
                )
            )
    else:
        print("Eval trace points: none (run training after this patch to generate them)")

    if train_rows:
        latest = train_rows[-1]
        deaths = dict(latest.get("deaths", {}))
        total_deaths = max(1, int(sum(int(v) for v in deaths.values())))
        ranked = sorted(deaths.items(), key=lambda kv: int(kv[1]), reverse=True)
        print("Latest training death mix:")
        for key, value in ranked:
            frac = float(int(value)) / float(total_deaths)
            print(f" {key}: {value} ({_format_pct(frac)})")
        score_summary = dict(latest.get("score_summary", {}))
        print(
            "Latest training score summary: mean={mean} median={median} p90={p90} best={best} last={last}".format(
                mean=score_summary.get("mean"),
                median=score_summary.get("median"),
                p90=score_summary.get("p90"),
                best=score_summary.get("best"),
                last=score_summary.get("last"),
            )
        )
    else:
        print("Training trace rows: none (run training after this patch to generate them)")


if __name__ == "__main__":
    main()
