from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuitePaths:
    project_root: Path
    artifact_dir: Path
    artifacts_root: Path
    out_dir: Path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _collect_checkpoints(checkpoints_dir: Path) -> list[dict[str, Any]]:
    if not checkpoints_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(checkpoints_dir.glob("step_*_steps.zip")):
        name = path.name
        try:
            step = int(name.split("_")[1])
        except Exception:
            continue
        stats = checkpoints_dir / f"step_vecnormalize_{step}_steps.pkl"
        rows.append(
            {
                "name": name,
                "step": step,
                "size_bytes": int(path.stat().st_size),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "has_vecnormalize": bool(stats.exists()),
            }
        )
    rows.sort(key=lambda row: int(row.get("step", 0)))
    return rows


def _collect_evaluations_npz(eval_npz: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(eval_npz),
        "exists": bool(eval_npz.exists()),
        "points": 0,
        "rows": [],
        "best": None,
        "last": None,
        "decay_from_best_to_last": None,
    }
    if not eval_npz.exists():
        return out
    try:
        data = np.load(str(eval_npz), allow_pickle=True)
    except Exception:
        out["error"] = "failed_to_read_npz"
        return out

    if "timesteps" not in data.files or "results" not in data.files:
        out["error"] = "missing_required_keys"
        return out
    steps = np.asarray(data["timesteps"])
    results = np.asarray(data["results"])
    if steps.ndim != 1 or results.ndim != 2:
        out["error"] = "unexpected_shape"
        return out

    rows: list[dict[str, Any]] = []
    means: list[float] = []
    for idx in range(min(len(steps), results.shape[0])):
        vals = np.asarray(results[idx], dtype=np.float64).reshape(-1)
        mean = float(vals.mean()) if vals.size else 0.0
        means.append(mean)
        rows.append(
            {
                "step": int(steps[idx]),
                "mean": mean,
                "std": float(vals.std(ddof=0)) if vals.size else 0.0,
                "min": float(vals.min()) if vals.size else 0.0,
                "max": float(vals.max()) if vals.size else 0.0,
                "p10": float(np.percentile(vals, 10)) if vals.size else 0.0,
                "p90": float(np.percentile(vals, 90)) if vals.size else 0.0,
            }
        )
    out["points"] = len(rows)
    out["rows"] = rows
    if rows:
        best_idx = max(range(len(rows)), key=lambda i: float(rows[i]["mean"]))
        out["best"] = rows[best_idx]
        out["last"] = rows[-1]
        out["decay_from_best_to_last"] = float(rows[best_idx]["mean"]) - float(rows[-1]["mean"])
    return out


def _latest_json(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _collect_external_summaries(artifacts_root: Path) -> dict[str, Any]:
    loop_candidates = list((artifacts_root / "loop_eval").glob("**/summary.json"))
    long_candidates = list((artifacts_root / "long_eval").glob("*summary*.json"))
    dashboard_latest = artifacts_root / "test_dashboard" / "latest" / "summary.json"

    loop_latest = _latest_json(loop_candidates)
    long_latest = _latest_json(long_candidates)
    dash_latest = dashboard_latest if dashboard_latest.exists() else None

    return {
        "loop_eval_latest_path": None if loop_latest is None else str(loop_latest),
        "loop_eval_latest": None if loop_latest is None else _read_json(loop_latest),
        "long_eval_latest_path": None if long_latest is None else str(long_latest),
        "long_eval_latest": None if long_latest is None else _read_json(long_latest),
        "dashboard_latest_path": None if dash_latest is None else str(dash_latest),
        "dashboard_latest": None if dash_latest is None else _read_json(dash_latest),
    }


def _build_insights(bundle: dict[str, Any]) -> list[str]:
    insights: list[str] = []
    meta = bundle.get("metadata") or {}
    eval_npz = bundle.get("eval_npz") or {}
    eval_trace = bundle.get("eval_trace") or {}
    train_trace = bundle.get("training_trace") or {}
    checkpoints = bundle.get("checkpoints") or []

    requested = _safe_int(meta.get("requested_total_timesteps"), 0)
    actual = _safe_int(meta.get("actual_total_timesteps"), 0)
    if requested > 0:
        ratio = float(actual) / float(requested) if requested else 0.0
        insights.append(f"actual/requested steps ratio={ratio:.3f} ({actual}/{requested})")
    best_eval_step = _safe_int(meta.get("best_eval_step"), 0)
    if actual > 0 and best_eval_step > actual:
        insights.append(f"warning: best_eval_step ({best_eval_step}) > actual_total_timesteps ({actual})")

    decay = eval_npz.get("decay_from_best_to_last")
    if decay is not None:
        decay_f = _safe_float(decay, 0.0)
        if decay_f > 0:
            insights.append(f"eval decay from best to last: {decay_f:.2f}")
        else:
            insights.append("no eval decay from best to last detected")

    latest_train = train_trace.get("latest") or {}
    deaths = latest_train.get("deaths") or {}
    if isinstance(deaths, dict) and deaths:
        top = sorted(((k, _safe_int(v, 0)) for k, v in deaths.items()), key=lambda kv: kv[1], reverse=True)
        if top:
            insights.append(f"top training death reason: {top[0][0]} ({top[0][1]})")

    if isinstance(checkpoints, list) and checkpoints:
        max_cp = max(_safe_int(row.get("step"), 0) for row in checkpoints if isinstance(row, dict))
        if actual > 0 and max_cp > actual:
            insights.append(f"warning: checkpoint step ({max_cp}) exceeds current actual steps ({actual})")
    score_stats = eval_trace.get("score_stats") or {}
    if isinstance(score_stats, dict) and score_stats:
        best_score = _safe_float(score_stats.get("best_mean_score"), 0.0)
        last_score = _safe_float(score_stats.get("last_mean_score"), 0.0)
        if best_score > 0.0:
            insights.append(f"eval score decay from best to last: {best_score - last_score:.2f}")

    ext = bundle.get("external_summaries") or {}
    loop = ext.get("loop_eval_latest") or {}
    if isinstance(loop, dict) and loop:
        status = str(loop.get("status", "")).strip()
        if status:
            insights.append(f"loop_eval status: {status}")

    return insights


def _markdown_report(bundle: dict[str, Any]) -> str:
    meta = bundle.get("metadata") or {}
    eval_npz = bundle.get("eval_npz") or {}
    checkpoints = bundle.get("checkpoints") or []
    eval_trace = bundle.get("eval_trace") or {}
    train_trace = bundle.get("training_trace") or {}
    insights = bundle.get("insights") or []

    lines: list[str] = []
    lines.append("# Post-Run Diagnostics Bundle")
    lines.append("")
    lines.append(f"- Generated: {bundle.get('generated_at_utc')}")
    lines.append(f"- Artifact Dir: `{bundle.get('artifact_dir')}`")
    lines.append(f"- Requested Steps: {meta.get('requested_total_timesteps')}")
    lines.append(f"- Actual Steps: {meta.get('actual_total_timesteps')}")
    lines.append(f"- Best Eval: {meta.get('best_eval_score')} @ {meta.get('best_eval_step')}")
    lines.append(f"- Last Eval: {meta.get('last_eval_score')}")
    prov = meta.get("provenance") or {}
    if isinstance(prov, dict):
        git = prov.get("git") or {}
        env = prov.get("environment") or {}
        if isinstance(git, dict):
            lines.append(
                f"- Provenance: commit={git.get('git_commit')} branch={git.get('git_branch')} dirty={git.get('git_dirty')}"
            )
        if isinstance(env, dict):
            lines.append(f"- Runtime: python={env.get('python_version')} platform={env.get('platform')}")
    lines.append("")
    lines.append("## Key Insights")
    if insights:
        for item in insights:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Eval NPZ")
    lines.append(f"- Points: {eval_npz.get('points')}")
    lines.append(f"- Decay Best->Last: {eval_npz.get('decay_from_best_to_last')}")
    best = eval_npz.get("best")
    last = eval_npz.get("last")
    if isinstance(best, dict):
        lines.append(f"- Best Row: step={best.get('step')} mean={best.get('mean')}")
    if isinstance(last, dict):
        lines.append(f"- Last Row: step={last.get('step')} mean={last.get('mean')}")
    lines.append("")
    lines.append("## Traces")
    lines.append(f"- eval_trace rows: {eval_trace.get('rows')}")
    score_stats = eval_trace.get("score_stats")
    if isinstance(score_stats, dict) and score_stats:
        lines.append(
            f"- eval score mean (last/best): {score_stats.get('last_mean_score')} / {score_stats.get('best_mean_score')}"
        )
    lines.append(f"- training_trace rows: {train_trace.get('rows')}")
    if train_trace.get("latest"):
        latest = train_trace["latest"]
        lines.append(f"- latest training deaths: {latest.get('deaths')}")
        lines.append(f"- latest training score_summary: {latest.get('score_summary')}")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append(f"- Count: {len(checkpoints)}")
    if checkpoints:
        last_cp = checkpoints[-1]
        lines.append(
            f"- Latest: {last_cp.get('name')} step={last_cp.get('step')} has_vecnormalize={last_cp.get('has_vecnormalize')}"
        )
    return "\n".join(lines) + "\n"


def build_bundle(paths: SuitePaths) -> dict[str, Any]:
    metadata_path = paths.artifact_dir / "metadata.json"
    eval_npz_path = paths.artifact_dir / "eval_logs" / "evaluations.npz"
    eval_trace_path = paths.artifact_dir / "eval_logs" / "evaluations_trace.jsonl"
    train_trace_path = paths.artifact_dir / "training_trace.jsonl"
    checkpoints_dir = paths.artifact_dir / "checkpoints"

    metadata = _read_json(metadata_path) or {}
    eval_trace = _read_jsonl(eval_trace_path)
    train_trace = _read_jsonl(train_trace_path)
    score_rows = [
        _safe_float(row.get("mean_score"))
        for row in eval_trace
        if isinstance(row, dict) and row.get("mean_score") is not None
    ]
    score_stats: dict[str, Any] | None = None
    if score_rows:
        score_stats = {
            "points": int(len(score_rows)),
            "best_mean_score": float(max(score_rows)),
            "last_mean_score": float(score_rows[-1]),
        }

    bundle: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(paths.artifact_dir),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "eval_npz": _collect_evaluations_npz(eval_npz_path),
        "eval_trace": {
            "path": str(eval_trace_path),
            "rows": len(eval_trace),
            "latest": eval_trace[-1] if eval_trace else None,
            "score_stats": score_stats,
        },
        "training_trace": {
            "path": str(train_trace_path),
            "rows": len(train_trace),
            "latest": train_trace[-1] if train_trace else None,
        },
        "checkpoints": _collect_checkpoints(checkpoints_dir),
        "external_summaries": _collect_external_summaries(paths.artifacts_root),
    }
    bundle["insights"] = _build_insights(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a complete post-run diagnostics suite into one bundle.")
    parser.add_argument("--artifact-dir", type=str, default="state/ppo/baseline")
    parser.add_argument("--artifacts-root", type=str, default="artifacts")
    parser.add_argument("--out-dir", type=str, default="artifacts/share")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = SuitePaths(
        project_root=root,
        artifact_dir=(root / args.artifact_dir).resolve(),
        artifacts_root=(root / args.artifacts_root).resolve(),
        out_dir=(root / args.out_dir).resolve(),
    )

    bundle = build_bundle(paths)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_json = paths.out_dir / "diagnostics_bundle.json"
    bundle_md = paths.out_dir / "diagnostics_bundle.md"
    bundle_json.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    bundle_md.write_text(_markdown_report(bundle), encoding="utf-8")

    print(f"Wrote: {bundle_json}")
    print(f"Wrote: {bundle_md}")
    if args.print_summary:
        print("")
        print(bundle_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
