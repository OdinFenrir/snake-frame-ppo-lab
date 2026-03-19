from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


_STEP_MODEL_RE = re.compile(r"^step_(\d+)_steps\.zip$", re.IGNORECASE)
_STEP_VEC_RE = re.compile(r"^step_vecnormalize_(\d+)_steps\.pkl$", re.IGNORECASE)
_STAMP_TOKEN_RE = re.compile(r"^\d{8}_\d{6}$")


def _prune_stamped_outputs(out_dir: Path, *, stem_prefix: str, suffix: str, retain: int) -> None:
    keep = max(0, int(retain))
    prefix = f"{stem_prefix}_"
    candidates: list[Path] = []
    for path in out_dir.glob(f"{prefix}*{suffix}"):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : len(name) - len(suffix)]
        if middle == "latest":
            continue
        if not _STAMP_TOKEN_RE.fullmatch(middle):
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in candidates[keep:]:
        stale.unlink(missing_ok=True)


def _resolve_default_artifact_dir(root: Path) -> Path:
    ui_prefs = root / "state" / "ui_prefs.json"
    active = "baseline"
    if ui_prefs.exists():
        payload = _read_json(ui_prefs)
        active = str(payload.get("activeExperiment", "baseline") or "baseline").strip() or "baseline"
    candidate = root / "state" / "ppo" / active
    if candidate.exists():
        return candidate.resolve()
    return (root / "state" / "ppo" / "baseline").resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _fmt_utc(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _load_vec_stats(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"obs_count": 0.0, "obs_mean_abs_avg": 0.0, "obs_var_avg": 0.0, "obs_var_max": 0.0}
    try:
        import pickle

        obj = pickle.loads(path.read_bytes())
        obs_rms = getattr(obj, "obs_rms", None)
        mean = np.asarray(getattr(obs_rms, "mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
        var = np.asarray(getattr(obs_rms, "var", np.asarray([], dtype=np.float64)), dtype=np.float64)
        count = _safe_float(getattr(obs_rms, "count", 0.0))
        if mean.ndim == 0:
            mean = mean.reshape(1)
        if var.ndim == 0:
            var = var.reshape(1)
        abs_mean = np.abs(mean)
        return {
            "obs_count": float(count),
            "obs_mean_abs_avg": float(abs_mean.mean()) if abs_mean.size > 0 else 0.0,
            "obs_var_avg": float(var.mean()) if var.size > 0 else 0.0,
            "obs_var_max": float(var.max()) if var.size > 0 else 0.0,
        }
    except Exception:
        return {"obs_count": 0.0, "obs_mean_abs_avg": 0.0, "obs_var_avg": 0.0, "obs_var_max": 0.0}


def build_timeline(artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints_dir = artifact_dir / "checkpoints"
    eval_trace = _read_jsonl(artifact_dir / "eval_logs" / "evaluations_trace.jsonl")
    metadata = _read_json(artifact_dir / "metadata.json")

    model_steps: set[int] = set()
    vec_paths: dict[int, Path] = {}
    if checkpoints_dir.exists():
        for p in checkpoints_dir.iterdir():
            if not p.is_file():
                continue
            m_model = _STEP_MODEL_RE.match(p.name)
            if m_model is not None:
                model_steps.add(int(m_model.group(1)))
                continue
            m_vec = _STEP_VEC_RE.match(p.name)
            if m_vec is not None:
                vec_paths[int(m_vec.group(1))] = p

    eval_by_step: dict[int, dict[str, Any]] = {}
    for row in eval_trace:
        step = _safe_int(row.get("step"), -1)
        if step < 0:
            continue
        eval_by_step[step] = row

    all_steps = sorted(set(model_steps) | set(vec_paths.keys()) | set(eval_by_step.keys()))
    timeline: list[dict[str, Any]] = []
    for step in all_steps:
        vec_path = vec_paths.get(step)
        vec_stats = _load_vec_stats(vec_path) if vec_path is not None else {
            "obs_count": 0.0,
            "obs_mean_abs_avg": 0.0,
            "obs_var_avg": 0.0,
            "obs_var_max": 0.0,
        }
        eval_row = eval_by_step.get(step, {})
        has_eval_point = bool(eval_row)
        timeline.append(
            {
                "step": int(step),
                "has_model_checkpoint": bool(step in model_steps),
                "has_vecnormalize_checkpoint": bool(vec_path is not None),
                "vec_file": "" if vec_path is None else vec_path.name,
                "vec_obs_count": _safe_float(vec_stats.get("obs_count")),
                "vec_obs_mean_abs_avg": _safe_float(vec_stats.get("obs_mean_abs_avg")),
                "vec_obs_var_avg": _safe_float(vec_stats.get("obs_var_avg")),
                "vec_obs_var_max": _safe_float(vec_stats.get("obs_var_max")),
                "eval_mean_reward": (_safe_float(eval_row.get("mean_reward")) if has_eval_point else None),
                "eval_mean_score": (_safe_float(eval_row.get("mean_score")) if has_eval_point else None),
                "eval_run_index": (_safe_int(eval_row.get("eval_run_index")) if has_eval_point else None),
            }
        )

    summary = {
        "latest_run_id": str(metadata.get("latest_run_id", "")),
        "requested_total_timesteps": _safe_int(metadata.get("requested_total_timesteps")),
        "actual_total_timesteps": _safe_int(metadata.get("actual_total_timesteps")),
        "checkpoint_steps_total": len(all_steps),
        "vec_checkpoint_count": len(vec_paths),
        "eval_trace_points": len(eval_by_step),
        "best_eval_mean_reward": max((_safe_float(v.get("mean_reward")) for v in eval_by_step.values()), default=0.0),
        "best_eval_mean_score": max((_safe_float(v.get("mean_score")) for v in eval_by_step.values()), default=0.0),
    }
    return timeline, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "step,has_model_checkpoint,has_vecnormalize_checkpoint,vec_file,"
        "vec_obs_count,vec_obs_mean_abs_avg,vec_obs_var_avg,vec_obs_var_max,"
        "eval_mean_reward,eval_mean_score,eval_run_index\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            (
                f"{_safe_int(row.get('step'))},{1 if row.get('has_model_checkpoint') else 0},"
                f"{1 if row.get('has_vecnormalize_checkpoint') else 0},{row.get('vec_file','')},"
                f"{_safe_float(row.get('vec_obs_count')):.6f},{_safe_float(row.get('vec_obs_mean_abs_avg')):.6f},"
                f"{_safe_float(row.get('vec_obs_var_avg')):.6f},{_safe_float(row.get('vec_obs_var_max')):.6f},"
                f"{'' if row.get('eval_mean_reward') is None else f'{_safe_float(row.get('eval_mean_reward')):.6f}'},"
                f"{'' if row.get('eval_mean_score') is None else f'{_safe_float(row.get('eval_mean_score')):.6f}'},"
                f"{'' if row.get('eval_run_index') is None else _safe_int(row.get('eval_run_index'))}\n"
            )
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_md(path: Path, artifact_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Training Input Timeline")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Artifact dir: `{artifact_dir}`")
    lines.append(f"- Run id: `{summary.get('latest_run_id')}`")
    lines.append(f"- Requested timesteps: {summary.get('requested_total_timesteps')}")
    lines.append(f"- Actual timesteps: {summary.get('actual_total_timesteps')}")
    lines.append(f"- Timeline points: {summary.get('checkpoint_steps_total')}")
    lines.append(f"- Vec checkpoints: {summary.get('vec_checkpoint_count')}")
    lines.append(f"- Eval points: {summary.get('eval_trace_points')}")
    lines.append(f"- Best eval mean_reward: {summary.get('best_eval_mean_reward')}")
    lines.append(f"- Best eval mean_score: {summary.get('best_eval_mean_score')}")
    lines.append("")
    lines.append("## Step Snapshot")
    for row in rows[-6:]:
        lines.append(
            "- step={step} vec_var_avg={vec_var_avg:.6f} vec_var_max={vec_var_max:.6f} eval_mean_reward={eval_mean_reward:.3f} eval_mean_score={eval_mean_score:.3f}".format(
                step=_safe_int(row.get("step")),
                vec_var_avg=_safe_float(row.get("vec_obs_var_avg")),
                vec_var_max=_safe_float(row.get("vec_obs_var_max")),
                eval_mean_reward=_safe_float(row.get("eval_mean_reward")),
                eval_mean_score=_safe_float(row.get("eval_mean_score")),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training-input timeline from checkpoint vecnorm + eval traces.")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="artifacts/training_input")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--retain-stamped", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    artifact_dir = (
        (root / args.artifact_dir).resolve()
        if str(args.artifact_dir or "").strip()
        else _resolve_default_artifact_dir(root)
    )
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline, summary = build_timeline(artifact_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip() or "latest"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "summary": summary,
        "timeline": timeline,
    }
    json_stamp = out_dir / f"training_input_timeline_{ts}.json"
    csv_stamp = out_dir / f"training_input_timeline_{ts}.csv"
    md_stamp = out_dir / f"training_input_timeline_{ts}.md"
    json_latest = out_dir / f"training_input_timeline_{tag}.json"
    csv_latest = out_dir / f"training_input_timeline_{tag}.csv"
    md_latest = out_dir / f"training_input_timeline_{tag}.md"

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_stamp.write_text(json_text, encoding="utf-8")
    _write_csv(csv_stamp, timeline)
    _write_md(md_stamp, artifact_dir, summary, timeline)
    json_latest.write_text(json_text, encoding="utf-8")
    csv_latest.write_text(csv_stamp.read_text(encoding="utf-8"), encoding="utf-8")
    md_latest.write_text(md_stamp.read_text(encoding="utf-8"), encoding="utf-8")
    _prune_stamped_outputs(
        out_dir,
        stem_prefix="training_input_timeline",
        suffix=".json",
        retain=int(args.retain_stamped),
    )
    _prune_stamped_outputs(
        out_dir,
        stem_prefix="training_input_timeline",
        suffix=".csv",
        retain=int(args.retain_stamped),
    )
    _prune_stamped_outputs(
        out_dir,
        stem_prefix="training_input_timeline",
        suffix=".md",
        retain=int(args.retain_stamped),
    )

    print(f"Wrote: {json_stamp}")
    print(f"Wrote: {csv_stamp}")
    print(f"Wrote: {md_stamp}")
    print(f"Wrote: {json_latest}")
    print(f"Wrote: {csv_latest}")
    print(f"Wrote: {md_latest}")


if __name__ == "__main__":
    main()
