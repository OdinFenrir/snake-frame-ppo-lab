from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import re
from typing import Any

import numpy as np


_STEP_VEC_RE = re.compile(r"^step_vecnormalize_(\d+)_steps\.pkl$", re.IGNORECASE)
_STAMP_TOKEN_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class ReportPaths:
    artifact_dir: Path
    out_dir: Path


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
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


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


def _format_ts(unix_s: float | int | None) -> str:
    if unix_s is None:
        return "n/a"
    try:
        dt = datetime.fromtimestamp(float(unix_s), tz=timezone.utc)
    except Exception:
        return "n/a"
    return dt.isoformat()


def _load_vecnormalize(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        obj = pickle.loads(path.read_bytes())
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": f"load_failed:{exc}"}

    obs_rms = getattr(obj, "obs_rms", None)
    mean = np.asarray(getattr(obs_rms, "mean", np.asarray([], dtype=np.float64)), dtype=np.float64)
    var = np.asarray(getattr(obs_rms, "var", np.asarray([], dtype=np.float64)), dtype=np.float64)
    count = _safe_float(getattr(obs_rms, "count", 0.0))
    if mean.ndim == 0:
        mean = mean.reshape(1)
    if var.ndim == 0:
        var = var.reshape(1)

    abs_mean = np.abs(mean)
    top_k = min(8, int(abs_mean.size))
    top_idx = np.argsort(-abs_mean)[:top_k] if top_k > 0 else np.asarray([], dtype=np.int64)
    feature_top = [
        {
            "feature_index": int(i),
            "obs_mean": float(mean[i]),
            "obs_var": float(var[i]) if i < var.size else 0.0,
        }
        for i in top_idx.tolist()
    ]

    return {
        "exists": True,
        "path": str(path),
        "obs_dim": int(mean.size),
        "obs_count": float(count),
        "obs_mean_abs_avg": float(abs_mean.mean()) if abs_mean.size > 0 else 0.0,
        "obs_mean_abs_max": float(abs_mean.max()) if abs_mean.size > 0 else 0.0,
        "obs_var_avg": float(var.mean()) if var.size > 0 else 0.0,
        "obs_var_min": float(var.min()) if var.size > 0 else 0.0,
        "obs_var_max": float(var.max()) if var.size > 0 else 0.0,
        "clip_obs": _safe_float(getattr(obj, "clip_obs", 0.0)),
        "clip_reward": _safe_float(getattr(obj, "clip_reward", 0.0)),
        "gamma": _safe_float(getattr(obj, "gamma", 0.0)),
        "epsilon": _safe_float(getattr(obj, "epsilon", 0.0)),
        "norm_obs": bool(getattr(obj, "norm_obs", False)),
        "norm_reward": bool(getattr(obj, "norm_reward", False)),
        "top_features_by_abs_mean": feature_top,
    }


def _checkpoint_vec_rows(checkpoints_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not checkpoints_dir.exists():
        return rows
    for p in sorted(checkpoints_dir.glob("step_vecnormalize_*_steps.pkl")):
        m = _STEP_VEC_RE.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        vec = _load_vecnormalize(p)
        rows.append(
            {
                "step": int(step),
                "file": p.name,
                "obs_dim": _safe_int(vec.get("obs_dim")),
                "obs_count": _safe_float(vec.get("obs_count")),
                "obs_mean_abs_avg": _safe_float(vec.get("obs_mean_abs_avg")),
                "obs_var_avg": _safe_float(vec.get("obs_var_avg")),
                "obs_var_max": _safe_float(vec.get("obs_var_max")),
                "mtime_utc": _format_ts(p.stat().st_mtime),
            }
        )
    return rows


def _pick_run_rows(
    metadata: dict[str, Any],
    training_trace_rows: list[dict[str, Any]],
    eval_trace_rows: list[dict[str, Any]],
    run_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    resolved_run_id = str(run_id or metadata.get("latest_run_id") or "").strip()
    train_row = None
    eval_rows: list[dict[str, Any]] = []
    if resolved_run_id:
        for row in training_trace_rows:
            if str(row.get("run_id", "")).strip() == resolved_run_id:
                train_row = row
        eval_rows = [row for row in eval_trace_rows if str(row.get("run_id", "")).strip() == resolved_run_id]
    else:
        if training_trace_rows:
            train_row = training_trace_rows[-1]
            resolved_run_id = str(train_row.get("run_id", "")).strip()
        eval_rows = eval_trace_rows
    return train_row, eval_rows, resolved_run_id


def _input_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(metadata.get("config", {})) if isinstance(metadata.get("config"), dict) else {}
    obs_cfg = dict(metadata.get("obs_config", {})) if isinstance(metadata.get("obs_config"), dict) else {}
    reward_cfg = dict(metadata.get("reward_config", {})) if isinstance(metadata.get("reward_config"), dict) else {}
    return {
        "env_count": _safe_int(cfg.get("env_count")),
        "n_steps": _safe_int(cfg.get("n_steps")),
        "batch_size": _safe_int(cfg.get("batch_size")),
        "n_epochs": _safe_int(cfg.get("n_epochs")),
        "gamma": _safe_float(cfg.get("gamma")),
        "gae_lambda": _safe_float(cfg.get("gae_lambda")),
        "learning_rate_start": _safe_float(cfg.get("learning_rate_start")),
        "learning_rate_end": _safe_float(cfg.get("learning_rate_end")),
        "clip_range": _safe_float(cfg.get("clip_range")),
        "ent_coef_start": _safe_float(cfg.get("ent_coef_start")),
        "ent_coef_end": _safe_float(cfg.get("ent_coef_end")),
        "policy_net_arch": cfg.get("policy_net_arch"),
        "obs_config": obs_cfg,
        "reward_config": reward_cfg,
    }


def _derive_checks(
    metadata: dict[str, Any],
    train_row: dict[str, Any] | None,
    eval_rows: list[dict[str, Any]],
    vec_latest: dict[str, Any],
    checkpoint_vec_rows: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "latest_run_id_present",
            "ok": bool(run_id),
            "detail": f"run_id={run_id or 'missing'}",
        }
    )
    checks.append(
        {
            "name": "training_trace_has_run_row",
            "ok": bool(train_row is not None),
            "detail": "training row found" if train_row is not None else "no training row for run_id",
        }
    )
    checks.append(
        {
            "name": "eval_trace_has_run_rows",
            "ok": bool(len(eval_rows) > 0),
            "detail": f"eval_rows={len(eval_rows)}",
        }
    )
    vec_dim = _safe_int(vec_latest.get("obs_dim"), -1)
    checks.append(
        {
            "name": "vecnormalize_obs_dim_valid",
            "ok": bool(vec_dim > 0),
            "detail": f"obs_dim={vec_dim}",
        }
    )
    config = dict(metadata.get("config", {})) if isinstance(metadata.get("config"), dict) else {}
    env_count = _safe_int(config.get("env_count"), 0)
    n_steps = _safe_int(config.get("n_steps"), 0)
    checks.append(
        {
            "name": "rollout_size_defined",
            "ok": bool(env_count > 0 and n_steps > 0),
            "detail": f"env_count={env_count} n_steps={n_steps} rollout={env_count * n_steps}",
        }
    )
    checkpoint_steps = [int(row.get("step", 0)) for row in checkpoint_vec_rows if _safe_int(row.get("step"), 0) > 0]
    checks.append(
        {
            "name": "checkpoint_vecnormalize_present",
            "ok": bool(len(checkpoint_steps) > 0),
            "detail": f"checkpoint_vec_files={len(checkpoint_steps)}",
        }
    )
    return checks


def _to_markdown(report: dict[str, Any]) -> str:
    run = report.get("run", {})
    contract = report.get("input_contract", {})
    vec = report.get("vecnormalize_latest", {})
    checks = report.get("checks", [])
    eval_rows = report.get("eval_trace_rows", [])
    lines: list[str] = []
    lines.append("# Training Input Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {report.get('generated_at_utc')}")
    lines.append(f"- Artifact dir: `{report.get('artifact_dir')}`")
    lines.append(f"- Run id: `{run.get('run_id')}`")
    lines.append(f"- Run started (UTC): {run.get('run_started_at_utc')}")
    lines.append("")
    lines.append("## Input Contract")
    lines.append(f"- env_count: {contract.get('env_count')}")
    lines.append(f"- n_steps: {contract.get('n_steps')}")
    lines.append(f"- batch_size: {contract.get('batch_size')}")
    lines.append(f"- n_epochs: {contract.get('n_epochs')}")
    lines.append(f"- gamma: {contract.get('gamma')}")
    lines.append(f"- gae_lambda: {contract.get('gae_lambda')}")
    lines.append(f"- learning_rate (start -> end): {contract.get('learning_rate_start')} -> {contract.get('learning_rate_end')}")
    lines.append(f"- entropy coef (start -> end): {contract.get('ent_coef_start')} -> {contract.get('ent_coef_end')}")
    lines.append("")
    lines.append("## VecNormalize (Latest)")
    lines.append(f"- obs_dim: {vec.get('obs_dim')}")
    lines.append(f"- obs_count: {vec.get('obs_count')}")
    lines.append(f"- obs_mean_abs_avg: {vec.get('obs_mean_abs_avg')}")
    lines.append(f"- obs_var_avg: {vec.get('obs_var_avg')}")
    lines.append(f"- obs_var_max: {vec.get('obs_var_max')}")
    lines.append("")
    lines.append("## Run-Level Signals")
    lines.append(f"- training row found: {bool(report.get('training_trace_row'))}")
    lines.append(f"- eval checkpoints in trace: {len(eval_rows)}")
    if eval_rows:
        best_eval = max((_safe_float(row.get('mean_reward')) for row in eval_rows), default=0.0)
        lines.append(f"- best eval mean_reward in run: {best_eval:.3f}")
    lines.append("")
    lines.append("## Checks")
    for chk in checks:
        icon = "OK" if bool(chk.get("ok")) else "FAIL"
        lines.append(f"- [{icon}] {chk.get('name')}: {chk.get('detail')}")
    lines.append("")
    return "\n".join(lines)


def _write_checkpoint_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "step,file,obs_dim,obs_count,obs_mean_abs_avg,obs_var_avg,obs_var_max,mtime_utc\n"
    out = [header]
    for row in rows:
        out.append(
            "{step},{file},{obs_dim},{obs_count:.6f},{obs_mean_abs_avg:.6f},{obs_var_avg:.6f},{obs_var_max:.6f},{mtime_utc}\n".format(
                step=_safe_int(row.get("step")),
                file=str(row.get("file", "")),
                obs_dim=_safe_int(row.get("obs_dim")),
                obs_count=_safe_float(row.get("obs_count")),
                obs_mean_abs_avg=_safe_float(row.get("obs_mean_abs_avg")),
                obs_var_avg=_safe_float(row.get("obs_var_avg")),
                obs_var_max=_safe_float(row.get("obs_var_max")),
                mtime_utc=str(row.get("mtime_utc", "")),
            )
        )
    path.write_text("".join(out), encoding="utf-8")


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


def build_report(paths: ReportPaths, run_id: str) -> dict[str, Any]:
    metadata_path = paths.artifact_dir / "metadata.json"
    training_trace_path = paths.artifact_dir / "training_trace.jsonl"
    eval_trace_path = paths.artifact_dir / "eval_logs" / "evaluations_trace.jsonl"
    vecnormalize_path = paths.artifact_dir / "vecnormalize.pkl"
    checkpoints_dir = paths.artifact_dir / "checkpoints"

    metadata = _read_json(metadata_path)
    training_rows = _read_jsonl(training_trace_path)
    eval_rows_all = _read_jsonl(eval_trace_path)
    train_row, eval_rows, resolved_run_id = _pick_run_rows(metadata, training_rows, eval_rows_all, run_id)
    vec_latest = _load_vecnormalize(vecnormalize_path)
    checkpoint_vec = _checkpoint_vec_rows(checkpoints_dir)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(paths.artifact_dir),
        "metadata_path": str(metadata_path),
        "training_trace_path": str(training_trace_path),
        "eval_trace_path": str(eval_trace_path),
        "vecnormalize_path": str(vecnormalize_path),
        "run": {
            "run_id": resolved_run_id,
            "run_started_at_unix_s": _safe_float((train_row or {}).get("run_started_at_unix_s"), 0.0),
            "run_started_at_utc": _format_ts((train_row or {}).get("run_started_at_unix_s")),
            "requested_total_timesteps": _safe_int((train_row or metadata).get("requested_total_timesteps")),
            "actual_total_timesteps": _safe_int((train_row or metadata).get("actual_total_timesteps")),
        },
        "input_contract": _input_contract(metadata),
        "training_trace_row": train_row,
        "eval_trace_rows": eval_rows,
        "vecnormalize_latest": vec_latest,
        "checkpoint_vecnormalize_rows": checkpoint_vec,
    }
    report["checks"] = _derive_checks(
        metadata=metadata,
        train_row=train_row,
        eval_rows=eval_rows,
        vec_latest=vec_latest,
        checkpoint_vec_rows=checkpoint_vec,
        run_id=resolved_run_id,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a training-input-only report from PPO artifacts (metadata, traces, vecnormalize)."
    )
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="artifacts/training_input")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--retain-stamped", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    artifact_dir = (
        (root / args.artifact_dir).resolve()
        if str(args.artifact_dir or "").strip()
        else _resolve_default_artifact_dir(root)
    )
    paths = ReportPaths(
        artifact_dir=artifact_dir,
        out_dir=(root / args.out_dir).resolve(),
    )
    report = build_report(paths=paths, run_id=str(args.run_id or "").strip())
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip()
    if not tag:
        tag = "latest"

    json_stamp = paths.out_dir / f"training_input_{ts}.json"
    md_stamp = paths.out_dir / f"training_input_{ts}.md"
    csv_stamp = paths.out_dir / f"training_input_checkpoint_vecnorm_{ts}.csv"
    json_latest = paths.out_dir / f"training_input_{tag}.json"
    md_latest = paths.out_dir / f"training_input_{tag}.md"
    csv_latest = paths.out_dir / f"training_input_checkpoint_vecnorm_{tag}.csv"

    json_payload = json.dumps(report, indent=2, ensure_ascii=False)
    md_payload = _to_markdown(report)
    _write_checkpoint_csv(csv_stamp, report.get("checkpoint_vecnormalize_rows", []))

    json_stamp.write_text(json_payload, encoding="utf-8")
    md_stamp.write_text(md_payload, encoding="utf-8")
    json_latest.write_text(json_payload, encoding="utf-8")
    md_latest.write_text(md_payload, encoding="utf-8")
    csv_latest.write_text(csv_stamp.read_text(encoding="utf-8"), encoding="utf-8")
    _prune_stamped_outputs(paths.out_dir, stem_prefix="training_input", suffix=".json", retain=int(args.retain_stamped))
    _prune_stamped_outputs(paths.out_dir, stem_prefix="training_input", suffix=".md", retain=int(args.retain_stamped))
    _prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="training_input_checkpoint_vecnorm",
        suffix=".csv",
        retain=int(args.retain_stamped),
    )

    print(f"Wrote: {json_stamp}")
    print(f"Wrote: {md_stamp}")
    print(f"Wrote: {csv_stamp}")
    print(f"Wrote: {json_latest}")
    print(f"Wrote: {md_latest}")
    print(f"Wrote: {csv_latest}")


if __name__ == "__main__":
    main()
