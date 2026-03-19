from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

_STAMP_TOKEN_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class ReportPaths:
    artifact_dir: Path
    run_log_path: Path
    out_dir: Path


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
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
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


def _episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        episode = _safe_int(row.get("episode_index"), -1)
        if episode < 1:
            continue
        out.append(row)
    out.sort(key=lambda r: _safe_int(r.get("episode_index")))
    return out


def _death_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"wall": 0, "body": 0, "starvation": 0, "fill": 0, "none": 0, "other": 0}
    for row in rows:
        reason = str(row.get("death_reason", "other") or "other").strip().lower()
        if reason not in counts:
            reason = "other"
        counts[reason] = int(counts.get(reason, 0) + 1)
    return counts


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("mode", "unknown") or "unknown").strip().lower()
        counts[mode] = int(counts.get(mode, 0) + 1)
    return counts


def _build_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append({"name": "run_session_rows_present", "ok": bool(rows), "detail": f"rows={len(rows)}"})
    if not rows:
        return checks

    episodes = [_safe_int(r.get("episode_index")) for r in rows]
    monotonic = all(episodes[i] > episodes[i - 1] for i in range(1, len(episodes)))
    checks.append({"name": "episode_index_strictly_increasing", "ok": monotonic, "detail": f"first={episodes[0]} last={episodes[-1]}"})

    bad_pct = [
        _safe_float(r.get("interventions_pct"))
        for r in rows
        if _safe_float(r.get("interventions_pct")) < 0.0 or _safe_float(r.get("interventions_pct")) > 100.0
    ]
    checks.append({"name": "interventions_pct_in_0_100", "ok": len(bad_pct) == 0, "detail": f"invalid_rows={len(bad_pct)}"})

    neg_decisions = [_safe_int(r.get("decisions_delta")) for r in rows if _safe_int(r.get("decisions_delta")) < 0]
    neg_interventions = [_safe_int(r.get("interventions_delta")) for r in rows if _safe_int(r.get("interventions_delta")) < 0]
    checks.append({"name": "decision_deltas_non_negative", "ok": len(neg_decisions) == 0, "detail": f"invalid_rows={len(neg_decisions)}"})
    checks.append({"name": "intervention_deltas_non_negative", "ok": len(neg_interventions) == 0, "detail": f"invalid_rows={len(neg_interventions)}"})
    return checks


def _build_report(paths: ReportPaths) -> dict[str, Any]:
    metadata = _read_json(paths.artifact_dir / "metadata.json")
    rows = _episode_rows(_read_jsonl(paths.run_log_path))
    scores = [_safe_int(r.get("score")) for r in rows]
    interventions_pct = [_safe_float(r.get("interventions_pct")) for r in rows]
    decisions_delta = [_safe_int(r.get("decisions_delta")) for r in rows]
    interventions_delta = [_safe_int(r.get("interventions_delta")) for r in rows]
    risk_total_series = [_safe_int(r.get("risk_total")) for r in rows]
    risk_last = risk_total_series[-1] if risk_total_series else 0

    trend = "unknown"
    if len(scores) >= 6:
        half = max(1, len(scores) // 2)
        early = sum(scores[:half]) / float(half)
        late = sum(scores[-half:]) / float(half)
        if late > early:
            trend = "up"
        elif late < early:
            trend = "down"
        else:
            trend = "flat"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(paths.artifact_dir),
        "run_session_log_path": str(paths.run_log_path),
        "run_id": str(metadata.get("latest_run_id", "")),
        "episodes": {
            "count": len(rows),
            "score_mean": (sum(scores) / float(len(scores))) if scores else 0.0,
            "score_best": max(scores) if scores else 0,
            "score_last": scores[-1] if scores else 0,
            "score_trend": trend,
            "deaths": _death_counts(rows),
            "modes": _mode_counts(rows),
        },
        "agent_control": {
            "interventions_pct_mean": (sum(interventions_pct) / float(len(interventions_pct))) if interventions_pct else 0.0,
            "interventions_delta_total": sum(interventions_delta) if interventions_delta else 0,
            "decisions_delta_total": sum(decisions_delta) if decisions_delta else 0,
            "risk_total_last": int(risk_last),
        },
        "rows": rows,
        "checks": _build_checks(rows),
    }
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    ep = dict(report.get("episodes", {}))
    ctl = dict(report.get("agent_control", {}))
    checks = list(report.get("checks", []))
    lines: list[str] = []
    lines.append("# Agent Performance Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {report.get('generated_at_utc')}")
    lines.append(f"- Artifact dir: `{report.get('artifact_dir')}`")
    lines.append(f"- Run log: `{report.get('run_session_log_path')}`")
    lines.append(f"- Run id: `{report.get('run_id')}`")
    lines.append("")
    lines.append("## Episode Performance")
    lines.append(f"- Episodes: {ep.get('count')}")
    lines.append(f"- Score mean / best / last: {ep.get('score_mean'):.2f} / {ep.get('score_best')} / {ep.get('score_last')}")
    lines.append(f"- Score trend: {ep.get('score_trend')}")
    lines.append(f"- Death counts: {ep.get('deaths')}")
    lines.append(f"- Mode counts: {ep.get('modes')}")
    lines.append("")
    lines.append("## Agent Control")
    lines.append(f"- Mean interventions %: {ctl.get('interventions_pct_mean'):.3f}")
    lines.append(f"- Interventions total: {ctl.get('interventions_delta_total')}")
    lines.append(f"- Decisions total: {ctl.get('decisions_delta_total')}")
    lines.append(f"- Risk total last: {ctl.get('risk_total_last')}")
    lines.append("")
    lines.append("## Checks")
    for chk in checks:
        icon = "OK" if bool(chk.get("ok")) else "FAIL"
        lines.append(f"- [{icon}] {chk.get('name')}: {chk.get('detail')}")
    lines.append("")
    return "\n".join(lines)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "episode_index,score,death_reason,mode,train_total_steps,interventions_pct,interventions_delta,decisions_delta,"
        "risk_total,stuck_episode_delta,loop_escape_activations_total,generated_at_unix_s\n"
    )
    out = [header]
    for row in rows:
        out.append(
            (
                f"{_safe_int(row.get('episode_index'))},{_safe_int(row.get('score'))},{str(row.get('death_reason',''))},"
                f"{str(row.get('mode',''))},{_safe_int(row.get('train_total_steps'))},{_safe_float(row.get('interventions_pct')):.6f},"
                f"{_safe_int(row.get('interventions_delta'))},{_safe_int(row.get('decisions_delta'))},"
                f"{_safe_int(row.get('risk_total'))},{_safe_int(row.get('stuck_episode_delta'))},"
                f"{_safe_int(row.get('loop_escape_activations_total'))},{_safe_float(row.get('generated_at_unix_s')):.6f}\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build agent-performance report from run session telemetry.")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--run-log", type=str, default="artifacts/live_eval/run_session_log.jsonl")
    parser.add_argument("--out-dir", type=str, default="artifacts/agent_performance")
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
        run_log_path=(root / args.run_log).resolve(),
        out_dir=(root / args.out_dir).resolve(),
    )
    report = _build_report(paths)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip() or "latest"
    json_stamp = paths.out_dir / f"agent_performance_{ts}.json"
    md_stamp = paths.out_dir / f"agent_performance_{ts}.md"
    csv_stamp = paths.out_dir / f"agent_performance_rows_{ts}.csv"
    json_latest = paths.out_dir / f"agent_performance_{tag}.json"
    md_latest = paths.out_dir / f"agent_performance_{tag}.md"
    csv_latest = paths.out_dir / f"agent_performance_rows_{tag}.csv"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    md_text = _to_markdown(report)
    _write_rows_csv(csv_stamp, list(report.get("rows", [])))

    json_stamp.write_text(json_text, encoding="utf-8")
    md_stamp.write_text(md_text, encoding="utf-8")
    json_latest.write_text(json_text, encoding="utf-8")
    md_latest.write_text(md_text, encoding="utf-8")
    csv_latest.write_text(csv_stamp.read_text(encoding="utf-8"), encoding="utf-8")
    _prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="agent_performance",
        suffix=".json",
        retain=int(args.retain_stamped),
    )
    _prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="agent_performance",
        suffix=".md",
        retain=int(args.retain_stamped),
    )
    _prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="agent_performance_rows",
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
