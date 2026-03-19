from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from scripts.reporting.common import (
        prune_stamped_outputs,
        read_json,
        read_jsonl,
        resolve_default_artifact_dir,
        safe_float,
        safe_int,
        validate_canonical_out_dir,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import REPORT_FAMILY_AGENT_PERFORMANCE
except ModuleNotFoundError:
    import sys

    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from scripts.reporting.common import (
        prune_stamped_outputs,
        read_json,
        read_jsonl,
        resolve_default_artifact_dir,
        safe_float,
        safe_int,
        validate_canonical_out_dir,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import REPORT_FAMILY_AGENT_PERFORMANCE


@dataclass(frozen=True)
class ReportPaths:
    artifact_dir: Path
    run_log_path: Path
    out_dir: Path


def _episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        episode = safe_int(row.get("episode_index"), -1)
        if episode < 1:
            continue
        out.append(row)
    out.sort(key=lambda r: safe_int(r.get("episode_index")))
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

    episodes = [safe_int(r.get("episode_index")) for r in rows]
    monotonic = all(episodes[i] > episodes[i - 1] for i in range(1, len(episodes)))
    checks.append({"name": "episode_index_strictly_increasing", "ok": monotonic, "detail": f"first={episodes[0]} last={episodes[-1]}"})

    bad_pct = [
        safe_float(r.get("interventions_pct"))
        for r in rows
        if safe_float(r.get("interventions_pct")) < 0.0 or safe_float(r.get("interventions_pct")) > 100.0
    ]
    checks.append({"name": "interventions_pct_in_0_100", "ok": len(bad_pct) == 0, "detail": f"invalid_rows={len(bad_pct)}"})

    neg_decisions = [safe_int(r.get("decisions_delta")) for r in rows if safe_int(r.get("decisions_delta")) < 0]
    neg_interventions = [safe_int(r.get("interventions_delta")) for r in rows if safe_int(r.get("interventions_delta")) < 0]
    checks.append({"name": "decision_deltas_non_negative", "ok": len(neg_decisions) == 0, "detail": f"invalid_rows={len(neg_decisions)}"})
    checks.append({"name": "intervention_deltas_non_negative", "ok": len(neg_interventions) == 0, "detail": f"invalid_rows={len(neg_interventions)}"})
    return checks


def _build_report(paths: ReportPaths) -> dict[str, Any]:
    metadata = read_json(paths.artifact_dir / "metadata.json")
    rows = _episode_rows(read_jsonl(paths.run_log_path))
    scores = [safe_int(r.get("score")) for r in rows]
    interventions_pct = [safe_float(r.get("interventions_pct")) for r in rows]
    decisions_delta = [safe_int(r.get("decisions_delta")) for r in rows]
    interventions_delta = [safe_int(r.get("interventions_delta")) for r in rows]
    risk_total_series = [safe_int(r.get("risk_total")) for r in rows]
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
                f"{safe_int(row.get('episode_index'))},{safe_int(row.get('score'))},{str(row.get('death_reason',''))},"
                f"{str(row.get('mode',''))},{safe_int(row.get('train_total_steps'))},{safe_float(row.get('interventions_pct')):.6f},"
                f"{safe_int(row.get('interventions_delta'))},{safe_int(row.get('decisions_delta'))},"
                f"{safe_int(row.get('risk_total'))},{safe_int(row.get('stuck_episode_delta'))},"
                f"{safe_int(row.get('loop_escape_activations_total'))},{safe_float(row.get('generated_at_unix_s')):.6f}\n"
            )
        )
    path.write_text("".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build agent-performance report from run session telemetry.")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--run-log", type=str, default="artifacts/live_eval/run_session_log.jsonl")
    parser.add_argument("--out-dir", type=str, default="artifacts/agent_performance")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--retain-stamped", type=int, default=5)
    args = parser.parse_args()
    try:
        validate_retain_stamped(int(args.retain_stamped))
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc

    root = Path(__file__).resolve().parents[2]
    artifact_dir = (
        (root / args.artifact_dir).resolve()
        if str(args.artifact_dir or "").strip()
        else resolve_default_artifact_dir(root)
    )
    out_dir = (root / args.out_dir).resolve()
    try:
        out_dir = validate_canonical_out_dir(root, REPORT_FAMILY_AGENT_PERFORMANCE, out_dir)
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc
    paths = ReportPaths(
        artifact_dir=artifact_dir,
        run_log_path=(root / args.run_log).resolve(),
        out_dir=out_dir,
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
    prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="agent_performance",
        suffix=".json",
        retain=int(args.retain_stamped),
    )
    prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="agent_performance",
        suffix=".md",
        retain=int(args.retain_stamped),
    )
    prune_stamped_outputs(
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
