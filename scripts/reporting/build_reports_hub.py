from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _latest_file(dir_path: Path, pattern: str) -> Path | None:
    files = [p for p in dir_path.glob(pattern) if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fmt(path: Path | None) -> str:
    return str(path) if path is not None else "missing"


def _training_input_summary(training_input_dir: Path) -> list[str]:
    lines: list[str] = []
    latest_json = training_input_dir / "training_input_latest.json"
    latest_timeline_json = training_input_dir / "training_input_timeline_latest.json"
    latest_md = training_input_dir / "training_input_latest.md"
    latest_timeline_md = training_input_dir / "training_input_timeline_latest.md"
    latest_vec_csv = training_input_dir / "training_input_checkpoint_vecnorm_latest.csv"
    latest_timeline_csv = training_input_dir / "training_input_timeline_latest.csv"
    latest_dashboard = training_input_dir / "training_input_dashboard_latest.html"

    report = _read_json(latest_json)
    timeline = _read_json(latest_timeline_json)

    run = dict(report.get("run", {})) if isinstance(report.get("run"), dict) else {}
    contract = dict(report.get("input_contract", {})) if isinstance(report.get("input_contract"), dict) else {}
    vec = dict(report.get("vecnormalize_latest", {})) if isinstance(report.get("vecnormalize_latest"), dict) else {}
    checks = list(report.get("checks", [])) if isinstance(report.get("checks"), list) else []

    ok_count = sum(1 for c in checks if isinstance(c, dict) and bool(c.get("ok")))
    fail_count = sum(1 for c in checks if isinstance(c, dict) and not bool(c.get("ok")))
    timeline_summary = dict(timeline.get("summary", {})) if isinstance(timeline.get("summary"), dict) else {}

    lines.append("## Training Input")
    lines.append(f"- JSON: `{_fmt(latest_json if latest_json.exists() else None)}`")
    lines.append(f"- Markdown: `{_fmt(latest_md if latest_md.exists() else None)}`")
    lines.append(f"- Timeline JSON: `{_fmt(latest_timeline_json if latest_timeline_json.exists() else None)}`")
    lines.append(f"- Timeline Markdown: `{_fmt(latest_timeline_md if latest_timeline_md.exists() else None)}`")
    lines.append(f"- Vec CSV: `{_fmt(latest_vec_csv if latest_vec_csv.exists() else None)}`")
    lines.append(f"- Timeline CSV: `{_fmt(latest_timeline_csv if latest_timeline_csv.exists() else None)}`")
    lines.append(f"- Dashboard HTML: `{_fmt(latest_dashboard if latest_dashboard.exists() else None)}`")
    lines.append("")
    lines.append("### Copy/Paste Summary")
    lines.append("```text")
    lines.append(f"run_id={run.get('run_id', '')}")
    lines.append(
        "timesteps requested={req} actual={actual}".format(
            req=run.get("requested_total_timesteps", 0),
            actual=run.get("actual_total_timesteps", 0),
        )
    )
    lines.append(
        "rollout env_count={env} n_steps={n_steps} batch_size={batch} n_epochs={epochs}".format(
            env=contract.get("env_count", 0),
            n_steps=contract.get("n_steps", 0),
            batch=contract.get("batch_size", 0),
            epochs=contract.get("n_epochs", 0),
        )
    )
    lines.append(
        "vecnormalize obs_dim={dim} obs_count={count} mean_abs_avg={maa:.6f} var_avg={vavg:.6f}".format(
            dim=vec.get("obs_dim", 0),
            count=vec.get("obs_count", 0),
            maa=float(vec.get("obs_mean_abs_avg", 0.0) or 0.0),
            vavg=float(vec.get("obs_var_avg", 0.0) or 0.0),
        )
    )
    lines.append(
        "timeline checkpoints={cp} vec_checkpoints={vc} eval_points={ep} best_eval_mean_reward={bemr}".format(
            cp=timeline_summary.get("checkpoint_steps_total", 0),
            vc=timeline_summary.get("vec_checkpoint_count", 0),
            ep=timeline_summary.get("eval_trace_points", 0),
            bemr=timeline_summary.get("best_eval_mean_reward", 0),
        )
    )
    lines.append(f"checks ok={ok_count} fail={fail_count}")
    lines.append("```")
    lines.append("")
    return lines


def _agent_performance_summary(agent_dir: Path) -> list[str]:
    lines: list[str] = []
    latest_json = agent_dir / "agent_performance_latest.json"
    latest_md = agent_dir / "agent_performance_latest.md"
    latest_csv = agent_dir / "agent_performance_rows_latest.csv"
    latest_dashboard = agent_dir / "agent_performance_dashboard_latest.html"

    report = _read_json(latest_json)
    episodes = dict(report.get("episodes", {})) if isinstance(report.get("episodes"), dict) else {}
    control = dict(report.get("agent_control", {})) if isinstance(report.get("agent_control"), dict) else {}
    checks = list(report.get("checks", [])) if isinstance(report.get("checks"), list) else []
    ok_count = sum(1 for c in checks if isinstance(c, dict) and bool(c.get("ok")))
    fail_count = sum(1 for c in checks if isinstance(c, dict) and not bool(c.get("ok")))

    lines.append("## Agent Performance")
    lines.append(f"- JSON: `{_fmt(latest_json if latest_json.exists() else None)}`")
    lines.append(f"- Markdown: `{_fmt(latest_md if latest_md.exists() else None)}`")
    lines.append(f"- Rows CSV: `{_fmt(latest_csv if latest_csv.exists() else None)}`")
    lines.append(f"- Dashboard HTML: `{_fmt(latest_dashboard if latest_dashboard.exists() else None)}`")
    lines.append("")
    lines.append("### Copy/Paste Summary")
    lines.append("```text")
    lines.append(f"run_id={report.get('run_id', '')}")
    lines.append(
        "episodes count={count} score_mean={mean:.3f} score_best={best} score_last={last} trend={trend}".format(
            count=int(episodes.get("count", 0) or 0),
            mean=float(episodes.get("score_mean", 0.0) or 0.0),
            best=int(episodes.get("score_best", 0) or 0),
            last=int(episodes.get("score_last", 0) or 0),
            trend=episodes.get("score_trend", "unknown"),
        )
    )
    lines.append(
        "agent_control interventions_mean_pct={pct:.3f} interventions_total={it} decisions_total={dt} risk_total_last={risk}".format(
            pct=float(control.get("interventions_pct_mean", 0.0) or 0.0),
            it=int(control.get("interventions_delta_total", 0) or 0),
            dt=int(control.get("decisions_delta_total", 0) or 0),
            risk=int(control.get("risk_total_last", 0) or 0),
        )
    )
    lines.append(f"checks ok={ok_count} fail={fail_count}")
    lines.append("```")
    lines.append("")
    return lines


def _generic_category_lines(title: str, dir_path: Path) -> list[str]:
    lines = [f"## {title}"]
    if not dir_path.exists():
        lines.append("- Directory missing")
        lines.append("")
        return lines
    latest_json = _latest_file(dir_path, "*.json")
    latest_md = _latest_file(dir_path, "*.md")
    latest_csv = _latest_file(dir_path, "*.csv")
    latest_jsonl = _latest_file(dir_path, "*.jsonl")
    if latest_json is None and latest_md is None and latest_csv is None and latest_jsonl is None:
        lines.append("- No reports found")
        lines.append("")
        return lines
    lines.append(f"- Latest JSON: `{_fmt(latest_json)}`")
    lines.append(f"- Latest Markdown: `{_fmt(latest_md)}`")
    lines.append(f"- Latest CSV: `{_fmt(latest_csv)}`")
    lines.append(f"- Latest JSONL: `{_fmt(latest_jsonl)}`")
    lines.append("")
    return lines


def build_hub(artifacts_root: Path, out_dir: Path) -> tuple[Path, Path]:
    training_input_dir = artifacts_root / "training_input"
    agent_performance_dir = artifacts_root / "agent_performance"
    live_eval_dir = artifacts_root / "live_eval"
    share_dir = artifacts_root / "share"
    generated_utc = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# Reports Hub")
    lines.append("")
    lines.append(f"- Generated (UTC): {generated_utc}")
    lines.append(f"- Artifacts root: `{artifacts_root}`")
    lines.append("")
    lines.extend(_training_input_summary(training_input_dir))
    lines.extend(_agent_performance_summary(agent_performance_dir))
    lines.extend(_generic_category_lines("Live Eval", live_eval_dir))
    lines.extend(_generic_category_lines("Share", share_dir))

    md = "\n".join(lines).rstrip() + "\n"
    txt = md.replace("`", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "reports_hub_latest.md"
    txt_path = out_dir / "reports_hub_latest.txt"
    md_path.write_text(md, encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    return md_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single reports hub file with latest report pointers and copy/paste summaries.")
    parser.add_argument("--artifacts-root", type=str, default="artifacts")
    parser.add_argument("--out-dir", type=str, default="artifacts/reports")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    artifacts_root = (root / args.artifacts_root).resolve()
    out_dir = (root / args.out_dir).resolve()
    md_path, txt_path = build_hub(artifacts_root, out_dir)
    print(f"Wrote: {md_path}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
