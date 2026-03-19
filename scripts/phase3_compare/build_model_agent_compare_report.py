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
        safe_float,
        safe_int,
        validate_canonical_out_dir,
        validate_non_empty,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import REPORT_FAMILY_PHASE3_COMPARE
except ModuleNotFoundError:
    import sys

    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from scripts.reporting.common import (
        prune_stamped_outputs,
        read_json,
        read_jsonl,
        safe_float,
        safe_int,
        validate_canonical_out_dir,
        validate_non_empty,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import REPORT_FAMILY_PHASE3_COMPARE

@dataclass(frozen=True)
class ComparePaths:
    ppo_root: Path
    out_dir: Path


def _artifact_size_bytes(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _eval_trace(exp_dir: Path, run_id: str) -> list[dict[str, Any]]:
    if run_id:
        run_log = exp_dir / "run_logs" / f"eval_trace_{run_id}.jsonl"
        rows = read_jsonl(run_log)
        if rows:
            return rows
    return read_jsonl(exp_dir / "eval_logs" / "evaluations_trace.jsonl")


def _death_pct(deaths: dict[str, Any], key: str) -> float:
    total = float(sum(safe_int(v) for v in deaths.values()))
    if total <= 0.0:
        return 0.0
    return 100.0 * float(safe_int(deaths.get(key))) / total


def _collect_experiment(paths: ComparePaths, exp_name: str) -> dict[str, Any]:
    exp_dir = (paths.ppo_root / exp_name).resolve()
    meta = read_json(exp_dir / "metadata.json")
    run_id = str(meta.get("latest_run_id", "") or "")
    eval_rows = _eval_trace(exp_dir, run_id)
    eval_rows_sorted = sorted(eval_rows, key=lambda r: safe_int(r.get("step")))

    train_sum = dict(meta.get("training_episode_summary", {})) if isinstance(meta.get("training_episode_summary"), dict) else {}
    score_sum = dict(train_sum.get("score_summary", {})) if isinstance(train_sum.get("score_summary"), dict) else {}
    deaths = dict(train_sum.get("deaths", {})) if isinstance(train_sum.get("deaths"), dict) else {}
    steps_sum = dict(train_sum.get("episode_steps_summary", {})) if isinstance(train_sum.get("episode_steps_summary"), dict) else {}
    latest_eval = dict(meta.get("latest_eval_trace", {})) if isinstance(meta.get("latest_eval_trace"), dict) else {}
    cfg = dict(meta.get("config", {})) if isinstance(meta.get("config"), dict) else {}

    eval_curve = []
    for row in eval_rows_sorted:
        eval_curve.append(
            {
                "step": safe_int(row.get("step")),
                "mean_reward": safe_float(row.get("mean_reward")),
                "mean_score": safe_float(row.get("mean_score")),
            }
        )

    return {
        "experiment": exp_name,
        "path": str(exp_dir),
        "exists": exp_dir.exists(),
        "run_id": run_id,
        "model": {
            "requested_total_timesteps": safe_int(meta.get("requested_total_timesteps")),
            "actual_total_timesteps": safe_int(meta.get("actual_total_timesteps")),
            "best_eval_reward": safe_float(meta.get("best_eval_score")),
            "last_eval_reward": safe_float(meta.get("last_eval_score")),
            "latest_eval_mean_score": safe_float(latest_eval.get("mean_score")),
            "best_eval_step": safe_int(meta.get("best_eval_step")),
            "eval_runs_completed": safe_int(meta.get("eval_runs_completed")),
            "rollout": {
                "env_count": safe_int(cfg.get("env_count")),
                "n_steps": safe_int(cfg.get("n_steps")),
                "batch_size": safe_int(cfg.get("batch_size")),
                "n_epochs": safe_int(cfg.get("n_epochs")),
                "gamma": safe_float(cfg.get("gamma")),
            },
        },
        "agent": {
            "episodes_total": safe_int(train_sum.get("episodes_total")),
            "score_mean": safe_float(score_sum.get("mean")),
            "score_p90": safe_float(score_sum.get("p90")),
            "score_best": safe_int(score_sum.get("best")),
            "score_last": safe_int(score_sum.get("last")),
            "steps_mean": safe_float(steps_sum.get("mean")),
            "deaths": deaths,
            "deaths_pct": {
                "body": _death_pct(deaths, "body"),
                "wall": _death_pct(deaths, "wall"),
                "starvation": _death_pct(deaths, "starvation"),
            },
        },
        "artifacts": {
            "has_last_model": (exp_dir / "last_model.zip").exists(),
            "has_vecnormalize": (exp_dir / "vecnormalize.pkl").exists(),
            "has_arbiter": (exp_dir / "arbiter_model.json").exists(),
            "has_tactic_memory": (exp_dir / "tactic_memory.json").exists(),
            "last_model_bytes": _artifact_size_bytes(exp_dir / "last_model.zip"),
            "vecnormalize_bytes": _artifact_size_bytes(exp_dir / "vecnormalize.pkl"),
            "arbiter_bytes": _artifact_size_bytes(exp_dir / "arbiter_model.json"),
            "tactic_memory_bytes": _artifact_size_bytes(exp_dir / "tactic_memory.json"),
        },
        "eval_curve": eval_curve,
    }


def _delta(left: float, right: float) -> tuple[float, float]:
    diff = right - left
    pct = 0.0
    if abs(left) > 1e-12:
        pct = (diff / left) * 100.0
    return diff, pct


def _metric_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    def g(side: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
        cur: Any = side
        for key in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key)
        return safe_float(cur, default)

    defs: list[tuple[str, tuple[str, ...], bool]] = [
        ("best_eval_reward", ("model", "best_eval_reward"), True),
        ("last_eval_reward", ("model", "last_eval_reward"), True),
        ("latest_eval_mean_score", ("model", "latest_eval_mean_score"), True),
        ("train_score_mean", ("agent", "score_mean"), True),
        ("train_score_p90", ("agent", "score_p90"), True),
        ("train_steps_mean", ("agent", "steps_mean"), True),
        ("body_death_pct", ("agent", "deaths_pct", "body"), False),
        ("wall_death_pct", ("agent", "deaths_pct", "wall"), False),
        ("starvation_death_pct", ("agent", "deaths_pct", "starvation"), False),
        ("timesteps_actual", ("model", "actual_total_timesteps"), True),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, higher_is_better in defs:
        left_value = g(left, path)
        right_value = g(right, path)
        d, p = _delta(left_value, right_value)
        rows.append(
            {
                "metric": name,
                "left_value": left_value,
                "right_value": right_value,
                "delta_right_minus_left": d,
                "delta_pct_vs_left": p,
                "higher_is_better": higher_is_better,
            }
        )
    return rows


def _compat_checks(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    l_rollout = dict(left.get("model", {}).get("rollout", {}))
    r_rollout = dict(right.get("model", {}).get("rollout", {}))
    same_rollout = (
        safe_int(l_rollout.get("env_count")) == safe_int(r_rollout.get("env_count"))
        and safe_int(l_rollout.get("n_steps")) == safe_int(r_rollout.get("n_steps"))
        and safe_int(l_rollout.get("batch_size")) == safe_int(r_rollout.get("batch_size"))
        and safe_int(l_rollout.get("n_epochs")) == safe_int(r_rollout.get("n_epochs"))
    )
    checks.append(
        {
            "name": "rollout_contract_match",
            "ok": same_rollout,
            "detail": f"left={l_rollout} right={r_rollout}",
        }
    )

    for side_name, side in (("left", left), ("right", right)):
        art = dict(side.get("artifacts", {}))
        checks.append(
            {
                "name": f"{side_name}_required_artifacts_present",
                "ok": bool(art.get("has_last_model")) and bool(art.get("has_vecnormalize")),
                "detail": f"has_last_model={art.get('has_last_model')} has_vecnormalize={art.get('has_vecnormalize')}",
            }
        )

    l_curve = list(left.get("eval_curve", []))
    r_curve = list(right.get("eval_curve", []))
    checks.append(
        {
            "name": "eval_curve_present_both",
            "ok": bool(l_curve) and bool(r_curve),
            "detail": f"left_points={len(l_curve)} right_points={len(r_curve)}",
        }
    )
    return checks


def _build_report(paths: ComparePaths, left_exp: str, right_exp: str) -> dict[str, Any]:
    left = _collect_experiment(paths, left_exp)
    right = _collect_experiment(paths, right_exp)
    metric_rows = _metric_rows(left, right)
    checks = _compat_checks(left, right)
    ok_count = sum(1 for c in checks if bool(c.get("ok")))
    fail_count = sum(1 for c in checks if not bool(c.get("ok")))

    win_right = 0
    win_left = 0
    ties = 0
    for row in metric_rows:
        d = safe_float(row.get("delta_right_minus_left"))
        hib = bool(row.get("higher_is_better"))
        if abs(d) < 1e-12:
            ties += 1
        elif (d > 0 and hib) or (d < 0 and not hib):
            win_right += 1
        else:
            win_left += 1

    verdict = "TIE"
    if win_right > win_left:
        verdict = "RIGHT_ADVANTAGE"
    elif win_left > win_right:
        verdict = "LEFT_ADVANTAGE"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "compare": {"left_experiment": left_exp, "right_experiment": right_exp, "verdict": verdict},
        "left": left,
        "right": right,
        "metric_rows": metric_rows,
        "checks": checks,
        "summary": {
            "wins_left": win_left,
            "wins_right": win_right,
            "ties": ties,
            "checks_ok": ok_count,
            "checks_fail": fail_count,
        },
    }


def _to_markdown(report: dict[str, Any]) -> str:
    compare = dict(report.get("compare", {}))
    left = dict(report.get("left", {}))
    right = dict(report.get("right", {}))
    summary = dict(report.get("summary", {}))
    checks = list(report.get("checks", []))
    rows = list(report.get("metric_rows", []))
    lines: list[str] = []
    lines.append("# Model + Agent Compare Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {report.get('generated_at_utc')}")
    lines.append(f"- Left: `{compare.get('left_experiment')}`")
    lines.append(f"- Right: `{compare.get('right_experiment')}`")
    lines.append(f"- Verdict: **{compare.get('verdict')}**")
    lines.append("")
    lines.append("## Quick Summary")
    lines.append(
        f"- Wins left / right / tie: {summary.get('wins_left', 0)} / {summary.get('wins_right', 0)} / {summary.get('ties', 0)}"
    )
    lines.append(f"- Checks ok/fail: {summary.get('checks_ok', 0)} / {summary.get('checks_fail', 0)}")
    lines.append(f"- Left run id: `{left.get('run_id', '')}`")
    lines.append(f"- Right run id: `{right.get('run_id', '')}`")
    lines.append("")
    lines.append("## Metrics (right - left)")
    lines.append("| metric | left | right | delta | delta % |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {m} | {l:.6f} | {r:.6f} | {d:.6f} | {p:.3f}% |".format(
                m=row.get("metric", ""),
                l=safe_float(row.get("left_value")),
                r=safe_float(row.get("right_value")),
                d=safe_float(row.get("delta_right_minus_left")),
                p=safe_float(row.get("delta_pct_vs_left")),
            )
        )
    lines.append("")
    lines.append("## Checks")
    for chk in checks:
        icon = "OK" if bool(chk.get("ok")) else "FAIL"
        lines.append(f"- [{icon}] {chk.get('name')}: {chk.get('detail')}")
    lines.append("")
    lines.append("### Copy/Paste Summary")
    lines.append("```text")
    lines.append(
        "compare left={left} right={right} verdict={verdict}".format(
            left=compare.get("left_experiment"),
            right=compare.get("right_experiment"),
            verdict=compare.get("verdict"),
        )
    )
    lines.append(
        "wins left={wl} right={wr} tie={ties} checks_ok={ok} checks_fail={fail}".format(
            wl=summary.get("wins_left", 0),
            wr=summary.get("wins_right", 0),
            ties=summary.get("ties", 0),
            ok=summary.get("checks_ok", 0),
            fail=summary.get("checks_fail", 0),
        )
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "metric,left_value,right_value,delta_right_minus_left,delta_pct_vs_left,higher_is_better\n"
    out = [header]
    for row in rows:
        out.append(
            "{m},{l:.10f},{r:.10f},{d:.10f},{p:.10f},{hib}\n".format(
                m=str(row.get("metric", "")),
                l=safe_float(row.get("left_value")),
                r=safe_float(row.get("right_value")),
                d=safe_float(row.get("delta_right_minus_left")),
                p=safe_float(row.get("delta_pct_vs_left")),
                hib="1" if bool(row.get("higher_is_better")) else "0",
            )
        )
    path.write_text("".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairwise model+agent compare report from experiment folders.")
    parser.add_argument("--left-exp", type=str, required=True)
    parser.add_argument("--right-exp", type=str, required=True)
    parser.add_argument("--ppo-root", type=str, default="state/ppo")
    parser.add_argument("--out-dir", type=str, default="artifacts/phase3_compare")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--retain-stamped", type=int, default=5)
    args = parser.parse_args()
    try:
        validate_retain_stamped(int(args.retain_stamped))
        validate_non_empty("--left-exp", str(args.left_exp))
        validate_non_empty("--right-exp", str(args.right_exp))
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc

    root = Path(__file__).resolve().parents[2]
    out_dir = (root / args.out_dir).resolve()
    try:
        out_dir = validate_canonical_out_dir(root, REPORT_FAMILY_PHASE3_COMPARE, out_dir)
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc
    paths = ComparePaths(
        ppo_root=(root / args.ppo_root).resolve(),
        out_dir=out_dir,
    )
    report = _build_report(paths, left_exp=str(args.left_exp).strip(), right_exp=str(args.right_exp).strip())
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = str(args.tag or "latest").strip() or "latest"
    slug = f"{report.get('compare', {}).get('left_experiment', 'left')}_vs_{report.get('compare', {}).get('right_experiment', 'right')}"
    slug = slug.replace(" ", "_")
    json_stamp = paths.out_dir / f"model_agent_compare_{slug}_{ts}.json"
    md_stamp = paths.out_dir / f"model_agent_compare_{slug}_{ts}.md"
    csv_stamp = paths.out_dir / f"model_agent_compare_rows_{slug}_{ts}.csv"
    json_latest = paths.out_dir / f"model_agent_compare_{tag}.json"
    md_latest = paths.out_dir / f"model_agent_compare_{tag}.md"
    csv_latest = paths.out_dir / f"model_agent_compare_rows_{tag}.csv"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    md_text = _to_markdown(report)
    _write_rows_csv(csv_stamp, list(report.get("metric_rows", [])))

    json_stamp.write_text(json_text, encoding="utf-8")
    md_stamp.write_text(md_text, encoding="utf-8")
    json_latest.write_text(json_text, encoding="utf-8")
    md_latest.write_text(md_text, encoding="utf-8")
    csv_latest.write_text(csv_stamp.read_text(encoding="utf-8"), encoding="utf-8")
    prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="model_agent_compare",
        suffix=".json",
        retain=int(args.retain_stamped),
    )
    prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="model_agent_compare",
        suffix=".md",
        retain=int(args.retain_stamped),
    )
    prune_stamped_outputs(
        paths.out_dir,
        stem_prefix="model_agent_compare_rows",
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
