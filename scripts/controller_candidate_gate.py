from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate controller candidates against a fixed baseline. "
            "Reject if full-suite delta regresses, worst-10 regresses, or intervention rate is too high."
        )
    )
    parser.add_argument("--baseline-full", type=str, required=True)
    parser.add_argument("--candidate-full", type=str, required=True)
    parser.add_argument("--baseline-worst", type=str, default="")
    parser.add_argument("--candidate-worst", type=str, default="")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--baseline-trace-dir", type=str, default="")
    parser.add_argument("--candidate-trace-dir", type=str, default="")
    parser.add_argument("--baseline-intervention-rate", type=float, default=None)
    parser.add_argument("--candidate-intervention-rate", type=float, default=None)
    parser.add_argument("--min-full-delta-gain", type=float, default=0.0)
    parser.add_argument("--min-worst-delta-gain", type=float, default=0.0)
    parser.add_argument("--max-intervention-rate", type=float, default=8.0)
    parser.add_argument("--max-intervention-rate-increase", type=float, default=1.5)
    parser.add_argument("--require-worst-improvement", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--out", type=str, default="artifacts/live_eval/controller_candidate_gate.json")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at: {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _suite_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ppo_rows = list(((payload.get("ppo_only") or {}).get("rows") or []))
    ctrl_rows = list(((payload.get("controller_on") or {}).get("rows") or []))
    return ppo_rows, ctrl_rows


def _paired_deltas_from_rows(ppo_rows: list[dict[str, Any]], ctrl_rows: list[dict[str, Any]]) -> list[float]:
    ppo: dict[int, float] = {}
    ctrl: dict[int, float] = {}
    for row in ppo_rows:
        if not isinstance(row, dict):
            continue
        try:
            ppo[int(row.get("seed"))] = float(row.get("score"))
        except Exception:
            continue
    for row in ctrl_rows:
        if not isinstance(row, dict):
            continue
        try:
            ctrl[int(row.get("seed"))] = float(row.get("score"))
        except Exception:
            continue
    paired = sorted(set(ppo.keys()) & set(ctrl.keys()))
    return [float(ctrl[s] - ppo[s]) for s in paired]


def extract_mean_delta(payload: dict[str, Any]) -> float:
    if "mean_delta" in payload:
        return _safe_float(payload.get("mean_delta"), 0.0)
    comparison = payload.get("comparison") or {}
    if isinstance(comparison, dict) and "mean_delta_controller_minus_ppo" in comparison:
        return _safe_float(comparison.get("mean_delta_controller_minus_ppo"), 0.0)
    ppo_rows, ctrl_rows = _suite_rows(payload)
    deltas = _paired_deltas_from_rows(ppo_rows, ctrl_rows)
    if deltas:
        return float(sum(deltas) / float(len(deltas)))
    raise RuntimeError("Could not extract mean delta from payload.")


def extract_worst_mean_delta(payload: dict[str, Any], *, top_n: int) -> float:
    if "worst_mean_delta_controller_minus_ppo" in payload:
        return _safe_float(payload.get("worst_mean_delta_controller_minus_ppo"), 0.0)
    ppo_rows, ctrl_rows = _suite_rows(payload)
    deltas = _paired_deltas_from_rows(ppo_rows, ctrl_rows)
    if not deltas:
        raise RuntimeError("Could not extract worst mean delta from payload.")
    deltas_sorted = sorted(float(v) for v in deltas)
    worst = deltas_sorted[: max(1, int(top_n))]
    return float(sum(worst) / float(len(worst)))


def trace_intervention_rate_percent(trace_dir: Path) -> float:
    if not trace_dir.exists():
        raise RuntimeError(f"Trace dir not found: {trace_dir}")
    rows_total = 0
    overrides_total = 0
    for path in sorted(trace_dir.rglob("seed_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rows_total += 1
            if bool(row.get("override_used", False)):
                overrides_total += 1
    if rows_total <= 0:
        raise RuntimeError(f"No trace decision rows found under: {trace_dir}")
    return float(100.0 * float(overrides_total) / float(rows_total))


def _enforce(report: dict[str, Any]) -> None:
    checks = list(report.get("checks") or [])
    failed = [c for c in checks if isinstance(c, dict) and not bool(c.get("passed", False))]
    if failed:
        labels = ", ".join(str(c.get("name", "unknown")) for c in failed)
        raise RuntimeError(f"Controller candidate gate failed: {labels}")


def _extract_intervention_rate_from_payload(payload: dict[str, Any]) -> float | None:
    comparison = payload.get("comparison")
    if isinstance(comparison, dict) and "mean_interventions_pct" in comparison:
        return _safe_float(comparison.get("mean_interventions_pct"), 0.0)
    return None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline_full = _load_json(Path(args.baseline_full))
    candidate_full = _load_json(Path(args.candidate_full))
    baseline_full_delta = extract_mean_delta(baseline_full)
    candidate_full_delta = extract_mean_delta(candidate_full)

    baseline_worst_delta = None
    candidate_worst_delta = None
    if str(args.baseline_worst).strip() and str(args.candidate_worst).strip():
        baseline_worst_delta = extract_worst_mean_delta(_load_json(Path(args.baseline_worst)), top_n=int(args.top_n))
        candidate_worst_delta = extract_worst_mean_delta(_load_json(Path(args.candidate_worst)), top_n=int(args.top_n))
    elif bool(args.require_worst_improvement):
        baseline_worst_delta = extract_worst_mean_delta(baseline_full, top_n=int(args.top_n))
        candidate_worst_delta = extract_worst_mean_delta(candidate_full, top_n=int(args.top_n))

    baseline_intv = args.baseline_intervention_rate
    candidate_intv = args.candidate_intervention_rate
    if baseline_intv is None and str(args.baseline_trace_dir).strip():
        baseline_intv = trace_intervention_rate_percent(Path(args.baseline_trace_dir))
    if candidate_intv is None and str(args.candidate_trace_dir).strip():
        candidate_intv = trace_intervention_rate_percent(Path(args.candidate_trace_dir))
    if baseline_intv is None:
        baseline_intv = _extract_intervention_rate_from_payload(baseline_full)
    if candidate_intv is None:
        candidate_intv = _extract_intervention_rate_from_payload(candidate_full)

    checks: list[dict[str, Any]] = []
    full_delta_gain = float(candidate_full_delta - baseline_full_delta)
    checks.append(
        {
            "name": "full_delta_gain",
            "passed": bool(full_delta_gain >= float(args.min_full_delta_gain)),
            "value": float(full_delta_gain),
            "threshold": float(args.min_full_delta_gain),
        }
    )

    if args.require_worst_improvement:
        if baseline_worst_delta is None or candidate_worst_delta is None:
            checks.append(
                {
                    "name": "worst_delta_gain",
                    "passed": False,
                    "value": None,
                    "threshold": float(args.min_worst_delta_gain),
                    "reason": "missing worst payloads",
                }
            )
        else:
            worst_delta_gain = float(candidate_worst_delta - baseline_worst_delta)
            checks.append(
                {
                    "name": "worst_delta_gain",
                    "passed": bool(worst_delta_gain >= float(args.min_worst_delta_gain)),
                    "value": float(worst_delta_gain),
                    "threshold": float(args.min_worst_delta_gain),
                }
            )

    if candidate_intv is not None:
        checks.append(
            {
                "name": "candidate_intervention_rate_cap",
                "passed": bool(float(candidate_intv) <= float(args.max_intervention_rate)),
                "value": float(candidate_intv),
                "threshold": float(args.max_intervention_rate),
            }
        )
    elif args.enforce:
        checks.append(
            {
                "name": "candidate_intervention_rate_cap",
                "passed": False,
                "value": None,
                "threshold": float(args.max_intervention_rate),
                "reason": "missing candidate intervention rate",
            }
        )

    if baseline_intv is not None and candidate_intv is not None:
        intv_increase = float(candidate_intv - baseline_intv)
        checks.append(
            {
                "name": "intervention_rate_increase_cap",
                "passed": bool(intv_increase <= float(args.max_intervention_rate_increase)),
                "value": float(intv_increase),
                "threshold": float(args.max_intervention_rate_increase),
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_full": str(args.baseline_full),
            "candidate_full": str(args.candidate_full),
            "baseline_worst": str(args.baseline_worst),
            "candidate_worst": str(args.candidate_worst),
            "baseline_trace_dir": str(args.baseline_trace_dir),
            "candidate_trace_dir": str(args.candidate_trace_dir),
        },
        "metrics": {
            "baseline_full_mean_delta": float(baseline_full_delta),
            "candidate_full_mean_delta": float(candidate_full_delta),
            "baseline_worst_mean_delta": None if baseline_worst_delta is None else float(baseline_worst_delta),
            "candidate_worst_mean_delta": None if candidate_worst_delta is None else float(candidate_worst_delta),
            "baseline_intervention_rate_pct": None if baseline_intv is None else float(baseline_intv),
            "candidate_intervention_rate_pct": None if candidate_intv is None else float(candidate_intv),
        },
        "checks": checks,
        "accepted": bool(all(bool(c.get("passed", False)) for c in checks)),
    }
    return report


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if bool(args.enforce):
        _enforce(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
