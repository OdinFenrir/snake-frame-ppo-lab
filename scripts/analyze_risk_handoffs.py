from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
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


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return str(default)
    return str(value)


def _latest_trace_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


@dataclass(frozen=True)
class RiskOnset:
    seed: int
    start_idx: int
    start_step: int
    run_len: int
    onset_mode: str
    onset_reason: str
    next_reason: str
    next_mode: str
    steps_to_death: int
    death_within_horizon: bool
    ate_food_within_horizon: bool
    no_progress_start: int
    no_progress_end: int
    no_progress_max: int
    horizon_len: int
    horizon_corridor_ratio: float
    horizon_food_pressure_mean: float
    horizon_food_pressure_max: float
    horizon_score_delta: int
    sequence_signature: str


def _terminal_index(rows: list[dict[str, Any]]) -> int:
    for i, row in enumerate(rows):
        if bool(row.get("game_over", False)):
            return int(i)
    return int(max(0, len(rows) - 1))


def _split_by_seed(all_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        seed = _safe_int(row.get("seed"), -1)
        out.setdefault(seed, []).append(row)
    for seed in list(out.keys()):
        out[seed] = sorted(out[seed], key=lambda r: _safe_int(r.get("step"), 0))
    return out


def _collect_risk_onsets(rows: list[dict[str, Any]], *, horizon: int, signature_len: int) -> list[RiskOnset]:
    if not rows:
        return []
    terminal_idx = _terminal_index(rows)
    out: list[RiskOnset] = []
    i = 0
    while i < len(rows):
        reason = _safe_str(rows[i].get("switch_reason"))
        if reason != "risk":
            i += 1
            continue
        prev_reason = _safe_str(rows[i - 1].get("switch_reason")) if i > 0 else ""
        if prev_reason == "risk":
            i += 1
            continue
        start = i
        j = i
        while j < len(rows) and _safe_str(rows[j].get("switch_reason")) == "risk":
            j += 1
        run_len = int(j - start)
        next_reason = _safe_str(rows[j].get("switch_reason")) if j < len(rows) else "terminal"
        next_mode = _safe_str(rows[j].get("mode")) if j < len(rows) else "terminal"
        horizon_end = min(len(rows) - 1, int(start + max(1, horizon)))
        window_rows = rows[start : horizon_end + 1]
        death_within_horizon = any(bool(rows[k].get("game_over", False)) for k in range(start, horizon_end + 1))
        ate_food_within_horizon = any(bool(rows[k].get("ate_food", False)) for k in range(start, horizon_end + 1))
        no_progress_vals = [_safe_int(rows[k].get("no_progress_steps"), 0) for k in range(start, min(j, len(rows)))]
        no_progress_start = no_progress_vals[0] if no_progress_vals else 0
        no_progress_end = no_progress_vals[-1] if no_progress_vals else 0
        no_progress_max = max(no_progress_vals) if no_progress_vals else 0
        safe_opts = [_safe_int(r.get("safe_option_count"), 3) for r in window_rows]
        corridor_ratio = (
            float(sum(1 for v in safe_opts if int(v) <= 1)) / float(max(1, len(safe_opts)))
            if safe_opts
            else 0.0
        )
        food_pressures = [float(r.get("food_pressure", 0.0)) for r in window_rows]
        food_pressure_mean = float(sum(food_pressures) / float(max(1, len(food_pressures)))) if food_pressures else 0.0
        food_pressure_max = float(max(food_pressures)) if food_pressures else 0.0
        score_start = _safe_int(rows[start].get("score_before"), 0)
        score_peak = max(_safe_int(r.get("score_after"), _safe_int(r.get("score_before"), 0)) for r in window_rows) if window_rows else score_start
        score_delta = int(score_peak - score_start)
        sig_parts: list[str] = []
        for k in range(start, min(len(rows), int(start + max(1, signature_len)))):
            sig_parts.append(_safe_str(rows[k].get("switch_reason")))
        signature = "->".join(sig_parts)
        out.append(
            RiskOnset(
                seed=_safe_int(rows[start].get("seed"), -1),
                start_idx=int(start),
                start_step=_safe_int(rows[start].get("step"), start),
                run_len=int(run_len),
                onset_mode=_safe_str(rows[start].get("mode")),
                onset_reason=reason,
                next_reason=next_reason,
                next_mode=next_mode,
                steps_to_death=max(0, int(terminal_idx - start)),
                death_within_horizon=bool(death_within_horizon),
                ate_food_within_horizon=bool(ate_food_within_horizon),
                no_progress_start=int(no_progress_start),
                no_progress_end=int(no_progress_end),
                no_progress_max=int(no_progress_max),
                horizon_len=int(len(window_rows)),
                horizon_corridor_ratio=float(corridor_ratio),
                horizon_food_pressure_mean=float(food_pressure_mean),
                horizon_food_pressure_max=float(food_pressure_max),
                horizon_score_delta=int(score_delta),
                sequence_signature=signature,
            )
        )
        i = j
    return out


def build_report(
    *,
    trace_files: list[Path],
    horizon: int,
    signature_len: int,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for path in trace_files:
        all_rows.extend(_read_jsonl(path))
    rows_by_seed = _split_by_seed(all_rows)
    onsets: list[RiskOnset] = []
    for rows in rows_by_seed.values():
        onsets.extend(_collect_risk_onsets(rows, horizon=int(horizon), signature_len=int(signature_len)))
    onsets_sorted = sorted(onsets, key=lambda x: (x.seed, x.start_step))

    next_reason_counts = Counter(x.next_reason for x in onsets_sorted)
    next_mode_counts = Counter(x.next_mode for x in onsets_sorted)
    signature_counts = Counter(x.sequence_signature for x in onsets_sorted)
    per_seed_counts = Counter(int(x.seed) for x in onsets_sorted)

    onset_count = len(onsets_sorted)
    run_len_mean = (sum(int(x.run_len) for x in onsets_sorted) / float(max(1, onset_count)))
    steps_to_death_mean = (sum(int(x.steps_to_death) for x in onsets_sorted) / float(max(1, onset_count)))
    corridor_ratio_mean = (
        sum(float(x.horizon_corridor_ratio) for x in onsets_sorted) / float(max(1, onset_count))
    )
    horizon_score_delta_mean = (
        sum(int(x.horizon_score_delta) for x in onsets_sorted) / float(max(1, onset_count))
    )
    death_within_h = sum(1 for x in onsets_sorted if bool(x.death_within_horizon))
    ate_within_h = sum(1 for x in onsets_sorted if bool(x.ate_food_within_horizon))

    out_onsets: list[dict[str, Any]] = []
    for x in onsets_sorted:
        out_onsets.append(
            {
                "seed": int(x.seed),
                "start_step": int(x.start_step),
                "run_len": int(x.run_len),
                "onset_mode": str(x.onset_mode),
                "next_reason": str(x.next_reason),
                "next_mode": str(x.next_mode),
                "steps_to_death": int(x.steps_to_death),
                "death_within_horizon": bool(x.death_within_horizon),
                "ate_food_within_horizon": bool(x.ate_food_within_horizon),
                "no_progress_start": int(x.no_progress_start),
                "no_progress_end": int(x.no_progress_end),
                "no_progress_max": int(x.no_progress_max),
                "horizon_len": int(x.horizon_len),
                "horizon_corridor_ratio": float(x.horizon_corridor_ratio),
                "horizon_food_pressure_mean": float(x.horizon_food_pressure_mean),
                "horizon_food_pressure_max": float(x.horizon_food_pressure_max),
                "horizon_score_delta": int(x.horizon_score_delta),
                "sequence_signature": str(x.sequence_signature),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "trace_files": int(len(trace_files)),
            "rows_scanned": int(len(all_rows)),
            "seed_count": int(len(rows_by_seed)),
            "risk_onset_count": int(onset_count),
            "horizon_steps": int(horizon),
            "signature_len": int(signature_len),
            "mean_risk_run_len": float(run_len_mean),
            "mean_steps_to_death_from_onset": float(steps_to_death_mean),
            "mean_horizon_corridor_ratio": float(corridor_ratio_mean),
            "mean_horizon_score_delta": float(horizon_score_delta_mean),
            "death_within_horizon_count": int(death_within_h),
            "death_within_horizon_pct": float(100.0 * float(death_within_h) / float(max(1, onset_count))),
            "ate_food_within_horizon_count": int(ate_within_h),
            "ate_food_within_horizon_pct": float(100.0 * float(ate_within_h) / float(max(1, onset_count))),
            "next_reason_counts": {str(k): int(v) for k, v in next_reason_counts.items()},
            "next_mode_counts": {str(k): int(v) for k, v in next_mode_counts.items()},
            "top_signatures": [[str(k), int(v)] for k, v in signature_counts.most_common(10)],
            "per_seed_onset_counts": {str(k): int(v) for k, v in per_seed_counts.items()},
        },
        "risk_onsets": out_onsets,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze post-handoff behavior from risk onsets in focused traces.")
    parser.add_argument("--trace-root", type=str, default="artifacts/live_eval/focused_traces")
    parser.add_argument("--trace-dir", type=str, default="")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--signature-len", type=int, default=6)
    parser.add_argument("--out", type=str, default="artifacts/live_eval/risk_handoff_report.json")
    return parser.parse_args(argv)


def _resolve_trace_files(args: argparse.Namespace) -> list[Path]:
    if str(args.trace_dir).strip():
        target = Path(str(args.trace_dir))
        return sorted(target.glob("seed_*.jsonl"))
    root = Path(str(args.trace_root))
    if bool(args.latest_only):
        latest = _latest_trace_dir(root)
        if latest is None:
            return []
        return sorted(latest.glob("seed_*.jsonl"))
    return sorted(root.glob("**/seed_*.jsonl"))


def main() -> None:
    args = parse_args()
    trace_files = _resolve_trace_files(args)
    if not trace_files:
        raise SystemExit("No trace files found.")
    report = build_report(
        trace_files=trace_files,
        horizon=max(1, int(args.horizon)),
        signature_len=max(1, int(args.signature_len)),
    )
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(json.dumps(report.get("summary") or {}, indent=2))


if __name__ == "__main__":
    main()
