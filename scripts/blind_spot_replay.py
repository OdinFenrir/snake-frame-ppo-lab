from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
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


def annotate_steps_until_death(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    terminal_idx = -1
    for i, row in enumerate(rows):
        if bool(row.get("game_over", False)):
            terminal_idx = int(i)
            break
    if terminal_idx < 0:
        terminal_idx = int(len(rows) - 1)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        enriched = dict(row)
        enriched["steps_until_death"] = int(max(0, terminal_idx - i))
        out.append(enriched)
    return out


def _seed_from_path(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "seed":
        return _safe_int(parts[1], -1)
    return -1


def _latest_trace_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _find_blind_indices(
    rows: list[dict[str, Any]],
    *,
    min_confidence: float,
    max_steps_to_death: int,
    only_no_override: bool,
) -> list[int]:
    idxs: list[int] = []
    for i, row in enumerate(rows):
        conf = _safe_float(row.get("predicted_confidence"), 0.0)
        steps = _safe_int(row.get("steps_until_death"), 0)
        if conf < float(min_confidence):
            continue
        if steps > int(max_steps_to_death):
            continue
        if only_no_override and bool(row.get("override_used", False)):
            continue
        idxs.append(int(i))
    return idxs


def build_blind_spot_report(
    *,
    trace_files: list[Path],
    min_confidence: float,
    max_steps_to_death: int,
    replay_window: int,
    max_spots: int,
    only_no_override: bool,
) -> dict[str, Any]:
    blind_spots: list[dict[str, Any]] = []
    per_seed_counts: dict[str, int] = {}
    scanned_rows = 0
    for trace_path in trace_files:
        rows = annotate_steps_until_death(_read_jsonl(trace_path))
        scanned_rows += int(len(rows))
        if not rows:
            continue
        seed = _seed_from_path(trace_path)
        idxs = _find_blind_indices(
            rows,
            min_confidence=float(min_confidence),
            max_steps_to_death=int(max_steps_to_death),
            only_no_override=bool(only_no_override),
        )
        for idx in idxs:
            row = rows[idx]
            start = max(0, int(idx - max(1, int(replay_window)) + 1))
            window_rows = rows[start : int(idx + 1)]
            spot = {
                "seed": int(seed),
                "trace_file": str(trace_path),
                "index": int(idx),
                "step": _safe_int(row.get("step"), idx),
                "predicted_confidence": _safe_float(row.get("predicted_confidence"), 0.0),
                "steps_until_death": _safe_int(row.get("steps_until_death"), 0),
                "score_before": _safe_int(row.get("score_before"), 0),
                "score_after": _safe_int(row.get("score_after"), 0),
                "mode": str(row.get("mode", "")),
                "switch_reason": str(row.get("switch_reason", "")),
                "override_used": bool(row.get("override_used", False)),
                "window_rows": window_rows,
            }
            blind_spots.append(spot)
            key = str(seed)
            per_seed_counts[key] = int(per_seed_counts.get(key, 0) + 1)

    blind_spots.sort(
        key=lambda s: (
            -_safe_float(s.get("predicted_confidence"), 0.0),
            _safe_int(s.get("steps_until_death"), 0),
        )
    )
    blind_spots = blind_spots[: max(0, int(max_spots))]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "trace_files": int(len(trace_files)),
            "rows_scanned": int(scanned_rows),
            "blind_spot_count": int(len(blind_spots)),
            "min_confidence": float(min_confidence),
            "max_steps_to_death": int(max_steps_to_death),
            "replay_window": int(replay_window),
            "only_no_override": bool(only_no_override),
            "per_seed_blind_spot_counts": per_seed_counts,
        },
        "blind_spots": blind_spots,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract high-confidence near-terminal blind spots from focused traces "
            "and save replay windows for manual inspection."
        )
    )
    parser.add_argument("--trace-root", type=str, default="artifacts/live_eval/focused_traces")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--max-steps-to-death", type=int, default=10)
    parser.add_argument("--replay-window", type=int, default=30)
    parser.add_argument("--max-spots", type=int, default=50)
    parser.add_argument("--only-no-override", action="store_true")
    parser.add_argument("--out", type=str, default="artifacts/live_eval/blind_spot_replay.json")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = Path(args.trace_root)
    if bool(args.latest_only):
        latest = _latest_trace_dir(root)
        if latest is None:
            raise SystemExit(f"No trace directories found under: {root}")
        trace_files = sorted(latest.glob("seed_*.jsonl"))
    else:
        trace_files = sorted(root.glob("**/seed_*.jsonl"))
    report = build_blind_spot_report(
        trace_files=trace_files,
        min_confidence=float(args.min_confidence),
        max_steps_to_death=int(args.max_steps_to_death),
        replay_window=int(args.replay_window),
        max_spots=int(args.max_spots),
        only_no_override=bool(args.only_no_override),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(json.dumps(report.get("summary") or {}, indent=2))


if __name__ == "__main__":
    main()
