from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract worst paired controller-vs-PPO seeds from a suite and optionally enforce focused gate thresholds."
    )
    parser.add_argument("--suite", type=str, default="artifacts/live_eval/suites/latest_suite.json")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", type=str, default="artifacts/live_eval/worst10_latest.json")
    parser.add_argument("--max-worse-count", type=int, default=8)
    parser.add_argument("--min-mean-delta", type=float, default=-25.0)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args(argv)


def _load_suite(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("suite payload must be a JSON object")
    return payload


def _rows_by_seed(rows: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            seed = int(row.get("seed"))
            score = int(row.get("score"))
        except Exception:
            continue
        out[seed] = score
    return out


def compute_worst_seed_report(suite: dict, top_n: int) -> dict:
    ppo_rows = list(((suite.get("ppo_only") or {}).get("rows") or []))
    ctrl_rows = list(((suite.get("controller_on") or {}).get("rows") or []))
    ppo = _rows_by_seed(ppo_rows)
    ctrl = _rows_by_seed(ctrl_rows)
    paired = sorted(set(ppo.keys()) & set(ctrl.keys()))
    deltas = [
        {
            "seed": int(seed),
            "ppo_score": int(ppo[seed]),
            "controller_score": int(ctrl[seed]),
            "delta_controller_minus_ppo": int(ctrl[seed] - ppo[seed]),
        }
        for seed in paired
    ]
    deltas.sort(key=lambda row: int(row["delta_controller_minus_ppo"]))
    worst = deltas[: max(1, int(top_n))]
    worst_deltas = [int(row["delta_controller_minus_ppo"]) for row in worst]
    mean_delta = float(sum(worst_deltas) / float(len(worst_deltas))) if worst_deltas else 0.0
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_generated_at_utc": str(suite.get("generated_at_utc", "")),
        "top_n": int(top_n),
        "paired_seed_count": int(len(deltas)),
        "worst_seed_count": int(len(worst)),
        "worst_mean_delta_controller_minus_ppo": float(mean_delta),
        "worst_worse_count": int(sum(1 for d in worst_deltas if d < 0)),
        "worst_improved_count": int(sum(1 for d in worst_deltas if d > 0)),
        "worst_equal_count": int(sum(1 for d in worst_deltas if d == 0)),
        "worst_rows": worst,
    }


def _enforce(report: dict, *, max_worse_count: int, min_mean_delta: float) -> None:
    worse_count = int(report.get("worst_worse_count", 0))
    mean_delta = float(report.get("worst_mean_delta_controller_minus_ppo", 0.0))
    if worse_count > int(max_worse_count):
        raise RuntimeError(f"Worst-seed gate failed: worse_count={worse_count} > max_worse_count={int(max_worse_count)}")
    if mean_delta < float(min_mean_delta):
        raise RuntimeError(
            "Worst-seed gate failed: "
            f"mean_delta={mean_delta:.2f} < min_mean_delta={float(min_mean_delta):.2f}"
        )


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite)
    if not suite_path.exists():
        raise SystemExit(f"suite not found: {suite_path}")
    suite = _load_suite(suite_path)
    report = compute_worst_seed_report(suite, top_n=max(1, int(args.top_n)))
    if bool(args.enforce):
        _enforce(
            report,
            max_worse_count=int(args.max_worse_count),
            min_mean_delta=float(args.min_mean_delta),
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
