from __future__ import annotations

import argparse
from collections import deque
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    return bool(default)


def _seed_trace_path(path_or_dir: Path, seed: int) -> Path:
    if path_or_dir.is_file():
        return path_or_dir
    return path_or_dir / f"seed_{int(seed)}.jsonl"


def _grid_size_from_rows(rows: list[dict[str, Any]], fallback: int = 20) -> int:
    max_coord = -1
    for row in rows:
        snake = row.get("snake_before")
        if not isinstance(snake, list):
            continue
        for cell in snake:
            if not isinstance(cell, list) or len(cell) < 2:
                continue
            max_coord = max(max_coord, _safe_int(cell[0], -1), _safe_int(cell[1], -1))
    if max_coord < 0:
        return int(fallback)
    return int(max_coord + 1)


def _free_components_and_head_area(row: dict[str, Any], board_cells: int) -> tuple[int, int]:
    snake = row.get("snake_after")
    head = row.get("head_after")
    if not isinstance(snake, list) or not snake:
        snake = row.get("snake_before")
        head = row.get("head_before")
    if not isinstance(snake, list) or not snake:
        return 0, 0
    blocked = set()
    for cell in snake:
        if isinstance(cell, list) and len(cell) >= 2:
            blocked.add((_safe_int(cell[0]), _safe_int(cell[1])))
    free = set()
    for y in range(int(board_cells)):
        for x in range(int(board_cells)):
            pos = (int(x), int(y))
            if pos not in blocked:
                free.add(pos)
    if not free:
        return 0, 0

    visited = set()
    components = 0
    for cell in list(free):
        if cell in visited:
            continue
        components += 1
        q: deque[tuple[int, int]] = deque([cell])
        visited.add(cell)
        while q:
            cx, cy = q.popleft()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                nxt = (int(nx), int(ny))
                if 0 <= nx < int(board_cells) and 0 <= ny < int(board_cells) and nxt in free and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

    head_area = 0
    hx = hy = -1
    if isinstance(head, list) and len(head) >= 2:
        hx = _safe_int(head[0], -1)
        hy = _safe_int(head[1], -1)
    if (hx, hy) != (-1, -1):
        q2: deque[tuple[int, int]] = deque()
        seen2 = set()
        for nx, ny in ((hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)):
            nxt = (int(nx), int(ny))
            if 0 <= nx < int(board_cells) and 0 <= ny < int(board_cells) and nxt in free:
                q2.append(nxt)
                seen2.add(nxt)
        while q2:
            cx, cy = q2.popleft()
            head_area += 1
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                nxt = (int(nx), int(ny))
                if 0 <= nx < int(board_cells) and 0 <= ny < int(board_cells) and nxt in free and nxt not in seen2:
                    seen2.add(nxt)
                    q2.append(nxt)
    return int(components), int(head_area)


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = _safe_int(row.get("step"), -1)
        if step < 0:
            continue
        out[step] = row
    return out


def _first_divergence(
    control: dict[int, dict[str, Any]],
    experiment: dict[int, dict[str, Any]],
) -> tuple[int | None, list[str]]:
    steps = sorted(set(control.keys()).intersection(experiment.keys()))
    watched = ("chosen_action", "mode", "switch_reason", "override_used")
    for step in steps:
        diffs: list[str] = []
        c = control[step]
        e = experiment[step]
        for key in watched:
            if c.get(key) != e.get(key):
                diffs.append(str(key))
        if diffs:
            return int(step), diffs
    return None, []


def _step_snapshot(row: dict[str, Any], board_cells: int) -> dict[str, Any]:
    comps, head_area = _free_components_and_head_area(row, board_cells)
    return {
        "step": _safe_int(row.get("step"), -1),
        "mode": str(row.get("mode", "")),
        "switch_reason": str(row.get("switch_reason", "")),
        "chosen_action": row.get("chosen_action"),
        "predicted_action": row.get("predicted_action"),
        "override_used": _safe_bool(row.get("override_used"), False),
        "safe_option_count": _safe_int(row.get("safe_option_count"), 0),
        "food_pressure": _safe_float(row.get("food_pressure"), 0.0),
        "no_progress_steps": _safe_int(row.get("no_progress_steps"), 0),
        "chosen_tail_reachable": _safe_bool(row.get("chosen_tail_reachable"), False),
        "chosen_capacity_shortfall": _safe_int(row.get("chosen_capacity_shortfall"), 0),
        "free_ratio": _safe_float(row.get("free_ratio"), 0.0),
        "score_before": _safe_int(row.get("score_before"), 0),
        "score_after": _safe_int(row.get("score_after"), 0),
        "ate_food": _safe_bool(row.get("ate_food"), False),
        "free_components": int(comps),
        "head_reachable_area": int(head_area),
    }


def _trend_summary(series: list[dict[str, Any]]) -> dict[str, Any]:
    if not series:
        return {}
    def _vals(key: str) -> list[float]:
        return [float(item.get(key, 0.0)) for item in series]

    safe_vals = _vals("safe_option_count")
    food_vals = _vals("food_pressure")
    tail_vals = _vals("chosen_tail_reachable")
    comp_vals = _vals("free_components")
    area_vals = _vals("head_reachable_area")
    return {
        "rows": int(len(series)),
        "safe_option_mean": float(sum(safe_vals) / max(1, len(safe_vals))),
        "safe_option_min": float(min(safe_vals)),
        "food_pressure_mean": float(sum(food_vals) / max(1, len(food_vals))),
        "food_pressure_max": float(max(food_vals)),
        "tail_reachable_pct": float(100.0 * sum(1.0 for v in tail_vals if float(v) > 0.5) / max(1, len(tail_vals))),
        "free_components_mean": float(sum(comp_vals) / max(1, len(comp_vals))),
        "head_reachable_area_mean": float(sum(area_vals) / max(1, len(area_vals))),
    }


def build_report(
    *,
    control_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    seed: int,
    window_before: int,
    window_after: int,
) -> dict[str, Any]:
    board_cells = _grid_size_from_rows(control_rows + experiment_rows, fallback=20)
    control_by_step = _rows_by_step(control_rows)
    exp_by_step = _rows_by_step(experiment_rows)
    step, diff_keys = _first_divergence(control_by_step, exp_by_step)
    if step is None:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "board_cells": int(board_cells),
            "first_divergence_step": None,
            "note": "No divergence found on overlapping steps for watched fields.",
        }

    prev_step = int(step - 1)
    control_prev = control_by_step.get(prev_step)
    exp_prev = exp_by_step.get(prev_step)
    control_div = control_by_step.get(step, {})
    exp_div = exp_by_step.get(step, {})

    begin = int(max(0, step - max(0, window_before)))
    end = int(step + max(0, window_after))

    control_window: list[dict[str, Any]] = []
    exp_window: list[dict[str, Any]] = []
    for s in range(begin, end + 1):
        if s in control_by_step:
            control_window.append(_step_snapshot(control_by_step[s], board_cells))
        if s in exp_by_step:
            exp_window.append(_step_snapshot(exp_by_step[s], board_cells))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "board_cells": int(board_cells),
        "first_divergence_step": int(step),
        "divergence_fields": [str(k) for k in diff_keys],
        "control_before_divergence": None if control_prev is None else _step_snapshot(control_prev, board_cells),
        "experiment_before_divergence": None if exp_prev is None else _step_snapshot(exp_prev, board_cells),
        "control_at_divergence": _step_snapshot(control_div, board_cells),
        "experiment_at_divergence": _step_snapshot(exp_div, board_cells),
        "control_window_trend": _trend_summary(control_window),
        "experiment_window_trend": _trend_summary(exp_window),
        "control_window_rows": control_window,
        "experiment_window_rows": exp_window,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a seed trace between control and experiment and locate first structural divergence.")
    parser.add_argument("--control-trace", type=str, required=True, help="Control trace dir or seed_*.jsonl path")
    parser.add_argument("--experiment-trace", type=str, required=True, help="Experiment trace dir or seed_*.jsonl path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window-before", type=int, default=30)
    parser.add_argument("--window-after", type=int, default=60)
    parser.add_argument("--out", type=str, default="artifacts/live_eval/seed_divergence_report.json")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    control_path = _seed_trace_path(Path(str(args.control_trace)), int(args.seed))
    exp_path = _seed_trace_path(Path(str(args.experiment_trace)), int(args.seed))
    control_rows = _read_jsonl(control_path)
    exp_rows = _read_jsonl(exp_path)
    if not control_rows:
        raise SystemExit(f"No rows in control trace: {control_path}")
    if not exp_rows:
        raise SystemExit(f"No rows in experiment trace: {exp_path}")
    report = build_report(
        control_rows=control_rows,
        experiment_rows=exp_rows,
        seed=int(args.seed),
        window_before=max(0, int(args.window_before)),
        window_after=max(0, int(args.window_after)),
    )
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(json.dumps({k: report.get(k) for k in ("seed", "first_divergence_step", "divergence_fields")}, indent=2))


if __name__ == "__main__":
    main()
