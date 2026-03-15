from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_risk_handoffs import build_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    path.write_text(text + "\n", encoding="utf-8")


def test_build_report_detects_risk_onsets_and_outcomes(tmp_path: Path) -> None:
    trace = tmp_path / "seed_17004.jsonl"
    rows = [
        {"seed": 17004, "step": 0, "switch_reason": "ppo_mode_viable", "mode": "ppo", "game_over": False, "ate_food": False, "no_progress_steps": 1},
        {"seed": 17004, "step": 1, "switch_reason": "risk", "mode": "escape", "game_over": False, "ate_food": False, "no_progress_steps": 10},
        {"seed": 17004, "step": 2, "switch_reason": "risk", "mode": "escape", "game_over": False, "ate_food": False, "no_progress_steps": 11},
        {"seed": 17004, "step": 3, "switch_reason": "food_pressure", "mode": "escape", "game_over": False, "ate_food": True, "no_progress_steps": 12},
        {"seed": 17004, "step": 4, "switch_reason": "risk", "mode": "escape", "game_over": False, "ate_food": False, "no_progress_steps": 20},
        {"seed": 17004, "step": 5, "switch_reason": "risk_cleared", "mode": "ppo", "game_over": True, "ate_food": False, "no_progress_steps": 21},
    ]
    _write_jsonl(trace, rows)
    report = build_report(trace_files=[trace], horizon=3, signature_len=4)
    summary = report["summary"]
    assert int(summary["risk_onset_count"]) == 2
    assert int(summary["death_within_horizon_count"]) == 1
    assert int(summary["ate_food_within_horizon_count"]) == 1
    assert float(summary["mean_horizon_corridor_ratio"]) >= 0.0
    assert float(summary["mean_horizon_score_delta"]) >= 0.0
    assert int(summary["next_reason_counts"]["food_pressure"]) == 1
    assert int(summary["next_reason_counts"]["risk_cleared"]) == 1
    onsets = report["risk_onsets"]
    assert int(onsets[0]["start_step"]) == 1
    assert int(onsets[0]["run_len"]) == 2
    assert str(onsets[0]["next_reason"]) == "food_pressure"
    assert "horizon_corridor_ratio" in onsets[0]
    assert "horizon_score_delta" in onsets[0]
    assert int(onsets[1]["start_step"]) == 4
    assert int(onsets[1]["run_len"]) == 1
    assert str(onsets[1]["next_mode"]) == "ppo"
