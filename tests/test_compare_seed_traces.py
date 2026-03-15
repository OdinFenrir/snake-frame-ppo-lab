from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_seed_traces import build_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    path.write_text(text + "\n", encoding="utf-8")


def test_build_report_finds_first_divergence_and_fields(tmp_path: Path) -> None:
    control = [
        {"seed": 17011, "step": 0, "mode": "ppo", "switch_reason": "ppo_mode_viable", "chosen_action": 0, "override_used": False, "snake_before": [[1, 1], [0, 1]], "head_before": [1, 1]},
        {"seed": 17011, "step": 1, "mode": "escape", "switch_reason": "risk", "chosen_action": 1, "override_used": True, "safe_option_count": 1, "food_pressure": 0.2, "chosen_tail_reachable": True, "snake_before": [[2, 1], [1, 1]], "head_before": [2, 1]},
        {"seed": 17011, "step": 2, "mode": "escape", "switch_reason": "risk", "chosen_action": 1, "override_used": True, "safe_option_count": 1, "food_pressure": 0.3, "chosen_tail_reachable": True, "snake_before": [[3, 1], [2, 1]], "head_before": [3, 1]},
    ]
    experiment = [
        {"seed": 17011, "step": 0, "mode": "ppo", "switch_reason": "ppo_mode_viable", "chosen_action": 0, "override_used": False, "snake_before": [[1, 1], [0, 1]], "head_before": [1, 1]},
        {"seed": 17011, "step": 1, "mode": "escape", "switch_reason": "risk_guard_hold", "chosen_action": 0, "override_used": False, "safe_option_count": 1, "food_pressure": 0.2, "chosen_tail_reachable": True, "snake_before": [[2, 1], [1, 1]], "head_before": [2, 1]},
        {"seed": 17011, "step": 2, "mode": "ppo", "switch_reason": "ppo_conf_trust", "chosen_action": 0, "override_used": False, "safe_option_count": 2, "food_pressure": 0.1, "chosen_tail_reachable": True, "snake_before": [[3, 1], [2, 1]], "head_before": [3, 1]},
    ]
    control_path = tmp_path / "control" / "seed_17011.jsonl"
    experiment_path = tmp_path / "experiment" / "seed_17011.jsonl"
    _write_jsonl(control_path, control)
    _write_jsonl(experiment_path, experiment)

    report = build_report(
        control_rows=control,
        experiment_rows=experiment,
        seed=17011,
        window_before=1,
        window_after=1,
    )
    assert int(report["first_divergence_step"]) == 1
    fields = set(report["divergence_fields"])
    assert "chosen_action" in fields
    assert "switch_reason" in fields
    assert report["control_at_divergence"]["switch_reason"] == "risk"
    assert report["experiment_at_divergence"]["switch_reason"] == "risk_guard_hold"
