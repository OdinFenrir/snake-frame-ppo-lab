from __future__ import annotations

from pathlib import Path

from scripts.compare_seed_traces import build_report


def test_seed_17011_first_divergence_pattern_is_locked() -> None:
    control = Path("artifacts/live_eval/focused_traces/20260315_094901_risk_diag_control_noexp/seed_17011.jsonl")
    experiment = Path("artifacts/live_eval/focused_traces/20260315_094808_risk_diag_narrow_np_margin0/seed_17011.jsonl")
    assert control.exists(), f"Missing control trace: {control}"
    assert experiment.exists(), f"Missing experiment trace: {experiment}"

    import json

    control_rows = [json.loads(x) for x in control.read_text(encoding="utf-8").splitlines() if x.strip()]
    experiment_rows = [json.loads(x) for x in experiment.read_text(encoding="utf-8").splitlines() if x.strip()]
    report = build_report(
        control_rows=control_rows,
        experiment_rows=experiment_rows,
        seed=17011,
        window_before=40,
        window_after=80,
    )
    assert int(report.get("first_divergence_step") or -1) == 1002
    fields = set(str(v) for v in (report.get("divergence_fields") or []))
    assert "mode" in fields
    assert "switch_reason" in fields
    control_at = report.get("control_at_divergence") or {}
    experiment_at = report.get("experiment_at_divergence") or {}
    assert str(control_at.get("switch_reason")) == "risk"
    assert str(experiment_at.get("switch_reason")) == "risk_guard_hold"
