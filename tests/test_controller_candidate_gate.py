from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.controller_candidate_gate import (
    _enforce,
    build_report,
    extract_mean_delta,
    extract_worst_mean_delta,
    trace_intervention_rate_percent,
)


class TestControllerCandidateGate(unittest.TestCase):
    def test_extract_mean_delta_from_suite_payload(self) -> None:
        suite = {
            "ppo_only": {"rows": [{"seed": 1, "score": 100}, {"seed": 2, "score": 100}]},
            "controller_on": {"rows": [{"seed": 1, "score": 96}, {"seed": 2, "score": 104}]},
        }
        self.assertAlmostEqual(extract_mean_delta(suite), 0.0)

    def test_extract_worst_mean_delta_from_suite_payload(self) -> None:
        suite = {
            "ppo_only": {"rows": [{"seed": 1, "score": 100}, {"seed": 2, "score": 100}, {"seed": 3, "score": 100}]},
            "controller_on": {"rows": [{"seed": 1, "score": 90}, {"seed": 2, "score": 101}, {"seed": 3, "score": 95}]},
        }
        # deltas = [-10, +1, -5], worst2 mean = -7.5
        self.assertAlmostEqual(extract_worst_mean_delta(suite, top_n=2), -7.5)

    def test_trace_intervention_rate_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace = root / "seed_17001.jsonl"
            rows = [
                {"override_used": True},
                {"override_used": False},
                {"override_used": True},
                {"override_used": False},
            ]
            trace.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertAlmostEqual(trace_intervention_rate_percent(root), 50.0)

    def test_build_report_and_enforce(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_full = root / "baseline_full.json"
            candidate_full = root / "candidate_full.json"
            baseline_worst = root / "baseline_worst.json"
            candidate_worst = root / "candidate_worst.json"
            baseline_trace = root / "baseline_traces"
            candidate_trace = root / "candidate_traces"
            baseline_trace.mkdir(parents=True, exist_ok=True)
            candidate_trace.mkdir(parents=True, exist_ok=True)

            baseline_full.write_text(json.dumps({"mean_delta": -3.0}), encoding="utf-8")
            candidate_full.write_text(json.dumps({"mean_delta": -2.0}), encoding="utf-8")
            baseline_worst.write_text(json.dumps({"worst_mean_delta_controller_minus_ppo": -20.0}), encoding="utf-8")
            candidate_worst.write_text(json.dumps({"worst_mean_delta_controller_minus_ppo": -15.0}), encoding="utf-8")
            (baseline_trace / "seed_1.jsonl").write_text(
                "\n".join([json.dumps({"override_used": False}) for _ in range(10)]),
                encoding="utf-8",
            )
            (candidate_trace / "seed_1.jsonl").write_text(
                "\n".join([json.dumps({"override_used": False}) for _ in range(9)] + [json.dumps({"override_used": True})]),
                encoding="utf-8",
            )

            args = SimpleNamespace(
                baseline_full=str(baseline_full),
                candidate_full=str(candidate_full),
                baseline_worst=str(baseline_worst),
                candidate_worst=str(candidate_worst),
                top_n=10,
                baseline_trace_dir=str(baseline_trace),
                candidate_trace_dir=str(candidate_trace),
                baseline_intervention_rate=None,
                candidate_intervention_rate=None,
                min_full_delta_gain=0.0,
                min_worst_delta_gain=0.0,
                max_intervention_rate=20.0,
                max_intervention_rate_increase=15.0,
                require_worst_improvement=True,
                enforce=True,
                out=str(root / "gate.json"),
            )

            report = build_report(args)
            self.assertTrue(bool(report.get("accepted")))
            _enforce(report)

    def test_enforce_raises_on_failed_check(self) -> None:
        report = {"checks": [{"name": "full_delta_gain", "passed": False}]}
        with self.assertRaisesRegex(RuntimeError, "full_delta_gain"):
            _enforce(report)


if __name__ == "__main__":
    unittest.main()
