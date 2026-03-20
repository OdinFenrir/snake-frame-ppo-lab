from __future__ import annotations

from pathlib import Path

from .analysis_tool_catalog import ToolSpec


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_experiments(root: Path | None = None) -> list[str]:
    base = (root or project_root()) / "state" / "ppo"
    if not base.exists():
        return ["baseline"]
    names = sorted(
        [
            p.name
            for p in base.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        ]
    )
    return names or ["baseline"]


def python_exe(root: Path) -> str:
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return "python"


def resolve_model_path(root: Path, exp_name: str) -> Path:
    preferred = root / "state" / "ppo" / exp_name / "best_score_model.zip"
    if preferred.exists():
        return preferred
    fallback = root / "state" / "ppo" / exp_name / "best_model.zip"
    if fallback.exists():
        return fallback
    return preferred


def _validate_experiment_exists(root: Path, exp_name: str, label: str) -> None:
    name = str(exp_name or "").strip()
    if not name:
        raise ValueError(f"{label} experiment must be non-empty")
    path = root / "state" / "ppo" / name
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{label} experiment does not exist: {name}")


def build_tool_commands(spec: ToolSpec, *, left_exp: str, right_exp: str) -> list[tuple[str, ...]]:
    root = project_root()
    py = python_exe(root)
    if spec.key not in ("report_artifacts", "report_artifacts_purge"):
        _validate_experiment_exists(root, left_exp, "Model 1")
    if spec.key == "phase3_compare":
        _validate_experiment_exists(root, right_exp, "Model 2")

    if spec.key == "training_input":
        return [
            (
                py,
                "scripts/training_input/build_training_input_report.py",
                "--artifact-dir",
                f"state/ppo/{left_exp}",
                "--out-dir",
                "artifacts/training_input",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/training_input/build_training_input_timeline.py",
                "--artifact-dir",
                f"state/ppo/{left_exp}",
                "--out-dir",
                "artifacts/training_input",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/training_input/build_training_input_visuals.py",
                "--in-dir",
                "artifacts/training_input",
                "--out-dir",
                "artifacts/training_input",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/reporting/build_reports_hub.py",
                "--artifacts-root",
                "artifacts",
                "--out-dir",
                "artifacts/reports",
            ),
        ]

    if spec.key == "agent_performance":
        return [
            (
                py,
                "scripts/agent_performance/build_agent_performance_report.py",
                "--artifact-dir",
                f"state/ppo/{left_exp}",
                "--out-dir",
                "artifacts/agent_performance",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/agent_performance/build_agent_performance_visuals.py",
                "--in-dir",
                "artifacts/agent_performance",
                "--out-dir",
                "artifacts/agent_performance",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/reporting/build_reports_hub.py",
                "--artifacts-root",
                "artifacts",
                "--out-dir",
                "artifacts/reports",
            ),
        ]

    if spec.key == "phase3_compare":
        return [
            (
                py,
                "scripts/phase3_compare/build_model_agent_compare_report.py",
                "--left-exp",
                left_exp,
                "--right-exp",
                right_exp,
                "--out-dir",
                "artifacts/phase3_compare",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/phase3_compare/build_model_agent_compare_visuals.py",
                "--in-dir",
                "artifacts/phase3_compare",
                "--out-dir",
                "artifacts/phase3_compare",
                "--tag",
                "latest",
                "--retain-stamped",
                "5",
            ),
            (
                py,
                "scripts/reporting/build_reports_hub.py",
                "--artifacts-root",
                "artifacts",
                "--out-dir",
                "artifacts/reports",
            ),
        ]

    if spec.key == "report_artifacts":
        return [
            (
                py,
                "scripts/reporting/manage_report_artifacts.py",
                "--retain-stamped",
                "5",
                "--apply",
                "--families",
                "training_input,agent_performance,phase3_compare,reports_hub",
                "--out-dir",
                "artifacts/reports",
                "--tag",
                "latest",
            )
        ]

    if spec.key == "report_artifacts_purge":
        return [
            (
                py,
                "scripts/reporting/manage_report_artifacts.py",
                "--apply",
                "--purge-all",
                "--families",
                "training_input,agent_performance,phase3_compare,reports_hub",
                "--out-dir",
                "artifacts/reports",
                "--tag",
                "latest",
            )
        ]

    if spec.key == "blind_spot":
        return [
            (
                py,
                "scripts/worst_seed_gate.py",
                "--suite",
                "artifacts/live_eval/suites/latest_suite.json",
                "--top-n",
                "10",
                "--out",
                "artifacts/live_eval/worst10_latest.json",
            ),
            (
                py,
                "scripts/focused_controller_trace.py",
                "--state-dir",
                "state",
                "--experiment",
                left_exp,
                "--out-dir",
                "artifacts/live_eval",
                "--model-selector",
                "last",
                "--worst-json",
                "artifacts/live_eval/worst10_latest.json",
                "--latest-summary",
                "artifacts/live_eval/latest_summary.json",
                "--top-n",
                "10",
                "--trace-tag",
                "blind_spot",
                "--reuse-latest-traces",
            ),
            (
                py,
                "scripts/blind_spot_replay.py",
                "--trace-root",
                "artifacts/live_eval/focused_traces",
                "--latest-only",
                "--min-confidence",
                "0.7",
                "--max-steps-to-death",
                "10",
                "--replay-window",
                "30",
                "--max-spots",
                "50",
                "--out",
                "artifacts/live_eval/blind_spot_replay_latest.json",
            ),
            (
                py,
                "scripts/blind_spot_replay_view.py",
                "--input",
                "artifacts/live_eval/blind_spot_replay_latest.json",
                "--out",
                "artifacts/live_eval/blind_spot_replay_latest.html",
            ),
        ]

    if spec.key == "postrun_suite":
        return [
            (
                py,
                "scripts/post_run_suite.py",
                "--artifact-dir",
                f"state/ppo/{left_exp}",
                "--artifacts-root",
                "artifacts",
                "--out-dir",
                "artifacts/share",
                "--print-summary",
            )
        ]

    if spec.key in ("policy_3d", "netron"):
        model_path = resolve_model_path(root, left_exp)
        if spec.key == "policy_3d":
            return [
                (
                    py,
                    "scripts/view_policy_3d.py",
                    "--model",
                    str(model_path),
                    "--episodes",
                    "8",
                    "--max-steps",
                    "800",
                    "--max-points",
                    "4000",
                    "--out",
                    "artifacts/share/policy_3d_latest.html",
                )
            ]
        netron_exe = root / ".venv" / "Scripts" / "netron.exe"
        trace_out = root / "artifacts" / "netron" / "policy_trace.pt"
        return [
            (
                py,
                "scripts/export_policy_trace.py",
                "--model",
                str(model_path),
                "--out",
                str(trace_out),
            ),
            ("__DETACH__", str(netron_exe), str(trace_out), "-b"),
        ]
    return []
