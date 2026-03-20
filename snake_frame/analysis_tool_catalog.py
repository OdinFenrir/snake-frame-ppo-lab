from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ToolSpec:
    key: str
    label: str
    category: str
    description: str
    outputs: tuple[str, ...]
    embeddable: bool


def build_tools(left_exp: str, right_exp: str) -> list[ToolSpec]:
    compare_desc = f"Compare model+agent between saves ({left_exp} vs {right_exp})"
    return [
        ToolSpec(
            key="training_input",
            label="Training Quality Report",
            category="Reports",
            description="Model-side training quality, checkpoints, and timeline dashboard",
            outputs=(
                "artifacts/training_input/training_input_dashboard_latest.html",
                "artifacts/training_input/training_input_latest.md",
                "artifacts/reports/reports_hub_latest.txt",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="agent_performance",
            label="Agent Runtime Report",
            category="Reports",
            description="Runtime behavior quality and intervention metrics",
            outputs=(
                "artifacts/agent_performance/agent_performance_dashboard_latest.html",
                "artifacts/agent_performance/agent_performance_latest.md",
                "artifacts/reports/reports_hub_latest.txt",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="phase3_compare",
            label="Model vs Model Compare",
            category="Reports",
            description=compare_desc,
            outputs=(
                "artifacts/phase3_compare/model_agent_compare_dashboard_latest.html",
                "artifacts/phase3_compare/model_agent_compare_latest.md",
                "artifacts/reports/reports_hub_latest.txt",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="report_artifacts",
            label="Report Artifact Manager",
            category="Reports",
            description="Prune old stamped report files and keep latest + last N per report type",
            outputs=(
                "artifacts/reports/report_artifact_cleanup_latest.md",
                "artifacts/reports/report_artifact_cleanup_latest.json",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="report_artifacts_purge",
            label="Purge Report Artifacts",
            category="Reports",
            description="Hard-delete report files in training_input/agent_performance/phase3_compare/reports (not live_eval/share/netron)",
            outputs=(
                "artifacts/reports/report_artifact_cleanup_latest.md",
                "artifacts/reports/report_artifact_cleanup_latest.json",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="blind_spot",
            label="Failure Replay",
            category="Diagnostics",
            description="Replay and summarize worst failure traces",
            outputs=(
                "artifacts/live_eval/blind_spot_replay_latest.json",
                "artifacts/live_eval/blind_spot_replay_latest.html",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="postrun_suite",
            label="Evaluation Suite",
            category="Diagnostics",
            description="Build post-run diagnostics bundle for the selected model",
            outputs=(
                "artifacts/share/diagnostics_bundle.json",
                "artifacts/share/diagnostics_bundle.md",
            ),
            embeddable=True,
        ),
        ToolSpec(
            key="policy_3d",
            label="Policy 3D Explorer",
            category="Model Views",
            description="Launch 3D policy visualization",
            outputs=("artifacts/share/policy_3d_latest.html",),
            embeddable=False,
        ),
        ToolSpec(
            key="netron",
            label="Model Graph (Netron)",
            category="Model Views",
            description="Open model graph with Netron",
            outputs=(),
            embeddable=False,
        ),
    ]


def get_tool_by_key(tools: Sequence[ToolSpec], key: str) -> ToolSpec | None:
    for tool in tools:
        if tool.key == key:
            return tool
    return None
