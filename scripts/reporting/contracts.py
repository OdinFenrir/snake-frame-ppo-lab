from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPORT_FAMILY_TRAINING_INPUT = "training_input"
REPORT_FAMILY_AGENT_PERFORMANCE = "agent_performance"
REPORT_FAMILY_PHASE3_COMPARE = "phase3_compare"
REPORT_FAMILY_REPORTS_HUB = "reports_hub"


@dataclass(frozen=True)
class ReportFamilyContract:
    family: str
    canonical_rel_dir: Path
    required_latest_files: tuple[str, ...]
    stamped_prefixes: tuple[str, ...]


CONTRACTS: dict[str, ReportFamilyContract] = {
    REPORT_FAMILY_TRAINING_INPUT: ReportFamilyContract(
        family=REPORT_FAMILY_TRAINING_INPUT,
        canonical_rel_dir=Path("artifacts") / "training_input",
        required_latest_files=(
            "training_input_latest.json",
            "training_input_latest.md",
            "training_input_checkpoint_vecnorm_latest.csv",
            "training_input_timeline_latest.json",
            "training_input_timeline_latest.md",
            "training_input_timeline_latest.csv",
            "training_input_dashboard_latest.html",
        ),
        stamped_prefixes=(
            "training_input",
            "training_input_checkpoint_vecnorm",
            "training_input_timeline",
            "training_input_dashboard",
        ),
    ),
    REPORT_FAMILY_AGENT_PERFORMANCE: ReportFamilyContract(
        family=REPORT_FAMILY_AGENT_PERFORMANCE,
        canonical_rel_dir=Path("artifacts") / "agent_performance",
        required_latest_files=(
            "agent_performance_latest.json",
            "agent_performance_latest.md",
            "agent_performance_rows_latest.csv",
            "agent_performance_dashboard_latest.html",
        ),
        stamped_prefixes=(
            "agent_performance",
            "agent_performance_rows",
            "agent_performance_dashboard",
        ),
    ),
    REPORT_FAMILY_PHASE3_COMPARE: ReportFamilyContract(
        family=REPORT_FAMILY_PHASE3_COMPARE,
        canonical_rel_dir=Path("artifacts") / "phase3_compare",
        required_latest_files=(
            "model_agent_compare_latest.json",
            "model_agent_compare_latest.md",
            "model_agent_compare_rows_latest.csv",
            "model_agent_compare_dashboard_latest.html",
        ),
        stamped_prefixes=(
            "model_agent_compare",
            "model_agent_compare_rows",
            "model_agent_compare_dashboard",
        ),
    ),
    REPORT_FAMILY_REPORTS_HUB: ReportFamilyContract(
        family=REPORT_FAMILY_REPORTS_HUB,
        canonical_rel_dir=Path("artifacts") / "reports",
        required_latest_files=(
            "reports_hub_latest.md",
            "reports_hub_latest.txt",
        ),
        stamped_prefixes=(),
    ),
}


def canonical_dir(root: Path, family: str) -> Path:
    contract = CONTRACTS[family]
    return (root / contract.canonical_rel_dir).resolve()

