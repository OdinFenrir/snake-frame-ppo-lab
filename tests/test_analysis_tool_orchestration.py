from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import patch

from snake_frame.analysis_tool_catalog import build_tools
from snake_frame.analysis_tool_commands import build_tool_commands
from snake_frame.analysis_tool_runner import pick_first_existing_output


def test_compare_command_requires_explicit_existing_model_1_and_model_2() -> None:
    tools = build_tools("Test_1", "baseline")
    compare = next(t for t in tools if t.key == "phase3_compare")
    try:
        build_tool_commands(compare, left_exp="", right_exp="baseline")
    except ValueError as exc:
        assert "Model 1 experiment must be non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing Model 1")


def test_compare_command_contains_selected_experiments() -> None:
    tools = build_tools("Test_1", "baseline")
    compare = next(t for t in tools if t.key == "phase3_compare")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        left = root / "state" / "ppo" / "Test_1"
        right = root / "state" / "ppo" / "baseline"
        left.mkdir(parents=True, exist_ok=True)
        right.mkdir(parents=True, exist_ok=True)
        with patch("snake_frame.analysis_tool_commands.project_root", return_value=root):
            cmds = build_tool_commands(compare, left_exp="Test_1", right_exp="baseline")
    flat = " ".join(" ".join(cmd) for cmd in cmds)
    assert "--left-exp Test_1" in flat
    assert "--right-exp baseline" in flat


def test_compare_command_rejects_missing_model_2_experiment_path() -> None:
    tools = build_tools("Test_1", "baseline")
    compare = next(t for t in tools if t.key == "phase3_compare")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        left = root / "state" / "ppo" / "Test_1"
        left.mkdir(parents=True, exist_ok=True)
        with patch("snake_frame.analysis_tool_commands.project_root", return_value=root):
            try:
                build_tool_commands(compare, left_exp="Test_1", right_exp="baseline")
            except ValueError as exc:
                assert "Model 2 experiment does not exist: baseline" in str(exc)
            else:
                raise AssertionError("expected ValueError for missing Model 2 experiment")


def test_pick_first_existing_output_is_deterministic_by_declared_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.txt").write_text("a", encoding="utf-8")
        (root / "b.txt").write_text("b", encoding="utf-8")
        chosen = pick_first_existing_output(root, ("b.txt", "a.txt"))
        assert chosen is not None
        assert chosen.name == "b.txt"
