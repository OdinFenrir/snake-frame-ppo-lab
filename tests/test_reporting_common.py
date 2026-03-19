from __future__ import annotations

from pathlib import Path
import tempfile

from scripts.reporting.common import (
    is_stamp_token,
    parse_stamped_middle,
    prune_stamped_outputs,
    read_jsonl,
    resolve_default_artifact_dir,
    safe_float,
    safe_int,
)


def test_stamp_token_parser() -> None:
    assert is_stamp_token("20260319_120001")
    assert not is_stamp_token("latest")
    assert parse_stamped_middle("training_input_20260319_120001.json", "training_input", ".json") == "20260319_120001"
    assert parse_stamped_middle("training_input_latest.json", "training_input", ".json") is None
    assert parse_stamped_middle("training_input_bad.json", "training_input", ".json") is None


def test_prune_keeps_latest_and_last_n_stamped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "training_input_latest.json").write_text("{}", encoding="utf-8")
        stamped = [
            out / "training_input_20260319_120001.json",
            out / "training_input_20260319_120002.json",
            out / "training_input_20260319_120003.json",
        ]
        for i, p in enumerate(stamped):
            p.write_text(str(i), encoding="utf-8")
            p.touch()
        # non-matching file must never be deleted by retention.
        other = out / "note.txt"
        other.write_text("keep", encoding="utf-8")

        prune_stamped_outputs(out, stem_prefix="training_input", suffix=".json", retain=2)
        remaining = sorted([p.name for p in out.glob("training_input_*.json") if p.name != "training_input_latest.json"])
        assert len(remaining) == 2
        assert "training_input_latest.json" in [p.name for p in out.glob("training_input_latest.json")]
        assert other.exists()


def test_safe_numeric_casts() -> None:
    assert safe_int("10") == 10
    assert safe_int("x", default=7) == 7
    assert abs(safe_float("1.5") - 1.5) < 1e-9
    assert abs(safe_float("x", default=2.5) - 2.5) < 1e-9


def test_resolve_default_artifact_dir_prefers_active_experiment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prefs = root / "state" / "ui_prefs.json"
        exp = root / "state" / "ppo" / "Test_1"
        exp.mkdir(parents=True, exist_ok=True)
        prefs.parent.mkdir(parents=True, exist_ok=True)
        prefs.write_text('{"activeExperiment":"Test_1"}', encoding="utf-8")
        assert resolve_default_artifact_dir(root) == exp.resolve()


def test_read_jsonl_skips_corrupt_and_partial_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "rows.jsonl"
        p.write_text('{"ok":1}\nnot-json\n{"also":"ok"}\n', encoding="utf-8")
        rows = read_jsonl(p)
        assert len(rows) == 2
        assert rows[0].get("ok") == 1
        assert rows[1].get("also") == "ok"
