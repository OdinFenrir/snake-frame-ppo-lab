from __future__ import annotations

import json
from pathlib import Path
import tempfile

from snake_frame.model_manager import (
    delete_model,
    list_archives,
    list_models,
    promote_to_baseline,
    recover_baseline,
)


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_promote_to_baseline_archives_existing_baseline_and_moves_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        baseline = state_root / "ppo" / "baseline"
        test1 = state_root / "ppo" / "Test_1"
        artifacts_root = Path(tmp) / "artifacts"
        _write_file(baseline / "metadata.json", '{"name":"baseline"}')
        _write_file(test1 / "metadata.json", '{"name":"test1","experiment_name":"Test_1"}')
        _write_file(artifacts_root / "training_input" / "training_input_latest.json", '{"artifact_dir":"state/ppo/baseline"}')
        _write_file(artifacts_root / "phase3_compare" / "model_agent_compare_latest.json", '{"left":"baseline","right":"test"}')

        result = promote_to_baseline(state_root, "Test_1")

        assert result.ok
        assert (state_root / "ppo" / "baseline" / "metadata.json").exists()
        promoted_meta = json.loads((state_root / "ppo" / "baseline" / "metadata.json").read_text(encoding="utf-8"))
        assert promoted_meta.get("experiment_name") == "baseline"
        assert promoted_meta.get("experiment") == "baseline"
        assert not (state_root / "ppo" / "Test_1").exists()
        archives = list_archives(state_root)
        assert len(archives) == 1
        import zipfile
        with zipfile.ZipFile(archives[0], "r") as zf:
            manifest = json.loads(zf.read("meta/manifest.json").decode("utf-8"))
            names = set(zf.namelist())
        assert manifest["operation"] == "archive_baseline"
        assert manifest["source_model"] == "baseline"
        assert int(manifest["summary"]["file_count"]) >= 1
        assert int(manifest["summary"]["total_size_bytes"]) >= 1
        assert len(str(manifest["summary"]["sha256_rollup"])) == 64
        assert "state/ppo/baseline/metadata.json" in names
        assert "artifacts/training_input/training_input_latest.json" in names
        assert "artifacts/phase3_compare/model_agent_compare_latest.json" not in names
        # Promote invalidates managed artifacts to prevent stale report context.
        assert not (artifacts_root / "training_input").exists()


def test_delete_model_removes_directory_tree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        model_dir = state_root / "ppo" / "ToDelete"
        _write_file(model_dir / "last_model.zip", "model")

        result = delete_model(state_root, "ToDelete")

        assert result.ok
        assert not model_dir.exists()


def test_recover_baseline_restores_from_archive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        artifacts_root = Path(tmp) / "artifacts"
        baseline = state_root / "ppo" / "baseline"
        source = state_root / "ppo" / "Test_1"
        _write_file(baseline / "metadata.json", json.dumps({"name": "old-baseline"}))
        _write_file(source / "metadata.json", json.dumps({"name": "test1"}))
        _write_file(artifacts_root / "training_input" / "training_input_latest.md", "baseline-report")
        promote = promote_to_baseline(state_root, "Test_1")
        assert promote.ok
        archive = promote.archive_path
        assert archive is not None

        _write_file(artifacts_root / "training_input" / "training_input_latest.md", "newer-report")
        recover = recover_baseline(state_root, archive)

        assert recover.ok
        payload = json.loads((state_root / "ppo" / "baseline" / "metadata.json").read_text(encoding="utf-8"))
        assert payload["name"] == "old-baseline"
        assert payload["experiment_name"] == "baseline"
        assert payload["experiment"] == "baseline"
        # Default recover is model-only and should leave artifacts untouched.
        still_current = (artifacts_root / "training_input" / "training_input_latest.md").read_text(encoding="utf-8")
        assert still_current == "newer-report"


def test_recover_baseline_workspace_restores_managed_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        artifacts_root = Path(tmp) / "artifacts"
        baseline = state_root / "ppo" / "baseline"
        source = state_root / "ppo" / "Test_1"
        _write_file(baseline / "metadata.json", json.dumps({"name": "old-baseline"}))
        _write_file(source / "metadata.json", json.dumps({"name": "test1"}))
        _write_file(artifacts_root / "training_input" / "training_input_latest.md", "baseline-report")
        _write_file(artifacts_root / "phase3_compare" / "model_agent_compare_latest.md", "compare-report")
        promote = promote_to_baseline(state_root, "Test_1")
        assert promote.ok
        archive = promote.archive_path
        assert archive is not None

        _write_file(artifacts_root / "training_input" / "training_input_latest.md", "newer-report")
        _write_file(artifacts_root / "phase3_compare" / "model_agent_compare_latest.md", "newer-compare")
        recover = recover_baseline(state_root, archive, include_artifacts=True)

        assert recover.ok
        restored = (artifacts_root / "training_input" / "training_input_latest.md").read_text(encoding="utf-8")
        assert restored == "baseline-report"
        # Compare artifacts are intentionally excluded from snapshot restore.
        still_compare = (artifacts_root / "phase3_compare" / "model_agent_compare_latest.md").read_text(encoding="utf-8")
        assert still_compare == "newer-compare"


def test_recover_failure_does_not_mutate_live_baseline_or_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        artifacts_root = Path(tmp) / "artifacts"
        baseline = state_root / "ppo" / "baseline"
        archives = state_root / "ppo" / "_archives"
        _write_file(baseline / "metadata.json", json.dumps({"name": "live-baseline"}))
        _write_file(artifacts_root / "training_input" / "training_input_latest.md", "live-report")
        archives.mkdir(parents=True, exist_ok=True)
        bad_archive = archives / "baseline_archive_bad.zip"
        import zipfile

        with zipfile.ZipFile(bad_archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta/manifest.json", json.dumps({"operation": "archive_baseline", "source_model": "baseline", "entries": [], "summary": {}}))
            # Path traversal should be rejected before any live mutation.
            zf.writestr("state/ppo/baseline/../../outside.txt", "bad")

        result = recover_baseline(state_root, bad_archive, include_artifacts=True)

        assert not result.ok
        payload = json.loads((baseline / "metadata.json").read_text(encoding="utf-8"))
        assert payload["name"] == "live-baseline"
        assert (artifacts_root / "training_input" / "training_input_latest.md").read_text(encoding="utf-8") == "live-report"


def test_list_models_excludes_internal_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        _write_file(state_root / "ppo" / "baseline" / "x.txt", "ok")
        _write_file(state_root / "ppo" / "_archives" / "x.txt", "no")
        _write_file(state_root / "ppo" / "_detached_session" / "x.txt", "no")

        names = list_models(state_root)

        assert names == ["baseline"]


def test_delete_baseline_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        _write_file(state_root / "ppo" / "baseline" / "last_model.zip", "model")

        result = delete_model(state_root, "baseline")

        assert not result.ok
        assert "blocked" in result.message.lower()
        assert (state_root / "ppo" / "baseline").exists()


def test_delete_baseline_allowed_with_explicit_override() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        artifacts_root = Path(tmp) / "artifacts"
        _write_file(state_root / "ppo" / "baseline" / "last_model.zip", "model")
        _write_file(artifacts_root / "training_input" / "training_input_latest.json", "{}")

        result = delete_model(state_root, "baseline", allow_delete_baseline=True)

        assert result.ok
        assert not (state_root / "ppo" / "baseline").exists()
        assert not (artifacts_root / "training_input").exists()


def test_recover_requires_valid_baseline_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "state"
        archives = state_root / "ppo" / "_archives"
        archives.mkdir(parents=True, exist_ok=True)
        bogus = archives / "baseline_archive_bogus.zip"
        import zipfile

        with zipfile.ZipFile(bogus, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("not_manifest.txt", "x")

        result = recover_baseline(state_root, bogus)

        assert not result.ok
        assert "invalid baseline archive" in result.message.lower()
