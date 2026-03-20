from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid
import zipfile


@dataclass(frozen=True)
class ModelManagerResult:
    ok: bool
    message: str
    archive_path: Path | None = None


_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_INTERNAL_PREFIX = "_"
_BASELINE = "baseline"
_MANAGED_ARTIFACT_SUBDIRS = (
    "training_input",
    "agent_performance",
    "live_eval",
    "share",
    "reports",
    "visuals",
    "netron",
)


def sanitize_model_name(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        return ""
    if not _NAME_RE.fullmatch(value):
        return ""
    return value


def is_internal_model_name(name: str) -> bool:
    n = sanitize_model_name(name)
    if not n:
        return True
    return n.startswith(_INTERNAL_PREFIX)


def list_models(state_root: Path) -> list[str]:
    ppo_root = state_root / "ppo"
    if not ppo_root.exists():
        return []
    names: list[str] = []
    for child in ppo_root.iterdir():
        if not child.is_dir():
            continue
        name = sanitize_model_name(child.name)
        if not name or is_internal_model_name(name):
            continue
        names.append(name)
    names.sort()
    return names


def list_archives(state_root: Path) -> list[Path]:
    archives_dir = state_root / "ppo" / "_archives"
    if not archives_dir.exists():
        return []
    archives = [p for p in archives_dir.glob("baseline_archive_*.zip") if p.is_file()]
    archives.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return archives


def promote_to_baseline(state_root: Path, source_model: str) -> ModelManagerResult:
    source = sanitize_model_name(source_model)
    if not source:
        return ModelManagerResult(ok=False, message="Invalid source model name.")
    if is_internal_model_name(source):
        return ModelManagerResult(ok=False, message="Cannot promote internal model.")
    if source == _BASELINE:
        return ModelManagerResult(ok=False, message="Source model is already baseline.")

    ppo_root = state_root / "ppo"
    source_dir = ppo_root / source
    baseline_dir = ppo_root / _BASELINE
    if not source_dir.exists() or not source_dir.is_dir():
        return ModelManagerResult(ok=False, message=f"Model not found: {source}")

    archive_path = _archive_baseline_if_present(state_root, include_artifacts=True)
    try:
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir)
        source_dir.replace(baseline_dir)
        _normalize_baseline_metadata(baseline_dir)
        # New baseline should start with fresh derived reports/visuals.
        _clear_managed_artifacts(state_root)
    except OSError as exc:
        return ModelManagerResult(ok=False, message=f"Promote failed: {exc}", archive_path=archive_path)

    if archive_path is not None:
        msg = f"Promoted {source} -> baseline. Archived previous baseline: {archive_path.name}"
    else:
        msg = f"Promoted {source} -> baseline."
    return ModelManagerResult(ok=True, message=msg, archive_path=archive_path)


def delete_model(
    state_root: Path,
    model_name: str,
    *,
    allow_delete_baseline: bool = False,
) -> ModelManagerResult:
    name = sanitize_model_name(model_name)
    if not name:
        return ModelManagerResult(ok=False, message="Invalid model name.")
    if is_internal_model_name(name):
        return ModelManagerResult(ok=False, message="Cannot delete internal model folder.")
    if name == _BASELINE and not bool(allow_delete_baseline):
        return ModelManagerResult(
            ok=False,
            message="Baseline delete is blocked in Model Manager. Promote another model to baseline first.",
        )
    model_dir = state_root / "ppo" / name
    if not model_dir.exists():
        return ModelManagerResult(ok=False, message=f"Model not found: {name}")
    try:
        shutil.rmtree(model_dir)
        if name == _BASELINE and bool(allow_delete_baseline):
            _clear_managed_artifacts(state_root)
    except OSError as exc:
        return ModelManagerResult(ok=False, message=f"Delete failed: {exc}")
    return ModelManagerResult(ok=True, message=f"Deleted model: {name}")


def recover_baseline(state_root: Path, archive_zip: Path, *, include_artifacts: bool = False) -> ModelManagerResult:
    ppo_root = state_root / "ppo"
    archives_dir = ppo_root / "_archives"
    archive_path = archive_zip.resolve()
    try:
        valid_parent = archive_path.parent.resolve() == archives_dir.resolve()
    except OSError:
        valid_parent = False
    if not valid_parent:
        return ModelManagerResult(ok=False, message="Archive must be selected from state/ppo/_archives.")
    if not archive_path.exists() or archive_path.suffix.lower() != ".zip":
        return ModelManagerResult(ok=False, message="Archive zip not found.")
    valid_manifest, manifest_error = _validate_baseline_archive_manifest(archive_path)
    if not valid_manifest:
        return ModelManagerResult(ok=False, message=f"Invalid baseline archive: {manifest_error}")

    baseline_dir = ppo_root / _BASELINE
    current_archive = _archive_baseline_if_present(state_root, include_artifacts=bool(include_artifacts))
    temp_root = ppo_root / f"_recover_tmp_{uuid.uuid4().hex[:10]}"
    temp_baseline = temp_root / "state" / "ppo" / _BASELINE
    temp_artifacts = temp_root / "artifacts"
    project_root = _project_root_from_state(state_root)
    try:
        temp_baseline.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                if not name or name.endswith("/"):
                    continue
                if name.startswith("meta/"):
                    continue
                if name.startswith("state/ppo/baseline/"):
                    rel = name[len("state/ppo/baseline/") :]
                    target = _safe_extract_target(temp_baseline, rel)
                elif name.startswith("artifacts/"):
                    if not include_artifacts:
                        continue
                    artifact_rel = name[len("artifacts/") :]
                    if not _artifact_rel_is_managed(artifact_rel):
                        continue
                    target = _safe_extract_target(temp_artifacts, artifact_rel)
                else:
                    # Backward compatibility with legacy archives that only stored baseline files.
                    target = _safe_extract_target(temp_baseline, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir)
        temp_baseline.replace(baseline_dir)
        _normalize_baseline_metadata(baseline_dir)
        if include_artifacts:
            _clear_managed_artifacts(state_root)
            for sub in _MANAGED_ARTIFACT_SUBDIRS:
                src = temp_artifacts / sub
                if not src.exists():
                    continue
                dst = project_root / "artifacts" / sub
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
    except Exception as exc:
        return ModelManagerResult(ok=False, message=f"Recover failed: {exc}", archive_path=current_archive)
    finally:
        try:
            if temp_root.exists():
                shutil.rmtree(temp_root)
        except OSError:
            pass

    suffix = f" (current baseline archived: {current_archive.name})" if current_archive else ""
    return ModelManagerResult(
        ok=True,
        message=f"Recovered baseline from archive: {archive_path.name}{suffix}",
        archive_path=current_archive,
    )


def _archive_baseline_if_present(state_root: Path, *, include_artifacts: bool) -> Path | None:
    ppo_root = state_root / "ppo"
    baseline_dir = ppo_root / _BASELINE
    if not baseline_dir.exists() or not baseline_dir.is_dir():
        return None

    archives_dir = ppo_root / "_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    archive_name = f"baseline_archive_{_BASELINE}_{ts}_{uuid.uuid4().hex[:8]}.zip"
    archive_path = archives_dir / archive_name

    manifest = {
        "schema_version": 1,
        "operation": "archive_baseline",
        "source_model": _BASELINE,
        "snapshot_scope": "model_plus_artifacts" if include_artifacts else "model_only",
        "includes_artifacts": bool(include_artifacts),
        "managed_artifact_roots": [f"artifacts/{v}" for v in _MANAGED_ARTIFACT_SUBDIRS] if include_artifacts else [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_name": archive_name,
        "entries": [],
        "summary": {
            "file_count": 0,
            "total_size_bytes": 0,
            "sha256_rollup": "",
        },
    }
    rollup = hashlib.sha256()
    total_size = 0
    file_count = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child, rel in _iter_archive_sources(state_root, include_artifacts=include_artifacts):
            blob = child.read_bytes()
            sha = hashlib.sha256(blob).hexdigest()
            zf.write(child, arcname=rel)
            size = int(len(blob))
            manifest["entries"].append({"path": rel, "size_bytes": size, "sha256": sha})
            rollup.update(rel.encode("utf-8"))
            rollup.update(b":")
            rollup.update(sha.encode("ascii"))
            rollup.update(b"\n")
            file_count += 1
            total_size += size
        manifest["summary"]["file_count"] = int(file_count)
        manifest["summary"]["total_size_bytes"] = int(total_size)
        manifest["summary"]["sha256_rollup"] = rollup.hexdigest()
        zf.writestr("meta/manifest.json", json.dumps(manifest, indent=2, ensure_ascii=True))
    return archive_path


def _validate_baseline_archive_manifest(archive_path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if "meta/manifest.json" not in zf.namelist():
                return False, "missing meta/manifest.json"
            raw = zf.read("meta/manifest.json")
    except Exception as exc:
        return False, str(exc)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return False, "manifest is not valid JSON"
    if not isinstance(payload, dict):
        return False, "manifest root must be an object"
    if str(payload.get("operation", "")).strip() != "archive_baseline":
        return False, "operation must be archive_baseline"
    if str(payload.get("source_model", "")).strip() != _BASELINE:
        return False, "source_model must be baseline"
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return False, "entries must be a list"
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False, "summary must be present"
    return True, ""


def _project_root_from_state(state_root: Path) -> Path:
    return state_root.resolve().parent


def _managed_artifact_roots(state_root: Path) -> list[Path]:
    project_root = _project_root_from_state(state_root)
    artifacts_root = project_root / "artifacts"
    return [artifacts_root / sub for sub in _MANAGED_ARTIFACT_SUBDIRS]


def _iter_archive_sources(state_root: Path, *, include_artifacts: bool) -> list[tuple[Path, str]]:
    ppo_root = state_root / "ppo"
    baseline_dir = ppo_root / _BASELINE
    out: list[tuple[Path, str]] = []
    for child in sorted(baseline_dir.rglob("*")):
        if child.is_file():
            rel = f"state/ppo/{_BASELINE}/{child.relative_to(baseline_dir).as_posix()}"
            out.append((child, rel))
    if include_artifacts:
        for root in _managed_artifact_roots(state_root):
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.rglob("*")):
                if not child.is_file():
                    continue
                rel = f"artifacts/{child.relative_to(root.parent).as_posix()}"
                out.append((child, rel))
    return out


def _clear_managed_artifacts(state_root: Path) -> None:
    for root in _managed_artifact_roots(state_root):
        try:
            if root.exists() and root.is_dir():
                shutil.rmtree(root)
        except OSError:
            # Ignore cleanup failure to avoid blocking critical model operations.
            continue


def _artifact_rel_is_managed(artifact_rel: str) -> bool:
    rel = str(artifact_rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        return False
    head = rel.split("/", 1)[0]
    return head in _MANAGED_ARTIFACT_SUBDIRS


def _safe_extract_target(base: Path, rel: str) -> Path:
    rel_norm = str(rel or "").replace("\\", "/").strip("/")
    if not rel_norm:
        raise ValueError("empty archive entry path")
    if rel_norm in (".", ".."):
        raise ValueError(f"invalid archive entry path: {rel_norm}")
    candidate = (base / rel_norm).resolve()
    base_resolved = base.resolve()
    if candidate == base_resolved or base_resolved not in candidate.parents:
        raise ValueError(f"archive entry escapes target root: {rel_norm}")
    return candidate


def _normalize_baseline_metadata(baseline_dir: Path) -> None:
    metadata_path = baseline_dir / "metadata.json"
    if not metadata_path.exists():
        return
    payload: dict = {}
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            payload = dict(raw)
    except Exception:
        payload = {}
    payload["experiment_name"] = _BASELINE
    payload["experiment"] = _BASELINE
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
