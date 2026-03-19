from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from scripts.reporting.contracts import CONTRACTS, canonical_dir


STAMP_TOKEN_RE = re.compile(r"^\d{8}_\d{6}$")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def resolve_default_artifact_dir(root: Path) -> Path:
    ui_prefs = root / "state" / "ui_prefs.json"
    active = "baseline"
    if ui_prefs.exists():
        payload = read_json(ui_prefs)
        active = str(payload.get("activeExperiment", "baseline") or "baseline").strip() or "baseline"
    candidate = root / "state" / "ppo" / active
    if candidate.exists():
        return candidate.resolve()
    return (root / "state" / "ppo" / "baseline").resolve()


def is_stamp_token(token: str) -> bool:
    return bool(STAMP_TOKEN_RE.fullmatch(token))


def parse_stamped_middle(filename: str, stem_prefix: str, suffix: str) -> str | None:
    prefix = f"{stem_prefix}_"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    middle = filename[len(prefix) : len(filename) - len(suffix)]
    if middle == "latest":
        return None
    return middle if is_stamp_token(middle) else None


def prune_stamped_outputs(out_dir: Path, *, stem_prefix: str, suffix: str, retain: int) -> None:
    keep = max(0, int(retain))
    prefix = f"{stem_prefix}_"
    candidates: list[Path] = []
    for path in out_dir.glob(f"{prefix}*{suffix}"):
        middle = parse_stamped_middle(path.name, stem_prefix, suffix)
        if middle is None:
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in candidates[keep:]:
        stale.unlink(missing_ok=True)


def validate_retain_stamped(retain: int) -> None:
    if int(retain) < 0:
        raise ValueError("--retain-stamped must be >= 0")


def validate_non_empty(name: str, value: str) -> None:
    if not str(value or "").strip():
        raise ValueError(f"{name} must be non-empty")


def validate_canonical_out_dir(root: Path, family: str, out_dir: Path) -> Path:
    expected = canonical_dir(root, family)
    resolved = out_dir.resolve()
    if resolved != expected:
        raise ValueError(
            f"--out-dir must be canonical for {family}: expected '{expected}', got '{resolved}'"
        )
    return resolved


def validate_required_latest_files(root: Path, family: str, out_dir: Path) -> None:
    contract = CONTRACTS[family]
    resolved = out_dir.resolve()
    expected = canonical_dir(root, family)
    if resolved != expected:
        raise ValueError(
            f"contract violation: {family} outputs must be in '{expected}', got '{resolved}'"
        )
    missing = [name for name in contract.required_latest_files if not (resolved / name).exists()]
    if missing:
        raise ValueError(f"contract violation: missing latest files for {family}: {missing}")

