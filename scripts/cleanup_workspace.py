from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str


_STEP_MODEL_RE = re.compile(r"^step_(\d+)_steps\.zip$", re.IGNORECASE)
_STEP_STATS_RE = re.compile(r"^step_vecnormalize_(\d+)_steps\.pkl$", re.IGNORECASE)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if path.is_file():
        return int(path.stat().st_size)
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += int(p.stat().st_size)
            except OSError:
                pass
    return total


def _human_bytes(num: int) -> str:
    value = float(max(0, num))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def _collect_candidates(root: Path, aggressive: bool) -> list[Candidate]:
    out: list[Candidate] = []
    artifacts = root / "artifacts"
    for p in artifacts.glob("tmp*"):
        if p.exists() and p.name != "tmp":
            out.append(Candidate(path=p, reason="temporary artifacts"))

    for p in (root / ".pytest_cache", root / ".ruff_cache"):
        if p.exists():
            out.append(Candidate(path=p, reason="tool cache"))

    # Remove project-local cache files (excluding venv/site-packages cache).
    pycache_dirs: set[Path] = set()
    for p in root.rglob("__pycache__"):
        if not p.is_dir():
            continue
        if ".venv" in p.parts:
            continue
        pycache_dirs.add(p.resolve())
        out.append(Candidate(path=p, reason="python bytecode cache"))
    for p in root.rglob("*.py[cod]"):
        if not p.is_file():
            continue
        if ".venv" in p.parts:
            continue
        # Avoid duplicate delete attempts for files already covered by
        # a parent __pycache__ directory candidate.
        if p.parent.resolve() in pycache_dirs:
            continue
        out.append(Candidate(path=p, reason="python bytecode file"))

    if aggressive:
        for p in (
            artifacts / "loop_eval" / "latest",
            artifacts / "loop_eval" / "quick",
            artifacts / "test_dashboard" / "latest",
        ):
            if p.exists():
                out.append(Candidate(path=p, reason="rolling summary directory"))
    return out


def _collect_checkpoint_candidates(root: Path) -> list[Candidate]:
    out: list[Candidate] = []
    checkpoints_dir = root / "state" / "ppo" / "baseline" / "checkpoints"
    if not checkpoints_dir.exists():
        return out

    model_steps: list[tuple[int, Path]] = []
    stats_steps: list[tuple[int, Path]] = []
    for path in checkpoints_dir.iterdir():
        if not path.is_file():
            continue
        model_match = _STEP_MODEL_RE.match(path.name)
        if model_match:
            model_steps.append((int(model_match.group(1)), path))
            continue
        stats_match = _STEP_STATS_RE.match(path.name)
        if stats_match:
            stats_steps.append((int(stats_match.group(1)), path))

    keep_step = None
    if model_steps:
        keep_step = max(step for step, _ in model_steps)
    elif stats_steps:
        keep_step = max(step for step, _ in stats_steps)

    if keep_step is None:
        return out

    for step, path in model_steps:
        if step != keep_step:
            out.append(Candidate(path=path, reason=f"older checkpoint (keeping step {keep_step})"))
    for step, path in stats_steps:
        if step != keep_step:
            out.append(Candidate(path=path, reason=f"older checkpoint stats (keeping step {keep_step})"))
    return out


def _delete_path(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean temporary workspace artifacts safely.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default is dry-run).")
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Also remove rolling latest/quick summary directories.",
    )
    parser.add_argument(
        "--last-run-only",
        action="store_true",
        help="Keep only the most recent checkpoint pair under state/ppo/baseline/checkpoints.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates = _collect_candidates(root=root, aggressive=bool(args.aggressive))
    if args.last_run_only:
        candidates.extend(_collect_checkpoint_candidates(root=root))
    # De-duplicate by path while preserving first reason.
    uniq: dict[Path, Candidate] = {}
    for c in candidates:
        if c.path not in uniq:
            uniq[c.path] = c
    candidates = list(uniq.values())
    if not candidates:
        print("No cleanup candidates found.")
        return

    rows: list[tuple[Candidate, int]] = []
    for c in candidates:
        try:
            size = _dir_size_bytes(c.path)
        except OSError:
            size = 0
        rows.append((c, size))

    total = sum(size for _, size in rows)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Candidates: {len(rows)}  Potential reclaim: {_human_bytes(total)}")
    for c, size in rows:
        rel = c.path.relative_to(root) if c.path.is_relative_to(root) else c.path
        print(f" - {rel}  ({_human_bytes(size)})  [{c.reason}]")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")
        return

    deleted = 0
    failed = 0
    for c, _size in rows:
        try:
            _delete_path(c.path)
            deleted += 1
        except Exception as exc:
            failed += 1
            print(f" ! Failed to delete {c.path}: {exc}")
    print(f"Deleted: {deleted}, Failed: {failed}")


if __name__ == "__main__":
    main()
