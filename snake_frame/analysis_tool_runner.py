from __future__ import annotations

import json
from pathlib import Path
import subprocess


def run_commands(commands: list[tuple[str, ...]], *, root: Path, timeout_s: int = 3600) -> str:
    merged_chunks: list[str] = []
    for cmd in commands:
        proc = subprocess.run(
            list(cmd),
            cwd=str(root),
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        block = "\n".join([x for x in [out, err] if x]).strip()
        if block:
            merged_chunks.append(f"$ {' '.join(cmd)}\n{block}")
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    merged = "\n\n".join(merged_chunks).strip()
    return merged if merged else "(no console output)"


def read_output_preview(path: Path, *, max_lines: int = 180) -> str:
    if not path.exists():
        return f"Missing output: {path}"
    suffix = path.suffix.lower()
    try:
        if suffix in (".md", ".txt", ".log", ".csv", ".jsonl"):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[:max_lines]) if lines else "(empty file)"
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return json.dumps(payload, indent=2)[:24000]
        if suffix == ".html":
            return f"HTML output generated:\n{path}\n\nOpen externally for full interactive view."
        return f"Output generated:\n{path}"
    except Exception as exc:
        return f"Failed reading output: {path}\n{exc}"


def pick_first_existing_output(root: Path, outputs: tuple[str, ...]) -> Path | None:
    for rel in outputs:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None
