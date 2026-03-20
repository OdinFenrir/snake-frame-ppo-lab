from __future__ import annotations

import json
from pathlib import Path
import subprocess


def run_commands(commands: list[tuple[str, ...]], *, root: Path, timeout_s: int = 3600) -> str:
    merged_chunks: list[str] = []
    detach_prefix = "__DETACH__"
    for cmd in commands:
        if len(cmd) >= 2 and str(cmd[0]) == detach_prefix:
            detached_cmd = list(cmd[1:])
            creationflags = 0
            if hasattr(subprocess, "DETACHED_PROCESS"):
                creationflags |= int(subprocess.DETACHED_PROCESS)
            if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                creationflags |= int(subprocess.CREATE_NEW_PROCESS_GROUP)
            subprocess.Popen(
                detached_cmd,
                cwd=str(root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=creationflags,
            )
            merged_chunks.append(f"$ {' '.join(detached_cmd)}\n(launched detached)")
            continue
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
