from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from scripts.reporting.common import (
        parse_stamped_middle,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import CONTRACTS, canonical_dir
except ModuleNotFoundError:
    import sys

    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from scripts.reporting.common import (
        parse_stamped_middle,
        validate_retain_stamped,
    )
    from scripts.reporting.contracts import CONTRACTS, canonical_dir


_KNOWN_SUFFIXES = (".json", ".md", ".csv", ".html", ".txt")


@dataclass(frozen=True)
class PrefixPruneResult:
    prefix: str
    total_stamped: int
    kept: int
    removed: int
    removed_files: tuple[str, ...]


def _collect_stamped_files(out_dir: Path, *, prefix: str) -> list[Path]:
    files: list[Path] = []
    for candidate in out_dir.glob(f"{prefix}_*"):
        if not candidate.is_file():
            continue
        name = candidate.name
        matched = False
        for suffix in _KNOWN_SUFFIXES:
            middle = parse_stamped_middle(name, prefix, suffix)
            if middle is None:
                continue
            matched = True
            break
        if matched:
            files.append(candidate)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _prune_prefix(out_dir: Path, *, prefix: str, retain: int, apply: bool) -> PrefixPruneResult:
    candidates = _collect_stamped_files(out_dir, prefix=prefix)
    keep = max(0, int(retain))
    stale = candidates[keep:]
    removed_files: list[str] = []
    if apply:
        for path in stale:
            path.unlink(missing_ok=True)
            removed_files.append(path.name)
    return PrefixPruneResult(
        prefix=prefix,
        total_stamped=len(candidates),
        kept=min(len(candidates), keep),
        removed=len(stale),
        removed_files=tuple(removed_files),
    )


def _family_results(root: Path, family: str, retain: int, apply: bool) -> dict[str, Any]:
    contract = CONTRACTS[family]
    out_dir = canonical_dir(root, family)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_rows: list[dict[str, Any]] = []
    removed_total = 0
    for prefix in contract.stamped_prefixes:
        result = _prune_prefix(out_dir, prefix=prefix, retain=retain, apply=apply)
        removed_total += int(result.removed)
        prefix_rows.append(
            {
                "prefix": result.prefix,
                "total_stamped": int(result.total_stamped),
                "kept": int(result.kept),
                "removed": int(result.removed),
                "removed_files": list(result.removed_files),
            }
        )
    return {
        "family": family,
        "out_dir": str(out_dir),
        "retain_stamped": int(retain),
        "apply": bool(apply),
        "mode": "prune",
        "removed_total": int(removed_total),
        "prefixes": prefix_rows,
    }


def _purge_family_files(root: Path, family: str, apply: bool) -> dict[str, Any]:
    out_dir = canonical_dir(root, family)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = [p for p in out_dir.rglob("*") if p.is_file()]
    removed_files: list[str] = []
    if apply:
        for path in files:
            path.unlink(missing_ok=True)
            removed_files.append(str(path.relative_to(out_dir)))
    return {
        "family": family,
        "out_dir": str(out_dir),
        "apply": bool(apply),
        "mode": "purge_all",
        "removed_total": int(len(files)),
        "prefixes": [],
        "removed_files": removed_files,
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Report Artifact Cleanup")
    lines.append("")
    lines.append(f"- Generated (UTC): {payload.get('generated_at_utc')}")
    lines.append(f"- Mode: {'APPLY' if bool(payload.get('apply')) else 'DRY-RUN'}")
    lines.append(f"- Retain stamped per prefix: {payload.get('retain_stamped')}")
    lines.append(f"- Families: {', '.join(payload.get('families', []))}")
    lines.append("- Scope note: cleanup only affects canonical report families (not `artifacts/live_eval`, `artifacts/share`, or `artifacts/netron`).")
    lines.append(f"- Removed files total: {payload.get('removed_files_total')}")
    lines.append("")
    for row in payload.get("results", []):
        lines.append(f"## {row.get('family')}")
        lines.append(f"- Out dir: `{row.get('out_dir')}`")
        lines.append(f"- Removed total: {row.get('removed_total')}")
        prefixes = row.get("prefixes", [])
        if not prefixes:
            lines.append("- No stamped prefixes configured.")
            lines.append("")
            continue
        for prefix in prefixes:
            lines.append(
                f"- `{prefix.get('prefix')}`: total={prefix.get('total_stamped')} kept={prefix.get('kept')} removed={prefix.get('removed')}"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage report artifacts by retaining latest + last N stamped files per report prefix."
    )
    parser.add_argument("--retain-stamped", type=int, default=5)
    parser.add_argument("--apply", action="store_true", help="Actually delete stale stamped files.")
    parser.add_argument("--purge-all", action="store_true", help="Delete all files under selected report families.")
    parser.add_argument(
        "--families",
        type=str,
        default="all",
        help="Comma-separated family names or 'all' (training_input,agent_performance,phase3_compare,reports_hub).",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts/reports")
    parser.add_argument("--tag", type=str, default="latest")
    return parser.parse_args()


def _resolve_families(raw: str) -> list[str]:
    value = str(raw or "all").strip().lower()
    if value in ("", "all"):
        return sorted(CONTRACTS.keys())
    requested = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in requested if v not in CONTRACTS]
    if unknown:
        raise ValueError(f"unknown family names: {unknown}")
    return requested


def main() -> None:
    args = _parse_args()
    try:
        validate_retain_stamped(int(args.retain_stamped))
        families = _resolve_families(str(args.families))
    except Exception as exc:
        raise SystemExit(f"invalid arguments: {exc}") from exc

    root = Path(__file__).resolve().parents[2]
    out_dir = (root / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag or "latest").strip() or "latest"

    mode = "purge_all" if bool(args.purge_all) else "prune"
    results: list[dict[str, Any]] = []
    removed_total = 0
    for family in families:
        if bool(args.purge_all):
            row = _purge_family_files(root, family, apply=bool(args.apply))
        else:
            row = _family_results(
                root,
                family,
                retain=int(args.retain_stamped),
                apply=bool(args.apply),
            )
        removed_total += int(row.get("removed_total", 0))
        results.append(row)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "mode": mode,
        "retain_stamped": int(args.retain_stamped),
        "families": families,
        "removed_files_total": int(removed_total),
        "results": results,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    md_text = _to_markdown(payload)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_stamp = out_dir / f"report_artifact_cleanup_{ts}.json"
    md_stamp = out_dir / f"report_artifact_cleanup_{ts}.md"
    json_latest = out_dir / f"report_artifact_cleanup_{tag}.json"
    md_latest = out_dir / f"report_artifact_cleanup_{tag}.md"

    json_stamp.write_text(json_text, encoding="utf-8")
    md_stamp.write_text(md_text, encoding="utf-8")
    json_latest.write_text(json_text, encoding="utf-8")
    md_latest.write_text(md_text, encoding="utf-8")

    print(f"Wrote: {json_stamp}")
    print(f"Wrote: {md_stamp}")
    print(f"Wrote: {json_latest}")
    print(f"Wrote: {md_latest}")


if __name__ == "__main__":
    main()
