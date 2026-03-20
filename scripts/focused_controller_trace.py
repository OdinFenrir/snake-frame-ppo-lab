from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_frame.app_factory import build_runtime  # noqa: E402
from snake_frame.holdout_eval import HoldoutEvalController  # noqa: E402
from snake_frame.settings import ObsConfig, RewardConfig, Settings, ppo_profile_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused controller holdout eval with per-step traces.")
    parser.add_argument("--state-dir", type=str, default="state")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--out-dir", type=str, default="artifacts/live_eval")
    parser.add_argument("--model-selector", choices=("best", "last"), default="last")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--worst-json", type=str, default="artifacts/live_eval/worst10_latest.json")
    parser.add_argument("--latest-summary", type=str, default="artifacts/live_eval/latest_summary.json")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--trace-tag", type=str, default="worst10")
    parser.add_argument("--reuse-latest-traces", action="store_true")
    parser.add_argument("--enable-risk-switch-guard", action="store_true")
    parser.add_argument("--risk-guard-allow-narrow-corridor", action="store_true")
    parser.add_argument("--risk-guard-narrow-min-no-progress-steps", type=int, default=16)
    parser.add_argument("--risk-guard-narrow-confidence-min", type=float, default=0.97)
    parser.add_argument("--risk-guard-narrow-no-progress-margin", type=int, default=0)
    return parser.parse_args(argv)


def _parse_seed_csv(text: str) -> list[int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out: list[int] = []
    for part in parts:
        out.append(int(part))
    return out


def _seeds_from_worst(path: Path, top_n: int) -> list[int]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("worst_rows") or [])
    out: list[int] = []
    for row in rows[: max(1, int(top_n))]:
        if not isinstance(row, dict):
            continue
        try:
            out.append(int(row.get("seed")))
        except Exception:
            continue
    return out


def _seeds_from_latest_summary(path: Path, top_n: int) -> list[int]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = list(payload.get("rows") or [])
    out: list[int] = []
    for row in rows[: max(1, int(top_n))]:
        if not isinstance(row, dict):
            continue
        try:
            out.append(int(row.get("seed")))
        except Exception:
            continue
    return out


def _latest_trace_dir(root: Path, *, trace_tag: str) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    safe_tag = "".join(ch for ch in str(trace_tag) if ch.isalnum() or ch in ("-", "_")).strip("_")
    if safe_tag:
        suffix = f"_{safe_tag}"
        dirs = [p for p in dirs if p.name.endswith(suffix)]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _seed_set_from_trace_dir(path: Path) -> set[int]:
    out: set[int] = set()
    for trace_file in sorted(path.glob("seed_*.jsonl")):
        stem = trace_file.stem
        try:
            out.add(int(stem.split("_", 1)[1]))
        except Exception:
            continue
    return out


def _trace_meta_path(trace_dir: Path) -> Path:
    return trace_dir / "trace_meta.json"


def _read_trace_meta(trace_dir: Path) -> dict[str, object] | None:
    meta_path = _trace_meta_path(trace_dir)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_trace_meta(
    trace_dir: Path,
    *,
    experiment: str,
    model_selector: str,
    max_steps: int,
    seeds: list[int],
    trace_tag: str,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": str(experiment),
        "model_selector": str(model_selector),
        "max_steps": int(max_steps),
        "trace_tag": str(trace_tag),
        "seeds": sorted(int(s) for s in seeds),
    }
    _trace_meta_path(trace_dir).write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _trace_meta_matches(
    meta: dict[str, object] | None,
    *,
    experiment: str,
    model_selector: str,
    max_steps: int,
    trace_tag: str,
    required_seeds: set[int],
) -> bool:
    if not isinstance(meta, dict):
        return False
    if str(meta.get("experiment", "")) != str(experiment):
        return False
    if str(meta.get("model_selector", "")) != str(model_selector):
        return False
    if str(meta.get("trace_tag", "")) != str(trace_tag):
        return False
    try:
        meta_max_steps = int(meta.get("max_steps", -1))
    except Exception:
        return False
    # Reuse only when existing traces were generated with at least the requested step cap.
    if meta_max_steps < int(max_steps):
        return False
    seeds_raw = meta.get("seeds")
    if not isinstance(seeds_raw, list):
        return False
    try:
        meta_seeds = {int(s) for s in seeds_raw}
    except Exception:
        return False
    return bool(required_seeds.issubset(meta_seeds))


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_csv(str(args.seeds)) if str(args.seeds).strip() else _seeds_from_worst(Path(args.worst_json), int(args.top_n))
    if not seeds:
        seeds = _seeds_from_latest_summary(Path(args.latest_summary), int(args.top_n))
    if not seeds:
        raise SystemExit("No seeds provided (use --seeds or a valid --worst-json).")
    if bool(args.reuse_latest_traces):
        trace_root = Path(args.out_dir) / "focused_traces"
        latest = _latest_trace_dir(trace_root, trace_tag=str(args.trace_tag))
        if latest is not None:
            need = set(int(s) for s in seeds)
            meta = _read_trace_meta(latest)
            if _trace_meta_matches(
                meta,
                experiment=str(args.experiment),
                model_selector=str(args.model_selector),
                max_steps=int(args.max_steps),
                trace_tag=str(args.trace_tag),
                required_seeds=need,
            ):
                have = _seed_set_from_trace_dir(latest)
                if need.issubset(have):
                    print(f"Reusing existing focused traces: {latest}")
                    return
            else:
                # Safe fallback: if metadata is missing/mismatched, force recompute.
                pass

    settings = Settings()
    if bool(args.enable_risk_switch_guard) or bool(args.risk_guard_allow_narrow_corridor):
        settings.dynamic_control = replace(
            settings.dynamic_control,
            enable_risk_switch_guard=bool(args.enable_risk_switch_guard),
            risk_switch_guard_allow_narrow_corridor=bool(args.risk_guard_allow_narrow_corridor),
            risk_switch_guard_narrow_min_no_progress_steps=max(0, int(args.risk_guard_narrow_min_no_progress_steps)),
            risk_switch_guard_narrow_confidence_min=float(args.risk_guard_narrow_confidence_min),
            risk_switch_guard_narrow_no_progress_margin=max(0, int(args.risk_guard_narrow_no_progress_margin)),
        )
    pygame.init()
    try:
        display_flags = 0
        if hasattr(pygame, "HIDDEN"):
            display_flags |= int(getattr(pygame, "HIDDEN"))
        pygame.display.set_mode((1, 1), display_flags)
        font = pygame.font.SysFont("Arial", 20, bold=True)
        small_font = pygame.font.SysFont("Arial", 16)
        runtime = build_runtime(
            settings=settings,
            font=font,
            small_font=small_font,
            on_score=lambda _score: None,
            state_dir=Path(args.state_dir),
            experiment_name=str(args.experiment),
            ppo_config=ppo_profile_config("app", seed=1337),
            reward_config=RewardConfig(),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True, use_free_space_features=True, use_tail_trend_features=True),
        )
        out_dir = Path(args.out_dir)
        holdout = HoldoutEvalController(
            agent=runtime.agent,
            settings=settings,
            obs_config=runtime.obs_config,
            reward_config=runtime.reward_config,
            out_dir=out_dir,
        )
        if not holdout.start(
            mode=HoldoutEvalController.MODE_CONTROLLER_ON,
            model_selector=str(args.model_selector),
            seeds=seeds,
            max_steps=int(args.max_steps),
            trace_enabled=True,
            trace_tag=str(args.trace_tag),
        ):
            snap = holdout.snapshot()
            raise SystemExit(f"Holdout trace did not start: {snap.last_error}")
        deadline = time.time() + 7200.0
        completion_msg: str | None = None
        while time.time() < deadline:
            msg = holdout.poll_completion()
            if msg is not None:
                completion_msg = str(msg)
                print(completion_msg)
                break
            time.sleep(0.05)
        else:
            raise SystemExit("Timed out waiting for focused controller trace completion.")
        if completion_msg is None:
            raise SystemExit("Holdout trace did not emit a completion message.")
        if "failed" in completion_msg.lower():
            raise SystemExit(completion_msg)
        latest = str(holdout.snapshot().latest_summary_path or "").strip()
        if not latest:
            raise SystemExit("Holdout trace finished without writing latest summary.")
        print(f"latest_summary={latest}")
        trace_root = Path(args.out_dir) / "focused_traces"
        latest_trace = _latest_trace_dir(trace_root, trace_tag=str(args.trace_tag))
        if latest_trace is not None:
            _write_trace_meta(
                latest_trace,
                experiment=str(args.experiment),
                model_selector=str(args.model_selector),
                max_steps=int(args.max_steps),
                seeds=[int(s) for s in seeds],
                trace_tag=str(args.trace_tag),
            )
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
