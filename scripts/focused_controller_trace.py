from __future__ import annotations

import argparse
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
    parser.add_argument("--out-dir", type=str, default="artifacts/live_eval")
    parser.add_argument("--model-selector", choices=("best", "last"), default="last")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--worst-json", type=str, default="artifacts/live_eval/worst10_latest.json")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--trace-tag", type=str, default="worst10")
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


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_csv(str(args.seeds)) if str(args.seeds).strip() else _seeds_from_worst(Path(args.worst_json), int(args.top_n))
    if not seeds:
        raise SystemExit("No seeds provided (use --seeds or a valid --worst-json).")

    settings = Settings()
    pygame.init()
    try:
        pygame.display.set_mode((1, 1))
        font = pygame.font.SysFont("Arial", 20, bold=True)
        small_font = pygame.font.SysFont("Arial", 16)
        runtime = build_runtime(
            settings=settings,
            font=font,
            small_font=small_font,
            on_score=lambda _score: None,
            state_dir=Path(args.state_dir),
            ppo_config=ppo_profile_config("app", seed=1337),
            reward_config=RewardConfig(),
            obs_config=ObsConfig(use_extended_features=True, use_path_features=True, use_tail_path_features=True),
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
        while time.time() < deadline:
            msg = holdout.poll_completion()
            if msg is not None:
                print(msg)
                break
            time.sleep(0.05)
        else:
            raise SystemExit("Timed out waiting for focused controller trace completion.")
        latest = holdout.snapshot().latest_summary_path
        print(f"latest_summary={latest}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
