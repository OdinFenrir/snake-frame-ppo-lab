from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pygame
import pytest

from snake_frame.controls_builder import build_controls
from snake_frame.layout_engine import LayoutEngine
from snake_frame.panel_ui import PanelRenderData, SidePanelsRenderer
from snake_frame.settings import Settings
from snake_frame.theme import get_design_tokens, get_theme
from snake_frame.game import SnakeGame

pytestmark = pytest.mark.render

_EXPECTED_HASHES = {
    "empty_default": "3dce7942a07095dbeea6642d8c76bf8ab8cf092e140dd828dae3bdcc3bc2e1d7",
    "training_populated": "27cbda54a1153bd02672f6f62e3717b65801d16f2bf5c04d08345479002dd03e",
    "run_populated": "c82f0aadf7239b8b9466cb5a58d3343fa36b48ce8ef669b2cfaffc611ddb551e",
    "small_window": "888100b8a339b91c30f21b39acf5ed3581d81ef6d0b52f7d37d6de5a544eaabd",
    "large_window": "060802028b4a168c4400b93b9a67a00554bf92513fdac191ff9eacedf7bcfff6",
}


def _surface_hash(surface: pygame.Surface) -> str:
    raw = pygame.image.tostring(surface, "RGB")
    return hashlib.sha256(raw).hexdigest()


_PINNED_FONT_PATH = Path(__file__).resolve().parents[1] / "external_assets" / "fonts" / "freesansbold.ttf"


def _render_view(*, width: int, height: int, training_scores: list[int], run_scores: list[int], theme_name: str) -> str:
    settings = Settings(theme_name=theme_name)
    layout = LayoutEngine(settings).update(width, height)
    compact = int(layout.window.height) < int(get_design_tokens(theme_name).spacing.graph_margin_compact_threshold)
    tokens = get_design_tokens(theme_name, compact=compact)
    theme = get_theme(theme_name)
    if not _PINNED_FONT_PATH.exists():
        raise AssertionError(f"Pinned render-regression font is missing: {_PINNED_FONT_PATH}")
    font = pygame.font.Font(str(_PINNED_FONT_PATH), tokens.typography.title_size)
    font.set_bold(bool(tokens.typography.title_bold))
    small = pygame.font.Font(str(_PINNED_FONT_PATH), tokens.typography.body_size)
    small.set_bold(bool(tokens.typography.body_bold))
    renderer = SidePanelsRenderer(settings=settings, font=font, small_font=small)
    game = SnakeGame(settings)
    game.snake = [(10, 10), (9, 10), (8, 10)]
    game.food = (14, 10)
    game.score = 2
    game.episode_scores = [int(v) for v in run_scores]
    controls = build_controls(
        settings,
        min_graph_height=layout.graph.min_graph_height,
        max_graph_height=layout.graph.max_graph_height,
        graph_margin=layout.graph.graph_margin,
        graph_top=layout.graph.graph_top,
        control_row_height=layout.graph.control_row_height,
        control_gap=layout.graph.control_gap,
        status_line_height=layout.graph.status_line_height,
        status_line_count=layout.graph.status_line_count,
    )
    surface = pygame.Surface((layout.window.width, layout.window.height))
    surface.fill(theme.surface_bg)
    renderer.draw(
        surface=surface,
        controls=controls.panel_controls,
        data=PanelRenderData(
            training_episode_scores=[int(v) for v in training_scores],
            run_episode_scores=[int(v) for v in run_scores],
            training_graph_rect=pygame.Rect(controls.training_graph_rect),
            run_graph_rect=pygame.Rect(controls.run_graph_rect),
            training_graph_badges=["Train 0/500000", f"Eps {len(training_scores)}", "Avg20 0.0", "Best 0", "Last 0"],
            run_graph_badges=["RunEps 0", "Avg20 0.0", "Best 0", "Last 0", "Intv 0.0%", "Risk 0"],
            run_status_lines=[
                "Algo: PPO (device=cpu)",
                "Model: ready",
                "Last train: in progress",
                "Game: running",
                "Control: agent",
                "Mode: ppo",
                "Switch: init",
            ],
            settings_lines=[
                "Board: 20x20 cell=54 tpm=5 fps=60",
                "Safety override: on",
                "Space strategy: on",
                "Theme: retro_forest_noir",
            ],
        ),
    )
    game.draw(surface, font)
    return _surface_hash(surface)


def test_render_regression_core_views() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        pygame.display.set_mode((1, 1))
        actual = {
            "empty_default": _render_view(width=1920, height=1080, training_scores=[], run_scores=[], theme_name="retro_forest_noir"),
            "training_populated": _render_view(width=1920, height=1080, training_scores=[1, 2, 0, 3, 4, 2, 5], run_scores=[], theme_name="retro_forest_noir"),
            "run_populated": _render_view(width=1920, height=1080, training_scores=[1, 2], run_scores=[0, 1, 2, 1, 3, 2], theme_name="retro_forest_noir"),
            "small_window": _render_view(width=1400, height=760, training_scores=[1, 2, 3], run_scores=[1, 0, 2], theme_name="retro_forest_noir"),
            "large_window": _render_view(width=2400, height=1360, training_scores=[1, 2, 3, 4, 5], run_scores=[1, 2, 0, 2, 4], theme_name="retro_forest_noir"),
        }
        missing = [k for k, v in _EXPECTED_HASHES.items() if not v]
        if missing:
            out = Path("tests") / "_render_hashes.new.txt"
            out.write_text("\n".join(f"{k}={actual[k]}" for k in actual), encoding="utf-8")
            raise AssertionError(f"Missing expected render hashes for: {missing}. Generated hashes at {out}")
        diffs = [name for name, expected in _EXPECTED_HASHES.items() if actual.get(name) != expected]
        if diffs:
            out = Path("tests") / "_render_hashes.diff.txt"
            out.write_text(
                "\n".join(f"{name} expected={_EXPECTED_HASHES[name]} actual={actual[name]}" for name in diffs),
                encoding="utf-8",
            )
            raise AssertionError(f"Render regression mismatch: {diffs}. See {out}")
    finally:
        pygame.quit()
