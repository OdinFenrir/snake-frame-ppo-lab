from __future__ import annotations

from dataclasses import dataclass

import pygame

from .settings import Settings


@dataclass(frozen=True)
class RightPanelLayout:
    """Fixed-grid layout for the right panel KPI dashboard."""
    # Top utility row
    utility_row_y: int
    utility_row_height: int
    
    # Training KPIs section
    training_header_y: int
    training_header_height: int
    training_badges_y: int
    training_badges_height: int
    training_graph_y: int
    
    # Run KPIs section  
    run_header_y: int
    run_header_height: int
    run_badges_y: int
    run_badges_height: int
    run_graph_y: int
    
    # Dimensions
    panel_width: int
    inner_x: int
    inner_width: int
    
    # Graph rectangles (calculated)
    training_graph_rect: pygame.Rect
    run_graph_rect: pygame.Rect


# Fixed layout constants for right panel
_RIGHT_PANEL_CONSTANTS = {
    "standard": {
        "utility_row_y": 0,
        "utility_row_height": 40,
        "header_to_badge_gap": 10,
        "badge_to_graph_gap": 10,
        "header_height": 24,
        "badges_height": 60,
        "section_gap": 20,
    },
}


def build_right_panel_layout(
    settings: Settings,
    *,
    panel_width: int | None = None,
    panel_x: int | None = None,
) -> RightPanelLayout:
    """Build a fixed-grid layout for the right panel."""
    p_width = int(panel_width if panel_width is not None else settings.right_panel_px)
    p_x = int(panel_x if panel_x is not None else settings.right_panel_offset_x)
    
    const = _RIGHT_PANEL_CONSTANTS["standard"]
    
    inner_margin = 18
    inner_x = p_x + inner_margin
    inner_width = p_width - (inner_margin * 2)
    
    # Fixed Y positions
    utility_y = const["utility_row_y"]
    utility_h = const["utility_row_height"]
    
    training_header_y = utility_y + utility_h + const["section_gap"]
    training_header_h = const["header_height"]
    
    training_badges_y = training_header_y + training_header_h + const["header_to_badge_gap"]
    training_badges_h = const["badges_height"]
    
    training_graph_y = training_badges_y + training_badges_h + const["badge_to_graph_gap"]
    
    # Run section starts after training graph
    # Calculate training graph height dynamically to fit window
    window_h = int(settings.window_height_px or settings.window_px)
    
    # Reserve space for run section
    run_header_y = training_graph_y + 200  # Reserve ~200 for training graph
    run_header_h = const["header_height"]
    
    run_badges_y = run_header_y + run_header_h + const["header_to_badge_gap"]
    run_badges_h = const["badges_height"]
    
    run_graph_y = run_badges_y + run_badges_h + const["badge_to_graph_gap"]
    
    # Calculate actual graph heights
    training_graph_h = max(150, run_header_y - training_graph_y - 10)
    run_graph_h = max(150, window_h - run_graph_y - 20)
    
    training_graph_rect = pygame.Rect(inner_x, training_graph_y, inner_width, training_graph_h)
    run_graph_rect = pygame.Rect(inner_x, run_graph_y, inner_width, run_graph_h)
    
    return RightPanelLayout(
        utility_row_y=utility_y,
        utility_row_height=utility_h,
        training_header_y=training_header_y,
        training_header_height=training_header_h,
        training_badges_y=training_badges_y,
        training_badges_height=training_badges_h,
        training_graph_y=training_graph_y,
        run_header_y=run_header_y,
        run_header_height=run_header_h,
        run_badges_y=run_badges_y,
        run_badges_height=run_badges_h,
        run_graph_y=run_graph_y,
        panel_width=p_width,
        inner_x=inner_x,
        inner_width=inner_width,
        training_graph_rect=training_graph_rect,
        run_graph_rect=run_graph_rect,
    )


@dataclass(frozen=True)
class PanelLayout:
    graph_rect: pygame.Rect
    controls_top: int
    x: int
    width: int
    half_width: int
    row_height: int
    gap: int


def build_panel_layout(
    settings: Settings,
    *,
    min_graph_height: int,
    max_graph_height: int,
    graph_margin: int,
    graph_top: int,
    control_row_height: int,
    control_gap: int,
    reserve_for_controls_and_status: int,
    panel_x: int | None = None,
    panel_width: int | None = None,
) -> PanelLayout:
    x0 = int(panel_x) if panel_x is not None else 0
    panel_w = int(panel_width) if panel_width is not None else int(settings.left_panel_px)
    usable_width = max(1, int(panel_w - (graph_margin * 2)))
    window_h = int(settings.window_height_px or settings.window_px)
    max_height_by_space = max(1, int(window_h - graph_top - graph_margin - reserve_for_controls_and_status))
    desired_min = max(1, int(min_graph_height))
    desired_max = max(desired_min, int(max_graph_height))
    if max_height_by_space < desired_min:
        graph_height = max_height_by_space
    else:
        graph_height = min(desired_max, max_height_by_space)
    controls_top = int(graph_top + graph_height + graph_margin)
    graph_rect = pygame.Rect(
        int(x0 + graph_margin),
        int(graph_top),
        usable_width,
        int(graph_height),
    )
    x = int(x0 + graph_margin)
    width = usable_width
    gap = int(control_gap)
    half_width = int((width - gap) // 2)
    return PanelLayout(
        graph_rect=graph_rect,
        controls_top=controls_top,
        x=x,
        width=width,
        half_width=half_width,
        row_height=int(control_row_height),
        gap=gap,
    )
