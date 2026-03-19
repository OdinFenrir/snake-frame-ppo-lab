from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
from typing import Literal
import webbrowser

import pygame

from .analysis_tool_catalog import ToolSpec, build_tools
from .analysis_tool_commands import build_tool_commands, list_experiments, project_root
from .analysis_tool_runner import pick_first_existing_output, read_output_preview, run_commands
from .theme import get_theme, normalize_theme_name

WelcomeRoute = Literal["live_training", "settings"]
ScreenState = Literal["menu", "tools", "viewer"]


def _project_root() -> Path:
    return project_root()


def _load_saved_theme_name() -> str:
    prefs_path = _project_root() / "state" / "ui_prefs.json"
    if not prefs_path.exists():
        return "retro_forest_noir"
    try:
        payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    except Exception:
        return "retro_forest_noir"
    if not isinstance(payload, dict):
        return "retro_forest_noir"
    return normalize_theme_name(str(payload.get("themeName", "retro_forest_noir")))


def _draw_main_style_background(surface: pygame.Surface, *, panel_bg: tuple[int, int, int], panel_bg_accent: tuple[int, int, int], surface_bg: tuple[int, int, int]) -> None:
    width, height = surface.get_size()
    surface.fill(surface_bg)
    h = max(1, int(height))
    for y in range(0, h, 4):
        t = float(y) / float(max(1, h - 1))
        shade = (
            int(panel_bg_accent[0] * (1.0 - t) + panel_bg[0] * t),
            int(panel_bg_accent[1] * (1.0 - t) + panel_bg[1] * t),
            int(panel_bg_accent[2] * (1.0 - t) + panel_bg[2] * t),
        )
        pygame.draw.line(surface, shade, (0, y), (width, y), 1)


def _shade(color: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return (
        max(0, min(255, color[0] + delta)),
        max(0, min(255, color[1] + delta)),
        max(0, min(255, color[2] + delta)),
    )


def _draw_box(surface: pygame.Surface, rect: pygame.Rect, *, bg: tuple[int, int, int], border: tuple[int, int, int], width: int = 1, radius: int = 12) -> None:
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def _fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    if max_width <= 8:
        return ""
    if font.size(text)[0] <= max_width:
        return text
    out = text
    while len(out) > 2 and font.size(out + "…")[0] > max_width:
        out = out[:-1]
    return out + "…"


def _open_html_in_browser(path: Path) -> None:
    if path.suffix.lower() != ".html":
        return
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception:
        # Browser launch failure should not break in-app viewer flow.
        pass


def _copy_text_to_clipboard(text: str) -> bool:
    payload = text or ""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(payload)
        root.update()
        root.destroy()
        return True
    except Exception:
        try:
            proc = subprocess.run(["clip"], input=payload, text=True, capture_output=True, check=False)
            return proc.returncode == 0
        except Exception:
            return False


def show_welcome_window() -> WelcomeRoute | None:
    pygame.init()
    info = pygame.display.Info()
    init_w = max(960, min(1400, int(info.current_w * 0.74)))
    init_h = max(640, min(920, int(info.current_h * 0.76)))
    surface = pygame.display.set_mode((init_w, init_h), pygame.RESIZABLE)
    pygame.display.set_caption("Snake Frame - Workspace")

    root = _project_root()
    theme = get_theme(_load_saved_theme_name())
    experiments = list_experiments(root)
    left_idx = 0
    right_idx = 1 if len(experiments) > 1 else 0

    screen_state: ScreenState = "menu"
    selected_route: WelcomeRoute | None = None
    selected_tool_idx = 0
    viewer_scroll = 0
    viewer_text = ""
    viewer_title = ""
    status_text = "Ready"
    mouse_down_y = None
    running = True

    worker: threading.Thread | None = None
    worker_lock = threading.Lock()
    worker_running = False
    worker_result = ""

    def _launch_tool(spec: ToolSpec) -> None:
        nonlocal worker, worker_running, status_text, worker_result, viewer_text, viewer_title, screen_state, viewer_scroll
        if worker_running:
            status_text = "A tool is already running."
            return

        def _run() -> None:
            nonlocal worker_running, status_text, worker_result, viewer_text, viewer_title, screen_state, viewer_scroll
            try:
                cmds = build_tool_commands(spec, left_exp=left_exp, right_exp=right_exp)
                merged = run_commands(cmds, root=root, timeout_s=60 * 60)
                with worker_lock:
                    worker_result = merged
                    worker_running = False
                    status_text = f"Finished: {spec.label}"
            except Exception as exc:
                with worker_lock:
                    worker_result = f"Tool failed: {exc}"
                    worker_running = False
                    status_text = f"Failed: {spec.label}"

            # Auto-load latest embeddable output into viewer after completion.
            output_path = pick_first_existing_output(root, spec.outputs)
            if output_path is not None:
                if output_path.suffix.lower() == ".html":
                    with worker_lock:
                        status_text = f"Opening {spec.label} in browser (loading...)"
                    _open_html_in_browser(output_path)
                    with worker_lock:
                        status_text = f"{spec.label} opened. Browser may still be loading large data."
                viewer_title = f"{spec.label} - {output_path.name}"
                viewer_text = read_output_preview(output_path)
                viewer_scroll = 0
                screen_state = "viewer"
            else:
                viewer_title = f"{spec.label} - Run Result"
                viewer_text = worker_result if worker_result else "Tool finished with no output file."
                viewer_scroll = 0
                screen_state = "viewer"
                if spec.key == "netron":
                    with worker_lock:
                        status_text = "Netron launched. Browser/model graph may still be loading."

        with worker_lock:
            worker_running = True
            worker_result = ""
            if spec.key in ("policy_3d", "netron"):
                status_text = f"Running: {spec.label} (building/opening, may take a while)"
            else:
                status_text = f"Running: {spec.label}"
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

    clock = pygame.time.Clock()
    while running:
        win_w, win_h = surface.get_size()
        mouse_pos = pygame.mouse.get_pos()
        title_font = pygame.font.SysFont("Segoe UI", max(30, min(50, int(win_h * 0.074))), bold=True)
        sub_font = pygame.font.SysFont("Segoe UI", max(18, min(30, int(win_h * 0.040))), bold=True)
        item_font = pygame.font.SysFont("Segoe UI", max(16, min(22, int(win_h * 0.030))), bold=True)
        body_font = pygame.font.SysFont("Segoe UI", max(14, min(19, int(win_h * 0.025))), bold=False)
        small_font = pygame.font.SysFont("Segoe UI", max(12, min(16, int(win_h * 0.021))), bold=False)
        cat_font = pygame.font.SysFont("Segoe UI", max(18, min(26, int(win_h * 0.032))), bold=True)
        sel_name_font = pygame.font.SysFont("Segoe UI", max(13, min(18, int(win_h * 0.023))), bold=True)
        sel_label_font = pygame.font.SysFont("Segoe UI", max(13, min(18, int(win_h * 0.022))), bold=True)
        action_btn_font = pygame.font.SysFont("Segoe UI", max(14, min(18, int(win_h * 0.023))), bold=True)

        left_exp = experiments[left_idx]
        right_exp = experiments[right_idx]
        tools = build_tools(left_exp=left_exp, right_exp=right_exp)
        if selected_tool_idx >= len(tools):
            selected_tool_idx = max(0, len(tools) - 1)
        active_tool = tools[selected_tool_idx]
        compare_mode = active_tool.key == "phase3_compare"

        # Precompute menu and tools hit regions for mouse interaction.
        menu_cards: list[tuple[pygame.Rect, str]] = []
        menu_card_w = max(500, min(int(win_w * 0.64), win_w - 60))
        menu_card_h = max(96, min(118, int(win_h * 0.145)))
        menu_gap = max(12, int(menu_card_h * 0.12))
        menu_x = (win_w - menu_card_w) // 2
        menu_y0 = max(172, int(win_h * 0.32))
        menu_cards.append((pygame.Rect(menu_x, menu_y0, menu_card_w, menu_card_h), "live_training"))
        menu_cards.append((pygame.Rect(menu_x, menu_y0 + menu_card_h + menu_gap, menu_card_w, menu_card_h), "analysis_tools"))
        menu_cards.append((pygame.Rect(menu_x, menu_y0 + (menu_card_h + menu_gap) * 2, menu_card_w, menu_card_h), "settings"))

        # Grid layout: title band + 2-column content band.
        margin = 22
        content_x = margin
        content_w = win_w - (margin * 2)
        gutter = 14
        left_w = max(360, int((content_w - gutter) * 0.52))
        right_w = max(320, content_w - gutter - left_w)

        title_y_probe = 14
        header_reserved_h = title_y_probe + title_font.get_height() + 14
        tools_left_rect = pygame.Rect(content_x, header_reserved_h, left_w, win_h - header_reserved_h - margin)
        tools_right_rect = pygame.Rect(tools_left_rect.right + gutter, tools_left_rect.y, right_w, tools_left_rect.height)

        selector_h = 30
        selector_panel_h = 100
        selector_panel_y = tools_right_rect.y + 8
        selector_row_y = selector_panel_y + 62
        selector_area_x = tools_right_rect.x + 12
        selector_area_w = tools_right_rect.width - 24
        selector_gap = 14
        if compare_mode:
            selector_block_w = max(190, int((selector_area_w - selector_gap) / 2))
            left_block_x = selector_area_x
            right_block_x = selector_area_x + selector_block_w + selector_gap
            tools_left_prev_btn = pygame.Rect(left_block_x, selector_row_y, 30, selector_h)
            tools_left_next_btn = pygame.Rect(left_block_x + selector_block_w - 30, selector_row_y, 30, selector_h)
            tools_right_prev_btn = pygame.Rect(right_block_x, selector_row_y, 30, selector_h)
            tools_right_next_btn = pygame.Rect(right_block_x + selector_block_w - 30, selector_row_y, 30, selector_h)
            tools_left_name_rect = pygame.Rect(tools_left_prev_btn.right + 6, selector_row_y, selector_block_w - 72, selector_h)
            tools_right_name_rect = pygame.Rect(tools_right_prev_btn.right + 6, selector_row_y, selector_block_w - 72, selector_h)
        else:
            single_total_w = min(max(280, selector_area_w - 120), selector_area_w)
            single_x = selector_area_x + (selector_area_w - single_total_w) // 2
            tools_left_prev_btn = pygame.Rect(single_x, selector_row_y, 30, selector_h)
            tools_left_next_btn = pygame.Rect(single_x + single_total_w - 30, selector_row_y, 30, selector_h)
            tools_left_name_rect = pygame.Rect(tools_left_prev_btn.right + 6, selector_row_y, single_total_w - 72, selector_h)
            tools_right_prev_btn = pygame.Rect(-1000, -1000, 1, 1)
            tools_right_next_btn = pygame.Rect(-1000, -1000, 1, 1)
            tools_right_name_rect = pygame.Rect(-1000, -1000, 1, 1)

        tools_row_rects: list[pygame.Rect] = []
        row_h_probe = max(66, int(tools_left_rect.height * 0.11))
        y_probe = tools_left_rect.y + 10
        current_cat_probe = ""
        for idx_probe, spec_probe in enumerate(tools):
            if spec_probe.category != current_cat_probe:
                current_cat_probe = spec_probe.category
                y_probe += cat_font.get_height() + 6
            row_probe = pygame.Rect(tools_left_rect.x + 8, y_probe, tools_left_rect.width - 16, row_h_probe)
            tools_row_rects.append(row_probe)
            y_probe += row_h_probe + 6
            if y_probe > tools_left_rect.bottom - row_h_probe:
                break

        tools_btn_h = 34
        tools_run_btn = pygame.Rect(tools_right_rect.x + 12, tools_right_rect.bottom - tools_btn_h - 12, 128, tools_btn_h)
        tools_view_btn = pygame.Rect(tools_run_btn.right + 10, tools_run_btn.y, 138, tools_btn_h)
        tools_back_btn = pygame.Rect(win_w - 108, 18, 86, 30)
        viewer_back_btn = pygame.Rect(win_w - 108, 20, 86, 30)
        viewer_copy_btn = pygame.Rect(viewer_back_btn.x - 102, 20, 92, 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                selected_route = None
                running = False
            elif event.type == pygame.VIDEORESIZE:
                surface = pygame.display.set_mode((max(920, int(event.w)), max(620, int(event.h))), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEWHEEL and screen_state == "viewer":
                viewer_scroll = max(0, viewer_scroll - int(event.y) * 24)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if screen_state == "viewer" and event.button == 1:
                    mouse_down_y = int(event.pos[1])
                if screen_state == "viewer" and event.button == 4:
                    viewer_scroll = max(0, viewer_scroll - 28)
                if screen_state == "viewer" and event.button == 5:
                    viewer_scroll = max(0, viewer_scroll + 28)
                if screen_state == "menu" and event.button == 1:
                    for rect, route in menu_cards:
                        if rect.collidepoint(event.pos):
                            if route == "analysis_tools":
                                screen_state = "tools"
                            elif route == "live_training":
                                selected_route = "live_training"
                                running = False
                            elif route == "settings":
                                selected_route = "settings"
                                running = False
                            break
                if screen_state == "tools" and event.button == 1:
                    for idx_row, row in enumerate(tools_row_rects):
                        if row.collidepoint(event.pos):
                            selected_tool_idx = idx_row
                            break
                    if tools_back_btn.collidepoint(event.pos):
                        screen_state = "menu"
                    elif tools_run_btn.collidepoint(event.pos):
                        _launch_tool(tools[selected_tool_idx])
                    elif tools_view_btn.collidepoint(event.pos):
                        spec = tools[selected_tool_idx]
                        out_path = pick_first_existing_output(root, spec.outputs)
                        if out_path is not None:
                            if out_path.suffix.lower() == ".html":
                                status_text = f"Opening {spec.label} in browser (loading...)"
                                _open_html_in_browser(out_path)
                                status_text = f"{spec.label} opened. Browser may still be loading large data."
                            viewer_title = f"{spec.label} - {out_path.name}"
                            viewer_text = read_output_preview(out_path)
                            viewer_scroll = 0
                            screen_state = "viewer"
                        else:
                            viewer_title = spec.label
                            viewer_text = "No output file found yet.\nRun the tool first."
                            viewer_scroll = 0
                            screen_state = "viewer"
                    elif tools_left_prev_btn.collidepoint(event.pos):
                        left_idx = (left_idx - 1) % len(experiments)
                    elif tools_left_next_btn.collidepoint(event.pos):
                        left_idx = (left_idx + 1) % len(experiments)
                    elif compare_mode and tools_right_prev_btn.collidepoint(event.pos):
                        right_idx = (right_idx - 1) % len(experiments)
                    elif compare_mode and tools_right_next_btn.collidepoint(event.pos):
                        right_idx = (right_idx + 1) % len(experiments)
                if screen_state == "viewer" and event.button == 1 and viewer_back_btn.collidepoint(event.pos):
                    screen_state = "tools"
                elif screen_state == "viewer" and event.button == 1 and viewer_copy_btn.collidepoint(event.pos):
                    ok = _copy_text_to_clipboard(viewer_text)
                    status_text = "Copied viewer output to clipboard." if ok else "Copy failed."
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down_y = None
            elif event.type == pygame.MOUSEMOTION and screen_state == "viewer" and mouse_down_y is not None:
                dy = int(event.pos[1]) - int(mouse_down_y)
                viewer_scroll = max(0, viewer_scroll - dy)
                mouse_down_y = int(event.pos[1])
            elif event.type == pygame.KEYDOWN:
                if screen_state == "menu":
                    if event.key == pygame.K_ESCAPE:
                        selected_route = None
                        running = False
                    elif event.key in (pygame.K_1, pygame.K_RETURN, pygame.K_SPACE):
                        selected_route = "live_training"
                        running = False
                    elif event.key == pygame.K_2:
                        screen_state = "tools"
                    elif event.key == pygame.K_3:
                        selected_route = "settings"
                        running = False
                elif screen_state == "tools":
                    if event.key == pygame.K_ESCAPE:
                        screen_state = "menu"
                    elif event.key == pygame.K_UP:
                        selected_tool_idx = max(0, selected_tool_idx - 1)
                    elif event.key == pygame.K_DOWN:
                        selected_tool_idx = min(len(tools) - 1, selected_tool_idx + 1)
                    elif event.key == pygame.K_q:
                        left_idx = (left_idx - 1) % len(experiments)
                    elif event.key == pygame.K_a:
                        left_idx = (left_idx + 1) % len(experiments)
                    elif compare_mode and event.key == pygame.K_w:
                        right_idx = (right_idx - 1) % len(experiments)
                    elif compare_mode and event.key == pygame.K_s:
                        right_idx = (right_idx + 1) % len(experiments)
                    elif event.key == pygame.K_r:
                        _launch_tool(tools[selected_tool_idx])
                    elif event.key in (pygame.K_RETURN, pygame.K_v):
                        spec = tools[selected_tool_idx]
                        out_path = pick_first_existing_output(root, spec.outputs)
                        if out_path is not None:
                            if out_path.suffix.lower() == ".html":
                                status_text = f"Opening {spec.label} in browser (loading...)"
                                _open_html_in_browser(out_path)
                                status_text = f"{spec.label} opened. Browser may still be loading large data."
                            viewer_title = f"{spec.label} - {out_path.name}"
                            viewer_text = read_output_preview(out_path)
                            viewer_scroll = 0
                            screen_state = "viewer"
                        else:
                            viewer_title = spec.label
                            viewer_text = "No output file found yet.\nPress R to run tool first."
                            viewer_scroll = 0
                            screen_state = "viewer"
                elif screen_state == "viewer":
                    if event.key == pygame.K_ESCAPE:
                        screen_state = "tools"
                    elif event.key == pygame.K_c and (event.mod & pygame.KMOD_CTRL):
                        ok = _copy_text_to_clipboard(viewer_text)
                        status_text = "Copied viewer output to clipboard." if ok else "Copy failed."
                    elif event.key == pygame.K_UP:
                        viewer_scroll = max(0, viewer_scroll - 24)
                    elif event.key == pygame.K_DOWN:
                        viewer_scroll = max(0, viewer_scroll + 24)
                    elif event.key == pygame.K_PAGEUP:
                        viewer_scroll = max(0, viewer_scroll - int(win_h * 0.6))
                    elif event.key == pygame.K_PAGEDOWN:
                        viewer_scroll = max(0, viewer_scroll + int(win_h * 0.6))

        _draw_main_style_background(
            surface,
            panel_bg=theme.panel_bg,
            panel_bg_accent=theme.panel_bg_accent,
            surface_bg=theme.surface_bg,
        )

        if screen_state == "menu":
            title = title_font.render("Snake Frame", True, theme.title_color)
            subtitle = sub_font.render("Choose your workspace", True, theme.status_color)
            surface.blit(title, ((win_w - title.get_width()) // 2, max(46, int(win_h * 0.11))))
            surface.blit(subtitle, ((win_w - subtitle.get_width()) // 2, max(98, int(win_h * 0.19))))

            cards = [
                (menu_cards[0][0], "Live Training", "Open game and training controls"),
                (menu_cards[1][0], "Analysis Tools", "Open reports and diagnostics workspace"),
                (menu_cards[2][0], "Application Settings", "Open options and preferences"),
            ]
            for idx, (rect, label, sub) in enumerate(cards):
                hovered = rect.collidepoint(mouse_pos)
                _draw_box(
                    surface,
                    rect,
                    bg=_shade(theme.panel_bg, 10 if hovered else 2),
                    border=_shade(theme.panel_border, 28 if hovered else 0),
                    width=2 if hovered else 1,
                )
                label_surf = item_font.render(label, True, theme.section_header)
                sub_surf = body_font.render(_fit_text(body_font, sub, rect.width - 40), True, theme.status_secondary)
                surface.blit(label_surf, (rect.x + 18, rect.y + max(10, int(rect.height * 0.14))))
                surface.blit(sub_surf, (rect.x + 18, rect.y + max(46, int(rect.height * 0.47))))
                key = small_font.render(f"[{idx + 1}]", True, theme.badge_text)
                surface.blit(key, (rect.right - key.get_width() - 16, rect.y + 12))

        elif screen_state == "tools":
            title = title_font.render("Analysis Tools", True, theme.title_color)
            title_y = title_y_probe
            surface.blit(title, ((win_w - title.get_width()) // 2, title_y))
            back_hovered = tools_back_btn.collidepoint(mouse_pos)
            _draw_box(
                surface,
                tools_back_btn,
                bg=_shade(theme.panel_bg, 14 if back_hovered else 8),
                border=_shade(theme.panel_border, 24 if back_hovered else 0),
                width=2 if back_hovered else 1,
                radius=8,
            )
            back_txt = action_btn_font.render("Back", True, theme.badge_text)
            surface.blit(back_txt, (tools_back_btn.centerx - back_txt.get_width() // 2, tools_back_btn.y + (tools_back_btn.height - back_txt.get_height()) // 2))

            left_rect = tools_left_rect
            right_rect = tools_right_rect
            _draw_box(surface, left_rect, bg=_shade(theme.panel_bg, 2), border=theme.panel_border)
            _draw_box(surface, right_rect, bg=_shade(theme.graph_bg, -2), border=theme.panel_border)

            # Model selectors at top of right column.
            selector_panel = pygame.Rect(right_rect.x + 8, selector_panel_y, right_rect.width - 16, selector_panel_h)
            _draw_box(surface, selector_panel, bg=_shade(theme.panel_bg, 5), border=theme.panel_border, width=1, radius=8)
            sel_row_title = item_font.render("Model Compare" if compare_mode else "Model Selection", True, theme.section_header)
            surface.blit(sel_row_title, (selector_panel.centerx - sel_row_title.get_width() // 2, selector_panel.y + 6))
            left_lbl = sel_label_font.render("Model 1" if compare_mode else "Model", True, theme.section_header)
            surface.blit(left_lbl, (tools_left_name_rect.centerx - left_lbl.get_width() // 2, selector_panel.y + 34))
            if compare_mode:
                right_lbl = sel_label_font.render("Model 2", True, theme.section_header)
                surface.blit(right_lbl, (tools_right_name_rect.centerx - right_lbl.get_width() // 2, selector_panel.y + 34))
            _draw_box(surface, tools_left_name_rect, bg=_shade(theme.panel_bg, 4), border=theme.panel_border, width=1, radius=6)
            if compare_mode:
                _draw_box(surface, tools_right_name_rect, bg=_shade(theme.panel_bg, 4), border=theme.panel_border, width=1, radius=6)

            row_h = max(66, int(left_rect.height * 0.11))
            y = left_rect.y + 10
            current_category = ""
            for idx, spec in enumerate(tools):
                if spec.category != current_category:
                    current_category = spec.category
                    cat = cat_font.render(current_category, True, theme.status_secondary)
                    surface.blit(cat, (left_rect.x + 12, y))
                    y += cat_font.get_height() + 6
                row = pygame.Rect(left_rect.x + 8, y, left_rect.width - 16, row_h)
                active = idx == selected_tool_idx
                hovered = row.collidepoint(mouse_pos)
                _draw_box(
                    surface,
                    row,
                    bg=_shade(theme.panel_bg, 12 if hovered else (8 if active else 0)),
                    border=_shade(theme.panel_border, 32 if hovered else (24 if active else 0)),
                    width=2 if (active or hovered) else 1,
                    radius=8,
                )
                name = item_font.render(_fit_text(item_font, spec.label, row.width - 24), True, theme.section_header)
                desc = small_font.render(_fit_text(small_font, spec.description, row.width - 24), True, theme.status_color)
                surface.blit(name, (row.x + 12, row.y + 9))
                surface.blit(desc, (row.x + 12, row.y + 37))
                y += row_h + 6
                if y > left_rect.bottom - row_h:
                    break

            sel = tools[selected_tool_idx]
            detail_y = selector_panel.bottom + 12
            title2 = item_font.render(_fit_text(item_font, sel.label, right_rect.width - 24), True, theme.section_header)
            desc2 = body_font.render(_fit_text(body_font, sel.description, right_rect.width - 24), True, theme.status_color)
            surface.blit(title2, (right_rect.x + 12, detail_y))
            surface.blit(desc2, (right_rect.x + 12, detail_y + 26))

            status = small_font.render(status_text, True, theme.badge_text if not worker_running else theme.banner_warn)
            surface.blit(status, (right_rect.x + 12, detail_y + 52))
            emb = "Embeddable output: yes" if sel.embeddable else "Embeddable output: limited (external viewer preferred)"
            emb_surf = small_font.render(emb, True, theme.status_color)
            surface.blit(emb_surf, (right_rect.x + 12, detail_y + 74))

            try:
                resolved_cmds = build_tool_commands(sel, left_exp=left_exp, right_exp=right_exp)
            except Exception:
                resolved_cmds = []
            if resolved_cmds:
                cmd = " ; ".join(" ".join(step) for step in resolved_cmds[:2])
                if len(resolved_cmds) > 2:
                    cmd += f" ; +{len(resolved_cmds)-2} step(s)"
            else:
                cmd = "(no command)"
            cmd_surf = small_font.render(_fit_text(small_font, f"Command: {cmd}", right_rect.width - 24), True, theme.status_color)
            surface.blit(cmd_surf, (right_rect.x + 12, detail_y + 98))

            outputs_title = small_font.render("Expected outputs:", True, theme.status_color)
            surface.blit(outputs_title, (right_rect.x + 12, detail_y + 128))
            oy = detail_y + 150
            for rel in sel.outputs:
                p = root / rel
                prefix = "OK " if p.exists() else ".. "
                out_surf = small_font.render(_fit_text(small_font, f"{prefix}{rel}", right_rect.width - 24), True, theme.status_color)
                surface.blit(out_surf, (right_rect.x + 12, oy))
                oy += 20

            if worker_result:
                preview_title = small_font.render("Last console output (preview):", True, theme.status_color)
                surface.blit(preview_title, (right_rect.x + 12, oy + 8))
                log_lines = worker_result.splitlines()[:14]
                py = oy + 30
                for line in log_lines:
                    line_surf = small_font.render(_fit_text(small_font, line, right_rect.width - 24), True, theme.status_color)
                    surface.blit(line_surf, (right_rect.x + 12, py))
                    py += 18

            # Mouse-friendly controls.
            run_hovered = tools_run_btn.collidepoint(mouse_pos)
            view_hovered = tools_view_btn.collidepoint(mouse_pos)
            _draw_box(
                surface,
                tools_run_btn,
                bg=_shade(theme.toggle_positive_bg, 10 if run_hovered else 2),
                border=_shade(theme.panel_border, 24 if run_hovered else 0),
                width=2 if run_hovered else 1,
                radius=8,
            )
            _draw_box(
                surface,
                tools_view_btn,
                bg=_shade(theme.toggle_info_bg, 10 if view_hovered else 2),
                border=_shade(theme.panel_border, 24 if view_hovered else 0),
                width=2 if view_hovered else 1,
                radius=8,
            )
            run_lbl = action_btn_font.render("Run", True, theme.badge_text)
            view_lbl = action_btn_font.render("View Latest", True, theme.badge_text)
            surface.blit(run_lbl, (tools_run_btn.centerx - run_lbl.get_width() // 2, tools_run_btn.y + (tools_run_btn.height - run_lbl.get_height()) // 2))
            surface.blit(view_lbl, (tools_view_btn.centerx - view_lbl.get_width() // 2, tools_view_btn.y + (tools_view_btn.height - view_lbl.get_height()) // 2))

            selector_buttons = [
                (tools_left_prev_btn, "<"),
                (tools_left_next_btn, ">"),
            ]
            if compare_mode:
                selector_buttons.extend(
                    [
                        (tools_right_prev_btn, "<"),
                        (tools_right_next_btn, ">"),
                    ]
                )
            for btn, txt in selector_buttons:
                hovered = btn.collidepoint(mouse_pos)
                _draw_box(
                    surface,
                    btn,
                    bg=_shade(theme.panel_bg, 14 if hovered else 8),
                    border=_shade(theme.panel_border, 24 if hovered else 0),
                    width=2 if hovered else 1,
                    radius=6,
                )
                t = small_font.render(txt, True, theme.badge_text)
                surface.blit(t, (btn.centerx - t.get_width() // 2, btn.y + 4))

            ltxt = sel_name_font.render(_fit_text(sel_name_font, left_exp, tools_left_name_rect.width - 8), True, theme.section_header)
            surface.blit(ltxt, (tools_left_name_rect.centerx - ltxt.get_width() // 2, tools_left_name_rect.y + 4))
            if compare_mode:
                rtxt = sel_name_font.render(_fit_text(sel_name_font, right_exp, tools_right_name_rect.width - 8), True, theme.section_header)
                surface.blit(rtxt, (tools_right_name_rect.centerx - rtxt.get_width() // 2, tools_right_name_rect.y + 4))

        else:  # viewer
            header_rect = pygame.Rect(18, 16, win_w - 36, 54)
            body_rect = pygame.Rect(18, 78, win_w - 36, win_h - 96)
            _draw_box(surface, header_rect, bg=theme.panel_bg, border=theme.panel_border, width=1, radius=10)
            _draw_box(surface, body_rect, bg=theme.graph_bg, border=theme.panel_border, width=1, radius=10)

            t = item_font.render(viewer_title or "Tool Output", True, theme.section_header)
            surface.blit(t, (header_rect.x + 12, header_rect.y + 10))
            hint_x_max = viewer_copy_btn.x - 10
            hint_text = _fit_text(small_font, "Esc back  Scroll mouse / PgUp/PgDn  Ctrl+C copy", max(80, hint_x_max - (header_rect.x + 260)))
            hint = small_font.render(hint_text, True, theme.status_secondary)
            surface.blit(hint, (hint_x_max - hint.get_width(), header_rect.y + 16))

            copy_hovered = viewer_copy_btn.collidepoint(mouse_pos)
            _draw_box(
                surface,
                viewer_copy_btn,
                bg=_shade(theme.panel_bg, 14 if copy_hovered else 8),
                border=_shade(theme.panel_border, 24 if copy_hovered else 0),
                width=2 if copy_hovered else 1,
                radius=8,
            )
            copy_txt = action_btn_font.render("Copy All", True, theme.badge_text)
            surface.blit(copy_txt, (viewer_copy_btn.centerx - copy_txt.get_width() // 2, viewer_copy_btn.y + (viewer_copy_btn.height - copy_txt.get_height()) // 2))

            back_hovered = viewer_back_btn.collidepoint(mouse_pos)
            _draw_box(
                surface,
                viewer_back_btn,
                bg=_shade(theme.panel_bg, 14 if back_hovered else 8),
                border=_shade(theme.panel_border, 24 if back_hovered else 0),
                width=2 if back_hovered else 1,
                radius=8,
            )
            back_txt = action_btn_font.render("Back", True, theme.badge_text)
            surface.blit(back_txt, (viewer_back_btn.centerx - back_txt.get_width() // 2, viewer_back_btn.y + (viewer_back_btn.height - back_txt.get_height()) // 2))

            lines = viewer_text.splitlines() if viewer_text else ["No output loaded."]
            line_h = max(16, int(body_font.get_height() + 4))
            visible_h = body_rect.height - 20
            total_h = max(0, len(lines) * line_h)
            max_scroll = max(0, total_h - visible_h)
            viewer_scroll = max(0, min(viewer_scroll, max_scroll))
            start_line = int(viewer_scroll // line_h)
            y = body_rect.y + 10 - int(viewer_scroll % line_h)
            max_lines = int(visible_h / line_h) + 3
            for i in range(start_line, min(len(lines), start_line + max_lines)):
                max_w = max(120, body_rect.width - 34)
                fitted = _fit_text(body_font, lines[i], max_w)
                rendered = body_font.render(fitted, True, theme.status_color)
                surface.blit(rendered, (body_rect.x + 12, y))
                y += line_h

            if max_scroll > 0:
                bar_h = max(26, int((visible_h / max(total_h, 1)) * visible_h))
                bar_track = pygame.Rect(body_rect.right - 10, body_rect.y + 10, 4, visible_h)
                bar_y = bar_track.y + int((viewer_scroll / max_scroll) * max(1, bar_track.height - bar_h))
                bar = pygame.Rect(bar_track.x, bar_y, bar_track.width, bar_h)
                pygame.draw.rect(surface, _shade(theme.panel_border, -16), bar_track, border_radius=2)
                pygame.draw.rect(surface, theme.panel_border, bar, border_radius=2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return selected_route
