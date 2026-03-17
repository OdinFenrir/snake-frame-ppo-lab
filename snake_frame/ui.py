"""
UI components for the Snake Frame application.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass
class Button:
    """
    A clickable button UI component.
    
    Attributes:
        label: Text displayed on the button
        rect: Position and dimensions of the button
        bg: Background color (RGB)
        bg_hover: Background color when hovered (RGB)
        fg: Foreground/text color (RGB)
        enabled: Whether the button is interactive
    """
    label: str
    rect: pygame.Rect
    bg: tuple[int, int, int] = (35, 89, 116)
    bg_hover: tuple[int, int, int] = (48, 120, 153)
    fg: tuple[int, int, int] = (236, 244, 248)
    enabled: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, mouse_pos: tuple[int, int]) -> None:
        """
        Draw the button on the given surface.
        
        Args:
            surface: pygame surface to draw on
            font: font to use for text rendering
            mouse_pos: current mouse position for hover detection
        """
        hover = bool(self.enabled) and self.rect.collidepoint(mouse_pos)
        color = self.bg_hover if hover else self.bg
        if not self.enabled:
            color = tuple(max(0, int(v * 0.45)) for v in color)
        shadow_rect = self.rect.move(0, 2)
        pygame.draw.rect(surface, (6, 10, 14), shadow_rect, border_radius=8)
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        highlight_rect = pygame.Rect(self.rect.x + 1, self.rect.y + 1, self.rect.width - 2, max(2, self.rect.height // 3))
        highlight = tuple(min(255, int(c + 26)) for c in color)
        pygame.draw.rect(surface, highlight, highlight_rect, border_radius=7)
        pygame.draw.rect(surface, (88, 122, 148), self.rect, width=1, border_radius=8)
        text_color = self.fg if self.enabled else (160, 170, 180)
        text = _safe_render(font, self.label, text_color)
        text_rect = text.get_rect(center=self.rect.center)
        shadow = _safe_render(font, self.label, (14, 22, 30))
        surface.blit(shadow, text_rect.move(0, 1))
        surface.blit(text, text_rect)

    def clicked(self, event: pygame.event.Event) -> bool:
        """
        Check if the button was clicked based on a pygame event.
        
        Args:
            event: pygame event to check
            
        Returns:
            True if the button was clicked, False otherwise
        """
        return bool(
            self.enabled
            and
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class NumericInput:
    """
    A numeric input field UI component.
    
    Attributes:
        rect: Position and dimensions of the input field
        value: Current string value in the input
        active: Whether the input is currently focused/editing
        max_len: Maximum allowed length of the input value
    """
    def __init__(self, rect: pygame.Rect, value: str = "100", max_len: int = 9) -> None:
        """
        Initialize a new numeric input.
        
        Args:
            rect: Position and dimensions of the input field
            value: Initial string value (default: "100")
            max_len: Maximum allowed length of input (default: 9)
        """
        self.rect = rect
        self.value = value
        self.active = False
        self.max_len = max(1, int(max_len))

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Handle pygame events for the numeric input.
        
        Args:
            event: pygame event to process
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
            return
        if not self.active or event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self.active = False
            return
        if event.key == pygame.K_BACKSPACE:
            self.value = self.value[:-1]
            return
        if event.unicode.isdigit() and len(self.value) < self.max_len:
            self.value += event.unicode

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Draw the numeric input on the given surface.
        
        Args:
            surface: pygame surface to draw on
            font: font to use for text rendering
        """
        border = (96, 176, 230) if self.active else (76, 106, 130)
        pygame.draw.rect(surface, (8, 16, 26), self.rect, border_radius=6)
        top = pygame.Rect(self.rect.x + 1, self.rect.y + 1, self.rect.width - 2, max(2, self.rect.height // 3))
        pygame.draw.rect(surface, (16, 30, 44), top, border_radius=5)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=6)
        shadow = _safe_render(font, self.value or "0", (10, 16, 24))
        text = _safe_render(font, self.value or "0", (236, 247, 255))
        text_rect = text.get_rect(midleft=(self.rect.x + 10, self.rect.centery - 1))
        surface.blit(shadow, text_rect.move(0, 1))
        surface.blit(text, text_rect)

    def as_int(self, minimum: int = 1, maximum: int = 100000) -> int:
        """
        Convert the input value to an integer within specified bounds.
        
        Args:
            minimum: Minimum allowed value (default: 1)
            maximum: Maximum allowed value (default: 100000)
            
        Returns:
            Integer value clamped to [minimum, maximum]
        """
        try:
            value = int(self.value.strip() or "0")
        except ValueError:
            value = minimum
        return max(minimum, min(maximum, value))


def _safe_render(font: pygame.font.Font, text: str, color: tuple[int, int, int]) -> pygame.Surface:
    """
    Safely render text with a fallback font.
    
    Attempts to render text with the provided font, falling back to Arial
    if the font fails to render the text.
    
    Args:
        font: pygame Font to use for rendering
        text: Text string to render
        color: RGB color tuple for the text
        
    Returns:
        pygame Surface containing the rendered text
    """
    try:
        return font.render(str(text), True, color)
    except Exception:
        fallback = pygame.font.SysFont("Arial", 16, bold=True)
        return fallback.render(str(text), True, color)
