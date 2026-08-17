"""
Theme System — Base Dataclass
==============================
All theme values are defined as pure Python dataclasses.
No Qt import here — themes are loaded before Qt starts.
Qt widgets read theme values directly and apply them.

Color format: CSS hex strings (#RRGGBB or #AARRGGBB)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThemeColors:
    """All color tokens for a single theme variant."""

    # Background layers
    bg_primary: str          # Main overlay background (semi-transparent)
    bg_secondary: str        # Chat area, panels
    bg_tertiary: str         # Input field, hover states
    bg_glass: str            # Glassmorphism tint (ARGB with alpha)

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str
    text_on_accent: str      # Text on colored buttons/bubbles

    # Accent
    accent_primary: str      # Spidy brand color (purple)
    accent_secondary: str    # Secondary accent (blue/teal)
    accent_success: str
    accent_warning: str
    accent_error: str

    # Chat bubbles
    bubble_user: str         # User message bubble background
    bubble_assistant: str    # Spidy message bubble background
    bubble_user_text: str
    bubble_assistant_text: str

    # UI elements
    border: str              # Subtle border/separator
    border_active: str       # Active/focused border
    scrollbar: str
    scrollbar_hover: str

    # State indicators
    mic_idle: str
    mic_listening: str       # Red when recording
    waveform_bar: str        # Waveform bar color
    thinking_dot: str        # Three-dot spinner color

    # Wake-ready edge glow
    wake_glow: str           # Animated glow color (for future wake-word)

    # Autonomous working state glow
    working_glow: str        # Gold/amber glow for WORKING state

    # Notification levels
    notify_info: str
    notify_success: str
    notify_warning: str
    notify_error: str

    # System tray
    tray_icon_bg: str

    # ── HUD / Futuristic Interface (M17.2) ─────────────────────────────────
    hud_primary: str     # Luminous cyan — main HUD accent
    hud_secondary: str   # Electric teal — secondary HUD elements
    hud_dim: str         # Dim cyan — inactive HUD elements
    hud_grid: str        # Near-black blue — HUD background/grid
    hud_text: str        # Bright HUD readout text (slightly off-white cyan)


@dataclass(frozen=True)
class ThemeFonts:
    """Font family and size tokens."""
    family_primary: str = "Segoe UI"
    family_mono: str = "Consolas"
    size_xs: int = 10
    size_sm: int = 12
    size_base: int = 13
    size_md: int = 14
    size_lg: int = 16
    size_xl: int = 20
    size_title: int = 24


@dataclass(frozen=True)
class ThemeGeometry:
    """Layout and sizing constants."""
    border_radius: int = 18
    border_radius_sm: int = 10
    border_radius_lg: int = 24
    border_width: int = 1
    padding: int = 16
    padding_sm: int = 8
    padding_lg: int = 24
    bubble_radius: int = 14
    margin_edge: int = 20      # Pixels from screen edge


@dataclass(frozen=True)
class Theme:
    """
    Complete visual theme for the Spidy overlay.

    Parameters
    ----------
    name:
        Theme identifier ("dark" | "light").
    display_name:
        Human-readable name shown in settings.
    colors:
        Full color token set.
    fonts:
        Font family and size definitions.
    geometry:
        Layout constants.
    overlay_opacity:
        Overall window transparency (0.0–1.0).
    blur_enabled:
        Whether to request Windows acrylic blur.
    """

    name: str
    display_name: str
    colors: ThemeColors
    fonts: ThemeFonts = field(default_factory=ThemeFonts)
    geometry: ThemeGeometry = field(default_factory=ThemeGeometry)
    overlay_opacity: float = 0.92
    blur_enabled: bool = True


class ThemeManager:
    """
    Manages theme selection and switching.

    Holds a registry of available themes and the currently active one.
    Notifies listeners when the theme changes (used to re-style all widgets).
    """

    def __init__(self) -> None:
        self._themes: dict[str, Theme] = {}
        self._current: Theme | None = None
        self._listeners: list = []

    def register(self, theme: Theme) -> None:
        """Register a theme in the registry."""
        self._themes[theme.name] = theme
        if self._current is None:
            self._current = theme

    def set_theme(self, name: str) -> Theme:
        """
        Switch to a named theme.

        Raises
        ------
        KeyError
            If the theme name is not registered.
        """
        if name not in self._themes:
            raise KeyError(
                f"Theme '{name}' not registered. "
                f"Available: {list(self._themes)}"
            )
        old = self._current
        self._current = self._themes[name]
        if old is not self._current:
            self._notify()
        return self._current

    @property
    def current(self) -> Theme:
        if self._current is None:
            raise RuntimeError("No theme registered. Call register() first.")
        return self._current

    @property
    def available(self) -> list[str]:
        return list(self._themes)

    def add_listener(self, fn) -> None:
        """Register a ``fn(theme: Theme) -> None`` callback."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        self._listeners = [l for l in self._listeners if l is not fn]

    def _notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn(self._current)
            except Exception:  # noqa: BLE001
                pass

    def toggle(self) -> Theme:
        """Switch between dark and light themes."""
        available = self.available
        if len(available) < 2:
            return self.current
        current_idx = available.index(self._current.name) if self._current else 0
        next_idx = (current_idx + 1) % len(available)
        return self.set_theme(available[next_idx])
