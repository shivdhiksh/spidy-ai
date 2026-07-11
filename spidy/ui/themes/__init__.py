"""spidy.ui.themes package — Theme registry and built-in themes."""

from spidy.ui.themes.base import Theme, ThemeColors, ThemeFonts, ThemeGeometry, ThemeManager
from spidy.ui.themes.dark import DARK_THEME
from spidy.ui.themes.light import LIGHT_THEME


def build_default_theme_manager(initial: str = "dark") -> ThemeManager:
    """
    Create a ThemeManager pre-loaded with the built-in dark and light themes.

    Parameters
    ----------
    initial:
        Name of the theme to activate first ("dark" | "light").
    """
    mgr = ThemeManager()
    mgr.register(DARK_THEME)
    mgr.register(LIGHT_THEME)
    mgr.set_theme(initial)
    return mgr


__all__ = [
    "Theme",
    "ThemeColors",
    "ThemeFonts",
    "ThemeGeometry",
    "ThemeManager",
    "DARK_THEME",
    "LIGHT_THEME",
    "build_default_theme_manager",
]
