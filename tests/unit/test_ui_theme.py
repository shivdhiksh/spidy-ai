"""
Tests for Milestone 5 — Theme System
======================================
Tests for ThemeColors, Theme, ThemeManager, DARK_THEME, LIGHT_THEME.
Pure Python; no Qt display required.
"""

from __future__ import annotations

import pytest

from spidy.ui.themes.base import Theme, ThemeColors, ThemeFonts, ThemeGeometry, ThemeManager
from spidy.ui.themes.dark import DARK_THEME
from spidy.ui.themes.light import LIGHT_THEME
from spidy.ui.themes import build_default_theme_manager


def _is_valid_hex(color: str) -> bool:
    """Check a string looks like a CSS hex color (#RGB, #RRGGBB, or #AARRGGBB)."""
    if not color.startswith("#"):
        return False
    rest = color[1:]
    if len(rest) not in (3, 6, 8):
        return False
    try:
        int(rest, 16)
        return True
    except ValueError:
        return False


class TestThemeColors:
    def test_dark_theme_all_colors_valid_hex(self):
        c = DARK_THEME.colors
        for field in c.__dataclass_fields__:
            val = getattr(c, field)
            assert _is_valid_hex(val), f"Field '{field}' = {val!r} is not valid hex"

    def test_light_theme_all_colors_valid_hex(self):
        c = LIGHT_THEME.colors
        for field in c.__dataclass_fields__:
            val = getattr(c, field)
            assert _is_valid_hex(val), f"Field '{field}' = {val!r} is not valid hex"

    def test_dark_accent_is_purple(self):
        assert DARK_THEME.colors.accent_primary == "#6C63FF"

    def test_light_accent_is_darker_purple(self):
        # Light theme has slightly darker violet for contrast
        assert LIGHT_THEME.colors.accent_primary == "#5C54EF"

    def test_mic_listening_is_red_in_dark(self):
        # Red-ish color for listening mic
        assert DARK_THEME.colors.mic_listening.startswith("#FF") or \
               "red" in DARK_THEME.colors.mic_listening.lower()

    def test_dark_and_light_have_different_bg(self):
        assert DARK_THEME.colors.bg_primary != LIGHT_THEME.colors.bg_primary

    def test_dark_text_is_near_white(self):
        # text_primary in dark theme should be light
        hex_val = DARK_THEME.colors.text_primary.lstrip("#")
        r = int(hex_val[0:2], 16)
        assert r > 200, "Dark theme text should be near-white"

    def test_light_text_is_near_black(self):
        hex_val = LIGHT_THEME.colors.text_primary.lstrip("#")
        r = int(hex_val[0:2], 16)
        assert r < 50, "Light theme text should be near-black"


class TestThemeFonts:
    def test_dark_theme_fonts(self):
        f = DARK_THEME.fonts
        assert f.family_primary == "Segoe UI"
        assert f.size_base == 13
        assert f.size_sm < f.size_base < f.size_lg

    def test_light_theme_fonts(self):
        f = LIGHT_THEME.fonts
        assert f.family_primary == "Segoe UI"
        assert f.size_base == 13

    def test_font_size_hierarchy(self):
        f = DARK_THEME.fonts
        assert f.size_xs < f.size_sm < f.size_base < f.size_md < f.size_lg < f.size_xl < f.size_title

    def test_mono_font_different_from_primary(self):
        f = DARK_THEME.fonts
        assert f.family_mono != f.family_primary


class TestThemeGeometry:
    def test_dark_geometry_defaults(self):
        g = DARK_THEME.geometry
        assert g.border_radius > 0
        assert g.padding > 0
        assert g.margin_edge > 0

    def test_radius_hierarchy(self):
        g = DARK_THEME.geometry
        assert g.border_radius_sm <= g.border_radius <= g.border_radius_lg

    def test_padding_hierarchy(self):
        g = DARK_THEME.geometry
        assert g.padding_sm <= g.padding <= g.padding_lg

    def test_same_geometry_for_dark_and_light(self):
        # Both themes use the same geometry
        assert DARK_THEME.geometry.border_radius == LIGHT_THEME.geometry.border_radius


class TestThemeProperties:
    def test_dark_theme_name(self):
        assert DARK_THEME.name == "dark"

    def test_light_theme_name(self):
        assert LIGHT_THEME.name == "light"

    def test_dark_opacity_high(self):
        assert 0.8 < DARK_THEME.overlay_opacity <= 1.0

    def test_light_opacity_high(self):
        assert 0.8 < LIGHT_THEME.overlay_opacity <= 1.0

    def test_blur_enabled_by_default(self):
        assert DARK_THEME.blur_enabled is True
        assert LIGHT_THEME.blur_enabled is True

    def test_themes_are_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            DARK_THEME.name = "hacked"


class TestThemeManager:
    def test_register_single_theme(self):
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        assert mgr.current == DARK_THEME

    def test_register_multiple_themes(self):
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.register(LIGHT_THEME)
        assert set(mgr.available) == {"dark", "light"}

    def test_set_theme_by_name(self):
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.register(LIGHT_THEME)
        result = mgr.set_theme("light")
        assert result == LIGHT_THEME
        assert mgr.current == LIGHT_THEME

    def test_set_unknown_theme_raises(self):
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        with pytest.raises(KeyError):
            mgr.set_theme("neon_pink")

    def test_current_before_register_raises(self):
        mgr = ThemeManager()
        with pytest.raises(RuntimeError):
            _ = mgr.current

    def test_toggle_switches_between_themes(self):
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.register(LIGHT_THEME)
        mgr.set_theme("dark")
        toggled = mgr.toggle()
        assert toggled.name == "light"

    def test_toggle_back(self):
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.register(LIGHT_THEME)
        mgr.set_theme("dark")
        mgr.toggle()
        toggled = mgr.toggle()
        assert toggled.name == "dark"

    def test_listener_called_on_change(self):
        events = []
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.register(LIGHT_THEME)
        mgr.set_theme("dark")
        mgr.add_listener(lambda t: events.append(t.name))
        mgr.set_theme("light")
        assert events == ["light"]

    def test_listener_not_called_when_same_theme(self):
        events = []
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.add_listener(lambda t: events.append(t))
        mgr.set_theme("dark")  # Already dark
        assert events == []

    def test_remove_listener(self):
        events = []
        fn = lambda t: events.append(t)
        mgr = ThemeManager()
        mgr.register(DARK_THEME)
        mgr.register(LIGHT_THEME)
        mgr.set_theme("dark")
        mgr.add_listener(fn)
        mgr.remove_listener(fn)
        mgr.set_theme("light")
        assert events == []


class TestBuildDefaultThemeManager:
    def test_dark_is_default(self):
        mgr = build_default_theme_manager()
        assert mgr.current.name == "dark"

    def test_light_initial(self):
        mgr = build_default_theme_manager(initial="light")
        assert mgr.current.name == "light"

    def test_both_themes_registered(self):
        mgr = build_default_theme_manager()
        assert set(mgr.available) == {"dark", "light"}

    def test_can_switch_themes(self):
        mgr = build_default_theme_manager()
        mgr.set_theme("light")
        assert mgr.current.name == "light"
        mgr.set_theme("dark")
        assert mgr.current.name == "dark"
