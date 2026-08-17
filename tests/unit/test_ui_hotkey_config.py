"""
Tests for Milestone 5 — Hotkey Manager and UIConfig
=====================================================
Pure Python; no Qt display required.
All GlobalHotkeyManager tests avoid actually registering OS hooks.
"""

from __future__ import annotations

import pytest

from spidy.ui.hotkey import GlobalHotkeyManager


class TestHotkeyParsing:
    def test_lowercase_normalisation(self):
        assert GlobalHotkeyManager.parse_hotkey("Ctrl+Space") == "ctrl+space"

    def test_strip_whitespace(self):
        assert GlobalHotkeyManager.parse_hotkey("ctrl + space") == "ctrl+space"

    def test_uppercase_modifiers(self):
        assert GlobalHotkeyManager.parse_hotkey("CTRL+SHIFT+S") == "ctrl+shift+s"

    def test_complex_hotkey(self):
        result = GlobalHotkeyManager.parse_hotkey("Alt+F4")
        assert result == "alt+f4"

    def test_single_key_normalised(self):
        assert GlobalHotkeyManager.parse_hotkey("F12") == "f12"


class TestHotkeyValidation:
    def test_ctrl_space_valid(self):
        assert GlobalHotkeyManager.validate_hotkey("ctrl+space") is True

    def test_ctrl_shift_s_valid(self):
        assert GlobalHotkeyManager.validate_hotkey("ctrl+shift+s") is True

    def test_alt_f4_valid(self):
        assert GlobalHotkeyManager.validate_hotkey("alt+f4") is True

    def test_win_valid(self):
        assert GlobalHotkeyManager.validate_hotkey("win+space") is True

    def test_empty_string_invalid(self):
        assert GlobalHotkeyManager.validate_hotkey("") is False

    def test_whitespace_only_invalid(self):
        assert GlobalHotkeyManager.validate_hotkey("   ") is False

    def test_bare_key_no_modifier_invalid(self):
        # Just "F12" with no modifier = invalid for a global hotkey
        assert GlobalHotkeyManager.validate_hotkey("f12") is False


class TestGlobalHotkeyManagerInit:
    def test_hotkey_property(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        mgr = GlobalHotkeyManager(bus=bus, hotkey="ctrl+space")
        assert mgr.hotkey == "ctrl+space"

    def test_not_registered_initially(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        mgr = GlobalHotkeyManager(bus=bus)
        assert not mgr.is_registered

    def test_repr(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        mgr = GlobalHotkeyManager(bus=bus, hotkey="ctrl+space")
        r = repr(mgr)
        assert "ctrl+space" in r
        assert "registered" in r

    def test_default_hotkey(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        mgr = GlobalHotkeyManager(bus=bus)
        assert mgr.hotkey == "ctrl+space"


class TestGlobalHotkeyManagerStop:
    def test_stop_when_not_registered_is_noop(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        mgr = GlobalHotkeyManager(bus=bus)
        mgr.stop()  # Should not raise
        assert not mgr.is_registered


class TestUIConfigValidation:
    def test_default_config_valid(self):
        from spidy.config.manager import UIConfig
        cfg = UIConfig()
        assert cfg.theme == "dark"
        assert cfg.position == "center"    # M17: default changed to centered
        assert cfg.opacity == 0.92
        assert cfg.hotkey == "ctrl+space"

    def test_invalid_theme_raises(self):
        from pydantic import ValidationError
        from spidy.config.manager import UIConfig
        with pytest.raises(ValidationError):
            UIConfig(theme="neon_pink")

    def test_invalid_position_raises(self):
        from pydantic import ValidationError
        from spidy.config.manager import UIConfig
        with pytest.raises(ValidationError):
            UIConfig(position="floating")

    def test_invalid_opacity_raises(self):
        from pydantic import ValidationError
        from spidy.config.manager import UIConfig
        with pytest.raises(ValidationError):
            UIConfig(opacity=0.0)  # must be > 0.0

    def test_opacity_max_valid(self):
        from spidy.config.manager import UIConfig
        cfg = UIConfig(opacity=1.0)
        assert cfg.opacity == 1.0

    def test_all_positions_valid(self):
        from spidy.config.manager import UIConfig
        for pos in ("top-right", "top-left", "bottom-right", "bottom-left"):
            cfg = UIConfig(position=pos)
            assert cfg.position == pos

    def test_both_themes_valid(self):
        from spidy.config.manager import UIConfig
        for theme in ("dark", "light"):
            cfg = UIConfig(theme=theme)
            assert cfg.theme == theme

    def test_new_m5_fields_have_defaults(self):
        from spidy.config.manager import UIConfig
        cfg = UIConfig()
        assert cfg.animation_speed_ms == 250
        assert cfg.notification_duration_ms == 4000
        assert cfg.edge_margin == 20
        assert cfg.glassmorphism_enabled is True
        assert cfg.accent_color == "#6C63FF"
        assert cfg.wake_ready_glow is True
        assert cfg.drag_to_reposition is True
        assert cfg.click_through_when_idle is False
        assert cfg.compact_mode is False
