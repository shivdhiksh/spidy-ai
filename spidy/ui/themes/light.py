"""
Light Theme — Spidy's clean light mode.

Design palette
--------------
Background:  Pearl white (#FAFBFF → #F0F1FA)
Accent:      Electric violet (#5C54EF) → teal (#0098CC)
Text:        Deep navy (#1A1B2E) → muted (#7B7D96)
Glass:       Near-white with frosted glass effect
"""

from __future__ import annotations

from spidy.ui.themes.base import Theme, ThemeColors, ThemeFonts, ThemeGeometry

LIGHT_THEME = Theme(
    name="light",
    display_name="Light (Frosted Glass)",
    overlay_opacity=0.94,
    blur_enabled=True,
    colors=ThemeColors(
        # Background layers
        bg_primary="#FAFBFF",           # Pearl white
        bg_secondary="#F0F1FA",         # Soft lavender-white
        bg_tertiary="#E8E9F5",          # Input fields, hover states
        bg_glass="#FAFAFACC",           # 80% transparent for frosted glass

        # Text
        text_primary="#1A1B2E",         # Deep navy
        text_secondary="#3D3F5E",
        text_muted="#7B7D96",
        text_on_accent="#FFFFFF",

        # Accent
        accent_primary="#5C54EF",       # Slightly darker violet for light bg
        accent_secondary="#0098CC",
        accent_success="#00A64E",
        accent_warning="#C07800",
        accent_error="#CC2B47",

        # Chat bubbles
        bubble_user="#EAEBF8",
        bubble_assistant="#FFFFFF",
        bubble_user_text="#2A2C4A",
        bubble_assistant_text="#1A1B2E",

        # UI elements
        border="#D8DAF0",
        border_active="#5C54EF",
        scrollbar="#D8DAF0",
        scrollbar_hover="#C0C2DC",

        # State indicators
        mic_idle="#5C54EF",
        mic_listening="#CC2B47",
        waveform_bar="#5C54EF",
        thinking_dot="#0098CC",

        # Wake-ready edge glow
        wake_glow="#5C54EF",

        # Autonomous working state glow — warm amber for light mode
        working_glow="#C07800",

        # Notifications
        notify_info="#0098CC",
        notify_success="#00A64E",
        notify_warning="#C07800",
        notify_error="#CC2B47",

        # Tray
        tray_icon_bg="#5C54EF",

        # HUD futuristic interface colors (M17.2) - slightly warmer for light mode
        hud_primary="#0097A7",    # Deep teal (readable on light)
        hud_secondary="#00838F",  # Darker teal
        hud_dim="#B2EBF2",        # Very light cyan for inactive
        hud_grid="#E0F7FA",       # Light cyan-white background
        hud_text="#006064",       # Dark teal text
    ),
    fonts=ThemeFonts(
        family_primary="Segoe UI",
        family_mono="Consolas",
        size_xs=10,
        size_sm=12,
        size_base=13,
        size_md=14,
        size_lg=16,
        size_xl=20,
        size_title=24,
    ),
    geometry=ThemeGeometry(
        border_radius=18,
        border_radius_sm=10,
        border_radius_lg=24,
        border_width=1,
        padding=16,
        padding_sm=8,
        padding_lg=24,
        bubble_radius=14,
        margin_edge=20,
    ),
)
