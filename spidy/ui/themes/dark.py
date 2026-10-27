"""
Dark Theme — Spidy's Spider-Verse Tactical HUD Theme.
=====================================================
Design palette
--------------
Background:  Near-black void (#0A0A0E) + Glassmorphic panels (#12141C @ 85-92%)
Accent:      Spider-red (#E62429) + Glow (#FF3B3F) + Navy Web-blue (#1B2A4A)
Alert/Info:  Spider-sense cyan (#3FA9F5)
Text:        Near-white (#F5F5F7) + Graphite secondary (#9A9AA5)
Web-lines:   Hairline dividers #0FFFFFFF
"""

from __future__ import annotations

from spidy.ui.themes.base import Theme, ThemeColors, ThemeFonts, ThemeGeometry

DARK_THEME = Theme(
    name="dark",
    display_name="Spider-Verse HUD (Dark Glass)",
    overlay_opacity=0.92,
    blur_enabled=True,
    colors=ThemeColors(
        # Background layers
        bg_primary="#0A0A0E",           # Void black — main overlay base
        bg_secondary="#12141C",         # Dark glassmorphic panels (card surfaces)
        bg_tertiary="#1B2A4A",          # Cool contrast panels, input fields, chip fills
        bg_glass="#E612141C",           # 90% opacity glassmorphism tint

        # Text
        text_primary="#F5F5F7",         # High-contrast crisp white/light gray
        text_secondary="#9A9AA5",       # Body & secondary readouts
        text_muted="#5F6170",           # Inactive / timestamps
        text_on_accent="#FFFFFF",       # Text on red badges/buttons

        # Accent — Spider-red & Web-blue & Spider-sense cyan
        accent_primary="#E62429",       # Iconic Spider-red
        accent_secondary="#1B2A4A",     # Web-blue navy
        accent_success="#2ECC71",       # Tactical green (success/allowed)
        accent_warning="#F39C12",       # Amber warning
        accent_error="#FF4D4F",         # Danger / permission denied red

        # Chat bubbles
        bubble_user="#161924",          # Dark slate glass
        bubble_assistant="#10121A",     # Void slate with red/blue accents
        bubble_user_text="#F5F5F7",
        bubble_assistant_text="#F5F5F7",

        # UI elements
        border="#14FFFFFF",             # Faint web-line border (8% white)
        border_active="#E62429",        # Red reticle active border
        scrollbar="#1B2A4A",
        scrollbar_hover="#E62429",

        # State indicators
        mic_idle="#1B2A4A",             # Deep navy when idle
        mic_listening="#E62429",        # Spider-red pulse when recording
        waveform_bar="#E62429",         # Spider-red waveform bars
        thinking_dot="#3FA9F5",         # Spider-sense cyan dots

        # Wake-ready edge glow
        wake_glow="#E62429",            # Spider-red glow bloom

        # Autonomous working state glow
        working_glow="#E62429",

        # Notifications
        notify_info="#3FA9F5",
        notify_success="#2ECC71",
        notify_warning="#F39C12",
        notify_error="#FF4D4F",

        # Tray
        tray_icon_bg="#E62429",

        # HUD Spider-Verse interface tokens
        hud_primary="#E62429",          # Spider-red
        hud_secondary="#1B2A4A",        # Web-blue navy
        hud_dim="#3A1014",              # Deep muted red-black for dim rings
        hud_grid="#0A0A0E",             # Void black background
        hud_text="#F5F5F7",             # Clean HUD readout text
        accent_tertiary="#3FA9F5",      # Spider-sense cyan
        line_web="#0FFFFFFF",           # 6% alpha white hairline
        bg_void="#0A0A0E",
    ),
    fonts=ThemeFonts(
        family_primary="Segoe UI",
        family_display="Rajdhani",
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
