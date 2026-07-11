"""
Dark Theme — Spidy's default glassmorphism dark theme.

Design palette
--------------
Background:  Deep navy/slate (#0D0E1A → #12131F)
Accent:      Electric violet (#6C63FF) → teal (#00D4FF)
Text:        Near-white (#E8E9F0) → muted (#8A8BA0)
Glass:       Semi-transparent dark with subtle purple tint
"""

from __future__ import annotations

from spidy.ui.themes.base import Theme, ThemeColors, ThemeFonts, ThemeGeometry

DARK_THEME = Theme(
    name="dark",
    display_name="Dark (Glassmorphism)",
    overlay_opacity=0.92,
    blur_enabled=True,
    colors=ThemeColors(
        # Background layers
        bg_primary="#0D0E1A",           # Deep navy — main overlay body
        bg_secondary="#12131F",         # Slightly lighter — chat/panels
        bg_tertiary="#1A1B2E",          # Input fields, hover states
        bg_glass="#1A1B2ECC",           # 80% transparent for glass effect

        # Text
        text_primary="#E8E9F0",
        text_secondary="#B0B2C8",
        text_muted="#6B6D88",
        text_on_accent="#FFFFFF",

        # Accent — electric violet + teal
        accent_primary="#6C63FF",       # Spidy brand purple
        accent_secondary="#00D4FF",     # Teal highlight
        accent_success="#00E96A",       # Emerald green
        accent_warning="#FFB800",       # Amber
        accent_error="#FF4D6A",         # Coral red

        # Chat bubbles
        bubble_user="#1E1F35",
        bubble_assistant="#252640",
        bubble_user_text="#C8CAE0",
        bubble_assistant_text="#E0E2F5",

        # UI elements
        border="#2A2B45",               # Subtle separator
        border_active="#6C63FF",        # Active/focused = accent
        scrollbar="#2A2B45",
        scrollbar_hover="#3D3F68",

        # State indicators
        mic_idle="#6C63FF",             # Purple when idle
        mic_listening="#FF4D6A",        # Red when recording
        waveform_bar="#6C63FF",         # Purple waveform bars
        thinking_dot="#00D4FF",         # Teal dots

        # Wake-ready edge glow (for future wake-word integration)
        wake_glow="#6C63FF",

        # Notifications
        notify_info="#00D4FF",
        notify_success="#00E96A",
        notify_warning="#FFB800",
        notify_error="#FF4D6A",

        # Tray
        tray_icon_bg="#6C63FF",
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
