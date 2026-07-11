"""spidy.ui package — Spidy Desktop Overlay UI (Milestone 5)."""

from spidy.ui.events import (
    UIMicButtonClickedEvent,
    UIClosedEvent,
    UIHideEvent,
    UIHotkeyPressedEvent,
    UIMessageEvent,
    UINotifyEvent,
    UIReadyEvent,
    UISettingsOpenedEvent,
    UIShowEvent,
    UIStateChangeEvent,
    UIThemeChangeEvent,
    UIWaveformDataEvent,
)
from spidy.ui.state import UIState, UIStateMachine, UIStateTransitionError
from spidy.ui.hotkey import GlobalHotkeyManager
from spidy.ui.themes import (
    Theme, ThemeManager, DARK_THEME, LIGHT_THEME,
    build_default_theme_manager,
)

__all__ = [
    # Events
    "UIShowEvent",
    "UIHideEvent",
    "UIStateChangeEvent",
    "UIMessageEvent",
    "UINotifyEvent",
    "UIWaveformDataEvent",
    "UIThemeChangeEvent",
    "UIReadyEvent",
    "UIClosedEvent",
    "UIHotkeyPressedEvent",
    "UIMicButtonClickedEvent",
    "UISettingsOpenedEvent",
    # State
    "UIState",
    "UIStateMachine",
    "UIStateTransitionError",
    # Hotkey
    "GlobalHotkeyManager",
    # Themes
    "Theme",
    "ThemeManager",
    "DARK_THEME",
    "LIGHT_THEME",
    "build_default_theme_manager",
]
