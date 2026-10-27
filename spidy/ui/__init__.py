"""spidy.ui package — Spidy Desktop Overlay UI (Milestone 5)."""

from spidy.ui.events import (
    UIMicButtonClickedEvent,
    UIClosedEvent,
    UIConfirmationDeniedEvent,
    UIConfirmationGrantedEvent,
    UIForegroundEvent,
    UIHideEvent,
    UIHotkeyPressedEvent,
    UIMessageEvent,
    UIMicButtonClickedEvent,
    UIMinimizeEvent,
    UINotifyEvent,
    UIReadyEvent,
    UIRestoreEvent,
    UISettingsOpenedEvent,
    UIShowEvent,
    UIStateChangeEvent,
    UIThemeChangeEvent,
    UIWakeAckEvent,
    UIWakeReceivedEvent,
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
    "UIRestoreEvent",
    "UIMinimizeEvent",
    "UIForegroundEvent",
    "UIWakeReceivedEvent",
    "UIWakeAckEvent",
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
    "UIConfirmationGrantedEvent",
    "UIConfirmationDeniedEvent",
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
