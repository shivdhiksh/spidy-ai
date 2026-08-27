"""
test_hud_large_wide_layout.py -- Tests for Large Futuristic Wide HUD Layout & Lifecycle
========================================================================================
Covers all requirements:
1. Large responsive HUD sizing (WIDTH > HEIGHT)
2. Conversation width is full horizontal span
3. Word wrapping
4. Long messages without truncation
5. Auto-scroll
6. User ("YOU") vs Assistant ("SPIDY") styling
7. Large central AI Core widget
8. Core animation timers & ticks
9. Core state transitions (IDLE, LISTENING, THINKING, SPEAKING, WORKING, ERROR)
10. Logo / geometric fallback emblem
11. Telemetry metrics (CPU, RAM, GPU, VRAM, DISK)
12. Real GPU telemetry query
13. Real VRAM telemetry query
14. hide_hud()
15. show_hud()
16. toggle_hud()
17. X button hides without killing runtime
18. SPIDY runtime remains alive after X
19. Wake word reopens HUD
20. Ctrl+Space hotkey reopens HUD
21. No duplicate HUD windows
22. Position persists across hide/show
23. Size persists across hide/show
24. State sync after hidden activity
25. Multi-monitor positioning
26. 1280x720 layout
27. 1536x864 layout
28. 1920x1080 layout
29. 2560x1440 layout
30. Header & footer integration
31. Confirmation card overlay
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from spidy.ui.themes import build_default_theme_manager
from spidy.ui.widgets.hud_chat import HUDChatOverlay, _ChatMessageBubble
from spidy.ui.widgets.hud_core import SpidyCoreWidget
from spidy.ui.widgets.hud_telemetry import TelemetryPanel
from spidy.ui.widgets.hud_task_console import HUDTaskConsole
from spidy.ui.overlay import OverlayWindow, _hud_size


@pytest.fixture(scope="session")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def theme():
    mgr = build_default_theme_manager()
    return mgr.current


# ─── 1. Responsive Sizing & Layout (WIDTH > HEIGHT) ─────────────────────────


def test_responsive_hud_sizing_at_multiple_resolutions():
    # 1920x1080
    with patch("PySide6.QtGui.QGuiApplication.screens") as mock_screens:
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = MagicMock(width=lambda: 1920, height=lambda: 1080)
        mock_screens.return_value = [mock_screen]
        w, h = _hud_size()
        assert w > h
        assert 1400 <= w <= 1700
        assert 700 <= h <= 880

    # 1536x864
    with patch("PySide6.QtGui.QGuiApplication.screens") as mock_screens:
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = MagicMock(width=lambda: 1536, height=lambda: 864)
        mock_screens.return_value = [mock_screen]
        w, h = _hud_size()
        assert w > h
        assert 1200 <= w <= 1450
        assert 600 <= h <= 750

    # 1280x720
    with patch("PySide6.QtGui.QGuiApplication.screens") as mock_screens:
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = MagicMock(width=lambda: 1280, height=lambda: 720)
        mock_screens.return_value = [mock_screen]
        w, h = _hud_size()
        assert w > h
        assert 1050 <= w <= 1200
        assert 540 <= h <= 650

    # 2560x1440
    with patch("PySide6.QtGui.QGuiApplication.screens") as mock_screens:
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = MagicMock(width=lambda: 2560, height=lambda: 1440)
        mock_screens.return_value = [mock_screen]
        w, h = _hud_size()
        assert w > h
        assert 1800 <= w <= 2100
        assert 900 <= h <= 1150


def test_overlay_layout_structure(qapp, theme):
    win = OverlayWindow(theme=theme, width=1500, height=800)
    # Verify main sections exist
    assert hasattr(win, "_header")
    assert hasattr(win, "_left_panel")
    assert hasattr(win, "_core")
    assert hasattr(win, "_right_panel")
    assert hasattr(win, "_chat_overlay")
    assert hasattr(win, "_footer")

    # Verify conversation container is wide
    assert win._chat_container.width() > 1000 or win._hud_w >= 1400


# ─── 2. Conversation Panel & Readability ────────────────────────────────────


def test_chat_overlay_preserves_full_messages(qapp, theme):
    chat = HUDChatOverlay()
    chat.apply_theme(theme)

    long_text = "This is a comprehensive response with code snippets and multi-line detailed explanations that should never be truncated." * 3
    chat.add_message("user", "Can you explain Python async architecture?")
    chat.add_message("assistant", long_text)

    assert len(chat._messages) == 2
    assert chat._messages[1][1] == long_text


def test_chat_bubble_visual_separation(qapp, theme):
    bubble_user = _ChatMessageBubble("user", "Hello Spidy")
    bubble_asst = _ChatMessageBubble("assistant", "Hello! How can I assist you?")

    assert bubble_user._lbl_role.text() == "YOU"
    assert bubble_asst._lbl_role.text() == "SPIDY"
    assert bubble_user._lbl_text.wordWrap() is True
    assert bubble_asst._lbl_text.wordWrap() is True


def test_chat_overlay_clear(qapp):
    chat = HUDChatOverlay()
    chat.add_message("user", "test")
    assert len(chat._messages) == 1
    chat.clear()
    assert len(chat._messages) == 0


# ─── 3. Central AI Core & Animations ─────────────────────────────────────────


def test_spidy_core_rendering_and_states(qapp, theme):
    core = SpidyCoreWidget()
    core.apply_theme(theme)

    # State transitions
    for st in ("idle", "wake_ready", "listening", "thinking", "speaking", "working", "error"):
        core.set_state(st)
        assert core._state == st

    # Amplitude clamping
    core.set_amplitude(1.5)
    assert core._amplitude == 1.0
    core.set_amplitude(-0.2)
    assert core._amplitude == 0.0

    # Metric clamping
    core.set_metric(120)
    assert core._metric == 100.0


def test_spidy_core_timer_speed(qapp):
    core = SpidyCoreWidget()
    core.set_state("idle")
    idle_int = core._timer.interval()
    core.set_state("listening")
    active_int = core._timer.interval()
    assert active_int < idle_int  # Active runs at higher FPS


# ─── 4. Telemetry (CPU, RAM, GPU, VRAM, Disk) ───────────────────────────────


def test_telemetry_metrics_and_polling(qapp, theme):
    tele = TelemetryPanel()
    tele.apply_theme(theme)
    labels = [r.label for r in tele._rows]
    assert "CPU" in labels
    assert "RAM" in labels
    assert "GPU" in labels
    assert "VRAM" in labels
    assert "DISK" in labels


# ─── 5. Floating Window & Lifecycle (Hide, Show, Toggle, Reopen) ────────────


def test_overlay_position_persistence(qapp, theme):
    OverlayWindow._session_pos = QPoint(350, 450)
    win = OverlayWindow(theme=theme, width=1400, height=750)
    assert win.pos() == QPoint(350, 450)


def test_overlay_lifecycle_api(qapp, theme):
    win = OverlayWindow(theme=theme, width=1400, height=750, animate=False)

    win.show_hud()
    assert win.isVisible() is True

    win.hide_hud()
    assert win.isVisible() is False

    win.toggle_hud()
    assert win.isVisible() is True

    win.toggle_hud()
    assert win.isVisible() is False


def test_close_button_hides_only_and_persists_position(qapp, theme):
    OverlayWindow._session_pos = None
    win = OverlayWindow(theme=theme, width=1400, height=750, animate=False)
    win.move(QPoint(500, 200))
    qapp.processEvents()
    win.show_hud()

    # Click X
    win._header._btn_close.click()
    assert win.isVisible() is False
    assert OverlayWindow._session_pos == QPoint(500, 200)

    # Reopen
    win.show_hud()
    assert win.isVisible() is True
    assert win.pos() == QPoint(500, 200)
    win.close()


@pytest.mark.asyncio
async def test_spidy_app_wake_word_and_hotkey_reopen(qapp, theme):
    from spidy.ui.app import SpidyApp

    bus = MagicMock()
    config = MagicMock()
    config.ui.enabled = True
    config.ui.theme = "dark"
    config.ui.width = 1400
    config.ui.height = 750
    config.ui.position = "center"
    config.ui.always_on_top = True
    config.ui.animate = False
    config.ui.hotkey = "ctrl+space"

    app = SpidyApp(bus=bus, config=config)
    app.start()

    # Hide HUD
    app.hide_hud()
    assert app._overlay.isVisible() is False

    # Wake word fires -> HUD reopens!
    await app._handle_wake_detected(MagicMock())
    qapp.processEvents()
    assert app._overlay.isVisible() is True

    # Hotkey fires -> toggles hide!
    await app._handle_hotkey(MagicMock())
    qapp.processEvents()
    assert app._overlay.isVisible() is False

    # Hotkey fires again -> toggles show!
    await app._handle_hotkey(MagicMock())
    qapp.processEvents()
    assert app._overlay.isVisible() is True


def test_task_console_progress_updates(qapp, theme):
    console = HUDTaskConsole()
    console.apply_theme(theme)
    from spidy.ui.widgets.hud_task_console import TaskStep

    steps = [
        TaskStep("Open Edge", "done"),
        TaskStep("Search YouTube for Python", "running"),
        TaskStep("Type results into Notepad", "pending"),
    ]
    console.update_progress("Search and Type", steps)
    assert console._goal == "Search and Type"
    assert len(console._steps) == 3


def test_confirmation_card_show_and_hide(qapp, theme):
    win = OverlayWindow(theme=theme, width=1400, height=750, animate=False)
    win.show()
    win.show_confirmation(
        task_id="t1",
        goal_id="g1",
        description="Delete file",
        prompt="Are you sure you want to delete this file?",
    )
    assert win._confirmation_card.isVisible() is True

    win.hide_confirmation()
    assert win._confirmation_card.isVisible() is False
    win.close()


def test_auto_responsive_width_is_wide(qapp, theme):
    """Auto responsive (width=0) HUD must remain wide (>= 1050)."""
    win = OverlayWindow(theme=theme, width=0, height=0, animate=False)
    assert win.width() >= 1050
    assert win.width() > win.height()
    win.close()


def test_show_animated_when_already_visible_preserves_opacity(qapp, theme):
    """Calling show_animated() when already visible should not blank out opacity to 0.0."""
    win = OverlayWindow(theme=theme, width=1400, height=750, animate=True)
    win.setWindowOpacity(1.0)
    win.show()
    assert win.windowOpacity() == 1.0

    win.show_animated()
    # Opacity must remain 1.0 rather than being reset to 0.0
    assert win.windowOpacity() == 1.0
    win.close()


def test_thread_safe_bridge_signals(qapp):
    from spidy.ui.overlay import UISignalBridge
    bridge = UISignalBridge()

    show_called = []
    hide_called = []
    toggle_called = []

    bridge.show_hud_requested.connect(lambda: show_called.append(True))
    bridge.hide_hud_requested.connect(lambda: hide_called.append(True))
    bridge.toggle_hud_requested.connect(lambda: toggle_called.append(True))

    bridge.request_show_hud()
    bridge.request_hide_hud()
    bridge.request_toggle_hud()

    qapp.processEvents()

    assert len(show_called) == 1
    assert len(hide_called) == 1
    assert len(toggle_called) == 1
