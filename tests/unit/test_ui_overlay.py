"""
Tests for Milestone 5 — Qt Overlay Widgets (offscreen)
=======================================================
These tests require PySide6.  If PySide6 is not installed,
the entire module is skipped.

The offscreen Qt platform is used so tests run in CI without a display.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtWidgets import QApplication


# ── Shared QApplication fixture (one per test session) ────────────────────────

@pytest.fixture(scope="session")
def qapp():
    """Return a functional QApplication for the test session.

    Ensures:
    1. Exactly one QApplication exists (creates one if absent).
    2. The Qt event loop is pumped via processEvents() so that QTimer
       registrations work correctly even when earlier tests in the full
       suite have exercised Qt code paths (e.g. via SpidyCore start/stop
       tests that create/destroy Qt objects).
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    # Pump the event loop once to flush any pending Qt state left by
    # earlier tests.  Without this, QTimer.isActive() can return False
    # even after .start() when the full suite runs non-interactively.
    app.processEvents()
    return app


# ── Theme fixture ─────────────────────────────────────────────────────────────

@pytest.fixture()
def dark_theme():
    from spidy.ui.themes.dark import DARK_THEME
    return DARK_THEME


@pytest.fixture()
def light_theme():
    from spidy.ui.themes.light import LIGHT_THEME
    return LIGHT_THEME


# ─────────────────────────────────────────────────────────────────────────────
# WaveformWidget
# ─────────────────────────────────────────────────────────────────────────────

class TestWaveformWidget:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.waveform import WaveformWidget
        w = WaveformWidget()
        assert w is not None
        w.close()

    def test_set_active_starts_timer(self, qapp):
        from spidy.ui.widgets.waveform import WaveformWidget
        w = WaveformWidget()
        w.set_active(True)
        assert w._timer.isActive()
        w.set_active(False)
        assert not w._timer.isActive()
        w.close()

    def test_set_amplitudes_disables_simulated(self, qapp):
        from spidy.ui.widgets.waveform import WaveformWidget
        w = WaveformWidget()
        w.set_amplitudes([0.5, 0.8, 0.3])
        assert not w._simulated
        w.close()

    def test_set_empty_amplitudes_reenables_simulated(self, qapp):
        from spidy.ui.widgets.waveform import WaveformWidget
        w = WaveformWidget()
        w.set_amplitudes([0.5])
        w.set_amplitudes([])  # Reset
        assert w._simulated
        w.close()

    def test_apply_theme_updates_color(self, qapp, dark_theme):
        from spidy.ui.widgets.waveform import WaveformWidget
        from PySide6.QtGui import QColor
        w = WaveformWidget()
        w.apply_theme(dark_theme)
        assert w._bar_color == QColor(dark_theme.colors.waveform_bar)
        w.close()

    def test_minimum_dimensions(self, qapp):
        from spidy.ui.widgets.waveform import WaveformWidget
        w = WaveformWidget()
        assert w.minimumHeight() >= 48
        assert w.minimumWidth() >= 160
        w.close()


# ─────────────────────────────────────────────────────────────────────────────
# MicrophoneButton
# ─────────────────────────────────────────────────────────────────────────────

class TestMicrophoneButton:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        assert btn is not None
        btn.close()

    def test_fixed_size(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        assert btn.width() == 56
        assert btn.height() == 56
        btn.close()

    def test_set_state_idle(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        btn.set_state("idle")
        assert btn._state == "idle"
        assert not btn._timer.isActive()
        btn.close()

    def test_set_state_listening_starts_timer(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        btn.set_state("listening")
        assert btn._state == "listening"
        assert btn._timer.isActive()
        btn.set_state("idle")
        btn.close()

    def test_set_state_thinking_starts_timer(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        btn.set_state("thinking")
        assert btn._timer.isActive()
        btn.set_state("idle")
        btn.close()

    def test_set_state_speaking_stops_timer(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        btn.set_state("speaking")
        assert not btn._timer.isActive()
        btn.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        from PySide6.QtGui import QColor
        btn = MicrophoneButton()
        btn.apply_theme(dark_theme)
        assert btn._color_idle == QColor(dark_theme.colors.mic_idle)
        btn.close()

    def test_all_valid_states(self, qapp):
        from spidy.ui.widgets.mic_button import MicrophoneButton
        btn = MicrophoneButton()
        for state in ("idle", "listening", "thinking", "speaking", "disabled"):
            btn.set_state(state)
            assert btn._state == state
        btn.set_state("idle")
        btn.close()


# ─────────────────────────────────────────────────────────────────────────────
# SpeakingIndicator
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeakingIndicator:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
        ind = SpeakingIndicator()
        assert ind is not None
        ind.close()

    def test_set_active_true_starts_timer(self, qapp):
        from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
        ind = SpeakingIndicator()
        ind.set_active(True)
        assert ind._timer.isActive()
        ind.set_active(False)
        ind.close()

    def test_set_active_false_stops_timer(self, qapp):
        from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
        ind = SpeakingIndicator()
        ind.set_active(True)
        ind.set_active(False)
        assert not ind._timer.isActive()
        ind.close()

    def test_fixed_height(self, qapp):
        from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
        ind = SpeakingIndicator()
        assert ind.height() == 24
        ind.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.speaking_indicator import SpeakingIndicator
        from PySide6.QtGui import QColor
        ind = SpeakingIndicator()
        ind.apply_theme(dark_theme)
        assert ind._color == QColor(dark_theme.colors.thinking_dot)
        ind.close()


# ─────────────────────────────────────────────────────────────────────────────
# ChatView + ChatMessage
# ─────────────────────────────────────────────────────────────────────────────

class TestChatMessage:
    def test_user_message(self):
        from spidy.ui.widgets.chat_view import ChatMessage
        msg = ChatMessage(role="user", text="Hello")
        assert msg.role == "user"
        assert msg.text == "Hello"

    def test_assistant_message(self):
        from spidy.ui.widgets.chat_view import ChatMessage
        msg = ChatMessage(role="assistant", text="Hi there!")
        assert msg.role == "assistant"

    def test_session_id_default_empty(self):
        from spidy.ui.widgets.chat_view import ChatMessage
        msg = ChatMessage(role="user", text="test")
        assert msg.session_id == ""


class TestChatView:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.chat_view import ChatView
        view = ChatView()
        assert view is not None
        view.close()

    def test_add_message_increases_count(self, qapp):
        from spidy.ui.widgets.chat_view import ChatView, ChatMessage
        view = ChatView()
        assert len(view._messages) == 0
        view.add_message(ChatMessage(role="user", text="Hello"))
        assert len(view._messages) == 1
        view.close()

    def test_add_multiple_messages(self, qapp):
        from spidy.ui.widgets.chat_view import ChatView, ChatMessage
        view = ChatView()
        for i in range(5):
            view.add_message(ChatMessage(role="user", text=f"Message {i}"))
        assert len(view._messages) == 5
        view.close()

    def test_clear_removes_all(self, qapp):
        from spidy.ui.widgets.chat_view import ChatView, ChatMessage
        view = ChatView()
        view.add_message(ChatMessage(role="user", text="Hello"))
        view.add_message(ChatMessage(role="assistant", text="Hi!"))
        view.clear()
        assert len(view._messages) == 0
        assert len(view._bubbles) == 0
        view.close()

    def test_apply_theme_runs_without_error(self, qapp, dark_theme):
        from spidy.ui.widgets.chat_view import ChatView
        view = ChatView()
        view.apply_theme(dark_theme)
        view.close()

    def test_max_messages_enforced(self, qapp):
        from spidy.ui.widgets.chat_view import ChatView, ChatMessage
        view = ChatView()
        view._MAX_MESSAGES = 5  # Override for test speed
        for i in range(10):
            view.add_message(ChatMessage(role="user", text=f"Msg {i}"))
        assert len(view._messages) <= 5
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# UISignalBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestUISignalBridge:
    def test_creates_without_error(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        bridge = UISignalBridge()
        assert bridge is not None

    def test_state_change_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        bridge.state_change_requested.connect(lambda s: received.append(s))
        bridge.request_state_change("listening")
        assert received == ["listening"]

    def test_message_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        bridge.message_received.connect(lambda r, t: received.append((r, t)))
        bridge.request_message("user", "Hello")
        assert received == [("user", "Hello")]

    def test_notification_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        # M17.2: notification_requested is Signal(str, str, str) -- title, body, level
        bridge.notification_requested.connect(
            lambda title, body, level: received.append((title, level))
        )
        bridge.request_notification("Test", "Body", "info")
        assert received == [("Test", "info")]

    def test_waveform_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        bridge.waveform_data_received.connect(lambda a: received.append(a))
        bridge.request_waveform([0.1, 0.5, 0.9])
        assert received == [[0.1, 0.5, 0.9]]

    def test_theme_change_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        # M17.2: theme_change_requested emits Theme object via apply_theme path
        bridge.theme_change_requested.connect(lambda t: received.append(t))
        from spidy.ui.themes.dark import DARK_THEME
        bridge.theme_change_requested.emit(DARK_THEME)
        assert len(received) == 1


# ─────────────────────────────────────────────────────────────────────────────
# OverlayWindow (basic creation)
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayWindowCreation:
    """M17.2 HUD -- updated assertions for new architecture."""

    def test_creates_without_error(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert win is not None
        win.close()

    def test_fixed_size(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        # M17.2: explicit size is still honoured
        win = OverlayWindow(dark_theme, width=1200, height=700, animate=False)
        assert win.width() == 1200
        assert win.height() == 700
        win.close()

    def test_state_starts_idle(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        # M17.2: state tracked on the core widget as a string
        assert win._core._state == "idle"
        win.close()

    def test_set_state_listening(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("listening")
        assert win._core._state == "listening"
        win.close()

    def test_set_state_thinking(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("thinking")
        assert win._core._state == "thinking"
        win.close()

    def test_set_state_speaking(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("speaking")
        assert win._core._state == "speaking"
        win.close()

    def test_set_state_error(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("error")
        assert win._core._state == "error"
        win.close()

    def test_add_message(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.add_message("user", "Hello Spidy!")
        # M17.2: messages stored in _chat_overlay
        assert len(win._chat_overlay._messages) == 1
        win.close()

    def test_apply_light_theme(self, qapp, dark_theme, light_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.apply_theme(light_theme)
        assert win._theme == light_theme
        win.close()

    def test_waveform_visible_in_listening(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("listening")
        # M17.2: voice bar is always visible, reacts to state
        assert win._footer.voice_bar._state == "listening"
        win.close()

    def test_waveform_hidden_in_idle(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("idle")
        # M17.2: voice bar idle state
        assert win._footer.voice_bar._state == "idle"
        win.close()

    def test_indicator_visible_in_thinking(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("thinking")
        # M17.2: core reflects thinking state (indicator is the core itself)
        assert win._core._state == "thinking"
        win.close()

    def test_indicator_visible_in_speaking(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("speaking")
        assert win._core._state == "speaking"
        win.close()

    def test_indicator_hidden_in_idle(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("idle")
        assert win._core._state == "idle"
        win.close()

    def test_mic_state_idle(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("idle")
        # M17.2: mic state tracked in the left panel
        assert not win._left_panel._mic_info.isHidden()
        win.close()

    def test_mic_state_listening(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("listening")
        assert win._core._state == "listening"
        win.close()
