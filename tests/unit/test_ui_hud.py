"""
tests/unit/test_ui_hud.py
=========================
Regression tests for Milestone 17.2 -- Full-screen HUD interface.

All tests use the offscreen Qt platform (no real display needed).
"""

from __future__ import annotations

import sys
import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    app.processEvents()
    return app


@pytest.fixture()
def dark_theme():
    from spidy.ui.themes.dark import DARK_THEME
    return DARK_THEME


# ==============================================================================
# HUD color tokens
# ==============================================================================

class TestHUDThemeTokens:
    def test_dark_theme_hud_primary(self, dark_theme):
        assert dark_theme.colors.hud_primary.startswith("#")

    def test_dark_theme_hud_secondary(self, dark_theme):
        assert dark_theme.colors.hud_secondary.startswith("#")

    def test_dark_theme_hud_dim(self, dark_theme):
        assert dark_theme.colors.hud_dim.startswith("#")

    def test_dark_theme_hud_grid(self, dark_theme):
        assert dark_theme.colors.hud_grid.startswith("#")

    def test_dark_theme_hud_text(self, dark_theme):
        assert dark_theme.colors.hud_text.startswith("#")

    def test_light_theme_has_hud_tokens(self):
        from spidy.ui.themes.light import LIGHT_THEME
        assert hasattr(LIGHT_THEME.colors, "hud_primary")


# ==============================================================================
# SpidyCoreWidget
# ==============================================================================

class TestSpidyCoreWidget:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        assert w is not None
        w.close()

    def test_timer_active_at_init(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        assert w._timer.isActive()
        w.close()

    def test_set_all_states_no_crash(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        for state in ("idle", "wake_ready", "listening", "thinking", "speaking", "working", "error"):
            w.set_state(state)
            assert w._state == state
        w.close()

    def test_set_amplitude_clamped(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        w.set_amplitude(2.5)
        assert w._amplitude == 1.0
        w.set_amplitude(-0.5)
        assert w._amplitude == 0.0
        w.close()

    def test_set_metric_clamped(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        w.set_metric(150)
        assert w._metric == 100.0
        w.set_metric(-10)
        assert w._metric == 0.0
        w.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        w.apply_theme(dark_theme)
        w.close()

    def test_boot_scale_property(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget
        w = SpidyCoreWidget()
        w.set_boot_scale(0.5)
        assert abs(w._boot_scale - 0.5) < 0.001
        w.close()

    def test_working_state_changes_timer(self, qapp):
        from spidy.ui.widgets.hud_core import SpidyCoreWidget, _FPS_IDLE, _FPS_ACTIVE
        w = SpidyCoreWidget()
        w.set_state("idle")
        assert w._timer.interval() == _FPS_IDLE
        w.set_state("listening")
        assert w._timer.interval() == _FPS_ACTIVE
        w.close()


# ==============================================================================
# TelemetryPanel
# ==============================================================================

class TestTelemetryPanel:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.hud_telemetry import TelemetryPanel
        p = TelemetryPanel()
        assert p is not None
        p.close()

    def test_custom_metrics_list(self, qapp):
        from spidy.ui.widgets.hud_telemetry import TelemetryPanel
        p = TelemetryPanel(["cpu", "ram"])
        assert len(p._rows) == 2
        p.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.hud_telemetry import TelemetryPanel
        p = TelemetryPanel()
        p.apply_theme(dark_theme)
        p.close()

    def test_timer_active(self, qapp):
        from spidy.ui.widgets.hud_telemetry import TelemetryPanel
        p = TelemetryPanel()
        assert p._timer.isActive()
        p.close()

    def test_row_labels(self, qapp):
        from spidy.ui.widgets.hud_telemetry import TelemetryPanel
        p = TelemetryPanel(["cpu", "ram", "gpu"])
        labels = [r.label for r in p._rows]
        assert "CPU" in labels and "RAM" in labels and "GPU" in labels
        p.close()


# ==============================================================================
# HUDChatOverlay
# ==============================================================================

class TestHUDChatOverlay:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay
        w = HUDChatOverlay()
        assert w is not None
        w.close()

    def test_add_message_user(self, qapp):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay
        w = HUDChatOverlay()
        w.add_message("user", "Hello Spidy")
        assert len(w._messages) == 1
        w.close()

    def test_add_message_assistant(self, qapp):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay
        w = HUDChatOverlay()
        w.add_message("assistant", "Hello! I am Spidy.")
        assert w._messages[-1][0] == "assistant"
        w.close()

    def test_max_visible_cards(self, qapp):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay, _MAX_CARDS
        w = HUDChatOverlay()
        for i in range(10):
            w.add_message("user", f"Message {i}")
        assert len(w._cards) == _MAX_CARDS
        w.close()

    def test_clear_removes_all(self, qapp):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay
        w = HUDChatOverlay()
        w.add_message("user", "test")
        w.clear()
        assert len(w._messages) == 0 and len(w._cards) == 0
        w.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay
        w = HUDChatOverlay()
        w.apply_theme(dark_theme)
        w.close()

    def test_opacity_fades_older_cards(self, qapp):
        from spidy.ui.widgets.hud_chat import HUDChatOverlay
        w = HUDChatOverlay()
        for i in range(3):
            w.add_message("user", f"msg {i}")
        opacities = [c.opacity for c in w._cards]
        # Oldest card should have lowest opacity
        assert opacities[0] < opacities[-1]
        w.close()


# ==============================================================================
# HUDVoiceBar
# ==============================================================================

class TestHUDVoiceBar:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.hud_voice_bar import HUDVoiceBar
        w = HUDVoiceBar()
        assert w is not None
        w.close()

    def test_set_state_listening(self, qapp):
        from spidy.ui.widgets.hud_voice_bar import HUDVoiceBar, _FPS_ACTIVE
        w = HUDVoiceBar()
        w.set_state("listening")
        assert w._state == "listening"
        assert w._timer.interval() == _FPS_ACTIVE
        w.close()

    def test_set_state_idle(self, qapp):
        from spidy.ui.widgets.hud_voice_bar import HUDVoiceBar, _FPS_IDLE
        w = HUDVoiceBar()
        w.set_state("idle")
        assert w._timer.interval() == _FPS_IDLE
        w.close()

    def test_set_amplitude(self, qapp):
        from spidy.ui.widgets.hud_voice_bar import HUDVoiceBar
        w = HUDVoiceBar()
        w.set_amplitude(0.7)
        assert abs(w._amplitude - 0.7) < 0.001
        w.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.hud_voice_bar import HUDVoiceBar
        w = HUDVoiceBar()
        w.apply_theme(dark_theme)
        w.close()


# ==============================================================================
# HUDTaskConsole
# ==============================================================================

class TestHUDTaskConsole:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.hud_task_console import HUDTaskConsole
        w = HUDTaskConsole()
        assert w is not None
        w.close()

    def test_update_progress(self, qapp):
        from spidy.ui.widgets.hud_task_console import HUDTaskConsole, TaskStep
        w = HUDTaskConsole()
        w.update_progress("Find my resume", [
            TaskStep("Search filesystem", "done"),
            TaskStep("Open file",         "running"),
        ])
        assert w._goal == "Find my resume"
        assert len(w._steps) == 2
        w.close()

    def test_task_step_icons(self):
        from spidy.ui.widgets.hud_task_console import TaskStep
        assert TaskStep("x", "done").icon    == "\u2713"
        assert TaskStep("x", "running").icon == "\u27f3"
        assert TaskStep("x", "pending").icon == "\u25cb"
        assert TaskStep("x", "failed").icon  == "\u2717"

    def test_clear(self, qapp):
        from spidy.ui.widgets.hud_task_console import HUDTaskConsole, TaskStep
        w = HUDTaskConsole()
        w.update_progress("Goal", [TaskStep("Step", "done")])
        w.clear()
        assert w._goal == "" and len(w._steps) == 0
        w.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.hud_task_console import HUDTaskConsole
        w = HUDTaskConsole()
        w.apply_theme(dark_theme)
        w.close()


# ==============================================================================
# HUDConfirmation
# ==============================================================================

class TestHUDConfirmation:
    def test_creates_hidden(self, qapp):
        from spidy.ui.widgets.hud_confirmation import HUDConfirmation
        w = HUDConfirmation()
        assert w.isHidden()
        w.close()

    def test_show_confirmation(self, qapp):
        from spidy.ui.widgets.hud_confirmation import HUDConfirmation
        w = HUDConfirmation()
        w.show_confirmation("t1", "g1", "Upload video", "Confirm?")
        assert not w.isHidden()
        assert w._description == "Upload video"
        w.close()

    def test_hide_confirmation(self, qapp):
        from spidy.ui.widgets.hud_confirmation import HUDConfirmation
        w = HUDConfirmation()
        w.show_confirmation("t1", "g1", "Test", "")
        w.hide_confirmation()
        assert w.isHidden()
        w.close()

    def test_confirmed_signal(self, qapp):
        from spidy.ui.widgets.hud_confirmation import HUDConfirmation
        received = []
        w = HUDConfirmation()
        w.confirmed.connect(lambda tid, gid: received.append((tid, gid)))
        w.show_confirmation("task-1", "goal-1", "Test action", "")
        w._on_confirm()
        assert received == [("task-1", "goal-1")]
        w.close()

    def test_cancelled_signal(self, qapp):
        from spidy.ui.widgets.hud_confirmation import HUDConfirmation
        received = []
        w = HUDConfirmation()
        w.cancelled.connect(lambda tid, gid: received.append((tid, gid)))
        w.show_confirmation("task-2", "goal-2", "Test action", "")
        w._on_cancel()
        assert received == [("task-2", "goal-2")]
        w.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.widgets.hud_confirmation import HUDConfirmation
        w = HUDConfirmation()
        w.apply_theme(dark_theme)
        w.close()


# ==============================================================================
# OverlayWindow (M17.2 HUD)
# ==============================================================================

class TestOverlayWindowHUD:
    def test_creates_without_error(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert win is not None
        win.close()

    def test_responsive_size_larger_than_1000(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert win.width() >= 1100
        win.close()

    def test_height_at_least_650(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert win.height() >= 650
        win.close()

    def test_core_widget_exists(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win, "_core")
        win.close()

    def test_left_panel_exists(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win, "_left_panel")
        win.close()

    def test_right_panel_exists(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win, "_right_panel")
        win.close()

    def test_confirmation_card_exists(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win, "_confirmation_card")
        win.close()

    def test_all_states_cycle_no_crash(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        for s in ("idle","wake_ready","listening","thinking","speaking","working","error","idle"):
            win.set_state(s)
        win.close()

    def test_rapid_state_changes(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        for _ in range(20):
            for s in ("idle","listening","thinking","working"):
                win.set_state(s)
        qapp.processEvents()
        win.close()

    def test_add_message(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.add_message("user", "Hello")
        assert len(win._chat_overlay._messages) == 1
        win.close()

    def test_long_message_no_crash(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.add_message("assistant", "x" * 5000)
        win.close()

    def test_update_waveform(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.update_waveform([0.1, 0.5, 0.9, 0.3])
        win.close()

    def test_task_progress_dict_format(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.update_task_progress("My goal", [
            {"description": "Step 1", "status": "done"},
            {"description": "Step 2", "status": "running"},
        ])
        win.close()

    def test_confirmation_show_hide(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.show_confirmation("t1", "g1", "Upload video", "Confirm?")
        assert not win._confirmation_card.isHidden()
        win.hide_confirmation()
        assert win._confirmation_card.isHidden()
        win.close()

    def test_apply_theme(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        from spidy.ui.themes.light import LIGHT_THEME
        win = OverlayWindow(dark_theme, animate=False)
        win.apply_theme(LIGHT_THEME)
        win.close()

    def test_working_state_shows_task_console(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("working")
        assert not win._right_panel._task_console.isHidden()
        win.close()

    def test_idle_state_hides_task_console(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("working")
        win.set_state("idle")
        assert win._right_panel._task_console.isHidden()
        win.close()


# ==============================================================================
# UISignalBridge (M17.2)
# ==============================================================================

class TestUISignalBridgeHUD:
    def test_task_progress_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        b = UISignalBridge()
        b.task_progress_received.connect(lambda g, s: received.append(g))
        b.request_task_progress("Goal X", [])
        assert received == ["Goal X"]

    def test_confirmation_show_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        b = UISignalBridge()
        b.confirmation_show.connect(lambda *a: received.append(a))
        b.request_confirmation_show("t", "g", "desc", "prompt")
        assert received == [("t", "g", "desc", "prompt")]

    def test_confirmation_hide_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        b = UISignalBridge()
        b.confirmation_hide.connect(lambda: received.append(True))
        b.request_confirmation_hide()
        assert received == [True]
