"""
Tests for Milestone 17 -- Glassy Desktop UI Redesign (offscreen)
=================================================================
Tests the new SpidyOrb, TaskPanel, ConfirmationCard widgets and
the extended OverlayWindow (800x600, centered, WORKING state).

All tests use the offscreen Qt platform and do NOT require a display.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtWidgets import QApplication


# -- Shared QApplication fixture -----------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture()
def dark_theme():
    from spidy.ui.themes.dark import DARK_THEME
    return DARK_THEME


# ==============================================================================
# UIState.WORKING
# ==============================================================================

class TestWorkingState:
    def test_working_state_exists(self):
        from spidy.ui.state import UIState
        assert hasattr(UIState, "WORKING")
        assert UIState.WORKING.value == "working"

    def test_working_state_is_string(self):
        from spidy.ui.state import UIState
        assert isinstance(UIState.WORKING.value, str)

    def test_working_transition_from_idle(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine()
        assert sm.transition(UIState.WORKING)
        assert sm.current == UIState.WORKING

    def test_working_transition_to_idle(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine(initial=UIState.WORKING)
        assert sm.transition(UIState.IDLE)
        assert sm.current == UIState.IDLE

    def test_working_transition_from_listening(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine(initial=UIState.LISTENING)
        assert sm.transition(UIState.WORKING)

    def test_working_error_reachable(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine(initial=UIState.WORKING)
        assert sm.transition(UIState.ERROR)

    def test_is_working_property(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine(initial=UIState.WORKING)
        assert sm.is_working

    def test_is_working_false_when_idle(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine()
        assert not sm.is_working

    def test_is_active_when_working(self):
        from spidy.ui.state import UIState, UIStateMachine
        sm = UIStateMachine(initial=UIState.WORKING)
        assert sm.is_active


# ==============================================================================
# SpidyOrb
# ==============================================================================

class TestSpidyOrb:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        assert orb is not None
        orb.close()

    def test_fixed_size(self, qapp):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        assert orb.width()  == 200
        assert orb.height() == 200
        orb.close()

    def test_set_state_idle(self, qapp):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        orb.set_state("idle")
        assert orb._state == "idle"
        orb.close()

    def test_set_all_states(self, qapp):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        for state in ("idle", "wake_ready", "listening", "thinking", "speaking", "working", "error"):
            orb.set_state(state)
            assert orb._state == state
        orb.close()

    def test_apply_theme_runs_without_error(self, qapp, dark_theme):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        orb.apply_theme(dark_theme)
        orb.close()

    def test_set_amplitude_clamps(self, qapp):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        orb.set_amplitude(2.5)
        assert orb._amplitude == 1.0
        orb.set_amplitude(-1.0)
        assert orb._amplitude == 0.0
        orb.close()

    def test_timer_active_at_init(self, qapp):
        from spidy.ui.widgets.spidy_orb import SpidyOrb
        orb = SpidyOrb()
        assert orb._timer.isActive()
        orb.close()


# ==============================================================================
# TaskPanel
# ==============================================================================

class TestTaskPanel:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.task_panel import TaskPanel
        panel = TaskPanel()
        assert panel is not None
        panel.close()

    def test_update_progress_sets_goal(self, qapp):
        from spidy.ui.widgets.task_panel import TaskPanel, TaskStep
        panel = TaskPanel()
        panel.update_progress(
            "Open Edge and search YouTube",
            [TaskStep("Open Edge", "done"), TaskStep("Navigate", "running")]
        )
        assert "Open Edge" in panel._goal_desc.text()
        panel.close()

    def test_update_progress_creates_rows(self, qapp):
        from spidy.ui.widgets.task_panel import TaskPanel, TaskStep
        panel = TaskPanel()
        steps = [TaskStep(f"Step {i}", "pending") for i in range(4)]
        panel.update_progress("My Goal", steps)
        assert len(panel._rows) == 4
        panel.close()

    def test_apply_theme_runs_without_error(self, qapp, dark_theme):
        from spidy.ui.widgets.task_panel import TaskPanel
        panel = TaskPanel()
        panel.apply_theme(dark_theme)
        panel.close()

    def test_task_step_icons(self):
        from spidy.ui.widgets.task_panel import TaskStep
        assert TaskStep("x", "done").icon    == "\u2713"
        assert TaskStep("x", "running").icon == "\u27f3"
        assert TaskStep("x", "pending").icon == "\u25cb"
        assert TaskStep("x", "failed").icon  == "\u2717"


# ==============================================================================
# ConfirmationCard
# ==============================================================================

class TestConfirmationCard:
    def test_creates_without_error(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        card = ConfirmationCard()
        assert card is not None
        card.close()

    def test_hidden_by_default(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        card = ConfirmationCard()
        assert card.isHidden()
        card.close()

    def test_show_confirmation_makes_visible(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        card = ConfirmationCard()
        card.show_confirmation("t1", "g1", "Upload to Instagram", "Confirm?")
        assert not card.isHidden()
        card.close()

    def test_show_confirmation_sets_description(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        card = ConfirmationCard()
        card.show_confirmation("t1", "g1", "Delete all files", "")
        assert "Delete all files" in card._desc_label.text()
        card.close()

    def test_hide_confirmation(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        card = ConfirmationCard()
        card.show_confirmation("t1", "g1", "Test", "")
        card.hide_confirmation()
        assert card.isHidden()
        card.close()

    def test_confirm_signal_emitted(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        received = []
        card = ConfirmationCard()
        card.confirmed.connect(lambda tid, gid: received.append((tid, gid)))
        card.show_confirmation("task-1", "goal-1", "Upload photo", "")
        card._on_confirm()
        assert received == [("task-1", "goal-1")]
        card.close()

    def test_cancel_signal_emitted(self, qapp):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        received = []
        card = ConfirmationCard()
        card.cancelled.connect(lambda tid, gid: received.append((tid, gid)))
        card.show_confirmation("task-2", "goal-2", "Send email", "")
        card._on_cancel()
        assert received == [("task-2", "goal-2")]
        card.close()

    def test_apply_theme_runs_without_error(self, qapp, dark_theme):
        from spidy.ui.widgets.confirmation_card import ConfirmationCard
        card = ConfirmationCard()
        card.apply_theme(dark_theme)
        card.close()


# ==============================================================================
# OverlayWindow (M17 -- extended)
# ==============================================================================

class TestOverlayWindowM17:
    """Tests for the overlay -- updated for M17.2 HUD."""

    def test_overlay_initializes(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert win is not None
        win.close()

    def test_default_size_responsive(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        # M17.2: responsive -- at least 1100px wide and 650px tall
        assert win.width()  >= 1100
        assert win.height() >= 650
        win.close()

    def test_core_widget_exists(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win, "_core")
        win.close()

    def test_task_panel_exists_in_right(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win._right_panel, "_task_console")
        win.close()

    def test_confirmation_card_exists(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        assert hasattr(win, "_confirmation_card")
        win.close()

    def test_sleeping_label(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("idle")
        # _HUDHeader._state_label is _tc (_HUDInfoWidget) -- check via core state
        assert win._core._state == "idle"
        win.close()

    def test_listening_label(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("listening")
        assert win._core._state == "listening"
        win.close()

    def test_processing_label(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("thinking")
        assert win._core._state == "thinking"
        win.close()

    def test_speaking_label(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("speaking")
        assert win._core._state == "speaking"
        win.close()

    def test_working_label(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("working")
        assert win._core._state == "working"
        win.close()

    def test_task_panel_shown_in_working(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("working")
        assert not win._right_panel._task_console.isHidden()
        win.close()

    def test_task_panel_hidden_in_idle(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.set_state("idle")
        assert win._right_panel._task_console.isHidden()
        win.close()

    def test_chat_overlay_receives_messages(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.add_message("user", "test message")
        assert len(win._chat_overlay._messages) == 1
        win.close()

    def test_confirmation_card_show(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.show_confirmation("t1", "g1", "Upload to Instagram", "Confirm?")
        assert not win._confirmation_card.isHidden()
        win.close()

    def test_confirmation_card_hide(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        win.show_confirmation("t1", "g1", "Upload", "")
        win.hide_confirmation()
        assert win._confirmation_card.isHidden()
        win.close()

    def test_all_states_cycle_without_crash(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        for state in ("idle", "wake_ready", "listening", "thinking", "speaking", "working", "error", "idle"):
            win.set_state(state)
        win.close()

    def test_long_text_does_not_break_layout(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        long_text = "This is a very long response from Spidy. " * 50
        win.add_message("assistant", long_text)
        assert len(win._chat_overlay._messages) == 1
        win.close()

    def test_update_task_progress_dict_format(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        steps = [
            {"description": "Open browser", "status": "done"},
            {"description": "Navigate",     "status": "running"},
            {"description": "Search",       "status": "pending"},
        ]
        win.update_task_progress("Test goal", steps)
        assert len(win._right_panel._task_console._steps) == 3
        win.close()

    def test_overlay_remains_responsive_after_state_changes(self, qapp, dark_theme):
        from spidy.ui.overlay import OverlayWindow
        win = OverlayWindow(dark_theme, animate=False)
        for _ in range(10):
            for state in ("idle", "listening", "thinking", "speaking", "working"):
                win.set_state(state)
        qapp.processEvents()   # Must not hang
        win.close()


# ==============================================================================
# UISignalBridge (M17 new signals)
# ==============================================================================

class TestUISignalBridgeM17:
    def test_task_progress_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        bridge.task_progress_received.connect(lambda g, s: received.append((g, s)))
        bridge.request_task_progress("My goal", [{"description": "step1", "status": "done"}])
        assert len(received) == 1
        assert received[0][0] == "My goal"

    def test_confirmation_show_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        bridge.confirmation_show.connect(lambda *a: received.append(a))
        bridge.request_confirmation_show("t1", "g1", "Upload", "Confirm?")
        assert received == [("t1", "g1", "Upload", "Confirm?")]

    def test_confirmation_hide_signal(self, qapp):
        from spidy.ui.overlay import UISignalBridge
        received = []
        bridge = UISignalBridge()
        bridge.confirmation_hide.connect(lambda: received.append(True))
        bridge.request_confirmation_hide()
        assert received == [True]


# ==============================================================================
# Confirmation events
# ==============================================================================

class TestConfirmationEvents:
    def test_granted_event_topic(self):
        from spidy.ui.events import UIConfirmationGrantedEvent
        ev = UIConfirmationGrantedEvent(goal_id="g1", task_id="t1")
        assert ev.topic == "agent.confirmation_granted"

    def test_denied_event_topic(self):
        from spidy.ui.events import UIConfirmationDeniedEvent
        ev = UIConfirmationDeniedEvent(goal_id="g1", task_id="t1")
        assert ev.topic == "agent.confirmation_denied"
