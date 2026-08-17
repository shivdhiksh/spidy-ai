"""
Tests for Milestone 5 — UI State Machine
==========================================
Pure Python tests; no Qt display required.
"""

from __future__ import annotations

import pytest

from spidy.ui.state import UIState, UIStateMachine, UIStateTransitionError


class TestUIState:
    def test_all_states_exist(self):
        states = {s.value for s in UIState}
        assert states == {"idle", "wake_ready", "listening", "thinking", "speaking", "working", "error"}

    def test_str_representation(self):
        assert str(UIState.IDLE) == "idle"
        assert str(UIState.LISTENING) == "listening"
        assert str(UIState.SPEAKING) == "speaking"

    def test_from_value(self):
        assert UIState("idle") == UIState.IDLE
        assert UIState("listening") == UIState.LISTENING

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            UIState("dancing")

    def test_all_states_are_strings(self):
        for s in UIState:
            assert isinstance(s.value, str)


class TestUIStateMachineInit:
    def test_default_state_is_idle(self):
        sm = UIStateMachine()
        assert sm.current == UIState.IDLE

    def test_custom_initial_state(self):
        sm = UIStateMachine(initial=UIState.WAKE_READY)
        assert sm.current == UIState.WAKE_READY

    def test_previous_equals_initial_on_start(self):
        sm = UIStateMachine(initial=UIState.IDLE)
        assert sm.previous == UIState.IDLE

    def test_repr(self):
        sm = UIStateMachine()
        r = repr(sm)
        assert "idle" in r


class TestUIStateMachineTransitions:
    def test_idle_to_listening(self):
        sm = UIStateMachine()
        assert sm.transition(UIState.LISTENING)
        assert sm.current == UIState.LISTENING

    def test_idle_to_wake_ready(self):
        sm = UIStateMachine()
        sm.transition(UIState.WAKE_READY)
        assert sm.current == UIState.WAKE_READY

    def test_wake_ready_to_listening(self):
        sm = UIStateMachine(initial=UIState.WAKE_READY)
        sm.transition(UIState.LISTENING)
        assert sm.current == UIState.LISTENING

    def test_listening_to_thinking(self):
        sm = UIStateMachine(initial=UIState.LISTENING)
        sm.transition(UIState.THINKING)
        assert sm.current == UIState.THINKING

    def test_thinking_to_speaking(self):
        sm = UIStateMachine(initial=UIState.THINKING)
        sm.transition(UIState.SPEAKING)
        assert sm.current == UIState.SPEAKING

    def test_speaking_to_idle(self):
        sm = UIStateMachine(initial=UIState.SPEAKING)
        sm.transition(UIState.IDLE)
        assert sm.current == UIState.IDLE

    def test_error_reachable_from_any_state(self):
        for initial in UIState:
            sm = UIStateMachine(initial=initial)
            sm.transition(UIState.ERROR)
            assert sm.current == UIState.ERROR

    def test_error_to_idle(self):
        sm = UIStateMachine(initial=UIState.ERROR)
        sm.transition(UIState.IDLE)
        assert sm.current == UIState.IDLE

    def test_string_transition(self):
        sm = UIStateMachine()
        sm.transition("listening")
        assert sm.current == UIState.LISTENING

    def test_invalid_string_raises(self):
        sm = UIStateMachine()
        with pytest.raises(UIStateTransitionError):
            sm.transition("flying")

    def test_no_op_same_state(self):
        sm = UIStateMachine()
        result = sm.transition(UIState.IDLE)
        assert result is True
        assert sm.current == UIState.IDLE

    def test_previous_state_tracked(self):
        sm = UIStateMachine()
        sm.transition(UIState.LISTENING)
        assert sm.previous == UIState.IDLE

    def test_previous_state_updated_on_each_transition(self):
        sm = UIStateMachine()
        sm.transition(UIState.LISTENING)
        sm.transition(UIState.THINKING)
        assert sm.previous == UIState.LISTENING
        assert sm.current == UIState.THINKING


class TestUIStateMachineProperties:
    def test_is_idle_true(self):
        sm = UIStateMachine()
        assert sm.is_idle

    def test_is_idle_false_when_listening(self):
        sm = UIStateMachine(initial=UIState.LISTENING)
        assert not sm.is_idle

    def test_is_listening(self):
        sm = UIStateMachine(initial=UIState.LISTENING)
        assert sm.is_listening

    def test_is_speaking(self):
        sm = UIStateMachine(initial=UIState.SPEAKING)
        assert sm.is_speaking

    def test_is_thinking(self):
        sm = UIStateMachine(initial=UIState.THINKING)
        assert sm.is_thinking

    def test_is_active_when_listening(self):
        sm = UIStateMachine(initial=UIState.LISTENING)
        assert sm.is_active

    def test_not_active_when_idle(self):
        sm = UIStateMachine()
        assert not sm.is_active

    def test_not_active_when_wake_ready(self):
        sm = UIStateMachine(initial=UIState.WAKE_READY)
        assert not sm.is_active

    def test_can_transition_to_true(self):
        sm = UIStateMachine()
        assert sm.can_transition_to(UIState.LISTENING)

    def test_can_transition_to_error_always(self):
        for initial in UIState:
            sm = UIStateMachine(initial=initial)
            assert sm.can_transition_to(UIState.ERROR)


class TestUIStateMachineListeners:
    def test_listener_called_on_transition(self):
        events = []
        sm = UIStateMachine()
        sm.add_listener(lambda old, new: events.append((old, new)))
        sm.transition(UIState.LISTENING)
        assert events == [(UIState.IDLE, UIState.LISTENING)]

    def test_listener_not_called_on_no_op(self):
        events = []
        sm = UIStateMachine()
        sm.add_listener(lambda old, new: events.append((old, new)))
        sm.transition(UIState.IDLE)  # same state
        assert events == []

    def test_multiple_listeners(self):
        counts = [0, 0]
        sm = UIStateMachine()
        sm.add_listener(lambda o, n: counts.__setitem__(0, counts[0] + 1))
        sm.add_listener(lambda o, n: counts.__setitem__(1, counts[1] + 1))
        sm.transition(UIState.LISTENING)
        assert counts == [1, 1]

    def test_listener_exception_does_not_crash(self):
        sm = UIStateMachine()
        sm.add_listener(lambda o, n: 1 / 0)  # Will raise ZeroDivisionError
        sm.transition(UIState.LISTENING)  # Should not raise
        assert sm.current == UIState.LISTENING

    def test_remove_listener(self):
        events = []
        fn = lambda old, new: events.append((old, new))
        sm = UIStateMachine()
        sm.add_listener(fn)
        sm.remove_listener(fn)
        sm.transition(UIState.LISTENING)
        assert events == []

    def test_duplicate_listener_not_added(self):
        events = []
        fn = lambda old, new: events.append(1)
        sm = UIStateMachine()
        sm.add_listener(fn)
        sm.add_listener(fn)  # Duplicate
        sm.transition(UIState.LISTENING)
        assert len(events) == 1  # Only called once


class TestUIStateMachineReset:
    def test_reset_returns_to_idle(self):
        sm = UIStateMachine(initial=UIState.SPEAKING)
        sm.reset()
        assert sm.current == UIState.IDLE

    def test_reset_does_not_notify(self):
        events = []
        sm = UIStateMachine(initial=UIState.SPEAKING)
        sm.add_listener(lambda o, n: events.append(1))
        sm.reset()
        assert events == []
