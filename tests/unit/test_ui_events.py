"""
Tests for Milestone 5 — UI Events
===================================
All pure-dataclass tests; no Qt display required.
"""

from __future__ import annotations

import pytest

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


class TestUIShowEvent:
    def test_topic(self):
        assert UIShowEvent.topic == "ui.show"

    def test_defaults(self):
        e = UIShowEvent()
        assert e.edge == "top-right"
        assert e.animate is True
        assert e.source == ""

    def test_custom_edge(self):
        e = UIShowEvent(edge="bottom-left", source="wake_word")
        assert e.edge == "bottom-left"
        assert e.source == "wake_word"


class TestUIHideEvent:
    def test_topic(self):
        assert UIHideEvent.topic == "ui.hide"

    def test_defaults(self):
        e = UIHideEvent()
        assert e.animate is True

    def test_no_animate(self):
        e = UIHideEvent(animate=False)
        assert e.animate is False


class TestUIStateChangeEvent:
    def test_topic(self):
        assert UIStateChangeEvent.topic == "ui.state_change"

    def test_all_states_constructible(self):
        for state in ("idle", "wake_ready", "listening", "thinking", "speaking", "error"):
            e = UIStateChangeEvent(state=state)
            assert e.state == state

    def test_detail_field(self):
        e = UIStateChangeEvent(state="listening", detail="Transcribing…")
        assert e.detail == "Transcribing…"


class TestUIMessageEvent:
    def test_topic(self):
        assert UIMessageEvent.topic == "ui.message"

    def test_user_message(self):
        e = UIMessageEvent(role="user", text="Hello Spidy")
        assert e.role == "user"
        assert e.text == "Hello Spidy"

    def test_assistant_message(self):
        e = UIMessageEvent(role="assistant", text="Hello! How can I help?")
        assert e.role == "assistant"

    def test_session_id(self):
        e = UIMessageEvent(role="user", text="test", session_id="sess-abc")
        assert e.session_id == "sess-abc"


class TestUINotifyEvent:
    def test_topic(self):
        assert UINotifyEvent.topic == "ui.notify"

    def test_defaults(self):
        e = UINotifyEvent(title="Test")
        assert e.level == "info"
        assert e.duration_ms == 4000
        assert e.body == ""

    def test_all_levels(self):
        for level in ("info", "success", "warning", "error"):
            e = UINotifyEvent(title="T", level=level)
            assert e.level == level

    def test_persistent(self):
        e = UINotifyEvent(title="Persistent", duration_ms=0)
        assert e.duration_ms == 0


class TestUIWaveformDataEvent:
    def test_topic(self):
        assert UIWaveformDataEvent.topic == "ui.waveform_data"

    def test_empty_amplitudes(self):
        e = UIWaveformDataEvent()
        assert e.amplitudes == []

    def test_amplitudes(self):
        amps = [0.1, 0.5, 0.9, 0.3]
        e = UIWaveformDataEvent(amplitudes=amps)
        assert e.amplitudes == amps


class TestUIThemeChangeEvent:
    def test_topic(self):
        assert UIThemeChangeEvent.topic == "ui.theme_change"

    def test_dark(self):
        e = UIThemeChangeEvent(theme="dark")
        assert e.theme == "dark"

    def test_light(self):
        e = UIThemeChangeEvent(theme="light")
        assert e.theme == "light"


class TestOutboundEvents:
    def test_ready_topic(self):
        assert UIReadyEvent.topic == "ui.ready"

    def test_closed_topic(self):
        assert UIClosedEvent.topic == "ui.closed"

    def test_hotkey_pressed_topic(self):
        assert UIHotkeyPressedEvent.topic == "ui.hotkey_pressed"

    def test_hotkey_pressed_fields(self):
        e = UIHotkeyPressedEvent(hotkey="ctrl+space")
        assert e.hotkey == "ctrl+space"

    def test_mic_clicked_topic(self):
        assert UIMicButtonClickedEvent.topic == "ui.mic_button_clicked"

    def test_settings_opened_topic(self):
        assert UISettingsOpenedEvent.topic == "ui.settings_opened"

    def test_ready_instantiable(self):
        e = UIReadyEvent()
        assert e.topic == "ui.ready"

    def test_all_outbound_events_instantiable(self):
        for cls in [
            UIReadyEvent, UIClosedEvent, UIMicButtonClickedEvent,
            UISettingsOpenedEvent,
        ]:
            obj = cls()
            assert obj.topic

    def test_hotkey_event_default_hotkey(self):
        e = UIHotkeyPressedEvent()
        assert e.hotkey == ""
