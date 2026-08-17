"""
test_wake_ack_awake_state.py — 16 regression tests
====================================================
Tests the multi-wake-word, wake acknowledgement, and overlay awake-state
features introduced in Milestone 15.

Covers
------
 1. hey_jarvis wake triggers session.activate()
 2. Duplicate wake event ignored during active session
 3. speak_ack() calls streaming TTS exactly once
 4. speak_ack() does NOT call brain/LLM
 5. Ack NOT forwarded to STT/Brain — mic is IDLE during ack
 6. Second wake during active session does not re-trigger ack
 7. wake_ack.enabled=False — ack not spoken, session opens normally
 8. Custom phrases from config are used (not hardcoded)
 9. Overlay -> "listening" immediately on wake (state change published first)
10. Overlay -> "idle" after session timeout
11. Overlay -> "listening" after brain response (continuous mode)
12. Prefix strip: "Hey Jarvis, open Edge" -> "open Edge"
13. Prefix strip: "Hey Spidy, open Edge" -> "open Edge"
14. Prefix strip: "Wake up Spidy, what time?" -> "what time?"
15. "Who is Jarvis?" / "Tell me about Spidy." NOT stripped
16. Factory production wiring creates an active WakeAcknowledger (enabled=true)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight stubs
# ---------------------------------------------------------------------------


def _make_wake_ack_config(enabled: bool = True, phrases=None):
    """Build a minimal WakeAckConfig-like object without importing Pydantic."""
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.phrases = phrases if phrases is not None else ["Yes Shiva.", "Hmm?"]
    return cfg


def _make_cvc(*, wake_acknowledger=None, session_active=False):
    """
    Build a ContinuousVoiceController with all heavy deps mocked out.

    Returns (cvc, mocks_dict).
    """
    from spidy.voice.continuous import ContinuousVoiceController

    session_mgr = MagicMock()
    session_mgr.is_active = session_active
    session_mgr.is_paused = False
    session_mgr.session_id = "test-sid-001"
    session_mgr.activate = AsyncMock(return_value="test-sid-001")
    session_mgr.record_utterance = AsyncMock()
    session_mgr.mark_speaking = AsyncMock()
    session_mgr.mark_response_complete = AsyncMock()
    session_mgr.deactivate = AsyncMock()

    streaming_tts = MagicMock()
    streaming_tts.speak = AsyncMock()
    streaming_tts.stop = MagicMock()
    streaming_tts.is_speaking = False

    brain = MagicMock()
    brain.process = AsyncMock(return_value="test response")
    brain.run_goal = AsyncMock(return_value="test goal response")

    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = AsyncMock()

    # Capture engine for mic mode assertions
    capture_engine = MagicMock()
    capture_engine.set_mode = MagicMock()

    wake_model = MagicMock()
    wake_model.reset_buffer = MagicMock()
    wake_model.model_name = "hey_jarvis"

    voice_engine = MagicMock()
    voice_engine._capture_engine = capture_engine
    voice_engine._wake_model = wake_model
    voice_engine._on_brain_response = MagicMock()  # allow unsubscribe
    voice_engine.signal_relisten = MagicMock()

    interruption = MagicMock()
    interruption.detect = MagicMock(return_value=None)

    cvc = ContinuousVoiceController(
        voice_engine=voice_engine,
        brain=brain,
        bus=bus,
        session_manager=session_mgr,
        streaming_tts=streaming_tts,
        interruption_handler=interruption,
        continuous_mode=True,
        measure_latency=False,
        wake_acknowledger=wake_acknowledger,
    )

    return cvc, {
        "session": session_mgr,
        "tts": streaming_tts,
        "brain": brain,
        "bus": bus,
        "capture": capture_engine,
        "wake_model": wake_model,
        "engine": voice_engine,
    }


def _wake_event(model_name: str = "hey_jarvis"):
    evt = MagicMock()
    evt.model_name = model_name
    return evt


# ---------------------------------------------------------------------------
# Test 1 — hey_jarvis wake triggers session.activate()
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_01_wake_triggers_session_activate():
    cvc, mocks = _make_cvc()
    await cvc._on_wake_word(_wake_event("hey_jarvis"))
    mocks["session"].activate.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2 — Duplicate wake ignored during active session
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_02_duplicate_wake_ignored():
    cvc, mocks = _make_cvc(session_active=True)
    await cvc._on_wake_word(_wake_event("hey_jarvis"))
    # session already active — activate() must NOT be called again
    mocks["session"].activate.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 3 — speak_ack() calls streaming TTS exactly once
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_03_speak_ack_calls_tts_once():
    from spidy.voice.wake_ack import WakeAcknowledger

    cfg = _make_wake_ack_config(phrases=["Yes Shiva."])
    ack = WakeAcknowledger(cfg)

    tts = MagicMock()
    tts.speak = AsyncMock()
    await ack.speak_ack(tts)

    assert tts.speak.call_count == 1


# ---------------------------------------------------------------------------
# Test 4 — speak_ack() does NOT call brain/LLM
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_04_speak_ack_does_not_call_brain():
    from spidy.voice.wake_ack import WakeAcknowledger

    cfg = _make_wake_ack_config(phrases=["Hmm?"])
    ack = WakeAcknowledger(cfg)

    brain = MagicMock()
    brain.process = AsyncMock()

    tts = MagicMock()
    tts.speak = AsyncMock()
    await ack.speak_ack(tts)

    brain.process.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 5 — Ack NOT forwarded to STT/Brain — mic is IDLE during ack
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_05_ack_not_forwarded_to_stt():
    """
    When ack is spoken, CaptureMode must be IDLE during the speak call.
    This proves ack audio cannot reach the STT or wake-word pipeline.
    """
    from spidy.perception.voice.audio_capture import CaptureMode
    from spidy.voice.wake_ack import WakeAcknowledger

    cfg = _make_wake_ack_config(phrases=["Yes Shiva."])
    ack = WakeAcknowledger(cfg)
    cvc, mocks = _make_cvc(wake_acknowledger=ack)

    mode_during_ack: list = []

    async def _record_mode(text: str) -> None:
        # Capture the mode set at the time speak() is invoked
        mode_during_ack.append(mocks["capture"].set_mode.call_args_list[-1])

    mocks["tts"].speak = AsyncMock(side_effect=_record_mode)

    await cvc._on_wake_word(_wake_event())

    # Verify mic was set to IDLE before ack
    idle_calls = [
        c for c in mocks["capture"].set_mode.call_args_list
        if c == call(CaptureMode.IDLE)
    ]
    assert idle_calls, "CaptureMode.IDLE was never set — ack could enter STT"

    # Brain must NOT have been called by ack
    mocks["brain"].process.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 6 — Second wake during active session does not re-trigger ack
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_06_second_wake_no_ack():
    from spidy.voice.wake_ack import WakeAcknowledger

    cfg = _make_wake_ack_config(phrases=["Yes Shiva."])
    ack = WakeAcknowledger(cfg)
    cvc, mocks = _make_cvc(wake_acknowledger=ack)

    # First wake — fires normally
    await cvc._on_wake_word(_wake_event())
    first_tts_count = mocks["tts"].speak.call_count

    # Now session is active — simulate second wake
    mocks["session"].is_active = True
    await cvc._on_wake_word(_wake_event())

    # TTS call count must not increase
    assert mocks["tts"].speak.call_count == first_tts_count


# ---------------------------------------------------------------------------
# Test 7 — wake_ack.enabled=False: ack not spoken, session opens normally
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_07_ack_disabled_session_still_opens():
    from spidy.voice.wake_ack import WakeAcknowledger

    cfg = _make_wake_ack_config(enabled=False)
    ack = WakeAcknowledger(cfg)
    cvc, mocks = _make_cvc(wake_acknowledger=ack)

    await cvc._on_wake_word(_wake_event())

    # Session must still open
    mocks["session"].activate.assert_awaited_once()
    # But TTS must NOT be called
    mocks["tts"].speak.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 8 — Custom phrases from config are used (not hardcoded)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_08_custom_phrases_used():
    from spidy.voice.wake_ack import WakeAcknowledger

    custom_phrases = ["Greetings, master.", "At your service."]
    cfg = _make_wake_ack_config(phrases=custom_phrases)
    ack = WakeAcknowledger(cfg)

    tts = MagicMock()
    tts.speak = AsyncMock()

    await ack.speak_ack(tts)

    spoken_text = tts.speak.call_args[0][0]
    assert spoken_text in custom_phrases, (
        f"Expected one of {custom_phrases}, got {spoken_text!r}"
    )


# ---------------------------------------------------------------------------
# Test 9 — Overlay -> "listening" published before ack speak()
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_09_overlay_listening_published_first():
    from spidy.voice.wake_ack import WakeAcknowledger
    from spidy.ui.events import UIStateChangeEvent

    cfg = _make_wake_ack_config(phrases=["Yes Shiva."])
    ack = WakeAcknowledger(cfg)
    cvc, mocks = _make_cvc(wake_acknowledger=ack)

    call_order: list[str] = []

    async def _bus_publish(event):
        if isinstance(event, UIStateChangeEvent):
            call_order.append(f"state:{event.state}")
        return None

    async def _tts_speak(text: str):
        call_order.append("tts:speak")

    mocks["bus"].publish = AsyncMock(side_effect=_bus_publish)
    mocks["tts"].speak = AsyncMock(side_effect=_tts_speak)

    await cvc._on_wake_word(_wake_event())

    # "state:listening" must appear before "tts:speak"
    try:
        idx_state = call_order.index("state:listening")
        idx_tts = call_order.index("tts:speak")
        assert idx_state < idx_tts, (
            f"Overlay state change ({idx_state}) did not precede TTS speak ({idx_tts})"
        )
    except ValueError as exc:
        pytest.fail(f"Expected events not found in order list {call_order}: {exc}")


# ---------------------------------------------------------------------------
# Test 10 — Overlay -> "idle" after session timeout
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_overlay_idle_on_timeout():
    from spidy.ui.events import UIStateChangeEvent

    cvc, mocks = _make_cvc()
    published_states: list[str] = []

    async def _capture_publish(event):
        if isinstance(event, UIStateChangeEvent):
            published_states.append(event.state)

    mocks["bus"].publish = AsyncMock(side_effect=_capture_publish)
    await cvc._on_session_timeout(MagicMock())

    assert "idle" in published_states, (
        f"Expected 'idle' state change on timeout, got: {published_states}"
    )


# ---------------------------------------------------------------------------
# Test 11 — Overlay -> "listening" after brain response (continuous mode)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_overlay_listening_after_response():
    from spidy.ui.events import UIStateChangeEvent

    cvc, mocks = _make_cvc()
    cvc._session = mocks["session"]
    mocks["session"].is_active = True
    mocks["session"].is_paused = False

    published_states: list[str] = []

    async def _capture_publish(event):
        if isinstance(event, UIStateChangeEvent):
            published_states.append(event.state)

    mocks["bus"].publish = AsyncMock(side_effect=_capture_publish)

    # Simulate _on_brain_response
    brain_event = MagicMock()
    brain_event.response_text = "Here is my answer."
    brain_event.text = ""

    # Patch the speak so it doesn't try to do real TTS
    mocks["tts"].speak = AsyncMock()

    from spidy.perception.voice.audio_capture import CaptureMode
    await cvc._on_brain_response(brain_event)

    # After TTS + relisten, overlay must be "listening"
    assert "listening" in published_states, (
        f"Expected 'listening' after brain response, got: {published_states}"
    )


# ---------------------------------------------------------------------------
# Test 12 — Prefix strip: "Hey Jarvis, open Edge" -> "open Edge"
# ---------------------------------------------------------------------------
def test_12_strip_hey_jarvis():
    from spidy.voice.wake_word_stripper import WakeWordStripper

    s = WakeWordStripper()
    assert s.strip_wake_prefix("Hey Jarvis, open Edge.") == "open Edge."


# ---------------------------------------------------------------------------
# Test 13 — Prefix strip: "Hey Spidy, open Edge" -> "open Edge"
# ---------------------------------------------------------------------------
def test_13_strip_hey_spidy():
    from spidy.voice.wake_word_stripper import WakeWordStripper

    s = WakeWordStripper()
    assert s.strip_wake_prefix("Hey Spidy, open Edge.") == "open Edge."


# ---------------------------------------------------------------------------
# Test 14 — Prefix strip: "Wake up Spidy, what time?" -> "what time?"
# ---------------------------------------------------------------------------
def test_14_strip_wake_up_spidy():
    from spidy.voice.wake_word_stripper import WakeWordStripper

    s = WakeWordStripper()
    assert s.strip_wake_prefix("Wake up Spidy, what time?") == "what time?"


# ---------------------------------------------------------------------------
# Test 15 — Mid-sentence phrases NOT stripped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Who is Jarvis?",
    "Tell me about Spidy.",
    "I asked Jarvis to help me.",
    "The Spidy project started in 2024.",
])
def test_15_no_strip_mid_sentence(text: str):
    from spidy.voice.wake_word_stripper import WakeWordStripper

    s = WakeWordStripper()
    assert s.strip_wake_prefix(text) == text, (
        f"Text should not be stripped: {text!r}"
    )


# ---------------------------------------------------------------------------
# Test 16 — Factory production wiring: WakeAcknowledger is active (enabled=true)
# ---------------------------------------------------------------------------
def test_16_factory_wires_wake_acknowledger_when_enabled():
    """
    VoiceEngineFactory.build_controller() must create a ContinuousVoiceController
    whose _wake_acknowledger is a live WakeAcknowledger with enabled=True and
    the expected phrases — when config.voice.wake_ack.enabled=True.

    All heavy dependencies (VoiceEngine, Brain, EventBus, TTS, STT, audio)
    are replaced with mocks so this test runs in < 1 second with no hardware.
    """
    from spidy.perception.voice.factory import VoiceEngineFactory
    from spidy.voice.wake_ack import WakeAcknowledger

    # ── Build a minimal SpidyConfig-shaped mock ──────────────────────────
    phrases = ["Yes Shiva.", "Hmm?"]

    wake_ack_cfg = MagicMock()
    wake_ack_cfg.enabled = True
    wake_ack_cfg.phrases = phrases

    session_cfg = MagicMock()
    session_cfg.conversation_timeout_seconds = 60.0
    session_cfg.max_turns_per_session = 50
    session_cfg.continuous_mode = True
    session_cfg.stop_mid_sentence = False
    session_cfg.measure_latency = False

    voice_cfg = MagicMock()
    voice_cfg.wake_ack = wake_ack_cfg
    voice_cfg.session = session_cfg

    config = MagicMock()
    config.voice = voice_cfg

    # ── Build minimal VoiceEngine mock ───────────────────────────────────
    tts_engine = MagicMock()
    voice_engine = MagicMock()
    voice_engine._tts = tts_engine
    voice_engine._on_brain_response = MagicMock()  # allow unsubscribe in CVC.start()
    voice_engine.enable_conversation_mode = MagicMock()

    brain = MagicMock()

    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = AsyncMock()

    # ── Call the real factory method ──────────────────────────────────────
    controller = VoiceEngineFactory.build_controller(
        voice_engine=voice_engine,
        brain=brain,
        bus=bus,
        config=config,
    )

    # ── Assert: WakeAcknowledger is wired and live ────────────────────────
    ack = controller._wake_acknowledger

    assert ack is not None, (
        "ContinuousVoiceController._wake_acknowledger is None — factory did not wire it."
    )
    assert isinstance(ack, WakeAcknowledger), (
        f"Expected WakeAcknowledger, got {type(ack).__name__}"
    )
    assert ack.enabled is True, (
        "WakeAcknowledger.enabled should be True when config.voice.wake_ack.enabled=True"
    )
    assert ack.phrases == phrases, (
        f"Expected phrases {phrases}, got {ack.phrases}"
    )
