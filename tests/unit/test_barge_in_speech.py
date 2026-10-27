"""Talking over Spidy must stop her mid-answer and start listening to the new question."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from spidy.perception.voice.audio_capture import CaptureMode
from spidy.perception.voice.engine import VoiceEngine
from spidy.voice.barge_in import BargeInDetector, is_echo_of
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.interruption import InterruptCommand
from spidy.voice.streaming_tts import StreamingTTSWrapper

CHUNK = 1280
RNG = np.random.default_rng(1)
SPOKEN = "The weather in Hyderabad is partly cloudy with a high of thirty degrees today."


def _chunk(rms: float) -> np.ndarray:
    x = RNG.normal(0.0, 1.0, CHUNK).astype(np.float32)
    return (x / np.sqrt(np.mean(x ** 2)) * rms).astype(np.float32)


class _Handler:
    def detect(self, text: str):
        return InterruptCommand.STOP if "stop" in text.lower() else None


class _TTS:
    is_speaking = True

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _detector(heard: str, **kw):
    bus = MagicMock()
    det = BargeInDetector(_Handler(), bus, **kw)
    det._model_loaded = True
    det._transcribe = lambda audio: heard          # type: ignore[method-assign]
    tts = _TTS()
    det.start(tts, spoken_text=SPOKEN)
    return det, tts, bus


def _wait(cond, timeout=3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _feed_echo(det, n=12, rms=0.05):
    for _ in range(n):
        det.feed_chunk(_chunk(rms))


# ── echo vs. the user ─────────────────────────────────────────────────────────

def test_spidys_own_words_are_echo():
    assert is_echo_of("the weather in hyderabad is partly cloudy", SPOKEN)
    assert is_echo_of("whether in hyderabad is partly clowdy", SPOKEN)        # garbled by tiny.en


def test_a_new_question_is_not_echo():
    assert not is_echo_of("open youtube and play some music", SPOKEN)
    assert not is_echo_of("what is the weather", SPOKEN)                      # one shared word only
    assert not is_echo_of("", SPOKEN)


# ── detector ──────────────────────────────────────────────────────────────────

def test_user_speaking_over_spidy_stops_tts_and_keeps_the_audio():
    det, tts, bus = _detector("open youtube please")
    try:
        _feed_echo(det)                                   # Spidy's echo sets the baseline
        loud = [_chunk(0.25) for _ in range(14)]
        for c in loud:
            det.feed_chunk(c)
        assert _wait(lambda: det.interrupted)
        assert tts.stopped
        event = bus.publish_threadsafe.call_args.args[0]
        assert event.command == "barge_in" and event.utterance == "open youtube please"

        chunks = det.consume_interrupt()
        assert chunks and len(chunks) >= 3
        assert max(float(np.sqrt(np.mean(c ** 2))) for c in chunks) > 0.2
    finally:
        det.stop()


def test_chunk_in_flight_after_hand_off_goes_to_the_recorder():
    det, tts, _ = _detector("open youtube please")
    got = []
    det.set_handoff_sink(got.append)
    try:
        _feed_echo(det)
        for _ in range(14):
            det.feed_chunk(_chunk(0.25))
        assert _wait(lambda: det.interrupted)
        det.consume_interrupt()
        late = _chunk(0.25)
        det.feed_chunk(late)
        assert got and got[0] is late
    finally:
        det.stop()


def test_spidys_own_echo_never_interrupts():
    det, tts, _ = _detector(SPOKEN)                       # even if it opened a candidate, it is echo
    try:
        _feed_echo(det, n=40, rms=0.05)                   # steady echo: no onset at all
        time.sleep(0.2)
        assert not det.interrupted and not tts.stopped

        for _ in range(14):                               # a louder echo burst is transcribed...
            det.feed_chunk(_chunk(0.25))
        time.sleep(0.5)
        assert not det.interrupted and not tts.stopped    # ...and rejected as Spidy's own words
    finally:
        det.stop()


def test_stop_phrase_still_stops_without_taking_over_the_mic():
    det, tts, bus = _detector("spidy stop")
    try:
        _feed_echo(det)
        for _ in range(14):
            det.feed_chunk(_chunk(0.25))
        assert _wait(lambda: tts.stopped)
        assert not det.interrupted                        # plain stop: normal flush + relisten path
        assert bus.publish_threadsafe.call_args.args[0].command == "stop"
    finally:
        det.stop()


def test_noise_hallucination_is_ignored():
    det, tts, _ = _detector("you")
    try:
        _feed_echo(det)
        for _ in range(30):
            det.feed_chunk(_chunk(0.25))
        time.sleep(0.5)
        assert not det.interrupted and not tts.stopped
    finally:
        det.stop()


def test_interrupt_on_speech_can_be_disabled():
    det, tts, _ = _detector("open youtube please", interrupt_on_speech=False)
    try:
        _feed_echo(det)
        for _ in range(14):
            det.feed_chunk(_chunk(0.25))
        time.sleep(0.5)
        assert not det.interrupted and not tts.stopped
    finally:
        det.stop()


# ── engine hand-off ───────────────────────────────────────────────────────────

class _Wake:
    chunk_size = CHUNK
    model_name = "hey_spidy"

    def process_chunk(self, chunk):
        return 0.0

    def reset_buffer(self):
        pass

    def unload(self):
        pass


def test_engine_keeps_the_interrupting_speech_in_front_of_new_audio():
    eng = VoiceEngine(bus=MagicMock(), wake_word_model=_Wake(),
                      recognizer=MagicMock(spec=["unload"]), tts=MagicMock(is_speaking=False))
    eng.set_speaking(True)
    newer = _chunk(0.2)
    older = [_chunk(0.2), _chunk(0.2)]
    eng._capture_engine.set_mode(CaptureMode.BARGE_IN)

    def preroll():
        assert eng._capture_engine.mode == CaptureMode.RECORDING   # mode flips before audio is taken
        eng._capture_buffer.append(newer)                           # a chunk arriving meanwhile
        return older

    eng.resume_listening_after_interrupt(preroll)

    assert eng._capture_buffer[:2] == older and eng._capture_buffer[2] is newer
    assert not eng.is_speaking and not eng.is_flushing
    assert eng._wake_detected.is_set() and eng._keep_buffered_audio is True


# ── controller ────────────────────────────────────────────────────────────────

def _controller(barge_interrupted: bool):
    engine = MagicMock()
    engine._capture_engine = MagicMock()
    barge = MagicMock()
    barge.is_loaded = True
    barge.interrupted = barge_interrupted
    barge.interrupt_text = "open youtube"
    barge.consume_interrupt.return_value = [_chunk(0.2)]
    session = SimpleNamespace(
        is_paused=False, is_active=True, session_id="s",
        mark_speaking=AsyncMock(), mark_audio_flush=AsyncMock(), mark_response_complete=AsyncMock(),
    )
    tts = MagicMock()
    tts.speak = AsyncMock()
    tts.is_speaking = False
    cvc = ContinuousVoiceController(
        voice_engine=engine, brain=MagicMock(), bus=MagicMock(publish=AsyncMock()),
        session_manager=session, streaming_tts=tts, interruption_handler=MagicMock(),
        barge_in=barge,
    )
    return cvc, engine, barge, session


def test_controller_resumes_listening_instead_of_flushing_when_interrupted():
    cvc, engine, barge, session = _controller(barge_interrupted=True)
    asyncio.run(cvc._on_brain_response(SimpleNamespace(response_text="Here is a long answer. And more.")))

    barge.start.assert_called_once()
    assert barge.start.call_args.kwargs["spoken_text"].startswith("Here is a long answer")
    engine.resume_listening_after_interrupt.assert_called_once_with(barge.consume_interrupt)
    engine.return_to_sleeping.assert_not_called()           # no flush, no wake-word mode
    engine.enter_audio_flush.assert_not_called()
    engine.signal_relisten.assert_not_called()
    session.mark_response_complete.assert_awaited()
    barge.stop.assert_called_once()


def test_controller_flushes_and_relistens_normally_when_not_interrupted():
    cvc, engine, barge, session = _controller(barge_interrupted=False)
    asyncio.run(cvc._on_brain_response(SimpleNamespace(response_text="Here is a long answer. And more.")))

    engine.resume_listening_after_interrupt.assert_not_called()
    engine.return_to_sleeping.assert_called()
    engine.signal_relisten.assert_called_once()


# ── a stopped engine must speak again ─────────────────────────────────────────

class _StopFlagEngine:
    """Mimics Piper: play_buffer() silently skips everything while its stop flag is set."""

    is_speaking = False
    voice_name = "fake"

    def __init__(self) -> None:
        self.stop_flag = False
        self.played: list[str] = []

    def stop(self) -> None:
        self.stop_flag = True

    def clear_stop(self) -> None:
        self.stop_flag = False

    async def synthesize(self, text: str):
        return text

    async def play_buffer(self, buf) -> None:
        if not self.stop_flag:
            self.played.append(buf)


def test_spidy_speaks_again_after_being_stopped():
    eng = _StopFlagEngine()
    tts = StreamingTTSWrapper(eng)
    tts.stop()
    asyncio.run(tts.speak("Hello there. Nice to see you."))
    assert eng.played == ["Hello there.", "Nice to see you."]
