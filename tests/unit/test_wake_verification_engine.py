"""VoiceEngine wake path with stage-2 Whisper verification (stubs, no audio hardware)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np

from spidy.perception.voice.engine import VoiceEngine, WakeWordDetectedEvent


class _AlwaysDetect:
    """Wake model stub that reports a detection on every chunk."""

    chunk_size = 1280
    model_name = "hey_spidy"
    wake_phrase = "Hey Spidy"
    last_detected_model = "hey_spidy"
    last_detected_phrase = "Hey Spidy"
    is_detected = True

    def process_chunk(self, chunk):
        return 0.9

    def reset_buffer(self):
        pass

    def unload(self):
        pass


class _Recognizer:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def quick_transcribe(self, audio):
        self.calls += 1
        return self.text

    def unload(self):
        pass


def _engine(text: str):
    bus = MagicMock()
    published: list = []
    done = threading.Event()

    def publish(event):
        published.append(event)
        done.set()

    bus.publish_threadsafe.side_effect = publish
    rec = _Recognizer(text)
    eng = VoiceEngine(
        bus=bus,
        wake_word_model=_AlwaysDetect(),
        recognizer=rec,
        tts=MagicMock(),
        wake_verify=True,
        wake_verify_window=2.0,
    )
    eng._wake_verifier._trailing = 0.0  # no real wait in tests
    return eng, rec, published, done


def _feed(eng, n=3):
    for _ in range(n):
        eng._process_wake_word(np.zeros(1280, dtype=np.float32))


def test_verified_wake_publishes_event():
    eng, rec, published, done = _engine("Hey Spidy.")
    _feed(eng)
    assert done.wait(2.0)
    assert any(isinstance(e, WakeWordDetectedEvent) for e in published)
    assert eng._wake_detected.is_set()


def test_rejected_wake_publishes_nothing():
    eng, rec, published, done = _engine("Hey buddy.")
    _feed(eng, n=1)
    # Wait for the worker to finish, then check nothing fired.
    for _ in range(200):
        if not eng._verifying and rec.calls:
            break
        threading.Event().wait(0.01)
    assert rec.calls == 1
    assert not published
    assert not eng._wake_detected.is_set()


def _engine_scored(text: str, score: float):
    eng, rec, published, done = _engine(text)
    eng._wake_model.process_chunk = lambda chunk: score        # type: ignore[method-assign]
    return eng, rec, published, done


def test_confident_detection_with_no_words_heard_is_accepted_after_one_retry():
    eng, rec, published, done = _engine_scored("", 0.95)
    _feed(eng, n=1)
    assert done.wait(2.0)
    assert rec.calls == 2                                       # first look, then a longer one
    assert any(isinstance(e, WakeWordDetectedEvent) for e in published)


def test_weak_detection_with_no_words_heard_is_rejected():
    eng, rec, published, done = _engine_scored("", 0.5)
    _feed(eng, n=1)
    for _ in range(200):
        if not eng._verifying and rec.calls:
            break
        threading.Event().wait(0.01)
    assert rec.calls == 1 and not published


def test_confident_detection_with_other_words_is_still_rejected():
    eng, rec, published, done = _engine_scored("Thank you.", 0.99)
    _feed(eng, n=1)
    for _ in range(200):
        if not eng._verifying and rec.calls:
            break
        threading.Event().wait(0.01)
    assert rec.calls == 1 and not published


def test_repeat_detections_while_verifying_start_one_check():
    eng, rec, published, done = _engine("Hey Buddy.")
    release = threading.Event()
    original = rec.quick_transcribe

    def slow(audio):
        release.wait(2.0)  # hold the first check open while more chunks arrive
        return original(audio)

    eng._wake_verifier._transcribe = slow
    _feed(eng, n=10)  # every chunk "detects", but only one check may run
    release.set()
    for _ in range(200):
        if not eng._verifying and rec.calls:
            break
        threading.Event().wait(0.01)
    assert rec.calls == 1


def test_verification_disabled_without_quick_transcribe():
    eng = VoiceEngine(
        bus=MagicMock(),
        wake_word_model=_AlwaysDetect(),
        recognizer=MagicMock(spec=["unload"]),
        tts=MagicMock(),
        wake_verify=True,
    )
    assert eng._wake_verifier is None
