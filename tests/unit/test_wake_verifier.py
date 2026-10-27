"""Tests for the stage-2 Whisper wake-word verifier."""

from __future__ import annotations

import numpy as np
import pytest

from spidy.perception.voice.wake_word.verifier import WakeVerifier, contains_wake_name


@pytest.mark.parametrize(
    "text",
    ["Hey Spidy.", "Hey Spidey!", "Wake up, Spidey.", "hey spiddy", "SPIDY", "Hey Spi dy"],
)
def test_accepts_spellings_of_the_name(text):
    assert contains_wake_name(text)


@pytest.mark.parametrize(
    "text",
    ["", "Hey Siri.", "Hey buddy!", "Hi Speedy!", "Hey Spiky.", "You", "What is the weather"],
)
def test_rejects_other_phrases(text):
    assert not contains_wake_name(text)


def _chunk(value: float = 0.0, n: int = 1280) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def test_ring_buffer_keeps_only_the_window():
    v = WakeVerifier(lambda a: "", sample_rate=16000, window_seconds=1.0)
    for i in range(20):
        v.push(_chunk(i))
    audio = v.snapshot()
    assert 16000 <= len(audio) < 16000 + 1280
    assert audio[-1] == 19  # newest audio is kept
    assert audio[0] > 0     # oldest audio was evicted


def test_snapshot_empty_buffer():
    v = WakeVerifier(lambda a: "", window_seconds=1.0)
    assert len(v.snapshot()) == 0


def test_verify_accepts_and_rejects_by_transcript():
    heard = {"text": "Hey Spidy."}
    v = WakeVerifier(lambda a: heard["text"])
    assert v.verify(_chunk()) == (True, "Hey Spidy.")
    heard["text"] = "Hey buddy."
    assert v.verify(_chunk()) == (False, "Hey buddy.")


def test_verify_fails_open_when_transcription_raises():
    def boom(_a):
        raise RuntimeError("whisper down")

    ok, text = WakeVerifier(boom).verify(_chunk())
    assert ok is True and text == ""


def test_collect_waits_for_trailing_audio(monkeypatch):
    slept = []
    monkeypatch.setattr("spidy.perception.voice.wake_word.verifier.time.sleep", slept.append)
    v = WakeVerifier(lambda a: "", trailing_seconds=0.4)
    v.push(_chunk(1.0))
    assert len(v.collect()) == 1280
    assert slept == [0.4]
