"""TV-proof wake word, background-speech filter, live captions, emotional pitch and the code writer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from spidy.perception.voice.engine import VoiceEngine, looks_like_background_speech, trim_process_memory
from spidy.perception.voice.wake_word.verifier import WakeVerifier, is_wake_phrase

CHUNK = 1280


# ── stage 2: only a real call counts ─────────────────────────────────────────

@pytest.mark.parametrize("heard", ["Hey Spidey!", "Hey Spidy", "Ok Spidey, who developed you?", "hey, spidey.",
                                   "Wake up Spidy", "Spidey, open Chrome", "Okay Spidy", "Hey Spidey.  Yeah."])
def test_real_calls_are_accepted(heard):
    assert is_wake_phrase(heard)


@pytest.mark.parametrize("heard", ["", "I will see you in the next video.", "Hey, it's Bodie",
                                   "and then Spidey jumped off the roof", "Subscribe to our channel.",
                                   "Thank you.", "Hey buddy", "Hey Siri", "speedy delivery"])
def test_tv_lines_are_rejected(heard):
    assert not is_wake_phrase(heard)


class _Wake:
    chunk_size = CHUNK
    model_name = "hey_spidy"
    _min_consecutive_frames = 2

    def __init__(self, frames):
        self.frames = list(frames)      # (score, is_detected) per chunk
        self.is_detected = False
        self.last_detected_model = "hey_spidy"
        self.last_detected_phrase = "Hey Spidy"

    def process_chunk(self, c):
        score, self.is_detected = self.frames.pop(0) if self.frames else (0.0, False)
        return score

    def reset_buffer(self):
        pass

    def unload(self):
        pass


def _engine(frames, heard="Hey Spidey!"):
    rec = MagicMock()
    rec.quick_transcribe = MagicMock(return_value=heard)
    eng = VoiceEngine(bus=MagicMock(), wake_word_model=_Wake(frames), recognizer=rec,
                      tts=MagicMock(is_speaking=False), wake_verify=True)
    eng._wake_verifier._trailing = 0.0
    eng._fire_wake = MagicMock()
    return eng, rec


def _feed(eng, n):
    for _ in range(n):
        eng._process_wake_word(np.zeros(CHUNK, np.float32))


def _wait(pred, t=2.0):
    end = time.monotonic() + t
    while not pred() and time.monotonic() < end:
        time.sleep(0.01)
    return pred()


def test_weak_candidate_never_reaches_whisper():
    """TV speech: the wake model fires at 0.3-0.8 but never climbs to 0.9 -- no Whisper run at all."""
    frames = [(0.3, True)] + [(0.7, False)] * 20
    eng, rec = _engine(frames)
    _feed(eng, 21)
    time.sleep(0.05)
    rec.quick_transcribe.assert_not_called()
    eng._fire_wake.assert_not_called()


def test_strong_candidate_is_verified_and_fires():
    frames = [(0.4, True), (0.8, False), (0.99, False)]
    eng, rec = _engine(frames)
    _feed(eng, 3)
    assert _wait(lambda: eng._fire_wake.called)


def test_silent_transcript_is_never_accepted():
    """Twice a TV woke Spidy because an empty transcript with a high score used to be accepted."""
    eng, rec = _engine([(0.95, True)], heard="")
    _feed(eng, 1)
    assert _wait(lambda: not eng._verifying and rec.quick_transcribe.call_count >= 2)
    time.sleep(0.05)
    eng._fire_wake.assert_not_called()


def test_tv_line_with_high_score_is_rejected():
    eng, rec = _engine([(0.97, True)], heard="I will see you in the next video.")
    _feed(eng, 1)
    assert _wait(lambda: rec.quick_transcribe.called and not eng._verifying)
    eng._fire_wake.assert_not_called()


# ── what to ignore after waking ──────────────────────────────────────────────

@pytest.mark.parametrize("text", ["Subscribe to our channel.", "I will see you in the next video.",
                                  "Open Discord. Open Teams. Open Teams. Open Teams. Open Teams. Open Teams.",
                                  "Move. Move. Move. Move.", "Okay.  Okay.  Okay.  Okay.  Okay.", "Thanks for watching!"])
def test_background_speech_is_ignored(text):
    assert looks_like_background_speech(text)


@pytest.mark.parametrize("text", ["Open Discord.", "What's the weather today?", "Thank you.", "Okay.",
                                  "Create a calculator website and open it in VS Code."])
def test_real_requests_pass(text):
    assert not looks_like_background_speech(text)


# ── live captions ────────────────────────────────────────────────────────────

def test_partial_captions_are_published_while_the_user_talks():
    from spidy.perception.voice.engine import VoiceCaptionEvent
    bus = MagicMock()
    rec = MagicMock()
    rec.quick_transcribe = MagicMock(return_value="open the calcu")
    eng = VoiceEngine(bus=bus, wake_word_model=_Wake([]), recognizer=rec,
                      tts=MagicMock(is_speaking=False), wake_verify=True)
    chunks = [np.full(CHUNK, 0.2, np.float32) for _ in range(12)]
    eng._maybe_caption(chunks, speech_chunks=12)
    assert _wait(lambda: bus.publish_threadsafe.called)
    ev = bus.publish_threadsafe.call_args[0][0]
    assert isinstance(ev, VoiceCaptionEvent) and ev.text == "open the calcu" and not ev.final


def test_outdated_partial_caption_is_dropped():
    bus = MagicMock()
    rec = MagicMock()
    rec.quick_transcribe = MagicMock(return_value="late words")
    eng = VoiceEngine(bus=bus, wake_word_model=_Wake([]), recognizer=rec, tts=MagicMock(is_speaking=False))
    eng._caption_token = 5
    eng._partial_caption(np.zeros(16000, np.float32), token=4)       # the recording already ended
    bus.publish_threadsafe.assert_not_called()


def test_caption_bar_shows_and_clears():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from spidy.ui.dashboard.window import CaptionBar
    bar = CaptionBar()
    bar.set_caption("make a calculator", False)
    assert bar.caption == "make a calculator" and not bar.final
    bar.set_caption("", False)
    assert bar.caption == ""
    assert app is not None


# ── memory ───────────────────────────────────────────────────────────────────

def test_memory_trim_runs():
    assert trim_process_memory() >= 0


# ── emotion ──────────────────────────────────────────────────────────────────

def test_emotions_change_pitch_direction():
    from spidy.voice.emotion import apply_pitch, style_for
    assert style_for("Wow, that's amazing!").name == "surprised"
    excited, sad = style_for("That's amazing, it worked!"), style_for("Oh no, I'm sorry, that failed.")
    assert excited.pitch > 1.0 and excited.pitch_end > excited.pitch          # high and rising
    assert sad.pitch < 1.0 and sad.pitch_end < sad.pitch                      # low and falling
    assert style_for("Do you want me to open it?").pitch_end > 1.0            # questions lift
    x = np.sin(np.arange(22050) * 2 * np.pi * 200 / 22050).astype(np.float32)
    up = apply_pitch(x, excited)
    assert len(up) < len(x)                    # read faster -> higher pitch (synthesised slower to compensate)
    assert np.abs(up).max() <= 1.0


def test_pitch_shift_keeps_sentence_length_overall():
    from spidy.voice.emotion import EXCITED, apply_pitch, synth_speed
    # Synthesised at synth_speed then shifted: the result matches the style's speed.
    n = 22050
    synth_len = n / synth_speed(EXCITED)
    out = apply_pitch(np.zeros(int(synth_len), np.float32), EXCITED)
    assert abs(len(out) - n / EXCITED.speed) < n * 0.02


# ── code writer ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,lang", [
    ("Create a website using HTML.", "html"),
    ("Just create any kind of website and open it in VS Code.", "html"),
    ("Make a calculator website.", "html"),
    ("Can you create a calculator website?", "html"),
    ("Open VS Code. Create HTML file. Write Calculator Code.", "html"),
    ("write a python script that renames my photos", "python"),
    ("build a snake game in JavaScript", "javascript"),
    ("make me a to-do app in React", "react"),
    ("write a C++ program to sort numbers", "c++"),
    ("create a REST API in Go", "go"),
    ("generate a SQL query to find duplicate emails", "sql"),
    ("make a calculator", "python"),
    ("develop a portfolio landing page", "html"),
])
def test_code_requests_are_recognised(text, lang):
    from spidy.brain.quick_actions import parse_code_request
    req = parse_code_request(text)
    assert req is not None and req[0] == lang


@pytest.mark.parametrize("text", ["create a file called notes", "make a folder on the desktop", "open VS Code",
                                  "make the chrome app full screen", "create a reminder for 5 pm",
                                  "what's the weather", "make a new note"])
def test_non_code_requests_are_left_alone(text):
    from spidy.brain.quick_actions import parse_code_request
    assert parse_code_request(text) is None
