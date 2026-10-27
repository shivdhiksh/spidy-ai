"""Phone-style resource use: speech models load when a wake word sounds and are freed when idle."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from spidy.perception.voice.engine import VoiceEngine, VoiceState
from spidy.perception.voice.stt.whisper import FasterWhisperRecognizer
from spidy.voice.barge_in import BargeInDetector

CHUNK = 1280


class _Wake:
    chunk_size = CHUNK
    model_name = "hey_spidy"

    def __init__(self, scores):
        self.scores = list(scores)

    def process_chunk(self, chunk):
        return self.scores.pop(0) if self.scores else 0.0

    def reset_buffer(self):
        pass

    def unload(self):
        pass


class _FakeModel:
    def transcribe(self, audio, **kw):
        return [SimpleNamespace(text="hello there", no_speech_prob=0.0, avg_logprob=-0.1, compression_ratio=1.0)], \
            SimpleNamespace(language="en")


def _recognizer() -> FasterWhisperRecognizer:
    rec = FasterWhisperRecognizer(model_size="base.en", device="cpu")
    loads = []

    def fake_load():
        loads.append(1)
        rec._model = _FakeModel()

    rec._load_locked = fake_load            # type: ignore[method-assign]
    rec.loads = loads                       # type: ignore[attr-defined]
    return rec


# ── recognizer ────────────────────────────────────────────────────────────────

def test_unloaded_recognizer_reloads_itself_when_asked_to_transcribe():
    rec = _recognizer()
    rec.load()
    rec.unload()
    assert not rec.is_loaded
    assert rec.quick_transcribe(np.zeros(16000, np.float32)) == "hello there"
    assert rec.is_loaded and len(rec.loads) == 2


def test_prewarm_loads_in_the_background_and_is_idempotent():
    rec = _recognizer()
    rec.prewarm()
    rec.prewarm()
    end = time.monotonic() + 3
    while not rec.is_loaded and time.monotonic() < end:
        time.sleep(0.01)
    assert rec.is_loaded and len(rec.loads) == 1
    rec.prewarm()                                       # already loaded: nothing to do
    assert len(rec.loads) == 1


def test_idle_seconds_restarts_after_each_transcription():
    rec = _recognizer()
    rec.load()
    rec._last_used -= 120
    assert rec.idle_seconds > 100
    rec.quick_transcribe(np.zeros(16000, np.float32))
    assert rec.idle_seconds < 5


# ── engine: governor and prewarm ──────────────────────────────────────────────

def _engine(scores=(), idle=60.0):
    rec = _recognizer()
    rec.load()
    eng = VoiceEngine(bus=MagicMock(), wake_word_model=_Wake(scores), recognizer=rec,
                      tts=MagicMock(is_speaking=False), idle_unload_seconds=idle)
    return eng, rec


def test_models_are_released_after_a_quiet_spell():
    eng, rec = _engine()
    hook = MagicMock()
    eng.add_resource_hooks(unload=hook)
    assert not eng._should_release_models()             # just started: not idle yet
    eng._last_active -= 120
    rec._last_used -= 120
    assert eng._should_release_models()
    eng._release_models()
    assert not rec.is_loaded
    hook.assert_called_once()
    assert not eng._should_release_models()             # nothing left to release


def test_models_are_kept_while_a_conversation_is_going_on():
    eng, rec = _engine()
    eng._last_active -= 120
    rec._last_used -= 120
    eng._set_state(VoiceState.LISTENING)                # also counts as activity
    assert not eng._should_release_models()
    eng._state = VoiceState.SLEEPING
    eng._last_active -= 120
    eng._verifying = True                               # wake word being verified right now
    assert not eng._should_release_models()


def test_recent_transcription_keeps_the_model_loaded():
    eng, rec = _engine()
    eng._last_active -= 120                             # nothing heard for a while...
    rec.quick_transcribe(np.zeros(16000, np.float32))   # ...but the model was just used
    assert not eng._should_release_models()


def test_wake_word_starting_to_sound_loads_everything():
    eng, rec = _engine(scores=[0.0, 0.1])               # 0.1 is below the detection threshold
    rec.unload()
    warmed = MagicMock()
    eng.add_resource_hooks(prewarm=warmed)
    quiet = np.zeros(CHUNK, np.float32)
    eng._process_wake_word(quiet)
    assert not rec.is_loaded                            # silence does not load anything
    eng._process_wake_word(quiet)                       # score 0.1: the wake word is starting
    end = time.monotonic() + 3
    while not rec.is_loaded and time.monotonic() < end:
        time.sleep(0.01)
    assert rec.is_loaded
    warmed.assert_called_once()


def test_disabled_governor_never_releases():
    eng, rec = _engine(idle=0.0)
    eng._last_active -= 10_000
    rec._last_used -= 10_000
    assert eng._governor_task is None and eng._idle_unload == 0.0


# ── barge-in model ────────────────────────────────────────────────────────────

def test_barge_in_model_loads_on_prewarm_and_frees_when_idle():
    det = BargeInDetector(MagicMock(), MagicMock())

    def fake_load():
        det._model, det._model_loaded = object(), True

    det.load_model = fake_load                          # type: ignore[method-assign]
    det.prewarm()
    end = time.monotonic() + 3
    while not det.is_loaded and time.monotonic() < end:
        time.sleep(0.01)
    assert det.is_loaded
    det._active = True
    det.unload_model()
    assert det.is_loaded                                # never while an utterance is being listened to
    det._active = False
    det.unload_model()
    assert not det.is_loaded and det._model is None
