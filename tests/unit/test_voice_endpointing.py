"""End-of-speech detection must adapt to the room's noise level (mic_gain makes noise loud)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from spidy.perception.voice.engine import VoiceEngine

CHUNK = 1280
RNG = np.random.default_rng(0)


class _Wake:
    chunk_size = CHUNK
    model_name = "hey_spidy"

    def process_chunk(self, chunk):
        return 0.0

    def reset_buffer(self):
        pass

    def unload(self):
        pass


def _chunk(rms: float) -> np.ndarray:
    x = RNG.normal(0.0, 1.0, CHUNK).astype(np.float32)
    return (x / np.sqrt(np.mean(x ** 2)) * rms).astype(np.float32)


def _engine(noise_rms: float | None, max_record: float = 10.0) -> VoiceEngine:
    eng = VoiceEngine(
        bus=MagicMock(),
        wake_word_model=_Wake(),
        recognizer=MagicMock(spec=["unload"]),
        tts=MagicMock(),
        max_record_seconds=max_record,
    )
    if noise_rms is not None:
        for _ in range(60):  # idle history, as _process_wake_word would have recorded
            eng._noise_energies.append(float(noise_rms * RNG.uniform(0.8, 1.2)))
    return eng


def _fill(eng: VoiceEngine, rms_sequence: list[float]) -> None:
    eng._capture_buffer.extend(_chunk(r) for r in rms_sequence)


def test_noisy_room_command_ends_on_silence_not_max_duration():
    # Real room: noise ~0.064 RMS at mic_gain=8 (the old fixed 0.03 threshold never fired).
    eng = _engine(noise_rms=0.064)
    seq = [0.064] * 5 + [0.35] * 15 + [0.064] * 60
    _fill(eng, seq)
    audio = eng._record_until_silence()
    assert audio is not None
    chunks = len(audio) // CHUNK
    assert chunks < len(seq)            # stopped early, did not consume the whole tail
    assert 5 + 15 <= chunks <= 5 + 15 + 30   # ~2 s (25 chunks) of silence after the speech


def test_noise_only_returns_none_so_whisper_never_sees_it():
    eng = _engine(noise_rms=0.064, max_record=0.3)
    _fill(eng, [0.064] * 30)
    assert eng._record_until_silence() is None


def test_quiet_room_still_works():
    eng = _engine(noise_rms=0.008)
    seq = [0.008] * 3 + [0.2] * 12 + [0.008] * 60
    _fill(eng, seq)
    audio = eng._record_until_silence()
    assert audio is not None and len(audio) // CHUNK < len(seq)


def test_no_idle_history_learns_floor_from_the_clip():
    eng = _engine(noise_rms=None)       # e.g. started from the hotkey
    seq = [0.064] * 14 + [0.35] * 15 + [0.064] * 60
    _fill(eng, seq)
    audio = eng._record_until_silence()
    assert audio is not None and len(audio) // CHUNK < len(seq)


def test_loudness_gate_ignores_quiet_background_speech_when_enabled():
    quiet_chatter = [0.064] * 5 + [0.15] * 15 + [0.064] * 60     # barely above the room
    loud_command = [0.064] * 5 + [0.6] * 15 + [0.064] * 60       # someone talking to the mic

    gated = _engine(noise_rms=0.064)
    gated._min_speech_ratio = 2.0
    _fill(gated, quiet_chatter)
    assert gated._record_until_silence() is None

    gated2 = _engine(noise_rms=0.064)
    gated2._min_speech_ratio = 2.0
    _fill(gated2, loud_command)
    assert gated2._record_until_silence() is not None


def test_loudness_gate_is_off_by_default():
    eng = _engine(noise_rms=0.064)
    _fill(eng, [0.064] * 5 + [0.15] * 15 + [0.064] * 60)
    assert eng._record_until_silence() is not None


def test_threshold_is_relative_to_noise_and_bounded():
    eng = _engine(noise_rms=0.064)
    thr = eng._silence_threshold(None, 0.03, 2.5, 0.25)
    assert 0.1 < thr < 0.25
    loud = _engine(noise_rms=0.5)
    assert loud._silence_threshold(None, 0.03, 2.5, 0.25) == 0.25
    quiet = _engine(noise_rms=0.001)
    assert quiet._silence_threshold(None, 0.03, 2.5, 0.25) == 0.03
    assert _engine(noise_rms=None)._silence_threshold(None, 0.03, 2.5, 0.25) is None
