"""The echo canceller must remove Spidy's own voice from the mic but keep the user's."""

from __future__ import annotations

import numpy as np

from spidy.voice.echo_canceller import EchoCanceller, PlaybackReference

N = 1280
SR = 16000


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    """Band-limited noise with a syllable-rate envelope: a stand-in for speech."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.normal(size=n)
    x = np.convolve(x, np.ones(4) / 4, mode="same")                       # darker spectrum
    t = np.arange(n) / SR
    env = np.clip(np.sin(2 * np.pi * 3.5 * t + rng.uniform(0, 6)) + 0.4, 0, None)
    env *= np.clip(np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)) + 0.9, 0.05, None)
    x = x * env
    return (x / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def _room(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    taps = np.zeros(int(0.30 * SR))
    delay = int(0.14 * SR)                                                # sound card + mic latency
    k = np.arange(2400)
    taps[delay:delay + k.size] = rng.normal(size=k.size) * np.exp(-k / 500.0)
    taps[delay] += 2.0                                                    # direct path
    return taps / np.abs(taps).sum() * 1.6


def _mic(ref: np.ndarray, room: np.ndarray, gain: float) -> np.ndarray:
    return (np.convolve(ref, room)[: ref.size] * gain).astype(np.float32)


def _blocks(x: np.ndarray):
    for i in range(0, x.size - N + 1, N):
        yield x[i:i + N]


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _run(aec, ref, mic):
    return np.concatenate([aec.process(m, r) for m, r in zip(_blocks(mic), _blocks(ref))])


def test_echo_is_cancelled_after_a_short_convergence():
    ref = 0.3 * _speechlike(14, 1)
    mic = _mic(ref, _room(), 7.5) + 0.01 * np.random.default_rng(3).normal(size=ref.size).astype(np.float32)
    aec = EchoCanceller()
    out = _run(aec, ref, mic)
    late = slice(int(8 * SR), None)
    erle = 20 * np.log10(_rms(mic[late]) / _rms(out[late]))
    assert erle > 12, f"only {erle:.1f} dB of echo removed"
    assert aec.erle_db > 8


def test_user_voice_survives_while_echo_is_removed():
    ref = 0.3 * _speechlike(16, 1)
    echo = _mic(ref, _room(), 7.5)
    user = np.zeros_like(ref)
    s = int(11 * SR)
    voice = 0.25 * _speechlike(3.0, 2)
    user[s:s + voice.size] = voice
    aec = EchoCanceller()
    _run(aec, ref[: int(9 * SR)], echo[: int(9 * SR)])                    # converge on echo alone
    aec.reset_history()
    out = _run(aec, ref[int(9 * SR):], (echo + user)[int(9 * SR):])
    off = int(9 * SR)
    quiet = out[: s - off]                                                # echo only
    talking = out[s - off + SR // 2: s - off + int(2.5 * SR)]             # user over the echo
    assert _rms(talking) > 3.0 * _rms(quiet), (_rms(talking), _rms(quiet))
    # what is left while the user talks is mostly the user, not echo
    u = user[s + SR // 2: s + int(2.5 * SR)]
    corr = np.corrcoef(u, talking[: u.size])[0, 1]
    assert corr > 0.7, corr


def test_playback_reference_places_audio_on_the_wall_clock():
    ref = PlaybackReference()
    x = np.ones(SR, dtype=np.float32)                                     # 1 s of ones starting at t=100
    ref.push(x, SR, start=100.0)
    blk = ref.block(100.5, N)                                             # window (100.42, 100.5]
    assert np.allclose(blk, 1.0)
    assert not ref.block(99.0, N).any() and not ref.block(102.0, N).any()
    straddle = ref.block(100.0 + N / SR / 2, N)                           # half before the start
    assert straddle[: N // 2].max() == 0 and np.allclose(straddle[N // 2:], 1.0)


def test_reference_resamples_to_16k():
    ref = PlaybackReference()
    ref.push(np.ones(22050, dtype=np.float32), 22050, start=10.0)
    assert np.allclose(ref.block(10.5, N), 1.0)
    assert not ref.block(11.2, N).any()
