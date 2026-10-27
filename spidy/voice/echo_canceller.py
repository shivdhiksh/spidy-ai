"""
Echo cancellation for barge-in
==============================
While Spidy talks, her own voice comes back through the microphone as loud as a person speaking
(measured on the dev machine: echo RMS median 0.09, peaks 0.6, vs ~0.25 for a user at the mic), so
neither loudness nor a transcript can tell the user from Spidy. But Spidy knows exactly what she is
playing. :class:`PlaybackReference` keeps that signal on a wall-clock timeline and
:class:`EchoCanceller` (a partitioned-block frequency-domain adaptive filter) learns how the speakers,
room and mic turn it into the echo, subtracts it, and leaves (mostly) the user's voice.
"""

from __future__ import annotations

import threading
import time

import numpy as np

_SR = 16000
_JOIN_SECONDS = 0.15      # a sentence starting this soon after the previous one ended continues it


class MicClock:
    """
    Wall-clock time of each microphone chunk's end, from the audio clock instead of the callbacks.

    Callbacks jitter by several milliseconds (+-130 samples), far too much for echo cancellation,
    while the samples themselves are perfectly regular. ``k * dt`` is exact; the offset is the smallest
    ``now - k * dt`` seen recently (a late callback only ever makes it larger).
    """

    def __init__(self, window: int = 60) -> None:
        self._k = 0
        self._offsets: list[float] = []
        self._window = window

    def tick(self, now: float, n: int) -> float:
        """Register one more chunk of ``n`` samples heard at ``now``; returns when it ended."""
        self._k += n
        off = now - self._k / _SR
        self._offsets.append(off)
        if len(self._offsets) > self._window:
            self._offsets.pop(0)
        return min(self._offsets) + self._k / _SR


class PlaybackReference:
    """What Spidy is playing, as 16 kHz audio placed on the ``time.monotonic()`` timeline."""

    def __init__(self, keep_seconds: float = 60.0) -> None:
        self._segments: list[tuple[float, np.ndarray]] = []   # (start time, samples)
        self._keep = keep_seconds
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray, sample_rate: int, start: float | None = None) -> None:
        """Register audio that starts playing now (call right before handing it to the sound card)."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        if sample_rate != _SR:
            n = max(1, int(round(x.size * _SR / sample_rate)))
            x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
        with self._lock:
            if start is None:
                # The sound card plays what is queued in order: a sentence handed over while (or just
                # after) the previous one plays starts exactly where that one ends. Only a real pause
                # puts it at "now". Sample-exact joins matter: the canceller learns from the waveform.
                start = time.monotonic()
                if self._segments:
                    last_t, last_s = self._segments[-1]
                    last_end = last_t + last_s.size / _SR
                    if start - last_end < _JOIN_SECONDS:
                        start = last_end
            t0 = start
            cutoff = time.monotonic() - self._keep
            self._segments = [(t, s) for t, s in self._segments if t + s.size / _SR > cutoff]
            self._segments.append((t0, x))

    def clear(self) -> None:
        with self._lock:
            self._segments = []

    def block(self, t_end: float, n: int) -> np.ndarray:
        """The ``n`` reference samples for the window that ends at ``t_end`` (zeros where nothing played)."""
        out = np.zeros(n, dtype=np.float32)
        t_start = t_end - n / _SR
        with self._lock:
            segments = list(self._segments)
        for t0, s in segments:
            end = t0 + s.size / _SR
            if end <= t_start or t0 >= t_end:
                continue
            a = int(round((t_start - t0) * _SR))          # index into s of the window start
            lo, hi = max(a, 0), min(a + n, s.size)
            if hi > lo:
                out[lo - a:hi - a] += s[lo:hi]
        return out

    @property
    def playing(self) -> bool:
        now = time.monotonic()
        with self._lock:
            return any(t <= now < t + s.size / _SR for t, s in self._segments)


class EchoCanceller:
    """
    Partitioned-block frequency-domain adaptive filter (MDF).

    ``process(mic, ref)`` takes one block of microphone audio and the reference block for the same
    window and returns the microphone audio with the echo of the reference removed. The filter spans
    ``partitions * block`` samples (about 0.64 s for 8 x 80 ms), which absorbs the unknown delay between
    "handed to the sound card" and "heard by the mic". Weights persist between utterances (the room does
    not change), only the reference history is reset.
    """

    def __init__(self, block: int = 1280, partitions: int = 8, mu: float = 1.0) -> None:
        self.block = block
        self.k = partitions
        self.mu = mu
        f = block + 1
        self._w = np.zeros((partitions, f), dtype=np.complex128)
        self._xh = np.zeros((partitions, f), dtype=np.complex128)
        self._x_prev = np.zeros(block)
        self._pow = np.full(f, 1e-4)
        self._converged = False
        self._frozen_blocks = 0
        self._d_energy = 1e-9
        self._e_energy = 1e-9
        self.erle_db = 0.0

    @property
    def converged(self) -> bool:
        """True once the filter has been seen to remove a good part of the echo."""
        return self._converged and self.erle_db > 4.0

    def reset_history(self) -> None:
        """Forget the reference history (new utterance); keep what was learned about the room."""
        self._xh[:] = 0
        self._x_prev[:] = 0

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        n = self.block
        d = mic.astype(np.float64, copy=False)
        x = ref.astype(np.float64, copy=False)
        spec = np.fft.rfft(np.concatenate([self._x_prev, x]))
        self._x_prev = x.copy()
        self._xh = np.roll(self._xh, 1, axis=0)
        self._xh[0] = spec

        y = np.fft.irfft((self._w * self._xh).sum(axis=0))[n:]
        e = d - y

        d_en, e_en = float(np.dot(d, d)) + 1e-9, float(np.dot(e, e)) + 1e-9
        ref_active = float(np.dot(x, x)) > 1e-6 * n
        if ref_active:
            self._d_energy = 0.95 * self._d_energy + 0.05 * d_en
            self._e_energy = 0.95 * self._e_energy + 0.05 * e_en
            self.erle_db = 10.0 * np.log10(self._d_energy / self._e_energy)
            if self.erle_db > 8.0:
                self._converged = True

        # Someone talking over the echo leaves the error as loud as the mic: don't learn from that.
        double_talk = self._converged and e_en > 0.5 * d_en
        if double_talk:
            self._frozen_blocks += 1
        else:
            self._frozen_blocks = 0
        if ref_active and (not double_talk or self._frozen_blocks > 25):
            mu = self.mu if not self._converged else (0.4 if not double_talk else 0.08)
            e_spec = np.fft.rfft(np.concatenate([np.zeros(n), e]))
            self._pow = 0.8 * self._pow + 0.2 * (np.abs(spec) ** 2)
            norm = (np.abs(self._xh) ** 2).sum(axis=0) + self._pow.sum() / (n + 1) * 0.05 + 1e-6
            grad = np.conj(self._xh) * e_spec / norm
            g = np.fft.irfft(grad, axis=1)
            g[:, n:] = 0.0                                   # keep the update causal (gradient constraint)
            self._w += mu * np.fft.rfft(g, axis=1)
        # While the filter is still learning it can add energy: never return something louder than the mic.
        return (e if e_en <= d_en else d).astype(np.float32)
