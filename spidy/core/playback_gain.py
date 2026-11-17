"""
Playback gain ("ducking")
=========================
One shared volume factor for Spidy's voice output, changed from any thread.

Full-duplex barge-in uses it to *duck* Spidy the instant your voice is detected over her (her volume
drops to a whisper within ~50 ms), while a short transcription confirms it really is you. Confirmed:
she stops and listens. Not you (her own echo, a cough, the TV): she comes straight back to full volume.

The audio player multiplies every chunk by :func:`factor` and glides toward it, so changes are
click-free.
"""

from __future__ import annotations

import threading

import numpy as np

_lock = threading.Lock()
_target = 1.0
_current = 1.0

DUCKED = 0.12          # how quiet Spidy gets while we check whether you're talking


def duck(on: bool = True) -> None:
    set_target(DUCKED if on else 1.0)


def set_target(x: float) -> None:
    global _target
    with _lock:
        _target = float(min(1.0, max(0.0, x)))


def target() -> float:
    with _lock:
        return _target


def apply(chunk: np.ndarray) -> np.ndarray:
    """Scale one output chunk, gliding from the current gain to the target across it (no clicks)."""
    global _current
    with _lock:
        start, end = _current, _target
        _current = end
    if start == 1.0 and end == 1.0:
        return chunk
    ramp = np.linspace(start, end, len(chunk), dtype=np.float32).reshape(-1, *([1] * (chunk.ndim - 1)))
    return (chunk * ramp).astype(chunk.dtype, copy=False)


def reset() -> None:
    global _target, _current
    with _lock:
        _target = _current = 1.0
