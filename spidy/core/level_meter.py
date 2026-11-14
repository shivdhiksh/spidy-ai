"""
Live voice level for the HUD.

The audio threads (Spidy's speaker output and the microphone while the user talks) report loudness
here; the dashboard's dot-blob reads it every animation frame so it waves in time with the voice.
A plain lock-free float per source is enough: writes are atomic in CPython and readers only need
"roughly now". Values fade out on their own, so a stopped stream never leaves the blob stuck.
"""

from __future__ import annotations

import math
import time

_FADE = 0.25            # seconds for a level to fade after the last report
_levels: dict[str, tuple[float, float]] = {}      # source -> (level 0..1, time reported)


def report(source: str, samples) -> None:
    """Report one audio chunk (float samples in -1..1) from ``source`` ("spidy" or "user")."""
    try:
        import numpy as np
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
    except Exception:  # noqa: BLE001
        return
    # perceptual-ish scale: quiet speech ~0.3, loud ~1.0
    level = max(0.0, min(1.0, (20 * math.log10(rms + 1e-6) + 48) / 36))
    _levels[source] = (level, time.monotonic())


def current(source: str | None = None) -> float:
    """Current level (0..1) of ``source``, or the loudest source if None; fades after reports stop."""
    now = time.monotonic()
    best = 0.0
    for name, (lvl, t) in list(_levels.items()):
        if source is not None and name != source:
            continue
        age = now - t
        if age < _FADE:
            best = max(best, lvl * (1 - age / _FADE))
    return best


def reset() -> None:
    _levels.clear()
