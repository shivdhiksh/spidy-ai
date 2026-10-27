"""
Wake Word Verifier
==================
Second-stage check for wake-word detections.

The ONNX wake models fire on many "Hey ___" phrases ("Hey buddy", "Hey Siri").
After a detection, the last ~2 seconds of microphone audio are transcribed with
Whisper and the detection is only accepted if the transcript actually contains
the name "Spidy".

Design
------
- ``push()`` is called from the audio thread for every chunk (cheap deque append).
- ``verify()`` is blocking and is meant to run in a worker thread.
- Fail-open: if transcription raises, the detection is accepted so a Whisper
  problem can never make Spidy deaf.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# "spidy", "spidey", "spiddy", "spidi", "spydy" — but not "speedy" or "siri".
_NAME_RE = re.compile(r"sp[iy]d{1,2}(?:y|ey|ie|i)")


def contains_wake_name(text: str) -> bool:
    """True if ``text`` contains a spelling of the name "Spidy"."""
    letters = re.sub(r"[^a-z]", "", text.lower())
    return bool(_NAME_RE.search(letters))


class WakeVerifier:
    """
    Keeps a rolling window of recent audio and verifies wake detections.

    Parameters
    ----------
    transcribe:
        Blocking callable ``(float32 16 kHz mono audio) -> text``.
    sample_rate:
        Sample rate of the pushed chunks.
    window_seconds:
        How much trailing audio is transcribed on each verification.
    """

    def __init__(
        self,
        transcribe: Callable[[np.ndarray], str],
        sample_rate: int = 16000,
        window_seconds: float = 2.0,
        trailing_seconds: float = 0.4,
    ) -> None:
        self._trailing = trailing_seconds
        self._transcribe = transcribe
        self._sample_rate = sample_rate
        self._max_samples = int(sample_rate * window_seconds)
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        """Append one audio chunk (audio thread)."""
        with self._lock:
            self._chunks.append(chunk)
            self._samples += len(chunk)
            while self._samples - len(self._chunks[0]) >= self._max_samples:
                self._samples -= len(self._chunks.popleft())

    def snapshot(self) -> np.ndarray:
        """Return the buffered audio as one float32 array."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self._chunks)).astype(np.float32, copy=False)

    def collect(self) -> np.ndarray:
        """
        Wait ``trailing_seconds`` for the end of the spoken word, then snapshot.

        The wake model fires just before the final syllable finishes; without the
        wait, Whisper hears a clipped word and returns nothing. Blocking — call
        from the worker thread, never the audio thread.
        """
        time.sleep(self._trailing)
        return self.snapshot()

    def verify(self, audio: np.ndarray) -> tuple[bool, str]:
        """
        Transcribe ``audio`` and check it for the wake name.

        Returns ``(accepted, transcript)``.
        """
        try:
            text = self._transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            log.warning("[WAKE VERIFY] transcription failed ({exc}); accepting wake.", exc=exc)
            return True, ""
        return contains_wake_name(text), text
