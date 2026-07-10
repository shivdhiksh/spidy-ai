"""
TTSEngine — Abstract Base Class
================================
Defines the contract for all text-to-speech engines.

Design
------
- The TTS layer is fully swappable: Piper → Kokoro → ElevenLabs via config.
- All engines produce audio at a standard sample rate (22050 Hz for Piper).
- speak() is the primary API: synthesize + play in one call.
- Audio playback is handled internally by each engine.
- stop() must be callable from any thread (e.g. wake word interrupt).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioBuffer:
    """
    Raw synthesised audio ready for playback.

    Attributes
    ----------
    samples:
        Float32 numpy array of PCM samples. Shape: (n_samples,).
    sample_rate:
        Audio sample rate in Hz (e.g. 22050 for Piper).
    duration_seconds:
        Total audio duration.
    """
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


class TTSEngine(ABC):
    """
    Abstract text-to-speech engine.

    The primary method for most use cases is ``speak()`` which synthesises
    and plays audio in one call. Use ``synthesize()`` when you need to
    process the audio buffer before playing (e.g. effects, logging).
    """

    @abstractmethod
    async def synthesize(self, text: str) -> AudioBuffer:
        """
        Convert text to an audio buffer without playing it.

        Parameters
        ----------
        text:
            The text to synthesise. May include punctuation for pacing.

        Returns
        -------
        AudioBuffer
            Raw PCM audio. Caller is responsible for playback.
        """
        ...

    @abstractmethod
    async def speak(self, text: str) -> None:
        """
        Synthesise and play audio, blocking until complete.

        Parameters
        ----------
        text:
            The text to synthesise and speak.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """
        Immediately stop any active playback.

        Must be thread-safe — callable from the wake word detection thread
        to interrupt Spidy mid-sentence.
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the TTS model (voice files, weights, etc.)."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release TTS resources."""
        ...

    @property
    @abstractmethod
    def voice_name(self) -> str:
        """The voice identifier being used."""
        ...

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """True while audio is being played."""
        ...
