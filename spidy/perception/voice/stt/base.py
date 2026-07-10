"""
SpeechRecognizer — Abstract Base Class
========================================
Defines the contract for all speech-to-text engines.

Audio contract
--------------
- Input: float32 numpy array, 16kHz, mono
- Comes from AudioCaptureEngine after wake word detection and VAD
- No streaming required for Milestone 1 (batch transcription)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptResult:
    """
    Result of a speech recognition request.

    Attributes
    ----------
    text:
        The transcribed text. Empty string if nothing was heard.
    confidence:
        Estimated confidence in [0.0, 1.0]. Some engines may return 1.0
        always if they don't provide per-segment confidence.
    language:
        Detected language code, e.g. "en". May be None if detection failed.
    duration_seconds:
        Duration of the input audio, in seconds.
    is_empty:
        True if no speech was detected in the audio.
    """
    text: str
    confidence: float = 1.0
    language: str | None = None
    duration_seconds: float = 0.0
    is_empty: bool = False

    def __post_init__(self) -> None:
        # Normalise whitespace
        self.text = self.text.strip()
        if not self.text:
            self.is_empty = True


class SpeechRecognizer(ABC):
    """
    Abstract speech-to-text engine.

    All concrete implementations must be safe to call from asyncio
    (use ``asyncio.to_thread()`` for CPU-bound inference internally).
    """

    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray) -> TranscriptResult:
        """
        Transcribe speech audio to text.

        Parameters
        ----------
        audio_data:
            Float32 numpy array at 16kHz mono. Values in [-1.0, 1.0].

        Returns
        -------
        TranscriptResult
            The transcription result. Never raises — exceptions return
            an empty TranscriptResult with is_empty=True.
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (and GPU if available)."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Returns the active device: 'cuda:0' or 'cpu'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
