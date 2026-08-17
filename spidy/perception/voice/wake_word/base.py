"""
WakeWordModel — Abstract Base Class
=====================================
Defines the contract every wake word engine must implement.

Design rationale
----------------
The wake word layer is deliberately thin and swappable. Today we use
openWakeWord with a pre-trained "hey_jarvis" model. In the future, we
will train a custom "hey_spidy" model. Swapping requires:
  1. Create a new subclass of WakeWordModel
  2. Change one line in spidy_config.yaml

Nothing else changes.

Audio contract
--------------
- Input: 16kHz mono PCM audio chunks as float32 numpy arrays
- Chunk size: 1280 samples (80ms at 16kHz)
- Audio comes from AudioCaptureEngine
- Wake word is detected if confidence score exceeds threshold
- On detection: publish WakeWordDetectedEvent via EventBus (thread-safe)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WakeWordModel(ABC):
    """
    Abstract wake word detector.

    Implementations must be thread-safe — process_chunk() is called from
    the AudioCaptureEngine background thread.

    Lifecycle
    ---------
    1. load(model_path) — called once at startup
    2. process_chunk(audio) — called continuously in audio thread
    3. unload() — called at shutdown
    """

    @abstractmethod
    def load(self, model_path: str | None = None) -> None:
        """
        Load the wake word model into memory.

        Parameters
        ----------
        model_path:
            Path to the model file(s). None uses the engine's default.
        """
        ...

    @abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process one audio chunk and return wake word confidence.

        Parameters
        ----------
        audio_chunk:
            Float32 numpy array of shape (chunk_size,) at 16kHz mono.
            Values should be in the range [-1.0, 1.0].

        Returns
        -------
        float
            Confidence score in [0.0, 1.0].
            Caller compares this to threshold to decide detection.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (GPU memory, file handles, etc.)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (for logging)."""
        ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """
        Required audio chunk size in samples.

        The AudioCaptureEngine will always provide chunks of exactly
        this size. Must match the model's expected input length.
        """
        ...

    def reset_buffer(self) -> None:
        """
        Reset the model's internal temporal smoothing buffer.

        Call after wake detection fires and after TTS playback ends, to
        prevent stale audio frames from influencing the next detection
        window.  The base implementation is a no-op — subclasses with
        stateful buffers (e.g. OpenWakeWordModel) should override.
        """
        pass
