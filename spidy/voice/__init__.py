"""
spidy.voice — Milestone 14: Voice Companion Layer
===================================================
High-level voice interaction layer that sits above the existing
spidy.perception.voice hardware abstraction.

Public API
----------
    from spidy.voice import (
        VoiceSessionManager,
        InterruptionHandler,
        VoiceActivityDetector,
        StreamingTTSWrapper,
        ContinuousVoiceController,
        VoiceSettings,
    )

This layer is purely behavioural — it never duplicates STT, TTS, or
wake word logic.  It orchestrates the existing VoiceEngine and Brain
to produce continuous, natural voice interaction.
"""

from __future__ import annotations

from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.interruption import InterruptionHandler, InterruptCommand
from spidy.voice.vad import VoiceActivityDetector
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.settings import VoiceSettings

__all__ = [
    "VoiceSessionManager",
    "VoiceSessionState",
    "InterruptionHandler",
    "InterruptCommand",
    "VoiceActivityDetector",
    "StreamingTTSWrapper",
    "ContinuousVoiceController",
    "VoiceSettings",
]
