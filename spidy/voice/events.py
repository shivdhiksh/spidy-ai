"""
Voice Companion Events — Milestone 14
======================================
All EventBus events introduced by the Voice Companion layer.

Topic namespace: ``voice.*``  (new M14 sub-topics)

These events extend (never replace) the existing voice events in
spidy.perception.voice.engine.  The M14 events carry higher-level
session and streaming information.

Existing events (unchanged):
  wake_word.detected      — wake word confidence crossed threshold
  voice.listening         — Spidy started recording
  voice.transcript        — final STT result
  voice.speaking_start    — TTS playback began
  voice.speaking_end      — TTS playback finished
  voice.error             — pipeline error

New M14 events:
  voice.partial_transcript  — streaming STT intermediate result
  voice.session_started     — continuous conversation session opened
  voice.session_ended       — session closed (timeout / cancel)
  voice.session_timeout     — session timed out (no speech for N seconds)
  voice.interrupt           — user issued an interrupt command
  voice.settings_changed    — runtime voice settings updated
  voice.paused              — voice session paused (user said "wait")
  voice.resumed             — voice session resumed
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


# ── Streaming STT ─────────────────────────────────────────────────────────────


@dataclass
class VoicePartialTranscriptEvent(Event):
    """
    Emitted during streaming STT as partial segments arrive.

    The UI can display a live "typewriter" effect using these events.
    is_final=True means the transcript is complete and will be sent
    to the Brain.  Partial events (is_final=False) are display-only.
    """
    topic = "voice.partial_transcript"
    text: str = ""
    is_final: bool = False
    confidence: float = 1.0


# ── Session lifecycle ─────────────────────────────────────────────────────────


@dataclass
class VoiceSessionStartedEvent(Event):
    """Emitted when a continuous conversation session opens (after wake word)."""
    topic = "voice.session_started"
    session_id: str = ""
    wake_word: str = ""


@dataclass
class VoiceSessionEndedEvent(Event):
    """Emitted when a voice session ends (timeout, cancel, or explicit stop)."""
    topic = "voice.session_ended"
    session_id: str = ""
    turn_count: int = 0
    reason: str = ""     # "timeout" | "cancelled" | "user_stopped" | "max_turns"


@dataclass
class VoiceSessionTimeoutEvent(Event):
    """Emitted when the session timeout fires (no speech for N seconds)."""
    topic = "voice.session_timeout"
    session_id: str = ""
    timeout_seconds: float = 0.0


# ── Interruption ──────────────────────────────────────────────────────────────


@dataclass
class VoiceInterruptEvent(Event):
    """
    Emitted when the user issues a voice interrupt command.

    The InterruptionHandler detects these commands before routing to Brain.
    The ContinuousVoiceController and Brain react accordingly.
    """
    topic = "voice.interrupt"
    command: str = ""    # "stop" | "cancel" | "never_mind" | "wait" | "pause" | "resume"
    utterance: str = ""  # The original phrase the user said


# ── Pause / Resume ────────────────────────────────────────────────────────────


@dataclass
class VoicePausedEvent(Event):
    """Emitted when the voice session is paused ('wait', 'pause')."""
    topic = "voice.paused"
    session_id: str = ""


@dataclass
class VoiceResumedEvent(Event):
    """Emitted when the voice session resumes from paused state."""
    topic = "voice.resumed"
    session_id: str = ""


# ── Settings ──────────────────────────────────────────────────────────────────


@dataclass
class VoiceSettingsChangedEvent(Event):
    """Emitted when runtime voice settings are updated via VoiceSettings."""
    topic = "voice.settings_changed"
    changes: dict = field(default_factory=dict)   # {setting_name: new_value}
