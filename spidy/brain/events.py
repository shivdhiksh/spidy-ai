"""
Brain Events
=============
All EventBus events published by the Brain Core (Milestone 3).

Topic namespace: ``brain.*``

These events decouple the Brain from:
- VoiceEngine (which publishes voice.user_spoke, listens for brain.response_ready)
- UI (which listens for brain.response_ready to render the reply)
- Memory system (which listens for brain.tool_result to store episodic memories)
- Learning system (which listens for all brain.* events for signal collection)
"""

from __future__ import annotations

from dataclasses import dataclass

from spidy.core.event_bus import Event


@dataclass
class BrainProcessingStartedEvent(Event):
    """Emitted when the Brain starts processing a user utterance."""
    topic = "brain.processing_started"
    session_id: str = ""
    utterance: str = ""


@dataclass
class BrainResponseReadyEvent(Event):
    """Emitted when the Brain has a response ready for TTS + UI."""
    topic = "brain.response_ready"
    session_id: str = ""
    response_text: str = ""
    decision_mode: str = ""     # DecisionMode.value


@dataclass
class BrainSessionStartedEvent(Event):
    """Emitted when a new conversation session begins."""
    topic = "brain.session_started"
    session_id: str = ""


@dataclass
class BrainSessionEndedEvent(Event):
    """Emitted when a conversation session ends."""
    topic = "brain.session_ended"
    session_id: str = ""
    turn_count: int = 0


@dataclass
class BrainToolCalledEvent(Event):
    """Emitted when the ToolRouter dispatches a step to a skill or LLM."""
    topic = "brain.tool_called"
    session_id: str = ""
    action: str = ""
    skill_name: str = ""
    step_type: str = ""


@dataclass
class BrainToolResultEvent(Event):
    """Emitted when a tool/skill call completes."""
    topic = "brain.tool_result"
    session_id: str = ""
    action: str = ""
    success: bool = False
    message: str = ""
