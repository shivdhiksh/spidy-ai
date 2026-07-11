"""
LLM Events — EventBus events published by the LLM layer
=========================================================
Topic namespace: ``llm.*``

These events decouple the LLM layer from:
- Brain (subscribes to llm.response_ready)
- UI (can show provider status, loading indicators)
- Analytics / telemetry (track usage, latency, costs)
- Learning Engine (track LLM call results for feedback loop)
"""

from __future__ import annotations

from dataclasses import dataclass

from spidy.core.event_bus import Event


@dataclass
class LLMRequestStartedEvent(Event):
    """Emitted when an LLM request begins."""
    topic = "llm.request_started"
    provider: str = ""
    model: str = ""
    session_id: str = ""


@dataclass
class LLMResponseReadyEvent(Event):
    """Emitted when an LLM response is received."""
    topic = "llm.response_ready"
    provider: str = ""
    model: str = ""
    session_id: str = ""
    success: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMProviderFailedEvent(Event):
    """Emitted when a provider fails to respond."""
    topic = "llm.provider_failed"
    provider: str = ""
    model: str = ""
    error_message: str = ""
    session_id: str = ""


@dataclass
class LLMProviderSwitchedEvent(Event):
    """Emitted when the LLMRouter switches to a fallback provider."""
    topic = "llm.provider_switched"
    from_provider: str = ""
    to_provider: str = ""
    session_id: str = ""
