"""
Brain Core — Shared Data Types
================================
Typed data structures used across all Brain components.

These are the fundamental "vocabulary" of the Brain:
- Intent       : what the user wants
- ConversationTurn : a single exchange in a conversation
- Decision     : what the Brain will do
- Plan / PlanStep  : how to do it
- ToolResult   : outcome of doing it

Design
------
All types are frozen dataclasses for thread-safety and immutability.
The Brain pipeline flows:
    utterance → Intent → Decision → Plan → ToolResult(s)

Lifelong Companion notes
------------------------
These types are designed to be compatible with future memory and
knowledge systems:
- Intent carries raw_utterance for future LLM-based classifiers
- ConversationTurn carries intent and session_id for memory indexing
- ToolResult carries structured data for learning systems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ─── Enumerations ─────────────────────────────────────────────────────────────


class DecisionMode(str, Enum):
    """How the Brain handles an Intent."""
    SKILL = "skill"           # Dispatch to a registered Skill
    LLM_DIRECT = "llm_direct" # Answer directly via LLM (no skill needed)
    CLARIFY = "clarify"       # Ask for clarification (low confidence)
    REJECT = "reject"         # Cannot/should not fulfil (permission, scope)


class TurnRole(str, Enum):
    """Role of a conversation turn participant."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ─── Intent ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entity:
    """A named entity extracted from the utterance."""
    name: str       # Entity type, e.g. "filename", "app_name", "query"
    value: str      # Extracted value


@dataclass(frozen=True)
class Intent:
    """
    The classified intent of a user utterance.

    Produced by IntentClassifier. Consumed by DecisionEngine.

    Attributes
    ----------
    action:
        Primary action string (e.g. "open_file", "search_web", "chat").
        ``"chat"`` is the fallback for open-ended conversational input.
    entities:
        Named entities extracted from the utterance.
    confidence:
        Classifier confidence (0.0–1.0). Low confidence → CLARIFY decision.
    raw_utterance:
        The original text, preserved for LLM context.
    source:
        Which classifier produced this intent (e.g. "heuristic", "llm").
    """
    action: str
    entities: tuple[Entity, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    raw_utterance: str = ""
    source: str = "heuristic"

    def get_entity(self, name: str, default: str = "") -> str:
        """Return the value of a named entity, or default."""
        for e in self.entities:
            if e.name == name:
                return e.value
        return default

    @property
    def is_confident(self) -> bool:
        """True if confidence meets the threshold for acting (>= 0.6)."""
        return self.confidence >= 0.6


# ─── Conversation ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a conversation."""
    role: TurnRole
    text: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    intent: Intent | None = None
    session_id: str = ""

    def to_llm_message(self) -> dict[str, str]:
        """Convert to the {role, content} format expected by LLM APIs."""
        return {"role": self.role.value, "content": self.text}


# ─── Decision ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    """
    What the Brain decided to do with an Intent.

    Produced by DecisionEngine. Consumed by Planner.
    """
    mode: DecisionMode
    intent: Intent
    skill_name: str = ""              # Populated when mode == SKILL
    rationale: str = ""               # Brief explanation (for logs + debug UI)
    clarification_question: str = ""  # Populated when mode == CLARIFY


# ─── Plan ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanStep:
    """
    A single atomic step in a Plan.

    Attributes
    ----------
    step_type:
        ``"skill"``   — dispatch to SkillRegistry
        ``"llm"``     — call LLMClient
        ``"clarify"`` — ask the user for more information
        ``"noop"``    — do nothing (rejected / empty intent)
    action:
        For skill steps: the action string (e.g. ``"open_file"``).
        For llm steps: ``"direct"`` or a prompt template identifier.
        For clarify/noop steps: empty.
    params:
        Key-value parameters for skill steps.
    prompt_context:
        Additional context passed to the LLM for llm steps.
    """
    step_type: str          # "skill" | "llm" | "clarify" | "noop"
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    prompt_context: str = ""


@dataclass(frozen=True)
class Plan:
    """An ordered sequence of steps the Brain will execute."""
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)
    session_id: str = ""
    decision: Decision | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    @property
    def first(self) -> PlanStep | None:
        return self.steps[0] if self.steps else None


# ─── Tool Result ──────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """
    Outcome of executing one PlanStep.

    Produced by ToolRouter. Used by Brain to compose the final response.
    """
    success: bool
    message: str                    # Human-readable response text
    action: str = ""
    data: Any = None                # Structured result (e.g. file list)
    error: str = ""                 # Error message if success=False
    step_type: str = ""             # Which step type produced this

    @classmethod
    def ok(
        cls,
        message: str,
        action: str = "",
        data: Any = None,
        step_type: str = "",
    ) -> "ToolResult":
        return cls(
            success=True,
            message=message,
            action=action,
            data=data,
            step_type=step_type,
        )

    @classmethod
    def fail(
        cls,
        message: str,
        action: str = "",
        error: str = "",
        step_type: str = "",
    ) -> "ToolResult":
        return cls(
            success=False,
            message=message,
            action=action,
            error=error,
            step_type=step_type,
        )
