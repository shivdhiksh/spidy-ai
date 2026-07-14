"""
Learning Engine — Shared Data Types
====================================
Frozen dataclasses for all Learning Engine domain objects.

These types are the currency passed between the five Learning components
(PreferenceLearner, HabitDetector, WorkflowLearner, FeedbackProcessor,
LearningManager) and surfaced to the Brain.

Design principles
-----------------
- All dataclasses are **frozen** — Learning data is immutable once created.
  Updates create new instances (value semantics, no hidden mutation).
- ``create()`` class methods provide sensible defaults and generate IDs.
- All types are JSON-serialisable (no custom objects in fields).
- Privacy: none of these types contain raw PII beyond the user's own words.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ─── Utility helpers ──────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _short_id() -> str:
    """Generate a short unique ID (first 12 hex chars of UUID4)."""
    return uuid.uuid4().hex[:12]


def _stable_id(parts: list[str]) -> str:
    """Generate a deterministic ID from a list of strings (SHA-256[:12])."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


# ─── Preference ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Preference:
    """
    A single learned user preference with a confidence score.

    Attributes
    ----------
    key:
        Preference identifier, e.g. ``"preferred_browser"``,
        ``"preferred_llm"``, ``"preferred_ide"``, ``"response_style"``.
    value:
        The preference value. Typically a string but can be any
        JSON-serialisable type.
    confidence:
        0.0 (wild guess) to 1.0 (explicitly stated by user).
        Implicit signals are capped at 0.8; explicit = 1.0.
    last_updated:
        UTC datetime of the most recent update.
    source:
        How this preference was established:
        ``"explicit"``  — user directly said "I prefer …"
        ``"implicit"``  — inferred from repeated behaviour
        ``"inferred"``  — derived from patterns (e.g. app → IDE)
    metadata:
        Optional dict with extra context (e.g. ``{"observed_count": 5}``).
    """

    key: str
    value: Any
    confidence: float
    last_updated: datetime
    source: Literal["explicit", "implicit", "inferred"] = "implicit"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        key: str,
        value: Any,
        confidence: float = 0.5,
        source: Literal["explicit", "implicit", "inferred"] = "implicit",
        metadata: dict[str, Any] | None = None,
    ) -> "Preference":
        """Create a new Preference with current UTC timestamp."""
        return cls(
            key=key,
            value=value,
            confidence=max(0.0, min(1.0, confidence)),
            last_updated=_utcnow(),
            source=source,
            metadata=metadata or {},
        )

    def with_confidence(self, new_confidence: float) -> "Preference":
        """Return a new Preference with updated confidence (immutable update)."""
        import dataclasses
        return dataclasses.replace(
            self,
            confidence=max(0.0, min(1.0, new_confidence)),
            last_updated=_utcnow(),
        )

    def with_value(self, new_value: Any, new_confidence: float | None = None) -> "Preference":
        """Return a new Preference with updated value."""
        import dataclasses
        return dataclasses.replace(
            self,
            value=new_value,
            confidence=max(0.0, min(1.0, new_confidence if new_confidence is not None else self.confidence)),
            last_updated=_utcnow(),
        )


# ─── Habit ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Habit:
    """
    A detected recurring user behaviour pattern.

    A habit is established when the same trigger-action pair is observed
    at least ``min_observations`` times (configurable, default 3).

    Attributes
    ----------
    id:
        Stable hash of (trigger_key, action) — same inputs → same ID.
    description:
        Human-readable summary, e.g. "Opens VS Code every morning".
    trigger:
        Context dict describing when the habit fires:
        ``{hour_bucket, day_of_week, active_app, topic, ...}``
    action:
        What the habit does / suggests, e.g. ``"open VS Code"``.
    confidence:
        0.0–1.0. Rises with observation count.
    observed_count:
        Number of times this trigger-action pair was observed.
    last_seen:
        UTC datetime of the most recent observation.
    """

    id: str
    description: str
    trigger: dict[str, Any]
    action: str
    confidence: float
    observed_count: int
    last_seen: datetime

    @classmethod
    def create(
        cls,
        trigger: dict[str, Any],
        action: str,
        description: str = "",
        observed_count: int = 1,
    ) -> "Habit":
        """Create a new Habit with a deterministic ID."""
        trigger_key = "|".join(f"{k}={v}" for k, v in sorted(trigger.items()))
        habit_id = _stable_id([trigger_key, action])
        confidence = min(1.0, observed_count / 10.0)  # saturates at 10 observations
        return cls(
            id=habit_id,
            description=description or f"Habit: {action}",
            trigger=trigger,
            action=action,
            confidence=confidence,
            observed_count=observed_count,
            last_seen=_utcnow(),
        )

    def observed(self) -> "Habit":
        """Return a new Habit with incremented observation count."""
        import dataclasses
        new_count = self.observed_count + 1
        return dataclasses.replace(
            self,
            observed_count=new_count,
            confidence=min(1.0, new_count / 10.0),
            last_seen=_utcnow(),
        )


# ─── Workflow ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Workflow:
    """
    A detected multi-step repeated sequence (e.g. VS Code → GitHub → Chrome).

    Attributes
    ----------
    id:
        Stable hash of the step sequence.
    name:
        Human-readable name, e.g. "Code → Review → Browse".
    steps:
        Ordered tuple of skill names / app names in the sequence.
    trigger_count:
        How many times this exact sequence was executed.
    confidence:
        0.0–1.0. Rises with trigger_count.
    last_seen:
        UTC datetime of the most recent execution.
    suggested:
        Whether Spidy has already suggested automating this workflow.
    """

    id: str
    name: str
    steps: tuple[str, ...]
    trigger_count: int
    confidence: float
    last_seen: datetime
    suggested: bool = False

    @classmethod
    def create(
        cls,
        steps: list[str],
        name: str = "",
        trigger_count: int = 1,
    ) -> "Workflow":
        """Create a new Workflow with a deterministic ID based on step sequence."""
        workflow_id = _stable_id(steps)
        auto_name = name or " → ".join(steps)
        confidence = min(1.0, trigger_count / 5.0)  # saturates at 5 executions
        return cls(
            id=workflow_id,
            name=auto_name,
            steps=tuple(steps),
            trigger_count=trigger_count,
            confidence=confidence,
            last_seen=_utcnow(),
        )

    def executed(self) -> "Workflow":
        """Return a new Workflow with incremented trigger count."""
        import dataclasses
        new_count = self.trigger_count + 1
        return dataclasses.replace(
            self,
            trigger_count=new_count,
            confidence=min(1.0, new_count / 5.0),
            last_seen=_utcnow(),
        )


# ─── FeedbackSignal ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeedbackSignal:
    """
    A single recorded feedback event from the user.

    Attributes
    ----------
    id:
        Unique signal ID.
    session_id:
        Brain session this signal belongs to.
    utterance:
        The user's original utterance.
    response:
        Spidy's response that was rated.
    rating:
        -1.0 (explicitly negative) to +1.0 (explicitly positive).
        0.0 = neutral or no feedback.
    tags:
        Intent tags, topic tags, or source tags attached to this signal.
    timestamp:
        UTC datetime when feedback was recorded.
    """

    id: str
    session_id: str
    utterance: str
    response: str
    rating: float
    tags: tuple[str, ...]
    timestamp: datetime

    @classmethod
    def create(
        cls,
        session_id: str,
        utterance: str,
        response: str,
        rating: float,
        tags: list[str] | None = None,
    ) -> "FeedbackSignal":
        """Create a new FeedbackSignal with current UTC timestamp and unique ID."""
        return cls(
            id=_short_id(),
            session_id=session_id,
            utterance=utterance[:500],   # truncate long utterances
            response=response[:500],
            rating=max(-1.0, min(1.0, rating)),
            tags=tuple(tags or []),
            timestamp=_utcnow(),
        )

    @property
    def is_positive(self) -> bool:
        return self.rating > 0.3

    @property
    def is_negative(self) -> bool:
        return self.rating < -0.3

    @property
    def is_neutral(self) -> bool:
        return -0.3 <= self.rating <= 0.3


# ─── LearningContext ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LearningContext:
    """
    Snapshot of the current context passed to habit detection.

    Created by LearningManager from live application state.
    """

    hour_of_day: int          # 0–23
    day_of_week: int          # 0=Monday … 6=Sunday
    active_app: str           # e.g. "Code.exe", "chrome.exe"
    active_window_title: str  # e.g. "main.py — VS Code"
    topic: str                # intent category, e.g. "code", "search"
    timestamp: datetime = field(default_factory=_utcnow)

    @classmethod
    def from_dict(cls, ctx: dict[str, Any]) -> "LearningContext":
        """Build a LearningContext from the dict passed to get_habit()."""
        now = _utcnow()
        return cls(
            hour_of_day=int(ctx.get("hour_of_day", now.hour)),
            day_of_week=int(ctx.get("day_of_week", now.weekday())),
            active_app=str(ctx.get("active_app", "")),
            active_window_title=str(ctx.get("active_window_title", "")),
            topic=str(ctx.get("topic", "")),
        )

    @property
    def time_bucket(self) -> str:
        """Coarse time-of-day bucket: morning / afternoon / evening / night."""
        if 6 <= self.hour_of_day < 12:
            return "morning"
        if 12 <= self.hour_of_day < 18:
            return "afternoon"
        if 18 <= self.hour_of_day < 23:
            return "evening"
        return "night"
