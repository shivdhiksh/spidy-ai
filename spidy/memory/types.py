"""
Memory Types — Shared Dataclasses for the Memory Engine
========================================================
All three memory tiers (Working, Episodic, Semantic) use these
types to represent stored memories and search results.

Frozen dataclasses give us:
- Immutability guarantees (no accidental mutation)
- Hash-ability (usable in sets for deduplication)
- IDE autocomplete on all fields
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ─── Enums ────────────────────────────────────────────────────────────────────


class MemoryType(str, Enum):
    """
    Tier of memory where the entry is stored.

    WORKING  — in-process dict; cleared on shutdown; fast lookup
    EPISODIC — SQLite-backed; persists across sessions; timestamped
    SEMANTIC — ChromaDB vector store; similarity-searchable embeddings
    """
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# ─── Core Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryEntry:
    """
    A single stored memory entry.

    Created by any of the three memory tiers and returned by all
    read operations. Immutable after creation.

    Fields
    ------
    id:
        UUID4 string — unique identifier for this memory.
    content:
        The text content to remember.
    session_id:
        Conversation session this memory belongs to. Empty = global.
    tags:
        Optional categorisation tags (e.g. ``["work", "code"]``).
    metadata:
        Arbitrary key-value metadata from the caller.
    timestamp:
        UTC datetime when the memory was created.
    memory_type:
        Which tier stored this entry.
    """

    id: str
    content: str
    session_id: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        compare=False,
        hash=False,
    )
    memory_type: MemoryType = MemoryType.WORKING

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serialisable dict for API responses."""
        return {
            "id": self.id,
            "content": self.content,
            "session_id": self.session_id,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type.value,
        }

    @classmethod
    def create(
        cls,
        content: str,
        session_id: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        memory_id: str | None = None,
    ) -> "MemoryEntry":
        """
        Factory method — creates a new MemoryEntry with a fresh UUID and
        current UTC timestamp.
        """
        return cls(
            id=memory_id or str(uuid.uuid4()),
            content=content,
            session_id=session_id,
            tags=tuple(tags) if tags else (),
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc),
            memory_type=memory_type,
        )


@dataclass(frozen=True)
class MemorySearchResult:
    """
    A memory entry paired with a relevance score from a search operation.

    The ``score`` is tier-dependent:
    - SemanticMemory: cosine similarity (0.0–1.0, higher = more similar)
    - EpisodicMemory: 1.0 for keyword matches, decaying with age
    - WorkingMemory: always 1.0 (exact session match)
    """

    entry: MemoryEntry
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Flatten into a single dict for MemoryInterface return values."""
        result = self.entry.to_dict()
        result["score"] = self.score
        return result
