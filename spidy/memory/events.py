"""
Memory Events — EventBus Events for the Memory Engine
======================================================
Topic namespace: ``memory.*``

All 6 event types are frozen dataclasses. The Brain and other modules
subscribe to these events to track memory activity.

Topics
------
memory.stored       — a new memory was successfully stored
memory.retrieved    — memories were recalled for a query
memory.searched     — a semantic/keyword search was performed
memory.deleted      — a specific memory entry was deleted
memory.updated      — a memory entry was updated (reserved for future)
memory.error        — any memory-layer error occurred
memory.cleared      — a session (or all) memories were cleared
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


# ─── Memory Lifecycle Events ──────────────────────────────────────────────────


@dataclass
class MemoryStoredEvent(Event):
    """Published when a memory entry is successfully stored."""
    topic = "memory.stored"
    memory_id: str = ""
    session_id: str = ""
    memory_type: str = ""          # "working" | "episodic" | "semantic"
    content_preview: str = ""      # first 80 chars of content
    tags: list[str] = field(default_factory=list)


@dataclass
class MemoryRetrievedEvent(Event):
    """Published when memories are recalled for a query."""
    topic = "memory.retrieved"
    session_id: str = ""
    query_preview: str = ""        # first 80 chars of query
    result_count: int = 0
    sources: list[str] = field(default_factory=list)   # ["working", "episodic", ...]


@dataclass
class MemorySearchedEvent(Event):
    """Published when a semantic or keyword search is performed."""
    topic = "memory.searched"
    query_preview: str = ""
    result_count: int = 0
    search_type: str = ""          # "semantic" | "keyword"
    tags_filter: list[str] = field(default_factory=list)


@dataclass
class MemoryDeletedEvent(Event):
    """Published when a specific memory entry is deleted."""
    topic = "memory.deleted"
    memory_id: str = ""
    session_id: str = ""


@dataclass
class MemoryClearedEvent(Event):
    """Published when memories for a session (or all) are cleared."""
    topic = "memory.cleared"
    session_id: str = ""           # empty = all memories cleared
    count_cleared: int = 0


@dataclass
class MemoryErrorEvent(Event):
    """Published when any memory-layer error occurs."""
    topic = "memory.error"
    operation: str = ""            # "store" | "recall" | "search" | "delete" | "clear"
    session_id: str = ""
    error: str = ""
