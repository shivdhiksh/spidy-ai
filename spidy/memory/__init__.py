"""
spidy.memory — Memory Engine (Milestone 8)
==========================================
Three-tier memory system for the Spidy AI companion.

Tiers
-----
WorkingMemory   — in-process rolling session buffer (no external deps)
EpisodicMemory  — persistent SQLite store via aiosqlite
SemanticMemory  — ChromaDB vector store for similarity search

Unified API
-----------
MemoryManager   — implements MemoryInterface; single entry point for the Brain

EventBus Events
---------------
All memory operations publish typed events to ``memory.*`` topics.
See ``spidy.memory.events`` for the full list.

Quick Start
-----------
    from spidy.memory import MemoryManager

    manager = MemoryManager(config=settings.memory, bus=bus)
    await manager.initialize()

    memory_id = await manager.store("User prefers dark mode", session_id="s1")
    results = await manager.recall("dark mode", session_id="s1")
"""

from spidy.memory.episodic import EpisodicMemory
from spidy.memory.events import (
    MemoryClearedEvent,
    MemoryDeletedEvent,
    MemoryErrorEvent,
    MemoryRetrievedEvent,
    MemorySearchedEvent,
    MemoryStoredEvent,
)
from spidy.memory.manager import MemoryManager
from spidy.memory.semantic import SemanticMemory
from spidy.memory.types import MemoryEntry, MemorySearchResult, MemoryType
from spidy.memory.working import WorkingMemory

__all__ = [
    # Main entry point
    "MemoryManager",
    # Tier implementations
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    # Types
    "MemoryEntry",
    "MemoryType",
    "MemorySearchResult",
    # Events
    "MemoryStoredEvent",
    "MemoryRetrievedEvent",
    "MemorySearchedEvent",
    "MemoryDeletedEvent",
    "MemoryClearedEvent",
    "MemoryErrorEvent",
]
