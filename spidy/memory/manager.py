"""
Memory Manager — Unified API for All Memory Tiers
==================================================
``MemoryManager`` is the single entry point for all memory operations.
It implements the ``MemoryInterface`` contract defined in M3 and wraps
all three memory tiers behind one clean API.

Architecture
------------
    MemoryManager  (MemoryInterface)
        ├── WorkingMemory    fast in-process session buffer
        ├── EpisodicMemory   persistent SQLite store
        └── SemanticMemory   ChromaDB vector search (optional)

All Brain interactions go through MemoryManager.
The Brain never touches SQLite or ChromaDB directly.

EventBus Integration
--------------------
Every public operation publishes a corresponding ``memory.*`` event:
    store()   → MemoryStoredEvent
    recall()  → MemoryRetrievedEvent
    search()  → MemorySearchedEvent
    clear()   → MemoryClearedEvent
    (errors)  → MemoryErrorEvent

The EventBus is optional — if no bus is passed, events are silently skipped.
This keeps tests simple (no bus required for unit tests).

Brain API
---------
The Brain's primary convenience method is ``store_interaction()``, which
packages a complete utterance → response turn into a single MemoryEntry
with automatic tags and metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from spidy.brain.interfaces import MemoryInterface
from spidy.logging.logger import get_logger
from spidy.memory.episodic import EpisodicMemory
from spidy.memory.events import (
    MemoryClearedEvent,
    MemoryErrorEvent,
    MemoryRetrievedEvent,
    MemorySearchedEvent,
    MemoryStoredEvent,
)
from spidy.memory.semantic import SemanticMemory
from spidy.memory.types import MemoryEntry, MemorySearchResult, MemoryType
from spidy.memory.working import WorkingMemory

if TYPE_CHECKING:
    from spidy.config.manager import MemoryConfig, PathsConfig
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class MemoryManager(MemoryInterface):
    """
    Unified memory API that coordinates all three memory tiers.

    Parameters
    ----------
    config:
        ``MemoryConfig`` from ``SpidyConfig.memory``.
    bus:
        Optional EventBus for publishing memory events.
    memory_dir:
        Directory for SQLite and ChromaDB persistent storage.
        Defaults to a temporary in-memory store if not provided.
    """

    def __init__(
        self,
        config: "MemoryConfig | None" = None,
        bus: "EventBus | None" = None,
        memory_dir: Path | None = None,
    ) -> None:
        self._bus = bus
        self._config = config
        self._memory_dir = memory_dir

        # Read flags from config (with safe defaults)
        enable_working = True
        enable_episodic = True
        enable_semantic = True
        max_messages = 20
        db_filename = "spidy_memory.db"
        collection_name = "spidy_memories"
        embedding_model = "all-MiniLM-L6-v2"

        if config is not None:
            enable_working = getattr(config, "enable_working", True)
            enable_episodic = getattr(config, "enable_episodic", True)
            enable_semantic = getattr(config, "enable_semantic", True)
            max_messages = config.short_term.max_messages
            db_filename = config.long_term.db_filename
            collection_name = config.semantic.collection_name
            embedding_model = config.semantic.embedding_model

        # ── Working Memory ─────────────────────────────────────────────────
        self._working: WorkingMemory | None = None
        if enable_working:
            self._working = WorkingMemory(max_messages=max_messages)

        # ── Episodic Memory ────────────────────────────────────────────────
        self._episodic: EpisodicMemory | None = None
        if enable_episodic:
            if memory_dir is not None:
                db_path = memory_dir / db_filename
            else:
                db_path = ":memory:"  # type: ignore[assignment]
            self._episodic = EpisodicMemory(db_path=db_path)

        # ── Semantic Memory ────────────────────────────────────────────────
        self._semantic: SemanticMemory | None = None
        if enable_semantic:
            if memory_dir is not None:
                chroma_dir = memory_dir / "chromadb"
            else:
                chroma_dir = ":memory:"  # type: ignore[assignment]
            self._semantic = SemanticMemory(
                collection_name=collection_name,
                embedding_model=embedding_model,
                persist_dir=chroma_dir,
            )

        self._initialized = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialise all memory tiers.

        Must be called once before any memory operations.
        Called by SpidyCore during application startup.
        """
        if self._initialized:
            return

        if self._episodic is not None:
            await self._episodic.initialize()

        # P1-4: SemanticMemory initialises lazily, but loading SentenceTransformer
        # on the first real request adds 1.5–3 s of blocking latency.  Instead, we
        # kick off the initialisation in a background thread immediately after
        # EpisodicMemory is ready.  This is fire-and-forget:
        #   - It is idempotent (_ensure_initialized is a no-op after first success).
        #   - If the warmup finishes before the first store()/search() call, the
        #     first request is instant.
        #   - If the first request arrives before the warmup finishes,
        #     _ensure_initialized() runs again synchronously (safe, idempotent).
        #   - Exceptions are caught and logged; they do not crash Spidy.
        if self._semantic is not None:
            import asyncio as _asyncio

            async def _warmup_semantic() -> None:
                try:
                    await _asyncio.to_thread(self._semantic._ensure_initialized)  # noqa: SLF001
                    log.info("SemanticMemory: background warmup complete.")
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "SemanticMemory: background warmup failed (non-fatal): {exc}",
                        exc=exc,
                    )

            _asyncio.create_task(_warmup_semantic())
            log.debug("SemanticMemory: background warmup task scheduled.")

        self._initialized = True
        log.info(
            "MemoryManager ready | working={w} | episodic={e} | semantic={s}",
            w="on" if self._working else "off",
            e="on" if self._episodic else "off",
            s="on" if self._semantic else "off",
        )

    async def close(self) -> None:
        """Graceful shutdown. Currently a no-op (SQLite connections are per-operation)."""
        log.debug("MemoryManager: shutdown.")

    # ── MemoryInterface Implementation ─────────────────────────────────────

    async def store(
        self,
        content: str,
        session_id: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a memory entry across all enabled tiers.

        The entry is stored in:
        1. WorkingMemory (always, if enabled)
        2. EpisodicMemory (always, if enabled)
        3. SemanticMemory (if enabled and available)

        All tiers use the same entry ID so they can be cross-referenced.

        Returns
        -------
        str
            The memory entry ID.
        """
        try:
            # Create one canonical entry (shared ID across all tiers)
            entry = MemoryEntry.create(
                content=content,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
                memory_type=MemoryType.EPISODIC,  # primary tier label
            )

            sources: list[str] = []

            if self._working is not None:
                self._working.add(
                    content=content,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )
                # Note: WorkingMemory creates its own entries; we track via session

            if self._episodic is not None:
                await self._episodic.store(entry)
                sources.append("episodic")

            if self._semantic is not None:
                await self._semantic.store(entry)
                sources.append("semantic")

            await self._publish(MemoryStoredEvent(
                memory_id=entry.id,
                session_id=session_id,
                memory_type="episodic",
                content_preview=content[:80],
                tags=tags or [],
            ))

            log.debug(
                "MemoryManager: stored {id} | sources={sources}",
                id=entry.id[:8],
                sources=sources,
            )
            return entry.id

        except Exception as exc:
            log.error("MemoryManager: store error: {exc}", exc=exc)
            await self._publish(MemoryErrorEvent(
                operation="store",
                session_id=session_id,
                error=str(exc),
            ))
            raise

    async def recall(
        self,
        query: str,
        session_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Recall memories relevant to a query.

        Strategy:
        1. Pull recent context from WorkingMemory (session-scoped)
        2. Retrieve keyword matches from EpisodicMemory
        3. Merge, deduplicate by ID, sort by recency, truncate to ``limit``

        Returns
        -------
        list[dict]
            Memory records, each with ``{id, content, score, ...}``.
        """
        try:
            results: dict[str, MemorySearchResult] = {}
            sources: list[str] = []

            # 1. Working Memory — recent session context
            if self._working is not None and session_id:
                working_entries = self._working.find(query, session_id=session_id, limit=limit)
                for entry in working_entries:
                    results[entry.id] = MemorySearchResult(entry=entry, score=1.0)
                if working_entries:
                    sources.append("working")

            # 2. Episodic — keyword recall
            if self._episodic is not None:
                episodic_entries = await self._episodic.recall(query, session_id, limit)
                for entry in episodic_entries:
                    if entry.id not in results:
                        results[entry.id] = MemorySearchResult(entry=entry, score=0.8)
                if episodic_entries:
                    sources.append("episodic")

            # Sort by timestamp (most recent first), truncate
            sorted_results = sorted(
                results.values(),
                key=lambda r: r.entry.timestamp,
                reverse=True,
            )[:limit]

            output = [r.to_dict() for r in sorted_results]

            await self._publish(MemoryRetrievedEvent(
                session_id=session_id,
                query_preview=query[:80],
                result_count=len(output),
                sources=sources,
            ))

            return output

        except Exception as exc:
            log.error("MemoryManager: recall error: {exc}", exc=exc)
            await self._publish(MemoryErrorEvent(
                operation="recall",
                session_id=session_id,
                error=str(exc),
            ))
            return []

    async def search(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search across all stored memories.

        Strategy:
        1. SemanticMemory (vector similarity) — if available
        2. Fall back to EpisodicMemory keyword search if semantic unavailable

        Returns
        -------
        list[dict]
            Ranked memory records.
        """
        try:
            search_type = "keyword"

            # Try semantic search first
            if self._semantic is not None and self._semantic.is_available:
                semantic_results = await self._semantic.search(query, limit=limit, tags=tags)
                if semantic_results:
                    search_type = "semantic"
                    output = [r.to_dict() for r in semantic_results]
                    await self._publish(MemorySearchedEvent(
                        query_preview=query[:80],
                        result_count=len(output),
                        search_type=search_type,
                        tags_filter=tags or [],
                    ))
                    return output

            # Fall back to episodic keyword search
            if self._episodic is not None:
                episodic_entries = await self._episodic.search(query, limit=limit, tags=tags)
                output = [
                    MemorySearchResult(entry=e, score=0.7).to_dict()
                    for e in episodic_entries
                ]
                await self._publish(MemorySearchedEvent(
                    query_preview=query[:80],
                    result_count=len(output),
                    search_type=search_type,
                    tags_filter=tags or [],
                ))
                return output

            return []

        except Exception as exc:
            log.error("MemoryManager: search error: {exc}", exc=exc)
            await self._publish(MemoryErrorEvent(
                operation="search",
                session_id="",
                error=str(exc),
            ))
            return []

    async def clear(self, session_id: str = "") -> int:
        """
        Clear memories for a session, or all memories if session_id is empty.

        Returns
        -------
        int
            Total number of entries cleared across all tiers.
        """
        try:
            total = 0

            if self._working is not None:
                total += self._working.clear(session_id)

            if self._episodic is not None:
                total += await self._episodic.clear(session_id)

            if self._semantic is not None:
                total += await self._semantic.clear(session_id)

            await self._publish(MemoryClearedEvent(
                session_id=session_id,
                count_cleared=total,
            ))

            log.info(
                "MemoryManager: cleared {n} entries | session='{sid}'",
                n=total,
                sid=session_id or "*",
            )
            return total

        except Exception as exc:
            log.error("MemoryManager: clear error: {exc}", exc=exc)
            await self._publish(MemoryErrorEvent(
                operation="clear",
                session_id=session_id,
                error=str(exc),
            ))
            return 0

    # ── Convenience Methods ────────────────────────────────────────────────

    async def store_interaction(
        self,
        utterance: str,
        response: str,
        session_id: str = "",
        intent: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Store a complete utterance → response interaction as a single memory.

        This is the primary method the Brain calls after each processed turn.
        The stored content includes both the user's utterance and Spidy's
        response, making it retrievable by either party's phrasing.

        Parameters
        ----------
        utterance:
            The user's input text.
        response:
            Spidy's response text.
        session_id:
            Current conversation session ID.
        intent:
            Classified intent label (added as a tag automatically).
        tags:
            Additional caller-supplied tags.

        Returns
        -------
        str
            The memory entry ID.
        """
        combined_tags = list(tags or [])
        if intent:
            combined_tags.append(f"intent:{intent}")

        content = f"User: {utterance}\nSpidy: {response}"
        metadata = {
            "utterance": utterance[:500],
            "response": response[:500],
            "intent": intent,
        }

        return await self.store(
            content=content,
            session_id=session_id,
            tags=combined_tags,
            metadata=metadata,
        )

    def get_working_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent N working memory entries for a session.

        Synchronous — safe to call from non-async contexts.
        Used by the Brain to inject conversation context into LLM prompts.

        Returns
        -------
        list[dict]
            Recent entries, oldest first.
        """
        if self._working is None:
            return []
        entries = self._working.get_context_window(session_id=session_id, limit=limit)
        return [e.to_dict() for e in entries]

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def has_working(self) -> bool:
        return self._working is not None

    @property
    def has_episodic(self) -> bool:
        return self._episodic is not None

    @property
    def has_semantic(self) -> bool:
        return self._semantic is not None and self._semantic.is_available

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _publish(self, event: Any) -> None:
        """Publish an event if a bus is available. Silently skip if not."""
        if self._bus is not None:
            try:
                await self._bus.publish(event)
            except Exception as exc:
                log.debug("MemoryManager: event publish failed: {exc}", exc=exc)
