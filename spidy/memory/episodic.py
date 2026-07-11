"""
Episodic Memory — SQLite-Backed Persistent Memory Store
========================================================
Provides durable, timestamped storage of every important interaction.

Architecture
------------
- Primary backend: ``aiosqlite`` for async non-blocking SQLite I/O
- Fallback: pure in-process list when ``aiosqlite`` is not installed
- DB file path: resolved from ``paths.memory_dir / config.db_filename``

Schema
------
    memories(
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        session_id  TEXT NOT NULL DEFAULT '',
        tags        TEXT NOT NULL DEFAULT '[]',   -- JSON array
        metadata    TEXT NOT NULL DEFAULT '{}',   -- JSON object
        timestamp   REAL NOT NULL,                -- Unix timestamp (UTC)
        memory_type TEXT NOT NULL DEFAULT 'episodic'
    )

Design decisions
----------------
1. ``aiosqlite`` wraps SQLite in a thread pool — safe for asyncio.
2. JSON-serialised tags + metadata — SQLite has no native array type.
3. Keyword recall uses LIKE for broad matching; SemanticMemory handles
   proper similarity search.
4. In-memory fallback is behaviorally identical but not persistent —
   data is lost on restart. Suitable for testing without disk I/O.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spidy.logging.logger import get_logger
from spidy.memory.types import MemoryEntry, MemoryType

log = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    session_id  TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '[]',
    metadata    TEXT NOT NULL DEFAULT '{}',
    timestamp   REAL NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'episodic'
);
"""

_INDEX_SESSION_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_session
ON memories (session_id);
"""

_INDEX_TIMESTAMP_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_timestamp
ON memories (timestamp);
"""


def _row_to_entry(row: tuple) -> MemoryEntry:
    """Convert a DB row tuple to a MemoryEntry."""
    mem_id, content, session_id, tags_json, meta_json, timestamp, mem_type = row
    from datetime import datetime, timezone
    return MemoryEntry(
        id=mem_id,
        content=content,
        session_id=session_id,
        tags=tuple(json.loads(tags_json)),
        metadata=json.loads(meta_json),
        timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
        memory_type=MemoryType(mem_type),
    )


class _InMemoryFallback:
    """
    Pure in-process list fallback used when ``aiosqlite`` is unavailable.

    All ops are synchronous internally; they're wrapped in async by EpisodicMemory.
    Data is NOT persistent — cleared on restart.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def store(self, entry: MemoryEntry) -> None:
        # Replace existing entry with same ID (INSERT OR REPLACE semantics)
        self._entries = [e for e in self._entries if e.id != entry.id]
        self._entries.append(entry)

    def recall(self, query: str, session_id: str, limit: int) -> list[MemoryEntry]:
        lower_q = query.lower()
        if session_id:
            candidates = [e for e in self._entries if e.session_id == session_id]
        else:
            candidates = list(self._entries)
        # Keyword filter
        matches = [e for e in candidates if lower_q in e.content.lower()]
        # Most recent first
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:limit]

    def search(
        self, query: str, limit: int, tags: list[str] | None
    ) -> list[MemoryEntry]:
        lower_q = query.lower()
        candidates = self._entries
        if tags:
            candidates = [e for e in candidates if any(t in e.tags for t in tags)]
        matches = [e for e in candidates if lower_q in e.content.lower()]
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:limit]

    def delete(self, memory_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != memory_id]
        return len(self._entries) < before

    def clear(self, session_id: str) -> int:
        before = len(self._entries)
        if session_id:
            self._entries = [e for e in self._entries if e.session_id != session_id]
        else:
            self._entries = []
        return before - len(self._entries)

    def count(self, session_id: str = "") -> int:
        if session_id:
            return sum(1 for e in self._entries if e.session_id == session_id)
        return len(self._entries)


class EpisodicMemory:
    """
    SQLite-backed persistent memory store for long-term episodic recall.

    Parameters
    ----------
    db_path:
        Absolute path to the SQLite database file.
        Pass ``:memory:`` for an in-process test database.
        If the parent directory does not exist it will be created.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._aiosqlite_available = False
        self._fallback: _InMemoryFallback | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create the DB schema. Must be called before any other operation."""
        if self._initialized:
            return

        try:
            import aiosqlite  # noqa: F401  (import probe only)
            self._aiosqlite_available = True
        except ImportError:
            log.warning(
                "EpisodicMemory: 'aiosqlite' is not installed. "
                "Falling back to in-process list (not persistent). "
                "Install with: pip install aiosqlite"
            )
            self._fallback = _InMemoryFallback()
            self._initialized = True
            return

        # Create parent directory if needed
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        await self._create_schema()
        self._initialized = True
        log.info(
            "EpisodicMemory: SQLite database ready at '{path}'",
            path=self._db_path,
        )

    async def _create_schema(self) -> None:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE_SQL)
            await db.execute(_INDEX_SESSION_SQL)
            await db.execute(_INDEX_TIMESTAMP_SQL)
            await db.commit()

    # ── Public API ──────────────────────────────────────────────────────────

    async def store(self, entry: MemoryEntry) -> str:
        """
        Persist a memory entry.

        Parameters
        ----------
        entry:
            The MemoryEntry to store. Its ``id`` is used as the primary key.

        Returns
        -------
        str
            The entry ID.
        """
        self._assert_initialized()

        if self._fallback is not None:
            self._fallback.store(entry)
            return entry.id

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, content, session_id, tags, metadata, timestamp, memory_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.content,
                    entry.session_id,
                    json.dumps(list(entry.tags)),
                    json.dumps(entry.metadata),
                    entry.timestamp.timestamp(),
                    entry.memory_type.value,
                ),
            )
            await db.commit()

        log.debug("EpisodicMemory: stored {id}", id=entry.id[:8])
        return entry.id

    async def recall(
        self,
        query: str,
        session_id: str = "",
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """
        Recall memories relevant to a query, optionally scoped to a session.

        Uses keyword LIKE matching; for semantic search use SemanticMemory.
        Results are returned most-recent first.

        Parameters
        ----------
        query:
            Keywords to search for (case-insensitive).
        session_id:
            Scope to this session. Empty = search all sessions.
        limit:
            Maximum results.

        Returns
        -------
        list[MemoryEntry]
            Most-recent matching entries.
        """
        self._assert_initialized()

        if self._fallback is not None:
            return self._fallback.recall(query, session_id, limit)

        import aiosqlite
        like_query = f"%{query}%"
        if session_id:
            sql = """
                SELECT id, content, session_id, tags, metadata, timestamp, memory_type
                FROM memories
                WHERE session_id = ? AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params: tuple = (session_id, like_query, limit)
        else:
            sql = """
                SELECT id, content, session_id, tags, metadata, timestamp, memory_type
                FROM memories
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (like_query, limit)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        return [_row_to_entry(row) for row in rows]

    async def search(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """
        Keyword search across all sessions, optionally filtered by tags.

        Parameters
        ----------
        query:
            Search query string.
        limit:
            Maximum results.
        tags:
            If provided, only entries with at least one matching tag are returned.

        Returns
        -------
        list[MemoryEntry]
            Most-recent matching entries.
        """
        self._assert_initialized()

        if self._fallback is not None:
            return self._fallback.search(query, limit, tags)

        import aiosqlite
        like_query = f"%{query}%"

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT id, content, session_id, tags, metadata, timestamp, memory_type
                FROM memories
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (like_query, limit * 3 if tags else limit),  # over-fetch if filtering
            ) as cursor:
                rows = await cursor.fetchall()

        entries = [_row_to_entry(row) for row in rows]

        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set.intersection(set(e.tags))]
            entries = entries[:limit]

        return entries

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory entry by ID.

        Returns
        -------
        bool
            True if the entry was found and deleted, False otherwise.
        """
        self._assert_initialized()

        if self._fallback is not None:
            return self._fallback.delete(memory_id)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            await db.commit()
            deleted = cursor.rowcount > 0

        log.debug(
            "EpisodicMemory: delete {id} → {result}",
            id=memory_id[:8],
            result="ok" if deleted else "not found",
        )
        return deleted

    async def clear(self, session_id: str = "") -> int:
        """
        Clear memories for a session, or all memories if session_id is empty.

        Returns
        -------
        int
            Number of entries removed.
        """
        self._assert_initialized()

        if self._fallback is not None:
            return self._fallback.clear(session_id)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            if session_id:
                cursor = await db.execute(
                    "DELETE FROM memories WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = await db.execute("DELETE FROM memories")
            await db.commit()
            count = cursor.rowcount

        log.info(
            "EpisodicMemory: cleared {n} entries (session='{sid}')",
            n=count,
            sid=session_id or "*",
        )
        return count

    async def count(self, session_id: str = "") -> int:
        """Return the number of stored memories."""
        self._assert_initialized()

        if self._fallback is not None:
            return self._fallback.count(session_id)

        import aiosqlite
        if session_id:
            sql = "SELECT COUNT(*) FROM memories WHERE session_id = ?"
            params: tuple = (session_id,)
        else:
            sql = "SELECT COUNT(*) FROM memories"
            params = ()

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(sql, params) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    @property
    def is_persistent(self) -> bool:
        """True if using real SQLite, False if using in-memory fallback."""
        return self._aiosqlite_available

    # ── Internal helpers ────────────────────────────────────────────────────

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "EpisodicMemory.initialize() must be called before use. "
                "Call 'await memory.initialize()' during application startup."
            )
