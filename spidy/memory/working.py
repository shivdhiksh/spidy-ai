"""
Working Memory — In-Process Rolling Session Buffer
===================================================
Provides ultra-fast, in-memory storage scoped to the current session.

Properties
----------
- Zero external dependencies (pure Python)
- Per-session isolation: sessions never share entries
- Configurable rolling window (oldest entries dropped when capacity reached)
- Thread-safe: all operations are synchronous dict/list operations;
  designed for use within a single asyncio event loop
- Cleared automatically on session end or application shutdown

This is the first tier of memory that recall() queries because it always
has the most recent conversation context.
"""

from __future__ import annotations

import collections
from typing import Any

from spidy.logging.logger import get_logger
from spidy.memory.types import MemoryEntry, MemoryType

log = get_logger(__name__)

_DEFAULT_MAX_MESSAGES = 20


class WorkingMemory:
    """
    In-process rolling buffer for active session context.

    Each session has its own bounded deque. When the deque is full,
    the oldest entry is automatically evicted (FIFO).

    Parameters
    ----------
    max_messages:
        Maximum number of entries to keep per session. Defaults to 20.
    """

    def __init__(self, max_messages: int = _DEFAULT_MAX_MESSAGES) -> None:
        self._max_messages = max(1, max_messages)
        # session_id → deque of MemoryEntry
        self._sessions: dict[str, collections.deque[MemoryEntry]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        session_id: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Add a new entry to the working memory for the given session.

        If the session buffer is full, the oldest entry is evicted.

        Parameters
        ----------
        content:
            Text to remember.
        session_id:
            Session this entry belongs to. Empty string = "default".
        tags:
            Optional categorisation tags.
        metadata:
            Optional arbitrary metadata.

        Returns
        -------
        MemoryEntry
            The newly created (and stored) entry.
        """
        sid = session_id or "default"
        entry = MemoryEntry.create(
            content=content,
            session_id=sid,
            tags=tags,
            metadata=metadata,
            memory_type=MemoryType.WORKING,
        )
        buf = self._get_or_create_buffer(sid)
        buf.append(entry)
        log.debug(
            "WorkingMemory: added entry {id} to session '{sid}' ({n}/{max})",
            id=entry.id[:8],
            sid=sid,
            n=len(buf),
            max=self._max_messages,
        )
        return entry

    def get_session(self, session_id: str = "") -> list[MemoryEntry]:
        """
        Return all entries for a session in chronological order.

        Parameters
        ----------
        session_id:
            Session to retrieve. Empty = "default".

        Returns
        -------
        list[MemoryEntry]
            All entries in insertion order (oldest first).
        """
        sid = session_id or "default"
        buf = self._sessions.get(sid)
        if buf is None:
            return []
        return list(buf)

    def get_context_window(
        self,
        session_id: str = "",
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """
        Return the most recent N entries for a session.

        Parameters
        ----------
        session_id:
            Session to query. Empty = "default".
        limit:
            Maximum entries to return. None = all entries in the buffer.

        Returns
        -------
        list[MemoryEntry]
            Most recent entries in chronological order (oldest first).
        """
        entries = self.get_session(session_id)
        if limit is not None:
            entries = entries[-limit:]
        return entries

    def find(
        self,
        query: str,
        session_id: str = "",
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """
        Simple keyword search within the working memory for a session.

        Searches are case-insensitive substring matches against ``content``.
        Returns the N most recently added matches.

        Parameters
        ----------
        query:
            Keyword(s) to search for.
        session_id:
            Session to search. Empty = search all sessions.
        limit:
            Maximum results to return.

        Returns
        -------
        list[MemoryEntry]
            Matching entries, most recent first.
        """
        lower_query = query.lower()

        if session_id:
            entries = self.get_session(session_id)
        else:
            # Search across all sessions when no session_id given
            entries = [e for buf in self._sessions.values() for e in buf]

        matches = [e for e in entries if lower_query in e.content.lower()]
        # Most recent first
        return list(reversed(matches))[:limit]

    def clear(self, session_id: str = "") -> int:
        """
        Clear all entries for a session, or all sessions if empty.

        Parameters
        ----------
        session_id:
            Session to clear. Empty string = clear ALL sessions.

        Returns
        -------
        int
            Number of entries removed.
        """
        if session_id:
            sid = session_id
            buf = self._sessions.pop(sid, None)
            count = len(buf) if buf else 0
            log.debug("WorkingMemory: cleared {n} entries for session '{sid}'", n=count, sid=sid)
            return count
        else:
            count = sum(len(b) for b in self._sessions.values())
            self._sessions.clear()
            log.debug("WorkingMemory: cleared all sessions ({n} entries)", n=count)
            return count

    def session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)

    def entry_count(self, session_id: str = "") -> int:
        """Return the number of entries for a session (or total)."""
        if session_id:
            buf = self._sessions.get(session_id or "default")
            return len(buf) if buf else 0
        return sum(len(b) for b in self._sessions.values())

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_or_create_buffer(self, session_id: str) -> collections.deque:
        if session_id not in self._sessions:
            self._sessions[session_id] = collections.deque(maxlen=self._max_messages)
        return self._sessions[session_id]
