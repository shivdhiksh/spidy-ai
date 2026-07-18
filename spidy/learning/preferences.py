"""
Preference Learner
==================
Stores and retrieves user preferences with confidence scores.

Persistence uses ``aiosqlite`` (same pattern as EpisodicMemory).
Falls back to an in-process dict when ``aiosqlite`` is unavailable.

Schema
------
    preferences(
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,          -- JSON-encoded
        confidence  REAL NOT NULL DEFAULT 0.5,
        source      TEXT NOT NULL DEFAULT 'implicit',
        last_updated REAL NOT NULL,          -- Unix timestamp (UTC)
        metadata    TEXT NOT NULL DEFAULT '{}'  -- JSON object
    )

Design decisions
----------------
- Key is the PRIMARY KEY so ``INSERT OR REPLACE`` handles upserts.
- ``value`` is JSON-encoded to support any serialisable type.
- Confidence is clamped to [0.0, 1.0] on every write.
- Implicit learning (from context observations) is capped at 0.8.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from spidy.learning.types import Preference
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS preferences (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    confidence   REAL NOT NULL DEFAULT 0.5,
    source       TEXT NOT NULL DEFAULT 'implicit',
    last_updated REAL NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
"""

# Cap for implicit / inferred confidence (explicit = 1.0 can exceed this)
_IMPLICIT_MAX_CONFIDENCE = 0.8


def _row_to_preference(row: tuple) -> Preference:
    key, value_json, confidence, source, last_updated_ts, meta_json = row
    return Preference(
        key=key,
        value=json.loads(value_json),
        confidence=confidence,
        source=source,  # type: ignore[arg-type]
        last_updated=datetime.fromtimestamp(last_updated_ts, tz=timezone.utc),
        metadata=json.loads(meta_json),
    )


class _InMemoryPrefFallback:
    """In-process dict fallback when aiosqlite is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, Preference] = {}

    def set(self, pref: Preference) -> None:
        self._store[pref.key] = pref

    def get(self, key: str) -> Preference | None:
        return self._store.get(key)

    def all(self) -> list[Preference]:
        return list(self._store.values())

    def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        return existed

    def count(self) -> int:
        return len(self._store)


class PreferenceLearner:
    """
    Stores and retrieves user preferences with confidence scores.

    Parameters
    ----------
    db_path:
        Path to the ``spidy_learning.db`` SQLite file.
        Pass ``:memory:`` for an ephemeral in-process store (tests).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._aiosqlite_available = False
        self._fallback: _InMemoryPrefFallback | None = None
        self._initialized = False
        self._db: Any = None  # persistent aiosqlite.Connection

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open a persistent DB connection and create the preferences table.

        A single connection is kept open for the lifetime of this object so
        that SQLite ``:memory:`` databases retain schema/data across calls.
        """
        if self._initialized:
            return
        try:
            import aiosqlite
            self._aiosqlite_available = True
        except ImportError:
            log.warning(
                "PreferenceLearner: 'aiosqlite' not installed — "
                "preferences will not persist across restarts. "
                "Install with: pip install aiosqlite"
            )
            self._fallback = _InMemoryPrefFallback()
            self._initialized = True
            return

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_CREATE_TABLE_SQL)
        await self._db.commit()

        self._initialized = True
        log.debug("PreferenceLearner: ready at '{path}'", path=self._db_path)

    async def close(self) -> None:
        """Close the persistent DB connection.  Safe to call multiple times."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False

    # ── Core API ───────────────────────────────────────────────────────────

    async def set(
        self,
        key: str,
        value: Any,
        confidence: float = 0.5,
        source: Literal["explicit", "implicit", "inferred"] = "implicit",
        metadata: dict[str, Any] | None = None,
    ) -> Preference:
        """
        Store or update a preference.

        Implicit/inferred confidence is capped at 0.8 so user explicit
        signals (1.0) always win.

        Returns
        -------
        Preference
            The stored preference (with final clamped confidence).
        """
        import time

        # Cap implicit confidence
        if source != "explicit":
            confidence = min(confidence, _IMPLICIT_MAX_CONFIDENCE)
        confidence = max(0.0, min(1.0, confidence))

        pref = Preference(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            last_updated=datetime.now(tz=timezone.utc),
            metadata=metadata or {},
        )

        if self._fallback is not None:
            self._fallback.set(pref)
            return pref

        value_json = json.dumps(value)
        meta_json = json.dumps(pref.metadata)
        ts = pref.last_updated.timestamp()

        await self._db.execute(
            """
            INSERT INTO preferences (key, value, confidence, source, last_updated, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                source=excluded.source,
                last_updated=excluded.last_updated,
                metadata=excluded.metadata
            """,
            (key, value_json, confidence, source, ts, meta_json),
        )
        await self._db.commit()

        return pref

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve the value for a preference key, or ``default`` if not set.

        Returns the raw value (not the Preference object).
        Use ``get_preference()`` for the full record.
        """
        pref = await self.get_preference(key)
        if pref is None:
            return default
        return pref.value

    async def get_preference(self, key: str) -> Preference | None:
        """
        Retrieve the full Preference record for a key, or None if absent.
        """
        if self._fallback is not None:
            return self._fallback.get(key)

        async with self._db.execute(
            "SELECT key, value, confidence, source, last_updated, metadata "
            "FROM preferences WHERE key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        return _row_to_preference(row)

    async def all_preferences(self) -> list[Preference]:
        """Return all stored preferences sorted by confidence (descending)."""
        if self._fallback is not None:
            prefs = self._fallback.all()
            return sorted(prefs, key=lambda p: p.confidence, reverse=True)

        async with self._db.execute(
            "SELECT key, value, confidence, source, last_updated, metadata "
            "FROM preferences ORDER BY confidence DESC"
        ) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_preference(r) for r in rows]

    async def update_confidence(self, key: str, delta: float) -> Preference | None:
        """
        Adjust the confidence of an existing preference by ``delta``.

        Clamps the result to [0.0, 1.0].
        Returns the updated Preference or None if the key doesn't exist.
        """
        pref = await self.get_preference(key)
        if pref is None:
            return None
        new_conf = max(0.0, min(1.0, pref.confidence + delta))
        return await self.set(
            key, pref.value, new_conf, pref.source, dict(pref.metadata)
        )

    async def delete(self, key: str) -> bool:
        """
        Delete a preference by key.

        Returns True if the key existed and was deleted.
        """
        if self._fallback is not None:
            return self._fallback.delete(key)

        cursor = await self._db.execute("DELETE FROM preferences WHERE key = ?", (key,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def count(self) -> int:
        """Return the total number of stored preferences."""
        if self._fallback is not None:
            return self._fallback.count()

        async with self._db.execute("SELECT COUNT(*) FROM preferences") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Implicit learning helpers ──────────────────────────────────────────

    async def observe_app(self, app_name: str) -> None:
        """
        Record an app observation to infer preferred IDE / browser / etc.

        Maps known app names to preference keys with medium confidence.
        These implicit inferences are overridden by explicit user statements.
        """
        _APP_PREFERENCE_MAP = {
            "code.exe": ("preferred_ide", "VS Code"),
            "code": ("preferred_ide", "VS Code"),
            "pycharm64.exe": ("preferred_ide", "PyCharm"),
            "idea64.exe": ("preferred_ide", "IntelliJ IDEA"),
            "chrome.exe": ("preferred_browser", "Chrome"),
            "chrome": ("preferred_browser", "Chrome"),
            "firefox.exe": ("preferred_browser", "Firefox"),
            "firefox": ("preferred_browser", "Firefox"),
            "msedge.exe": ("preferred_browser", "Edge"),
            "spotify.exe": ("preferred_music_app", "Spotify"),
            "slack.exe": ("preferred_chat", "Slack"),
            "teams.exe": ("preferred_chat", "Teams"),
            "discord.exe": ("preferred_chat", "Discord"),
        }
        lower = app_name.lower()
        mapping = _APP_PREFERENCE_MAP.get(lower)
        if mapping is None:
            return

        pref_key, pref_value = mapping
        existing = await self.get_preference(pref_key)
        if existing is not None and existing.source == "explicit":
            # Never overwrite an explicit preference with an inferred one
            return

        new_conf = (existing.confidence + 0.05) if existing else 0.3
        await self.set(pref_key, pref_value, new_conf, "inferred")

    async def to_context_string(self) -> str:
        """
        Return a compact natural-language summary of preferences for
        LLM prompt injection.

        Example output:
            User preferences: preferred_browser=Chrome (0.9), preferred_ide=VS Code (0.7)
        """
        prefs = await self.all_preferences()
        if not prefs:
            return ""
        parts = [f"{p.key}={p.value} ({p.confidence:.1f})" for p in prefs[:8]]
        return "User preferences: " + ", ".join(parts)
