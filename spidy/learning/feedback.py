"""
Feedback Processor
==================
Processes explicit and implicit feedback signals from the user.

Feedback integration
--------------------
Positive feedback (+1.0)
    → Boost confidence on preferences matching the signal's tags (+delta).
    → Log for sentiment analysis.

Negative feedback (-1.0)
    → Decay confidence on matching preferences (-delta).
    → Tag the signal as "correction" for future analysis.

Neutral (0.0)
    → Stored for aggregate analysis only; no confidence change.

Persistence
-----------
    feedback(
        id          TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL,
        utterance   TEXT NOT NULL,
        response    TEXT NOT NULL,
        rating      REAL NOT NULL,
        tags        TEXT NOT NULL DEFAULT '[]',   -- JSON array
        timestamp   REAL NOT NULL
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from spidy.learning.types import FeedbackSignal
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus
    from spidy.learning.preferences import PreferenceLearner

log = get_logger(__name__)

_CREATE_FEEDBACK_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    utterance   TEXT NOT NULL,
    response    TEXT NOT NULL,
    rating      REAL NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',
    timestamp   REAL NOT NULL
);
"""

_INDEX_FEEDBACK_SESSION_SQL = """
CREATE INDEX IF NOT EXISTS idx_feedback_session
ON feedback (session_id);
"""

_INDEX_FEEDBACK_TIMESTAMP_SQL = """
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
ON feedback (timestamp);
"""


def _row_to_signal(row: tuple) -> FeedbackSignal:
    sig_id, session_id, utterance, response, rating, tags_json, timestamp_ts = row
    return FeedbackSignal(
        id=sig_id,
        session_id=session_id,
        utterance=utterance,
        response=response,
        rating=rating,
        tags=tuple(json.loads(tags_json)),
        timestamp=datetime.fromtimestamp(timestamp_ts, tz=timezone.utc),
    )


class _InMemoryFeedbackFallback:
    """In-process fallback when aiosqlite is unavailable."""

    def __init__(self) -> None:
        self._signals: list[FeedbackSignal] = []

    def store(self, signal: FeedbackSignal) -> None:
        self._signals.append(signal)

    def recent(self, limit: int) -> list[FeedbackSignal]:
        return sorted(self._signals, key=lambda s: s.timestamp, reverse=True)[:limit]

    def by_tag(self, tag: str) -> list[FeedbackSignal]:
        return [s for s in self._signals if tag in s.tags]

    def count(self) -> int:
        return len(self._signals)


class FeedbackProcessor:
    """
    Records and processes user feedback signals.

    Parameters
    ----------
    db_path:
        Path to ``spidy_learning.db``. Pass ``:memory:`` for tests.
    feedback_boost:
        Confidence delta applied for positive signals.
    feedback_decay:
        Confidence delta (negative) applied for negative signals.
    bus:
        Optional EventBus for publishing ``learning.feedback_recorded`` events.
    preferences:
        Optional PreferenceLearner to update confidence on feedback.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        feedback_boost: float = 0.15,
        feedback_decay: float = 0.10,
        bus: "EventBus | None" = None,
        preferences: "PreferenceLearner | None" = None,
    ) -> None:
        self._db_path = str(db_path)
        self._boost = feedback_boost
        self._decay = feedback_decay
        self._bus = bus
        self._preferences = preferences
        self._aiosqlite_available = False
        self._fallback: _InMemoryFeedbackFallback | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create schema. Must be called before any other method."""
        if self._initialized:
            return
        try:
            import aiosqlite  # noqa: F401
            self._aiosqlite_available = True
        except ImportError:
            log.warning(
                "FeedbackProcessor: 'aiosqlite' not installed — "
                "feedback will not persist across restarts. "
                "Install with: pip install aiosqlite"
            )
            self._fallback = _InMemoryFeedbackFallback()
            self._initialized = True
            return

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_FEEDBACK_SQL)
            await db.execute(_INDEX_FEEDBACK_SESSION_SQL)
            await db.execute(_INDEX_FEEDBACK_TIMESTAMP_SQL)
            await db.commit()

        self._initialized = True
        log.debug("FeedbackProcessor: ready at '{path}'", path=self._db_path)

    # ── Core API ───────────────────────────────────────────────────────────

    async def process(self, signal: FeedbackSignal) -> None:
        """
        Store a feedback signal and apply confidence updates to preferences.

        Parameters
        ----------
        signal:
            The FeedbackSignal to process (created by LearningManager).
        """
        # Store the signal
        await self._store(signal)

        # Apply preference confidence adjustments
        if self._preferences is not None:
            await self._apply_feedback(signal)

        # Publish event
        if self._bus is not None:
            from spidy.learning.events import LearningFeedbackRecordedEvent
            await self._bus.publish(LearningFeedbackRecordedEvent(
                signal_id=signal.id,
                session_id=signal.session_id,
                rating=signal.rating,
                tags=list(signal.tags),
            ))

        log.debug(
            "FeedbackProcessor: recorded signal {sid} rating={r:.2f}",
            sid=signal.id,
            r=signal.rating,
        )

    async def get_recent(self, limit: int = 20) -> list[FeedbackSignal]:
        """Return the most recent feedback signals (newest first)."""
        if self._fallback is not None:
            return self._fallback.recent(limit)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, session_id, utterance, response, rating, tags, timestamp "
                "FROM feedback ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [_row_to_signal(r) for r in rows]

    async def sentiment_by_tag(self, tag: str) -> float:
        """
        Return the average rating for signals that include ``tag``.

        Returns 0.0 if no signals match.
        """
        if self._fallback is not None:
            signals = self._fallback.by_tag(tag)
        else:
            # Read matching signals from DB (tags stored as JSON array)
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT id, session_id, utterance, response, rating, tags, timestamp "
                    "FROM feedback WHERE tags LIKE ?",
                    (f'%"{tag}"%',),
                ) as cursor:
                    rows = await cursor.fetchall()
            signals = [_row_to_signal(r) for r in rows]
            # Filter accurately (LIKE can produce false positives for similar tags)
            signals = [s for s in signals if tag in s.tags]

        if not signals:
            return 0.0
        return sum(s.rating for s in signals) / len(signals)

    async def count(self) -> int:
        """Return the total number of stored feedback signals."""
        if self._fallback is not None:
            return self._fallback.count()

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM feedback") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Private helpers ────────────────────────────────────────────────────

    async def _store(self, signal: FeedbackSignal) -> None:
        """Persist a FeedbackSignal to the database."""
        if self._fallback is not None:
            self._fallback.store(signal)
            return

        import aiosqlite
        tags_json = json.dumps(list(signal.tags))
        ts = signal.timestamp.timestamp()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO feedback
                    (id, session_id, utterance, response, rating, tags, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.id, signal.session_id, signal.utterance,
                    signal.response, signal.rating, tags_json, ts,
                ),
            )
            await db.commit()

    async def _apply_feedback(self, signal: FeedbackSignal) -> None:
        """
        Adjust preference confidence based on signal rating.

        Positive → boost preferences whose key appears in signal tags.
        Negative → decay those preferences.
        """
        assert self._preferences is not None

        if signal.is_neutral:
            return

        delta = self._boost if signal.is_positive else -self._decay

        for tag in signal.tags:
            # Tags like "preferred_browser", "preferred_ide" map directly to pref keys
            pref = await self._preferences.get_preference(tag)
            if pref is not None:
                await self._preferences.update_confidence(tag, delta)
                log.debug(
                    "FeedbackProcessor: updated confidence for '{k}' by {d:+.2f}",
                    k=tag,
                    d=delta,
                )
