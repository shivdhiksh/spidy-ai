"""
Habit Detector
==============
Detects recurring user behaviour patterns from interaction observations.

A habit is a (trigger, action) pair that has been observed at least
``min_observations`` times (default: 3).

Trigger signals
---------------
- ``hour_bucket`` — time-of-day: morning / afternoon / evening / night
- ``day_of_week``  — 0=Mon … 6=Sun
- ``active_app``   — foreground application name
- ``topic``        — intent category from the Brain

Detection algorithm
-------------------
1. Every ``observe()`` call logs a (trigger_key, action) pair.
2. After each observation the count is incremented.
3. When count crosses ``min_observations``, the habit is promoted to
   the ``habits`` table and a ``LearningHabitDetectedEvent`` is published.
4. ``get_matching_habit()`` checks the current context against all stored
   habits and returns the best match (highest confidence, all trigger
   fields must match or be empty).

Persistence
-----------
Uses the shared ``spidy_learning.db`` SQLite file.
Falls back to an in-process dict when ``aiosqlite`` is unavailable.

Schema
------
    habit_observations(
        trigger_key  TEXT NOT NULL,
        action       TEXT NOT NULL,
        count        INTEGER NOT NULL DEFAULT 1,
        last_seen    REAL NOT NULL,
        PRIMARY KEY (trigger_key, action)
    )

    habits(
        id           TEXT PRIMARY KEY,
        description  TEXT NOT NULL,
        trigger_json TEXT NOT NULL,  -- JSON dict
        action       TEXT NOT NULL,
        confidence   REAL NOT NULL,
        observed_count INTEGER NOT NULL,
        last_seen    REAL NOT NULL
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from spidy.learning.types import Habit, LearningContext
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

_CREATE_OBSERVATIONS_SQL = """
CREATE TABLE IF NOT EXISTS habit_observations (
    trigger_key  TEXT NOT NULL,
    action       TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 1,
    last_seen    REAL NOT NULL,
    PRIMARY KEY (trigger_key, action)
);
"""

_CREATE_HABITS_SQL = """
CREATE TABLE IF NOT EXISTS habits (
    id             TEXT PRIMARY KEY,
    description    TEXT NOT NULL,
    trigger_json   TEXT NOT NULL,
    action         TEXT NOT NULL,
    confidence     REAL NOT NULL,
    observed_count INTEGER NOT NULL,
    last_seen      REAL NOT NULL
);
"""


def _row_to_habit(row: tuple) -> Habit:
    habit_id, description, trigger_json, action, confidence, observed_count, last_seen_ts = row
    return Habit(
        id=habit_id,
        description=description,
        trigger=json.loads(trigger_json),
        action=action,
        confidence=confidence,
        observed_count=observed_count,
        last_seen=datetime.fromtimestamp(last_seen_ts, tz=timezone.utc),
    )


def _make_trigger_key(trigger: dict[str, Any]) -> str:
    """Deterministic string key from trigger dict."""
    return "|".join(f"{k}={v}" for k, v in sorted(trigger.items()) if v)


def _build_trigger(ctx: LearningContext) -> dict[str, Any]:
    """Extract the habit trigger signals from a LearningContext."""
    return {
        "time_bucket": ctx.time_bucket,
        "day_of_week": ctx.day_of_week,
        "active_app": ctx.active_app.lower() if ctx.active_app else "",
        "topic": ctx.topic,
    }


class _InMemoryHabitFallback:
    """In-process fallback when aiosqlite is unavailable."""

    def __init__(self) -> None:
        self._observations: dict[tuple[str, str], int] = {}  # (trigger_key, action) → count
        self._habits: dict[str, Habit] = {}

    def increment_observation(self, trigger_key: str, action: str) -> int:
        k = (trigger_key, action)
        self._observations[k] = self._observations.get(k, 0) + 1
        return self._observations[k]

    def store_habit(self, habit: Habit) -> None:
        self._habits[habit.id] = habit

    def list_habits(self) -> list[Habit]:
        return list(self._habits.values())

    def get_habit(self, habit_id: str) -> Habit | None:
        return self._habits.get(habit_id)

    def count_habits(self) -> int:
        return len(self._habits)


class HabitDetector:
    """
    Observes user interactions and detects recurring behaviour patterns.

    Parameters
    ----------
    db_path:
        Path to the ``spidy_learning.db`` SQLite file.
        Pass ``:memory:`` for an ephemeral in-process store (tests).
    min_observations:
        Number of observations required before a habit is promoted.
    bus:
        Optional EventBus for publishing ``learning.habit_detected`` events.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        min_observations: int = 3,
        bus: "EventBus | None" = None,
    ) -> None:
        self._db_path = str(db_path)
        self._min_observations = min_observations
        self._bus = bus
        self._aiosqlite_available = False
        self._fallback: _InMemoryHabitFallback | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create schema tables. Must be called before any other method."""
        if self._initialized:
            return
        try:
            import aiosqlite  # noqa: F401
            self._aiosqlite_available = True
        except ImportError:
            log.warning(
                "HabitDetector: 'aiosqlite' not installed — "
                "habits will not persist across restarts."
            )
            self._fallback = _InMemoryHabitFallback()
            self._initialized = True
            return

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_OBSERVATIONS_SQL)
            await db.execute(_CREATE_HABITS_SQL)
            await db.commit()

        self._initialized = True
        log.debug("HabitDetector: ready at '{path}'", path=self._db_path)

    # ── Core API ───────────────────────────────────────────────────────────

    async def observe(
        self,
        context: dict[str, Any],
        utterance: str,
        intent: str,
    ) -> Habit | None:
        """
        Record one interaction observation and check for habit promotion.

        Called by LearningManager.observe_context() on every Brain turn.

        Parameters
        ----------
        context:
            Current context dict (same format as get_habit() context param).
        utterance:
            The user's utterance (used to build action label).
        intent:
            Intent category string (e.g. ``"code"``, ``"search"``).

        Returns
        -------
        Habit | None
            The promoted Habit if this observation crossed the threshold,
            else None.
        """
        ctx = LearningContext.from_dict({**context, "topic": intent})
        trigger = _build_trigger(ctx)
        trigger_key = _make_trigger_key(trigger)
        action = f"intent:{intent}"   # the 'action' is the intent type

        count = await self._increment_observation(trigger_key, action)

        if count < self._min_observations:
            return None

        # Habit threshold crossed — store or update
        habit = await self._promote_habit(trigger, action, count)
        if habit:
            log.info(
                "HabitDetector: habit detected — '{desc}' (confidence={conf:.2f})",
                desc=habit.description,
                conf=habit.confidence,
            )
            if self._bus is not None:
                from spidy.learning.events import LearningHabitDetectedEvent
                await self._bus.publish(LearningHabitDetectedEvent(
                    habit_id=habit.id,
                    description=habit.description,
                    confidence=habit.confidence,
                    observed_count=habit.observed_count,
                ))
        return habit

    async def get_matching_habit(self, context: dict[str, Any]) -> Habit | None:
        """
        Check the current context against all stored habits and return the best match.

        A habit matches when ALL non-empty trigger fields match the context.
        Among multiple matches, returns the one with the highest confidence.

        Returns None if no habit matches or no habits are stored.
        """
        habits = await self.list_habits()
        if not habits:
            return None

        ctx = LearningContext.from_dict(context)
        candidates: list[tuple[float, Habit]] = []

        for habit in habits:
            if _habit_matches(habit, ctx):
                candidates.append((habit.confidence, habit))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    async def list_habits(self) -> list[Habit]:
        """Return all detected habits sorted by confidence descending."""
        if self._fallback is not None:
            habits = self._fallback.list_habits()
            return sorted(habits, key=lambda h: h.confidence, reverse=True)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, description, trigger_json, action, confidence, "
                "observed_count, last_seen FROM habits ORDER BY confidence DESC"
            ) as cursor:
                rows = await cursor.fetchall()

        return [_row_to_habit(r) for r in rows]

    async def count(self) -> int:
        """Return the number of detected habits."""
        if self._fallback is not None:
            return self._fallback.count_habits()

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM habits") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Private helpers ────────────────────────────────────────────────────

    async def _increment_observation(self, trigger_key: str, action: str) -> int:
        """Increment the observation count for a (trigger_key, action) pair."""
        import time

        if self._fallback is not None:
            return self._fallback.increment_observation(trigger_key, action)

        import aiosqlite
        now_ts = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO habit_observations (trigger_key, action, count, last_seen)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(trigger_key, action) DO UPDATE SET
                    count = count + 1,
                    last_seen = excluded.last_seen
                """,
                (trigger_key, action, now_ts),
            )
            await db.commit()
            async with db.execute(
                "SELECT count FROM habit_observations WHERE trigger_key=? AND action=?",
                (trigger_key, action),
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 1

    async def _promote_habit(
        self, trigger: dict[str, Any], action: str, count: int
    ) -> Habit | None:
        """Store or update a habit once it crosses the min_observations threshold."""
        import time

        description = _describe_habit(trigger, action)
        habit = Habit.create(trigger, action, description, count)

        if self._fallback is not None:
            existing = self._fallback.get_habit(habit.id)
            if existing is not None:
                habit = existing.observed()
            self._fallback.store_habit(habit)
            return habit if count == self._min_observations else None  # only return on first crossing

        import aiosqlite
        now_ts = time.time()
        trigger_json = json.dumps(trigger)
        async with aiosqlite.connect(self._db_path) as db:
            # Check if already stored (returning None avoids duplicate events)
            async with db.execute("SELECT id FROM habits WHERE id=?", (habit.id,)) as cur:
                already_existed = await cur.fetchone() is not None

            await db.execute(
                """
                INSERT INTO habits
                    (id, description, trigger_json, action, confidence, observed_count, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    confidence=excluded.confidence,
                    observed_count=excluded.observed_count,
                    last_seen=excluded.last_seen
                """,
                (
                    habit.id, description, trigger_json, action,
                    habit.confidence, count, now_ts,
                ),
            )
            await db.commit()

        # Only return on first crossing (so caller only publishes event once)
        if already_existed:
            return None
        return habit


def _habit_matches(habit: Habit, ctx: LearningContext) -> bool:
    """
    Check whether all non-empty trigger fields in ``habit.trigger`` match ``ctx``.
    """
    t = habit.trigger
    if t.get("time_bucket") and t["time_bucket"] != ctx.time_bucket:
        return False
    if t.get("day_of_week") is not None and t["day_of_week"] != ctx.day_of_week:
        return False
    if t.get("active_app") and t["active_app"] != ctx.active_app.lower():
        return False
    if t.get("topic") and t["topic"] != ctx.topic:
        return False
    return True


def _describe_habit(trigger: dict[str, Any], action: str) -> str:
    """Build a human-readable habit description from trigger fields."""
    parts: list[str] = []
    if trigger.get("time_bucket"):
        parts.append(f"In the {trigger['time_bucket']}")
    if trigger.get("active_app"):
        parts.append(f"while using {trigger['active_app']}")
    intent = action.replace("intent:", "")
    parts.append(f"frequently does: {intent}")
    return ", ".join(parts).capitalize()
