"""
Workflow Learner
================
Detects and stores multi-step repeated skill/app sequences.

A workflow is an ordered sequence of N≥2 steps (skill names) that
the user executes repeatedly. When the same sequence is observed at
least ``min_observations`` times, it becomes a stored ``Workflow``.

Session Tracking
----------------
Steps are accumulated per session in an in-process buffer.
When a session ends (or the buffer window is flushed), all sub-sequences
of length ≥2 are extracted and their occurrence counts incremented.

Example
-------
Session steps: ["open_file", "edit_code", "git_commit", "open_browser"]
Sub-sequences extracted:
    ("open_file", "edit_code")
    ("open_file", "edit_code", "git_commit")
    ("edit_code", "git_commit")
    ("edit_code", "git_commit", "open_browser")
    ("git_commit", "open_browser")
    (etc.)

Persistence
-----------
    workflow_observations(
        steps_key    TEXT PRIMARY KEY,  -- stable hash of steps
        steps_json   TEXT NOT NULL,     -- JSON array
        count        INTEGER NOT NULL DEFAULT 1,
        last_seen    REAL NOT NULL
    )

    workflows(
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        steps_json   TEXT NOT NULL,
        trigger_count INTEGER NOT NULL,
        confidence   REAL NOT NULL,
        last_seen    REAL NOT NULL,
        suggested    INTEGER NOT NULL DEFAULT 0  -- boolean
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from spidy.learning.types import Workflow
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

_CREATE_OBSERVATIONS_SQL = """
CREATE TABLE IF NOT EXISTS workflow_observations (
    steps_key    TEXT PRIMARY KEY,
    steps_json   TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 1,
    last_seen    REAL NOT NULL
);
"""

_CREATE_WORKFLOWS_SQL = """
CREATE TABLE IF NOT EXISTS workflows (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    steps_json    TEXT NOT NULL,
    trigger_count INTEGER NOT NULL,
    confidence    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    suggested     INTEGER NOT NULL DEFAULT 0
);
"""


def _row_to_workflow(row: tuple) -> Workflow:
    wid, name, steps_json, trigger_count, confidence, last_seen_ts, suggested = row
    return Workflow(
        id=wid,
        name=name,
        steps=tuple(json.loads(steps_json)),
        trigger_count=trigger_count,
        confidence=confidence,
        last_seen=datetime.fromtimestamp(last_seen_ts, tz=timezone.utc),
        suggested=bool(suggested),
    )


def _steps_key(steps: list[str] | tuple[str, ...]) -> str:
    """Stable hash key for a step sequence."""
    from spidy.learning.types import _stable_id
    return _stable_id(list(steps))


def _extract_subsequences(
    steps: list[str], min_len: int = 2, max_len: int = 10
) -> list[tuple[str, ...]]:
    """Extract all contiguous sub-sequences of length [min_len, max_len]."""
    result: list[tuple[str, ...]] = []
    n = len(steps)
    for length in range(min_len, min(max_len, n) + 1):
        for start in range(n - length + 1):
            result.append(tuple(steps[start:start + length]))
    return result


class _InMemoryWorkflowFallback:
    """In-process fallback when aiosqlite is unavailable."""

    def __init__(self) -> None:
        self._observations: dict[str, tuple[list[str], int]] = {}
        self._workflows: dict[str, Workflow] = {}

    def increment(self, steps: list[str]) -> int:
        key = _steps_key(steps)
        _, count = self._observations.get(key, (steps, 0))
        new_count = count + 1
        self._observations[key] = (steps, new_count)
        return new_count

    def store(self, workflow: Workflow) -> None:
        self._workflows[workflow.id] = workflow

    def get(self, wid: str) -> Workflow | None:
        return self._workflows.get(wid)

    def list(self) -> list[Workflow]:
        return list(self._workflows.values())

    def count(self) -> int:
        return len(self._workflows)


class WorkflowLearner:
    """
    Detects and stores multi-step repeated skill/app sequences.

    Parameters
    ----------
    db_path:
        Path to ``spidy_learning.db``. Pass ``:memory:`` for tests.
    min_observations:
        Times a sequence must be seen before becoming a Workflow.
    max_steps:
        Maximum number of steps in a tracked sequence.
    bus:
        Optional EventBus for publishing workflow events.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        min_observations: int = 3,
        max_steps: int = 10,
        bus: "EventBus | None" = None,
    ) -> None:
        self._db_path = str(db_path)
        self._min_observations = min_observations
        self._max_steps = max_steps
        self._bus = bus
        self._aiosqlite_available = False
        self._fallback: _InMemoryWorkflowFallback | None = None
        self._initialized = False
        # Per-session step buffer: session_id → list of step names
        self._session_steps: dict[str, list[str]] = {}

    async def initialize(self) -> None:
        """Create schema. Must be called before any other method."""
        if self._initialized:
            return
        try:
            import aiosqlite  # noqa: F401
            self._aiosqlite_available = True
        except ImportError:
            log.warning(
                "WorkflowLearner: 'aiosqlite' not installed — "
                "workflows will not persist across restarts. "
                "Install with: pip install aiosqlite"
            )
            self._fallback = _InMemoryWorkflowFallback()
            self._initialized = True
            return

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_OBSERVATIONS_SQL)
            await db.execute(_CREATE_WORKFLOWS_SQL)
            await db.commit()

        self._initialized = True
        log.debug("WorkflowLearner: ready at '{path}'", path=self._db_path)

    # ── Core API ───────────────────────────────────────────────────────────

    async def record_step(self, session_id: str, skill_name: str) -> None:
        """
        Record a single skill/app step within a session.

        Call this from Brain.process() after each plan step is executed.
        Steps accumulate in a per-session buffer until ``flush_session()``
        is called or the session ends.
        """
        if session_id not in self._session_steps:
            self._session_steps[session_id] = []
        self._session_steps[session_id].append(skill_name)

        # Keep the buffer bounded
        if len(self._session_steps[session_id]) > self._max_steps * 2:
            self._session_steps[session_id] = self._session_steps[session_id][-self._max_steps:]

    async def flush_session(self, session_id: str) -> list[Workflow]:
        """
        Process all steps recorded in a session and detect new workflows.

        Should be called when the Brain session ends.
        Returns any newly promoted Workflows.
        """
        steps = self._session_steps.pop(session_id, [])
        if len(steps) < 2:
            return []

        new_workflows: list[Workflow] = []
        subsequences = _extract_subsequences(steps, 2, self._max_steps)

        for seq in subsequences:
            count = await self._increment_observation(list(seq))
            if count >= self._min_observations:
                workflow = await self._promote_workflow(list(seq), count)
                if workflow is not None:
                    new_workflows.append(workflow)
                    log.info(
                        "WorkflowLearner: workflow detected — '{name}' ({count}x)",
                        name=workflow.name,
                        count=workflow.trigger_count,
                    )
                    if self._bus is not None:
                        from spidy.learning.events import LearningWorkflowDetectedEvent
                        await self._bus.publish(LearningWorkflowDetectedEvent(
                            workflow_id=workflow.id,
                            name=workflow.name,
                            step_count=len(workflow.steps),
                            trigger_count=workflow.trigger_count,
                        ))

        return new_workflows

    async def suggest_next_step(self, current_steps: list[str]) -> str | None:
        """
        Predict the most likely next step given the current step sequence.

        Searches all stored workflows for a prefix match and returns
        the step that follows the prefix in the most confident workflow.

        Returns None if no prediction can be made.
        """
        workflows = await self.list_workflows()
        best: tuple[float, str] | None = None

        for wf in workflows:
            wf_steps = list(wf.steps)
            n = len(current_steps)
            if len(wf_steps) > n and list(wf_steps[:n]) == current_steps:
                next_step = wf_steps[n]
                if best is None or wf.confidence > best[0]:
                    best = (wf.confidence, next_step)

        return best[1] if best else None

    async def list_workflows(self) -> list[Workflow]:
        """Return all detected workflows sorted by confidence descending."""
        if self._fallback is not None:
            wfs = self._fallback.list()
            return sorted(wfs, key=lambda w: w.confidence, reverse=True)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, name, steps_json, trigger_count, confidence, last_seen, suggested "
                "FROM workflows ORDER BY confidence DESC"
            ) as cursor:
                rows = await cursor.fetchall()

        return [_row_to_workflow(r) for r in rows]

    async def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Retrieve a specific workflow by ID."""
        if self._fallback is not None:
            return self._fallback.get(workflow_id)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, name, steps_json, trigger_count, confidence, last_seen, suggested "
                "FROM workflows WHERE id=?",
                (workflow_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return _row_to_workflow(row)

    async def count(self) -> int:
        """Return the number of detected workflows."""
        if self._fallback is not None:
            return self._fallback.count()

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM workflows") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Private helpers ────────────────────────────────────────────────────

    async def _increment_observation(self, steps: list[str]) -> int:
        """Increment the observation count for a step sequence."""
        import time

        if self._fallback is not None:
            return self._fallback.increment(steps)

        key = _steps_key(steps)
        now_ts = time.time()
        steps_json = json.dumps(steps)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO workflow_observations (steps_key, steps_json, count, last_seen)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(steps_key) DO UPDATE SET
                    count = count + 1,
                    last_seen = excluded.last_seen
                """,
                (key, steps_json, now_ts),
            )
            await db.commit()
            async with db.execute(
                "SELECT count FROM workflow_observations WHERE steps_key=?", (key,)
            ) as cursor:
                row = await cursor.fetchone()

        return row[0] if row else 1

    async def _promote_workflow(self, steps: list[str], count: int) -> Workflow | None:
        """Store or update a workflow; return it only on first promotion."""
        import time

        workflow = Workflow.create(steps, trigger_count=count)

        if self._fallback is not None:
            existing = self._fallback.get(workflow.id)
            if existing is not None:
                return None  # Already promoted
            self._fallback.store(workflow)
            return workflow

        import aiosqlite
        now_ts = time.time()
        steps_json = json.dumps(steps)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id FROM workflows WHERE id=?", (workflow.id,)
            ) as cur:
                already_existed = await cur.fetchone() is not None

            await db.execute(
                """
                INSERT INTO workflows
                    (id, name, steps_json, trigger_count, confidence, last_seen, suggested)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    trigger_count=excluded.trigger_count,
                    confidence=excluded.confidence,
                    last_seen=excluded.last_seen
                """,
                (workflow.id, workflow.name, steps_json, count, workflow.confidence, now_ts),
            )
            await db.commit()

        if already_existed:
            return None
        return workflow
