"""
Learning Manager — Unified Learning Engine Entry Point
=======================================================
``LearningManager`` is the single entry point for all learning operations.
It implements the ``LearningInterface`` contract defined in M3
(``spidy.brain.interfaces.LearningInterface``) and coordinates four
sub-components:

    PreferenceLearner  — key-value preferences with confidence
    HabitDetector      — recurring trigger-action patterns
    WorkflowLearner    — multi-step skill sequences
    FeedbackProcessor  — explicit and implicit feedback integration

Architecture
------------
    LearningManager  (LearningInterface)
        ├── PreferenceLearner
        ├── HabitDetector
        ├── WorkflowLearner
        └── FeedbackProcessor

All four components share the same SQLite database file
(``spidy_learning.db``). Each uses separate tables so they don't
interfere with each other.

No memory duplication
----------------------
The Learning Engine does NOT write to the Memory Engine's database.
It reads interaction signals from LearningManager.observe_context()
(called by the Brain on every turn) rather than from MemoryManager.

This keeps Memory and Learning orthogonal: Memory is the episodic log;
Learning is the behaviour inference layer built on top of it.

Privacy
-------
- All data is stored locally on the user's device only.
- No telemetry, no network calls, no model fine-tuning.
- Data retention is configurable (default: 365 days).

EventBus
--------
LearningManager uses the EventBus to publish learning.* events but
never subscribes — it is driven by Brain calls, not by events.

Brain Integration
-----------------
The Brain calls:
    1. observe_context() — on every turn (feeds HabitDetector + PreferenceLearner)
    2. get_preference() — before plan execution (personalise LLM context)
    3. get_habit() — after execution (check for habit suggestions)
    4. record_feedback() — when explicit feedback is given
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from spidy.brain.interfaces import LearningInterface
from spidy.learning.feedback import FeedbackProcessor
from spidy.learning.habits import HabitDetector
from spidy.learning.preferences import PreferenceLearner
from spidy.learning.types import FeedbackSignal, Habit, Preference, Workflow
from spidy.learning.workflows import WorkflowLearner
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.config.manager import LearningConfig
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class LearningManager(LearningInterface):
    """
    Unified learning engine that coordinates all learning sub-components.

    Parameters
    ----------
    config:
        ``LearningConfig`` from ``SpidyConfig.learning``. Optional;
        uses sensible defaults when None.
    bus:
        Optional EventBus for publishing ``learning.*`` events.
    learning_dir:
        Directory for the ``spidy_learning.db`` SQLite file.
        Defaults to ``:memory:`` when None (useful for tests).
    """

    def __init__(
        self,
        config: "LearningConfig | None" = None,
        bus: "EventBus | None" = None,
        learning_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._learning_dir = learning_dir

        # Resolve settings from config (with defaults)
        db_filename = "spidy_learning.db"
        min_obs_habit = 3
        min_obs_workflow = 3
        max_workflow_steps = 10
        feedback_boost = 0.15
        feedback_decay = 0.10

        if config is not None:
            db_filename = getattr(config, "db_filename", db_filename)
            min_obs_habit = getattr(config, "min_observations_for_habit", min_obs_habit)
            min_obs_workflow = getattr(config, "min_observations_for_workflow", min_obs_workflow)
            max_workflow_steps = getattr(config, "max_workflow_steps", max_workflow_steps)
            feedback_boost = getattr(config, "feedback_boost", feedback_boost)
            feedback_decay = getattr(config, "feedback_decay", feedback_decay)

        # Resolve DB path
        if learning_dir is not None:
            db_path = learning_dir / db_filename
        else:
            db_path = ":memory:"  # type: ignore[assignment]

        # Instantiate sub-components (all share the same DB file)
        self._preferences = PreferenceLearner(db_path=db_path)
        self._habits = HabitDetector(
            db_path=db_path,
            min_observations=min_obs_habit,
            bus=bus,
        )
        self._workflows = WorkflowLearner(
            db_path=db_path,
            min_observations=min_obs_workflow,
            max_steps=max_workflow_steps,
            bus=bus,
        )
        self._feedback = FeedbackProcessor(
            db_path=db_path,
            feedback_boost=feedback_boost,
            feedback_decay=feedback_decay,
            bus=bus,
            preferences=self._preferences,
        )
        self._initialized = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialise all Learning sub-components.

        Must be called once before any learning operations.
        Called by SpidyCore during application startup.
        """
        if self._initialized:
            return

        if self._learning_dir is not None:
            self._learning_dir.mkdir(parents=True, exist_ok=True)

        await self._preferences.initialize()
        await self._habits.initialize()
        await self._workflows.initialize()
        await self._feedback.initialize()

        self._initialized = True
        log.info("LearningManager: ready.")

    async def close(self) -> None:
        """Graceful shutdown. No-op for now (SQLite connections are per-operation)."""
        log.debug("LearningManager: shutdown.")

    # ── LearningInterface Implementation ───────────────────────────────────

    async def record_feedback(
        self,
        session_id: str,
        utterance: str,
        response: str,
        rating: float,
        tags: list[str] | None = None,
    ) -> None:
        """
        Record user feedback on a Brain response.

        This is the primary feedback injection point called by the Brain
        when explicit feedback is available (e.g. "that's wrong",
        "perfect", thumbs up/down).

        Implicit signals can also be passed here with rating=0.0 and
        appropriate tags for aggregate analysis.
        """
        try:
            signal = FeedbackSignal.create(
                session_id=session_id,
                utterance=utterance,
                response=response,
                rating=rating,
                tags=tags,
            )
            await self._feedback.process(signal)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "LearningManager.record_feedback failed (non-fatal): {exc}", exc=exc
            )
            await self._publish_error("FeedbackProcessor", str(exc))

    async def get_preference(
        self,
        category: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a learned user preference value.

        Parameters
        ----------
        category:
            Preference key, e.g. ``"preferred_browser"``, ``"preferred_llm"``.

        Returns
        -------
        Any
            The preference value, or ``default`` if not established.
        """
        try:
            return await self._preferences.get(category, default)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "LearningManager.get_preference failed (non-fatal): {exc}", exc=exc
            )
            return default

    async def get_habit(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Check if a learned habit matches the current context.

        Parameters
        ----------
        context:
            Current signals: ``{hour_of_day, day_of_week, active_app, topic, ...}``

        Returns
        -------
        dict | None
            ``{action, confidence, description}`` if a habit matches, else None.
        """
        try:
            habit = await self._habits.get_matching_habit(context)
            if habit is None:
                return None
            return {
                "action": habit.action,
                "confidence": habit.confidence,
                "description": habit.description,
            }
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "LearningManager.get_habit failed (non-fatal): {exc}", exc=exc
            )
            return None

    # ── Extended API ───────────────────────────────────────────────────────

    async def observe_context(
        self,
        context: dict[str, Any],
        utterance: str,
        intent: str,
    ) -> None:
        """
        Feed the current context into the learning sub-components.

        Called by Brain.process() on every interaction turn (before decide).
        Updates:
        - HabitDetector: records one observation
        - PreferenceLearner: infers app preference from ``context["active_app"]``

        Never raises — all errors are caught and logged.
        """
        try:
            await self._habits.observe(context, utterance, intent)
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: habit observe failed (non-fatal): {exc}", exc=exc)

        try:
            active_app = context.get("active_app", "")
            if active_app:
                await self._preferences.observe_app(active_app)
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: app observe failed (non-fatal): {exc}", exc=exc)

    async def record_step(self, session_id: str, skill_name: str) -> None:
        """
        Record a skill step for workflow detection.

        Called by Brain.process() for each plan step that executes.
        """
        try:
            await self._workflows.record_step(session_id, skill_name)
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: record_step failed (non-fatal): {exc}", exc=exc)

    async def flush_session(self, session_id: str) -> list[Workflow]:
        """
        Finalise workflow detection for a completed session.

        Call when the Brain session ends. Returns any newly detected Workflows.
        """
        try:
            return await self._workflows.flush_session(session_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: flush_session failed (non-fatal): {exc}", exc=exc)
            return []

    async def set_preference(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> Preference:
        """
        Explicitly set a user preference.

        Explicit preferences (confidence=1.0, source="explicit") always
        override implicit inferences.
        """
        return await self._preferences.set(
            key, value, confidence, source  # type: ignore[arg-type]
        )

    async def suggest_workflow(self, current_steps: list[str]) -> str | None:
        """
        Predict the next step in a known workflow given current steps.

        Returns the predicted next step name, or None if no prediction.
        """
        try:
            return await self._workflows.suggest_next_step(current_steps)
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: suggest_workflow failed (non-fatal): {exc}", exc=exc)
            return None

    async def get_preference_context(self) -> str:
        """
        Return a compact natural-language summary of user preferences
        suitable for injection into the LLM system prompt.

        Example:
            "User preferences: preferred_browser=Chrome (0.9), preferred_ide=VS Code (0.8)"

        Returns an empty string if no preferences are established.
        """
        try:
            return await self._preferences.to_context_string()
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: get_preference_context failed: {exc}", exc=exc)
            return ""

    async def list_preferences(self) -> list[Preference]:
        """Return all stored preferences sorted by confidence."""
        try:
            return await self._preferences.all_preferences()
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: list_preferences failed: {exc}", exc=exc)
            return []

    async def list_habits(self) -> list[Habit]:
        """Return all detected habits."""
        try:
            return await self._habits.list_habits()
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: list_habits failed: {exc}", exc=exc)
            return []

    async def list_workflows(self) -> list[Workflow]:
        """Return all detected workflows."""
        try:
            return await self._workflows.list_workflows()
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: list_workflows failed: {exc}", exc=exc)
            return []

    async def get_feedback_sentiment(self, tag: str) -> float:
        """Return the average feedback rating for signals with a specific tag."""
        try:
            return await self._feedback.sentiment_by_tag(tag)
        except Exception as exc:  # noqa: BLE001
            log.debug("LearningManager: get_feedback_sentiment failed: {exc}", exc=exc)
            return 0.0

    # ── Private helpers ────────────────────────────────────────────────────

    async def _publish_error(self, component: str, message: str) -> None:
        """Publish a LearningErrorEvent (best-effort, never raises)."""
        if self._bus is None:
            return
        try:
            from spidy.learning.events import LearningErrorEvent
            await self._bus.publish(LearningErrorEvent(
                component=component,
                error_message=message,
            ))
        except Exception:  # noqa: BLE001
            pass
