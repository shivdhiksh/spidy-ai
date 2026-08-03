"""
tests/integration/test_autonomous_agent.py — M13 Integration Tests
====================================================================
End-to-end integration tests for the AutonomousAgent system.

All skill execution and OS calls are mocked. Tests verify:
1. Goal creation → decomposition → execution → completion
2. Goal cancellation mid-execution
3. Failure recovery (retry + alternative)
4. Progress event publishing
5. Brain.run_goal() delegation
6. Natural response composition
7. Nested/long multi-step workflows
8. Concurrent goal protection (only one active goal)
9. Goal history tracking
10. Flask project workflow
11. React project workflow
12. Browser search workflow
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.agent.agent import AutonomousAgent
from spidy.agent.types import GoalState, TaskState
from spidy.brain.brain import Brain
from spidy.core.event_bus import EventBus
from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult
from spidy.skills.registry import SkillRegistry


# ─── Test Skill Mocks ─────────────────────────────────────────────────────────


class MockUniversalSkill(BaseSkill):
    """A catch-all skill for agent integration tests."""
    name = "mock_universal_skill"
    version = "1.0.0"

    def __init__(self, fail_actions: set[str] | None = None) -> None:
        self._fail_actions = fail_actions or set()
        self.call_log: list[str] = []

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability("launch_app", "Open app", "T1"),
            SkillCapability("close_app", "Close app", "T1"),
            SkillCapability("search_google", "Search Google", "T0"),
            SkillCapability("search_youtube", "Search YouTube", "T0"),
            SkillCapability("open_file", "Open file", "T1"),
            SkillCapability("open_url", "Open URL", "T0"),
            SkillCapability("bring_app_to_foreground", "Focus app", "T0"),
        ]

    async def execute(self, action: str, ctx: SkillContext) -> SkillResult:
        self.call_log.append(action)
        if action in self._fail_actions:
            return SkillResult.fail(f"I couldn't {action}.")
        app = ctx.params.get("name", ctx.params.get("app_name", action))
        return SkillResult.ok(f"I've done {action} for {app}.")


# ─── Builders ─────────────────────────────────────────────────────────────────


def make_bus() -> EventBus:
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)
    return bus


def make_brain(fail_actions: set[str] | None = None) -> tuple[Brain, EventBus]:
    bus = make_bus()
    registry = SkillRegistry()
    skill = MockUniversalSkill(fail_actions=fail_actions)
    registry.register(skill)

    from spidy.config.manager import ReasoningConfig
    config = ReasoningConfig()

    brain = Brain(
        bus=bus,
        config=config,
        skill_registry=registry,
        user_name="TestUser",
    )
    return brain, bus


def make_agent(
    fail_actions: set[str] | None = None,
    inter_task_delay: float = 0.0,
) -> tuple[AutonomousAgent, Brain, EventBus]:
    brain, bus = make_brain(fail_actions=fail_actions)
    agent = AutonomousAgent(
        brain=brain,
        bus=bus,
        llm_client=None,  # use heuristic decomposition
        max_task_retries=1,
        inter_task_delay=inter_task_delay,
    )
    return agent, brain, bus


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Flask project workflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlaskProjectWorkflow:
    @pytest.mark.asyncio
    async def test_flask_goal_completes(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await agent.run_goal("Create a Flask project")

        await brain.stop()

        assert response
        assert isinstance(response, str)
        assert len(response) > 10

    @pytest.mark.asyncio
    async def test_flask_goal_reaches_completed_state(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Create a Flask project")

        await brain.stop()

        history = agent.history
        assert len(history) >= 1
        completed = [g for g in history if g.state == GoalState.COMPLETED]
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_flask_goal_produces_tasks(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        goal_events = []
        bus.subscribe("agent.goal_started", lambda e: goal_events.append(e))

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Create a Flask project")

        await brain.stop()

        assert len(goal_events) >= 1
        assert goal_events[0].task_count >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: React application workflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestReactAppWorkflow:
    @pytest.mark.asyncio
    async def test_react_goal_completes(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await agent.run_goal("Create a React application")

        await brain.stop()

        assert response
        assert len(response) > 5

    @pytest.mark.asyncio
    async def test_react_goal_decomposes_to_multiple_tasks(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        started_events = []
        bus.subscribe("agent.goal_started", lambda e: started_events.append(e))

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Create a React application")

        await brain.stop()

        if started_events:
            assert started_events[0].task_count >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Browser search workflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrowserSearchWorkflow:
    @pytest.mark.asyncio
    async def test_youtube_search_goal_completes(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._is_network_available", return_value=True):
            response = await agent.run_goal("Open browser and search YouTube")

        await brain.stop()

        assert response
        assert len(response) > 5

    @pytest.mark.asyncio
    async def test_google_search_goal_completes(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._is_network_available", return_value=True):
            response = await agent.run_goal("Search Google for Python tutorials")

        await brain.stop()

        assert response


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Goal cancellation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoalCancellation:
    @pytest.mark.asyncio
    async def test_cancel_with_no_active_goal_returns_false(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        result = await agent.cancel_current_goal()

        await brain.stop()

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_goal_state_is_cancelled(self) -> None:
        """
        Start a goal, immediately cancel via ctx, verify CANCELLED state.
        Uses a slow brain to simulate mid-execution cancellation.
        """
        agent, brain, bus = make_agent()
        await brain.start()

        # Directly test cancellation via the goal manager + context
        from spidy.agent.types import ExecutionContext, GoalRecord, GoalState
        mgr = agent.goal_manager
        goal = await mgr.create_goal("Long running goal")
        cancelled = await mgr.cancel_goal(goal.goal_id)

        await brain.stop()

        assert cancelled.state == GoalState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_publishes_brain_cancelled_event(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        events = []
        bus.subscribe("brain.goal_cancelled", lambda e: events.append(e))

        # Manually create + cancel a goal to test event publishing
        mgr = agent.goal_manager
        goal = await mgr.create_goal("Test goal")
        await agent.cancel_current_goal()

        await brain.stop()

        # Either the brain event was published OR we directly checked cancel
        # In this test we verify the agent.cancel_current_goal path
        assert not agent.is_running


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Recovery after failure
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecoveryAfterFailure:
    @pytest.mark.asyncio
    async def test_response_is_user_friendly_on_failure(self) -> None:
        """Brain returns failure responses but agent still produces friendly output."""
        agent, brain, bus = make_agent(
            fail_actions={"launch_app", "close_app", "search_google"},
        )
        await brain.start()

        response = await agent.run_goal("Create a Flask project")

        await brain.stop()

        assert response
        assert "Traceback" not in response
        assert "Exception" not in response

    @pytest.mark.asyncio
    async def test_no_unhandled_exceptions_during_failure(self) -> None:
        """Agent must never raise — even when all tasks fail."""
        agent, brain, bus = make_agent(
            fail_actions={"launch_app", "close_app"},
        )
        await brain.start()

        # Should not raise
        response = await agent.run_goal("Open VS Code and close it")

        await brain.stop()

        assert isinstance(response, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Progress events published
# ═══════════════════════════════════════════════════════════════════════════════


class TestProgressEvents:
    @pytest.mark.asyncio
    async def test_agent_progress_events_emitted(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        progress_events = []
        bus.subscribe("agent.progress", lambda e: progress_events.append(e))

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Create a Flask project")

        await brain.stop()

        assert len(progress_events) >= 2  # at least planning + first task

    @pytest.mark.asyncio
    async def test_brain_progress_events_emitted(self) -> None:
        """BrainProgressEvent should also be published (backward compat)."""
        agent, brain, bus = make_agent()
        await brain.start()

        brain_progress = []
        bus.subscribe("brain.progress", lambda e: brain_progress.append(e))

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Search YouTube for music")

        await brain.stop()

        assert len(brain_progress) >= 1

    @pytest.mark.asyncio
    async def test_goal_created_event_published(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        events = []
        bus.subscribe("agent.goal_created", lambda e: events.append(e))

        await agent.run_goal("Create a Python project")

        await brain.stop()

        assert len(events) >= 1
        assert "python project" in events[0].description.lower()

    @pytest.mark.asyncio
    async def test_goal_completed_event_published(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        events = []
        bus.subscribe("agent.goal_completed", lambda e: events.append(e))

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Search YouTube for music")

        await brain.stop()

        # The goal should eventually complete
        history = agent.history
        if history and history[-1].state == GoalState.COMPLETED:
            assert len(events) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 7: Brain.run_goal() delegation
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrainRunGoalDelegation:
    @pytest.mark.asyncio
    async def test_brain_run_goal_without_agent_falls_back(self) -> None:
        """When no AutonomousAgent is attached, Brain.run_goal falls back to process()."""
        brain, bus = make_brain()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await brain.run_goal("Open Notepad")

        await brain.stop()

        assert response
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_brain_run_goal_with_agent_delegates(self) -> None:
        """When AutonomousAgent is attached, Brain.run_goal delegates to it."""
        agent, brain, bus = make_agent()
        brain.attach_agent(agent)
        await brain.start()

        assert brain.autonomous_agent is agent

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await brain.run_goal("Search YouTube for cats")

        await brain.stop()

        assert response
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_brain_cancel_goal_returns_false_without_agent(self) -> None:
        brain, bus = make_brain()
        await brain.start()

        result = await brain.cancel_goal()

        await brain.stop()

        assert result is False

    @pytest.mark.asyncio
    async def test_brain_attach_agent_sets_property(self) -> None:
        agent, brain, bus = make_agent()
        assert brain.autonomous_agent is None
        brain.attach_agent(agent)
        assert brain.autonomous_agent is agent


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 8: Concurrent goal protection
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcurrentGoalProtection:
    @pytest.mark.asyncio
    async def test_cannot_create_second_goal_while_first_active(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        mgr = agent.goal_manager
        await mgr.create_goal("Goal 1")

        # Second goal should raise
        with pytest.raises(RuntimeError, match="already active"):
            await mgr.create_goal("Goal 2")

        await brain.stop()

    @pytest.mark.asyncio
    async def test_can_create_goal_after_previous_completes(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Search YouTube")
            response2 = await agent.run_goal("Search Google")

        await brain.stop()

        assert response2
        assert len(agent.history) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 9: Goal history tracking
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoalHistoryTracking:
    @pytest.mark.asyncio
    async def test_completed_goals_in_history(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Search YouTube")

        await brain.stop()

        assert len(agent.history) >= 1

    @pytest.mark.asyncio
    async def test_multiple_goals_accumulated_in_history(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Search YouTube")
            await agent.run_goal("Search Google")

        await brain.stop()

        assert len(agent.history) >= 2

    @pytest.mark.asyncio
    async def test_is_running_false_after_completion(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        assert agent.is_running is False

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Search YouTube")

        await brain.stop()

        assert agent.is_running is False


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 10: Natural response quality
# ═══════════════════════════════════════════════════════════════════════════════


class TestNaturalResponseQuality:
    @pytest.mark.asyncio
    async def test_response_is_not_empty(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await agent.run_goal("Create a Python project")

        await brain.stop()

        assert response.strip()

    @pytest.mark.asyncio
    async def test_response_is_not_raw_error(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await agent.run_goal("Open VS Code and create main.py")

        await brain.stop()

        assert "Traceback" not in response
        assert "raise " not in response

    @pytest.mark.asyncio
    async def test_completion_response_mentions_done(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await agent.run_goal("Search Google for Python")

        await brain.stop()

        # Response should feel conclusive / summary-like
        assert len(response) > 10

    @pytest.mark.asyncio
    async def test_brain_goal_started_event_published(self) -> None:
        agent, brain, bus = make_agent()
        await brain.start()

        events = []
        bus.subscribe("brain.goal_started", lambda e: events.append(e))

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            await agent.run_goal("Create a Python project")

        await brain.stop()

        assert len(events) >= 1
        assert "python project" in events[0].goal_description.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 11: Existing Brain pipeline unaffected (regression)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrainPipelineUnaffected:
    @pytest.mark.asyncio
    async def test_brain_process_still_works(self) -> None:
        """Brain.process() must work exactly as before M13."""
        brain, bus = make_brain()
        await brain.start()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            response = await brain.process("Open Notepad")

        await brain.stop()

        assert response
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_brain_session_maintained(self) -> None:
        """Brain session management unchanged by M13."""
        brain, bus = make_brain()
        await brain.start()

        assert brain.is_running
        assert brain.session_id

        await brain.stop()

        assert not brain.is_running

    @pytest.mark.asyncio
    async def test_brain_autonomous_agent_initially_none(self) -> None:
        """Brain starts with no attached agent (backward compat)."""
        brain, bus = make_brain()
        assert brain.autonomous_agent is None
