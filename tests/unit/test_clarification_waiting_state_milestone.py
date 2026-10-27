"""
Unit Tests: Clarification & WAITING_FOR_USER Lifecycle State
============================================================
Verifies the explicit WAITING_FOR_USER state machine:
- Goal enters WAITING_FOR_USER when dialog.ask_user is encountered
- ExecutionLoop pauses immediately with NO retries and NO LLM calls
- User can provide answer to resume the goal from the exact paused point
- Variable binding propagates user answer to subsequent tasks
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from spidy.core.event_bus import EventBus
from spidy.agent.types import (
    GoalRecord,
    GoalState,
    TaskRecord,
    TaskState,
    ExecutionContext,
)
from spidy.agent.goal_manager import GoalManager
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.agent import AutonomousAgent
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.structured_router import StructuredRouter


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def goal_manager(bus: EventBus) -> GoalManager:
    return GoalManager(bus=bus)


class TestWaitingForUserState:
    """Test WAITING_FOR_USER transitions and properties."""

    @pytest.mark.asyncio
    async def test_goal_manager_wait_for_user_and_resume(self, goal_manager: GoalManager):
        """GoalManager transitions to WAITING_FOR_USER and resumes back to EXECUTING."""
        goal = await goal_manager.create_goal("Test Goal")
        goal = await goal_manager.begin_planning(goal.goal_id)
        task1 = TaskRecord(description="Step 1", utterance="do step 1")
        goal = await goal_manager.begin_execution(goal.goal_id, [task1])
        assert goal.state == GoalState.EXECUTING
        assert goal.is_active

        # Transition to WAITING_FOR_USER
        waiting_goal = await goal_manager.wait_for_user(goal.goal_id, prompt="Which file?")
        assert waiting_goal.state == GoalState.WAITING_FOR_USER
        assert waiting_goal.is_active
        assert not waiting_goal.is_terminal

        # Resume to EXECUTING
        resumed_goal = await goal_manager.resume_goal(goal.goal_id)
        assert resumed_goal.state == GoalState.EXECUTING
        assert resumed_goal.is_active

    @pytest.mark.asyncio
    async def test_execution_loop_pauses_on_clarification_without_retry(self, goal_manager: GoalManager, bus: EventBus):
        """ExecutionLoop must pause on ask_user with 0 retries and return WAITING_FOR_USER."""
        brain_mock = MagicMock()
        brain_mock.process = AsyncMock(return_value="unexpected brain call")

        loop = ExecutionLoop(
            goal_manager=goal_manager,
            brain=brain_mock,
            bus=bus,
        )

        goal = await goal_manager.create_goal("Search something")
        task = TaskRecord(
            task_id="t-clarify",
            goal_id=goal.goal_id,
            description="Clarify search query",
            utterance="What would you like me to search for?",
            action={"skill": "dialog", "action": "ask_user", "prompt": "What would you like to search for?"},
            state=TaskState.PENDING,
        )
        goal = await goal_manager.begin_execution(goal.goal_id, [task])
        ctx = ExecutionContext(goal=goal, session_id="test-session")

        final_goal, obs = await loop.run(goal=goal, tasks=[task], ctx=ctx)

        assert final_goal.state == GoalState.WAITING_FOR_USER
        assert ctx.waiting_task_id == "t-clarify"
        assert "What would you like to search for?" in ctx.waiting_prompt
        # Brain.process was NOT called
        assert brain_mock.process.call_count == 0

    @pytest.mark.asyncio
    async def test_autonomous_agent_resume_with_user_input(self, goal_manager: GoalManager, bus: EventBus):
        """AutonomousAgent resumes paused goal with user input and completes next step."""
        brain_mock = MagicMock()
        brain_mock.session_id = "test-session"
        brain_mock.process = AsyncMock(return_value="Found results for Python")

        decomposer_mock = MagicMock()
        t1 = TaskRecord(
            task_id="t1",
            description="Clarify query",
            utterance="What would you like to search for?",
            action={"skill": "dialog", "action": "ask_user", "prompt": "What would you like to search for?"},
            terminal=False,
        )
        t2 = TaskRecord(
            task_id="t2",
            description="Search query",
            utterance="search for ${t1.result}",
            action={"skill": "browser", "action": "search", "query": "${t1.result}"},
            terminal=True,
        )
        decomposer_mock.decompose = AsyncMock(return_value=[t1, t2])

        router_mock = MagicMock()
        router_mock.execute = AsyncMock(return_value=("Search results for Python are visible", True))

        agent = AutonomousAgent(
            brain=brain_mock,
            goal_manager=goal_manager,
            decomposer=decomposer_mock,
            bus=bus,
            router=router_mock,
        )

        # 1. Run goal -> pauses in WAITING_FOR_USER
        resp1 = await agent.run_goal("search something")
        assert "What would you like to search for?" in resp1
        assert agent.has_waiting_goal()

        # 2. User provides answer: "Python tutorials"
        resp2 = await agent.resume_with_user_input("Python tutorials")
        assert not agent.has_waiting_goal()
        assert agent.active_goal is None or agent.active_goal.state == GoalState.COMPLETED
        # router executed search for Python tutorials
        assert router_mock.execute.call_count == 1
        call_action = router_mock.execute.call_args[0][0]
        assert call_action["query"] == "Python tutorials"
