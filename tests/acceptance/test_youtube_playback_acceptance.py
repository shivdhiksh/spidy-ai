"""
Acceptance test for YouTube playback end-to-end task.
Verifies decomposition, structured routing, video playback, and GoalVerifier validation.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from spidy.agent.agent import AutonomousAgent
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.types import TaskState
from spidy.core.event_bus import EventBus
from spidy.skills.browser.browser_skill import BrowserSkill
from spidy.skills.registry import SkillRegistry


@pytest.mark.asyncio
async def test_youtube_playback_decomposition():
    """Verify decomposer produces 3 structured tasks without invoking an LLM."""
    decomposer = TaskDecomposer()
    goal = "Open Chrome and search YouTube Python videos. Play any one video random which you like."

    tasks = await decomposer.decompose(goal, goal_id="goal_yt_1")

    assert len(tasks) == 3

    # Task 1: Open Chrome
    t1 = tasks[0]
    assert "open chrome" in t1.description.lower()
    assert t1.action["action"] == "open_app"
    assert t1.action["target"] == "Chrome"

    # Task 2: Search YouTube
    t2 = tasks[1]
    assert "search" in t2.description.lower() and "youtube" in t2.description.lower()
    assert t2.action["skill"] == "browser"
    assert t2.action["action"] == "search"
    assert t2.action["target"] == "youtube"
    assert "python" in t2.action["query"].lower()

    # Task 3: Play YouTube video
    t3 = tasks[2]
    assert "play" in t3.description.lower() and "video" in t3.description.lower()
    assert t3.action["skill"] == "browser"
    assert t3.action["action"] == "play_youtube_video"
    assert t3.action["random_pick"] is True
    assert t3.terminal is True


@pytest.mark.asyncio
async def test_youtube_playback_execution_loop_and_verification():
    """Verify execution loop executes tasks, collects evidence, and GoalVerifier strictly confirms."""
    registry = SkillRegistry()

    # Mock BrowserAgent
    browser_agent = MagicMock()
    browser_agent.is_running = True
    browser_agent.open_browser = AsyncMock(return_value=True)
    browser_agent.search_youtube = AsyncMock(return_value={
        "url": "https://www.youtube.com/results?search_query=Python+videos",
        "title": "Python videos - YouTube",
    })
    browser_agent.play_youtube_video = AsyncMock(return_value={
        "status": "playing",
        "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
        "title": "Python Full Course for Beginners [2026]",
        "video_id": "kqtD5dpn9C8",
        "player_state": 1,
        "is_playing": True,
        "is_paused": False,
        "current_time": 3.2,
        "duration": 3600.0,
        "playback_verified": True,
    })

    browser_skill = BrowserSkill(browser_agent=browser_agent)
    registry.register(browser_skill)

    # Mock Brain
    brain = MagicMock()
    brain._skills = registry
    brain._registry = registry
    brain.process = AsyncMock(return_value="Action executed successfully.")

    bus = EventBus()
    router = StructuredRouter(brain=brain, skill_registry=registry)
    agent = AutonomousAgent(
        brain=brain,
        bus=bus,
        router=router,
        verify_enabled=True,
    )

    goal = "Open Chrome and search YouTube Python videos. Play any one video random which you like."
    summary = await agent.run_goal(goal, session_id="test_session")

    # Verify agent marked completion and verified playback
    assert "completed successfully" in summary.lower() or "achieved" in summary.lower() or "playing" in summary.lower()

    # Check goal record in goal manager history
    history = agent._goal_manager.get_history()
    assert len(history) >= 1
    goal_record = history[-1]

    from spidy.agent.types import GoalState
    assert goal_record.state == GoalState.COMPLETED
    assert len(goal_record.tasks) == 3
    assert all(t.state == TaskState.COMPLETED for t in goal_record.tasks)

    # Verify task 3 evidence
    t3 = goal_record.tasks[2]
    assert t3.evidence is not None
    assert t3.evidence.get("playback_verified") is True
    assert t3.evidence.get("is_playing") is True
    assert "watch?v=" in t3.evidence.get("url", "")
