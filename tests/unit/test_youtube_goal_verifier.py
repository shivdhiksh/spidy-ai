"""
Unit tests for GoalVerifier strict YouTube playback validation.
Ensures the verifier rejects search-result-only states and unverified/paused states.
"""

import pytest

from spidy.agent.types import GoalRecord, TaskRecord, TaskState
from spidy.agent.verifier import GoalVerifier


def make_goal_record(goal_desc: str, tasks: list[TaskRecord]) -> GoalRecord:
    return GoalRecord(
        goal_id="g1",
        description=goal_desc,
        tasks=tasks,
    )


class TestYouTubeGoalVerifier:
    @pytest.mark.asyncio
    async def test_rejects_search_results_only_state(self):
        """Verifier MUST reject if page is still on YouTube search results without playback."""
        verifier = GoalVerifier()
        goal = make_goal_record(
            "Open Chrome and search YouTube Python videos. Play any one video random which you like.",
            [
                TaskRecord(
                    task_id="t1",
                    goal_id="g1",
                    description="Open Chrome",
                    utterance="open chrome",
                    state=TaskState.COMPLETED,
                    result_message="Chrome opened.",
                ),
                TaskRecord(
                    task_id="t2",
                    goal_id="g1",
                    description="Search YouTube for Python videos",
                    utterance="search youtube for Python videos",
                    state=TaskState.COMPLETED,
                    result_message="Navigated to https://www.youtube.com/results?search_query=Python+videos",
                    evidence={
                        "url": "https://www.youtube.com/results?search_query=Python+videos",
                        "title": "Python videos - YouTube",
                    },
                ),
            ],
        )

        decision = await verifier.verify(goal)
        assert decision.verified is False
        assert "playback was not confirmed" in decision.summary.lower() or "not playing" in decision.summary.lower()

    @pytest.mark.asyncio
    async def test_rejects_when_playback_unverified_or_paused(self):
        """Verifier MUST reject if video opened but is paused or unverified."""
        verifier = GoalVerifier()
        goal = make_goal_record(
            "Open Chrome and search YouTube Python videos. Play any one video random which you like.",
            [
                TaskRecord(
                    task_id="t1",
                    goal_id="g1",
                    description="Play YouTube video",
                    utterance="play any one video",
                    state=TaskState.COMPLETED,
                    result_message="Video opened at https://www.youtube.com/watch?v=123",
                    evidence={
                        "url": "https://www.youtube.com/watch?v=123",
                        "is_playing": False,
                        "playback_verified": False,
                        "is_paused": True,
                    },
                )
            ],
        )

        decision = await verifier.verify(goal)
        assert decision.verified is False
        assert "not playing" in decision.summary.lower() or "unverified" in decision.summary.lower()

    @pytest.mark.asyncio
    async def test_succeeds_when_playback_strictly_verified(self):
        """Verifier marks goal satisfied only when video is active and playback is verified."""
        verifier = GoalVerifier()
        goal = make_goal_record(
            "Open Chrome and search YouTube Python videos. Play any one video random which you like.",
            [
                TaskRecord(
                    task_id="t1",
                    goal_id="g1",
                    description="Open Chrome",
                    utterance="open chrome",
                    state=TaskState.COMPLETED,
                    result_message="Launched Chrome.",
                ),
                TaskRecord(
                    task_id="t2",
                    goal_id="g1",
                    description="Search YouTube",
                    utterance="search youtube for python",
                    state=TaskState.COMPLETED,
                    result_message="Navigated to search results.",
                ),
                TaskRecord(
                    task_id="t3",
                    goal_id="g1",
                    description="Play YouTube video",
                    utterance="play any one video random which you like",
                    state=TaskState.COMPLETED,
                    result_message="Playing 'Python Tutorial'. Current playback: 3.5s / 3600s.",
                    evidence={
                        "status": "playing",
                        "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
                        "title": "Python Tutorial - YouTube",
                        "is_playing": True,
                        "playback_verified": True,
                        "current_time": 3.5,
                        "duration": 3600.0,
                    },
                ),
            ],
        )

        decision = await verifier.verify(goal)
        assert decision.verified is True
        assert "play youtube video" in decision.summary.lower() or "playback" in decision.summary.lower()
