"""
test_timezone_and_time_skill.py
================================
Tests for TimeSkill, local timezone determination, and zero-LLM deterministic routing.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import zoneinfo

import pytest

from spidy.brain.brain import Brain
from spidy.brain.goal_intent_classifier import GoalIntentClassifier
from spidy.brain.intent_classifier import IntentClassifier
from spidy.core.event_bus import EventBus
from spidy.skills.builtin.time_skill import TimeSkill
from spidy.skills.registry import SkillRegistry
from spidy.utils.datetime_utils import get_local_datetime, get_local_timezone


def test_get_local_timezone_and_datetime():
    """Verify local timezone resolves to OS timezone or configured timezone."""
    dt = get_local_datetime()
    assert dt.tzinfo is not None

    # Test explicit override
    dt_ist = get_local_datetime("Asia/Kolkata")
    assert dt_ist.tzinfo is not None
    # Asia/Kolkata is UTC+05:30 (19800 seconds offset)
    assert dt_ist.utcoffset().total_seconds() == 19800

    dt_ny = get_local_datetime("America/New_York")
    assert dt_ny.tzinfo is not None


@pytest.mark.asyncio
async def test_time_skill_deterministic_execution():
    """Verify TimeSkill produces formatted time in local timezone."""
    skill = TimeSkill()
    ctx = MagicMock()
    ctx.get.return_value = "Asia/Kolkata"

    res = await skill.execute("get_time", ctx)
    assert res.success is True
    assert "The current time is" in res.message
    assert "time" in res.data
    assert "timezone" in res.data

    res_date = await skill.execute("get_date", ctx)
    assert res_date.success is True
    assert "Today is" in res_date.message
    assert "date" in res_date.data

    res_day = await skill.execute("get_day", ctx)
    assert res_day.success is True
    assert "Today is" in res_day.message
    assert "day" in res_day.data

    res_dt = await skill.execute("get_datetime", ctx)
    assert res_dt.success is True
    assert "datetime" in res_dt.data


@pytest.mark.asyncio
async def test_intent_classifier_time_queries():
    """Verify natural time queries are classified with high confidence without LLM."""
    clf = IntentClassifier()

    queries = [
        "what time is it",
        "what time is it now",
        "what's the time",
        "whats the time",
        "tell me the time",
        "current time",
        "what is the time right now",
    ]
    for q in queries:
        intent = await clf.classify(q)
        assert intent.action == "get_time", f"Query '{q}' failed: got '{intent.action}'"
        assert intent.confidence >= 0.9

    date_queries = [
        "what's today's date",
        "what date is it",
        "today's date",
        "tell me today's date",
    ]
    for q in date_queries:
        intent = await clf.classify(q)
        assert intent.action == "get_date", f"Query '{q}' failed: got '{intent.action}'"
        assert intent.confidence >= 0.9

    day_queries = [
        "what day is it",
        "what day of the week is it",
        "what day is today",
    ]
    for q in day_queries:
        intent = await clf.classify(q)
        assert intent.action == "get_day", f"Query '{q}' failed: got '{intent.action}'"
        assert intent.confidence >= 0.9


@pytest.mark.asyncio
async def test_brain_process_time_query_bypasses_llm():
    """Verify that asking 'What time is it now?' executes TimeSkill with 0 LLM calls."""
    bus = EventBus()
    registry = SkillRegistry()
    time_skill = TimeSkill()
    registry.register(time_skill)

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="LLM should not be called")
    mock_llm.stream = AsyncMock()

    config = MagicMock()
    config.user_name = "Shiva"

    brain = Brain(
        bus=bus,
        config=config,
        skill_registry=registry,
        llm_client=mock_llm,
        user_name="Shiva",
    )
    await brain.start()

    response = await brain.process("What time is it now?")

    assert "The current time is" in response
    # Crucial assertion: LLM MUST NOT BE CALLED
    mock_llm.complete.assert_not_called()
    mock_llm.stream.assert_not_called()

    await brain.stop()
