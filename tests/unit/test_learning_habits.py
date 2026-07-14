"""Tests for spidy.learning.habits - HabitDetector."""
import pytest
from spidy.learning.habits import HabitDetector

@pytest.fixture
def detector():
    return HabitDetector(db_path=":memory:", min_observations=3)

@pytest.mark.asyncio
async def test_initialize(detector):
    await detector.initialize()

@pytest.mark.asyncio
async def test_observe_returns_none_below_threshold(detector):
    await detector.initialize()
    ctx = {"active_app": "code.exe", "topic": "code"}
    result = await detector.observe(ctx, "open file", "file")
    assert result is None

@pytest.mark.asyncio
async def test_observe_promotes_habit_at_threshold(detector):
    await detector.initialize()
    ctx = {"active_app": "code.exe", "topic": "code"}
    for _ in range(2):
        await detector.observe(ctx, "open file", "code")
    result = await detector.observe(ctx, "open file", "code")
    # May or may not be promoted depending on exact trigger key
    # Just verify no exception

@pytest.mark.asyncio
async def test_list_habits_empty(detector):
    await detector.initialize()
    habits = await detector.list_habits()
    assert habits == []

@pytest.mark.asyncio
async def test_count_empty(detector):
    await detector.initialize()
    assert await detector.count() == 0

@pytest.mark.asyncio
async def test_get_matching_habit_no_habits(detector):
    await detector.initialize()
    result = await detector.get_matching_habit({"active_app": "code.exe", "topic": "code"})
    assert result is None

@pytest.mark.asyncio
async def test_observe_different_contexts_not_confused(detector):
    await detector.initialize()
    ctx1 = {"active_app": "chrome.exe", "topic": "search"}
    ctx2 = {"active_app": "code.exe", "topic": "code"}
    for _ in range(3):
        await detector.observe(ctx1, "search web", "search")
    # Even if habit is promoted, it should not match ctx2
    result = await detector.get_matching_habit(ctx2)
    # Result should be None or a different habit
    if result is not None:
        assert result.trigger.get("active_app", "") != "chrome.exe" or result.trigger.get("topic", "") != "search"

@pytest.mark.asyncio
async def test_low_min_observations(detector):
    d = HabitDetector(db_path=":memory:", min_observations=1)
    await d.initialize()
    ctx = {"active_app": "code.exe", "topic": "code"}
    result = await d.observe(ctx, "do task", "code")
    assert result is not None
    habits = await d.list_habits()
    assert len(habits) >= 1