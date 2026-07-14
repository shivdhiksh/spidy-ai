"""Tests for spidy.learning.manager - LearningManager (main integration point)."""
import pytest
from spidy.learning.manager import LearningManager

@pytest.fixture
def manager():
    return LearningManager(config=None, bus=None, learning_dir=None)

@pytest.mark.asyncio
async def test_initialize(manager):
    await manager.initialize()

@pytest.mark.asyncio
async def test_initialize_idempotent(manager):
    await manager.initialize()
    await manager.initialize()  # second call is no-op

@pytest.mark.asyncio
async def test_close(manager):
    await manager.initialize()
    await manager.close()  # should not raise

@pytest.mark.asyncio
async def test_record_feedback_positive(manager):
    await manager.initialize()
    # Should not raise
    await manager.record_feedback("sess1", "hello", "hi", rating=1.0, tags=["greeting"])

@pytest.mark.asyncio
async def test_record_feedback_negative(manager):
    await manager.initialize()
    await manager.record_feedback("sess1", "wrong", "bad", rating=-1.0)

@pytest.mark.asyncio
async def test_get_preference_missing(manager):
    await manager.initialize()
    val = await manager.get_preference("nonexistent", default="fallback")
    assert val == "fallback"

@pytest.mark.asyncio
async def test_get_habit_empty(manager):
    await manager.initialize()
    result = await manager.get_habit({"active_app": "code.exe", "topic": "code"})
    assert result is None

@pytest.mark.asyncio
async def test_observe_context_no_crash(manager):
    await manager.initialize()
    ctx = {"active_app": "Code.exe", "topic": "code"}
    await manager.observe_context(ctx, "open file", "file")

@pytest.mark.asyncio
async def test_set_preference(manager):
    await manager.initialize()
    pref = await manager.set_preference("browser", "Chrome", confidence=1.0, source="explicit")
    assert pref.key == "browser"
    assert pref.value == "Chrome"
    assert pref.confidence == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_get_preference_after_set(manager):
    await manager.initialize()
    await manager.set_preference("ide", "VS Code", confidence=1.0, source="explicit")
    val = await manager.get_preference("ide")
    assert val == "VS Code"

@pytest.mark.asyncio
async def test_list_preferences_empty(manager):
    await manager.initialize()
    prefs = await manager.list_preferences()
    assert prefs == []

@pytest.mark.asyncio
async def test_list_preferences_after_set(manager):
    await manager.initialize()
    await manager.set_preference("k", "v")
    prefs = await manager.list_preferences()
    assert len(prefs) == 1

@pytest.mark.asyncio
async def test_list_habits_empty(manager):
    await manager.initialize()
    habits = await manager.list_habits()
    assert habits == []

@pytest.mark.asyncio
async def test_list_workflows_empty(manager):
    await manager.initialize()
    workflows = await manager.list_workflows()
    assert workflows == []

@pytest.mark.asyncio
async def test_record_step_and_flush(manager):
    await manager.initialize()
    await manager.record_step("sess1", "open_file")
    result = await manager.flush_session("sess1")
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_suggest_workflow_empty(manager):
    await manager.initialize()
    result = await manager.suggest_workflow(["open_file"])
    assert result is None

@pytest.mark.asyncio
async def test_get_preference_context_empty(manager):
    await manager.initialize()
    ctx = await manager.get_preference_context()
    assert ctx == ""

@pytest.mark.asyncio
async def test_get_preference_context_with_prefs(manager):
    await manager.initialize()
    await manager.set_preference("preferred_browser", "Chrome", confidence=0.9, source="explicit")
    ctx = await manager.get_preference_context()
    assert "preferred_browser" in ctx

@pytest.mark.asyncio
async def test_get_feedback_sentiment_empty(manager):
    await manager.initialize()
    s = await manager.get_feedback_sentiment("unknown_tag")
    assert s == 0.0

@pytest.mark.asyncio
async def test_learning_interface_contract(manager):
    """Verify LearningManager satisfies the LearningInterface contract."""
    from spidy.brain.interfaces import LearningInterface
    assert isinstance(manager, LearningInterface)