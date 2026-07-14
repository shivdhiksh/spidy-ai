"""Tests for spidy.learning.preferences - PreferenceLearner."""
import asyncio, pytest, pytest_asyncio
from spidy.learning.preferences import PreferenceLearner

@pytest.fixture
def learner():
    return PreferenceLearner(db_path=":memory:")

@pytest.mark.asyncio
async def test_initialize(learner):
    await learner.initialize()

@pytest.mark.asyncio
async def test_set_and_get(learner):
    await learner.initialize()
    await learner.set("browser", "Chrome", confidence=0.8)
    val = await learner.get("browser")
    assert val == "Chrome"

@pytest.mark.asyncio
async def test_get_missing_returns_default(learner):
    await learner.initialize()
    val = await learner.get("missing", default="fallback")
    assert val == "fallback"

@pytest.mark.asyncio
async def test_get_preference_object(learner):
    await learner.initialize()
    await learner.set("ide", "VS Code", confidence=0.7)
    pref = await learner.get_preference("ide")
    assert pref is not None
    assert pref.key == "ide"
    assert pref.confidence == pytest.approx(0.7)

@pytest.mark.asyncio
async def test_get_preference_missing(learner):
    await learner.initialize()
    assert await learner.get_preference("nonexistent") is None

@pytest.mark.asyncio
async def test_implicit_confidence_capped(learner):
    await learner.initialize()
    await learner.set("k", "v", confidence=0.95, source="implicit")
    pref = await learner.get_preference("k")
    assert pref.confidence <= 0.8

@pytest.mark.asyncio
async def test_explicit_not_capped(learner):
    await learner.initialize()
    await learner.set("k", "v", confidence=0.95, source="explicit")
    pref = await learner.get_preference("k")
    assert pref.confidence == pytest.approx(0.95)

@pytest.mark.asyncio
async def test_update_confidence(learner):
    await learner.initialize()
    await learner.set("k", "v", confidence=0.5)
    updated = await learner.update_confidence("k", 0.1)
    assert updated is not None
    assert updated.confidence == pytest.approx(0.6)

@pytest.mark.asyncio
async def test_update_confidence_clamps(learner):
    await learner.initialize()
    # Implicit preferences are capped at 0.8 by _IMPLICIT_MAX_CONFIDENCE
    await learner.set("k", "v", confidence=0.5)
    updated = await learner.update_confidence("k", 5.0)
    # update_confidence delegates to set() which caps implicit at 0.8
    assert updated.confidence <= 1.0
    assert updated.confidence > 0.5

@pytest.mark.asyncio
async def test_update_confidence_missing(learner):
    await learner.initialize()
    result = await learner.update_confidence("nonexistent", 0.1)
    assert result is None

@pytest.mark.asyncio
async def test_delete_existing(learner):
    await learner.initialize()
    await learner.set("k", "v")
    deleted = await learner.delete("k")
    assert deleted is True
    assert await learner.get_preference("k") is None

@pytest.mark.asyncio
async def test_delete_missing(learner):
    await learner.initialize()
    assert await learner.delete("nonexistent") is False

@pytest.mark.asyncio
async def test_count(learner):
    await learner.initialize()
    assert await learner.count() == 0
    await learner.set("k1", "v1")
    await learner.set("k2", "v2")
    assert await learner.count() == 2

@pytest.mark.asyncio
async def test_all_preferences_sorted(learner):
    await learner.initialize()
    await learner.set("low", "v", confidence=0.2)
    await learner.set("high", "v", confidence=0.9)
    prefs = await learner.all_preferences()
    assert prefs[0].confidence >= prefs[1].confidence

@pytest.mark.asyncio
async def test_observe_app_known(learner):
    await learner.initialize()
    await learner.observe_app("chrome.exe")
    val = await learner.get("preferred_browser")
    assert val == "Chrome"

@pytest.mark.asyncio
async def test_observe_app_unknown_ignored(learner):
    await learner.initialize()
    await learner.observe_app("unknown_app.exe")
    assert await learner.count() == 0

@pytest.mark.asyncio
async def test_observe_app_explicit_not_overwritten(learner):
    await learner.initialize()
    await learner.set("preferred_browser", "Firefox", confidence=1.0, source="explicit")
    await learner.observe_app("chrome.exe")
    val = await learner.get("preferred_browser")
    assert val == "Firefox"

@pytest.mark.asyncio
async def test_to_context_string_empty(learner):
    await learner.initialize()
    s = await learner.to_context_string()
    assert s == ""

@pytest.mark.asyncio
async def test_to_context_string_with_prefs(learner):
    await learner.initialize()
    await learner.set("preferred_browser", "Chrome", confidence=0.9)
    s = await learner.to_context_string()
    assert "preferred_browser" in s
    assert "Chrome" in s

@pytest.mark.asyncio
async def test_upsert_overwrites(learner):
    await learner.initialize()
    await learner.set("k", "old", confidence=0.5)
    await learner.set("k", "new", confidence=0.7)
    val = await learner.get("k")
    assert val == "new"