"""Tests for spidy.learning.feedback - FeedbackProcessor."""
import pytest
from spidy.learning.feedback import FeedbackProcessor
from spidy.learning.types import FeedbackSignal

@pytest.fixture
def processor():
    return FeedbackProcessor(db_path=":memory:", feedback_boost=0.15, feedback_decay=0.10)

@pytest.mark.asyncio
async def test_initialize(processor):
    await processor.initialize()

@pytest.mark.asyncio
async def test_count_empty(processor):
    await processor.initialize()
    assert await processor.count() == 0

@pytest.mark.asyncio
async def test_process_positive(processor):
    await processor.initialize()
    signal = FeedbackSignal.create("sess1", "hello", "hi", rating=1.0, tags=["greeting"])
    await processor.process(signal)
    assert await processor.count() == 1

@pytest.mark.asyncio
async def test_process_negative(processor):
    await processor.initialize()
    signal = FeedbackSignal.create("sess1", "bad", "wrong", rating=-1.0)
    await processor.process(signal)
    assert await processor.count() == 1

@pytest.mark.asyncio
async def test_get_recent_empty(processor):
    await processor.initialize()
    recent = await processor.get_recent(limit=10)
    assert recent == []

@pytest.mark.asyncio
async def test_get_recent_returns_signals(processor):
    await processor.initialize()
    for i in range(5):
        s = FeedbackSignal.create("sess", f"u{i}", f"r{i}", rating=float(i % 2))
        await processor.process(s)
    recent = await processor.get_recent(limit=3)
    assert len(recent) == 3

@pytest.mark.asyncio
async def test_sentiment_by_tag_no_matches(processor):
    await processor.initialize()
    sentiment = await processor.sentiment_by_tag("unknown_tag")
    assert sentiment == 0.0

@pytest.mark.asyncio
async def test_sentiment_by_tag_positive(processor):
    await processor.initialize()
    s = FeedbackSignal.create("sess", "u", "r", rating=1.0, tags=["code"])
    await processor.process(s)
    sentiment = await processor.sentiment_by_tag("code")
    assert sentiment == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_sentiment_by_tag_mixed(processor):
    await processor.initialize()
    s1 = FeedbackSignal.create("sess", "u", "r", rating=1.0, tags=["code"])
    s2 = FeedbackSignal.create("sess", "u", "r", rating=-1.0, tags=["code"])
    await processor.process(s1)
    await processor.process(s2)
    sentiment = await processor.sentiment_by_tag("code")
    assert sentiment == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_process_multiple_independent_signals(processor):
    await processor.initialize()
    # Each create() call produces a unique id, so both are stored independently
    s1 = FeedbackSignal.create("sess", "u", "r", rating=1.0)
    s2 = FeedbackSignal.create("sess", "u", "r", rating=-1.0)
    await processor.process(s1)
    await processor.process(s2)
    count = await processor.count()
    assert count == 2