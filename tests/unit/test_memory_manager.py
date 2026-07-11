"""
Tests for spidy.memory.manager — MemoryManager (unified API)

Tests cover:
- All three tiers working together
- EventBus event publishing
- Graceful degradation with disabled tiers
- store_interaction() convenience method
- Working context window access
- Error handling and isolation
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.core.event_bus import EventBus
from spidy.memory.events import (
    MemoryClearedEvent,
    MemoryErrorEvent,
    MemoryRetrievedEvent,
    MemorySearchedEvent,
    MemoryStoredEvent,
)
from spidy.memory.manager import MemoryManager
from spidy.memory.types import MemoryEntry, MemoryType


# ─── Helpers / Fixtures ───────────────────────────────────────────────────────


class MockMemoryConfig:
    """Minimal config object that MemoryManager accepts."""
    enabled = True
    enable_working = True
    enable_episodic = True
    enable_semantic = True

    class _ShortTerm:
        max_messages = 10
    class _LongTerm:
        db_filename = "spidy_memory.db"
    class _Semantic:
        collection_name = "test_memories"
        embedding_model = "all-MiniLM-L6-v2"

    short_term = _ShortTerm()
    long_term = _LongTerm()
    semantic = _Semantic()


def make_manager(
    *,
    with_bus: bool = False,
    enable_working: bool = True,
    enable_episodic: bool = True,
    enable_semantic: bool = False,
) -> tuple[MemoryManager, EventBus | None]:
    """Build a MemoryManager with in-memory storage. Returns (manager, bus)."""
    cfg = MockMemoryConfig()
    cfg.enable_working = enable_working
    cfg.enable_episodic = enable_episodic
    cfg.enable_semantic = enable_semantic

    bus = EventBus() if with_bus else None
    mgr = MemoryManager(config=cfg, bus=bus, memory_dir=None)  # None = :memory:
    return mgr, bus


@pytest.fixture
async def mgr() -> MemoryManager:
    """Ready-to-use MemoryManager with no bus (unit test style)."""
    manager, _ = make_manager()
    await manager.initialize()
    return manager


@pytest.fixture
async def mgr_with_bus() -> tuple[MemoryManager, EventBus]:
    """Ready-to-use MemoryManager with an EventBus attached."""
    manager, bus = make_manager(with_bus=True)
    await manager.initialize()
    return manager, bus


# ─── Initialisation ───────────────────────────────────────────────────────────


class TestMemoryManagerInit:
    async def test_initialize_is_idempotent(self):
        m, _ = make_manager()
        await m.initialize()
        await m.initialize()  # should not raise
        assert m._initialized

    async def test_has_working_by_default(self, mgr):
        assert mgr.has_working is True

    async def test_has_episodic_by_default(self, mgr):
        assert mgr.has_episodic is True

    async def test_semantic_disabled_by_default(self, mgr):
        # Default fixture disables semantic (enable_semantic=False)
        assert mgr.has_semantic is False

    async def test_disable_working_tier(self):
        m, _ = make_manager(enable_working=False)
        await m.initialize()
        assert m.has_working is False

    async def test_disable_episodic_tier(self):
        m, _ = make_manager(enable_episodic=False)
        await m.initialize()
        assert m.has_episodic is False

    async def test_close_is_safe(self, mgr):
        await mgr.close()  # Should not raise


# ─── store() ──────────────────────────────────────────────────────────────────


class TestMemoryManagerStore:
    async def test_store_returns_str_id(self, mgr):
        mem_id = await mgr.store("hello world", session_id="s1")
        assert isinstance(mem_id, str)
        assert len(mem_id) == 36  # UUID4

    async def test_store_persists_to_episodic(self, mgr):
        await mgr.store("persistent memory", session_id="s1")
        assert await mgr._episodic.count() == 1

    async def test_store_adds_to_working(self, mgr):
        await mgr.store("working context", session_id="s1")
        assert mgr._working.entry_count("s1") == 1

    async def test_store_with_tags(self, mgr):
        mem_id = await mgr.store(
            "tagged content",
            session_id="s1",
            tags=["work", "code"],
        )
        # Verify in episodic
        results = await mgr._episodic.recall("tagged", session_id="s1")
        assert len(results) == 1
        assert "work" in results[0].tags

    async def test_store_with_metadata(self, mgr):
        await mgr.store("meta content", session_id="s1", metadata={"key": "val"})
        results = await mgr._episodic.recall("meta", session_id="s1")
        assert results[0].metadata == {"key": "val"}

    async def test_store_publishes_event(self, mgr_with_bus):
        mgr, bus = mgr_with_bus
        events = []
        bus.subscribe("memory.stored", lambda e: events.append(e) or None)

        # Make the handler async
        received = []
        async def capture(event):
            received.append(event)
        bus.subscribe("memory.stored", capture)

        await mgr.store("event test", session_id="s1")
        assert len(received) == 1
        assert isinstance(received[0], MemoryStoredEvent)
        assert received[0].session_id == "s1"


# ─── recall() ─────────────────────────────────────────────────────────────────


class TestMemoryManagerRecall:
    async def test_recall_finds_stored_memory(self, mgr):
        await mgr.store("I prefer dark mode", session_id="s1")
        results = await mgr.recall("dark mode", session_id="s1")
        assert len(results) > 0
        assert any("dark mode" in r["content"] for r in results)

    async def test_recall_returns_list_of_dicts(self, mgr):
        await mgr.store("some content", session_id="s1")
        results = await mgr.recall("content", session_id="s1")
        for r in results:
            assert isinstance(r, dict)
            assert "id" in r
            assert "content" in r
            assert "score" in r

    async def test_recall_respects_limit(self, mgr):
        for i in range(10):
            await mgr.store(f"python message {i}", session_id="s1")
        results = await mgr.recall("python", session_id="s1", limit=3)
        assert len(results) <= 3

    async def test_recall_empty_result(self, mgr):
        results = await mgr.recall("xyznonexistent", session_id="s1")
        assert results == []

    async def test_recall_publishes_event(self, mgr_with_bus):
        mgr, bus = mgr_with_bus
        received = []
        async def capture(event):
            received.append(event)
        bus.subscribe("memory.retrieved", capture)

        await mgr.store("recall test", session_id="s1")
        await mgr.recall("recall", session_id="s1")
        assert len(received) == 1
        assert isinstance(received[0], MemoryRetrievedEvent)


# ─── search() ─────────────────────────────────────────────────────────────────


class TestMemoryManagerSearch:
    async def test_search_finds_keyword(self, mgr):
        await mgr.store("python programming is great", session_id="s1")
        await mgr.store("java is also popular", session_id="s2")
        results = await mgr.search("python")
        assert any("python" in r["content"].lower() for r in results)

    async def test_search_returns_dicts_with_score(self, mgr):
        await mgr.store("searchable content", session_id="s1")
        results = await mgr.search("searchable")
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    async def test_search_empty_result(self, mgr):
        results = await mgr.search("xyznonexistent999")
        assert results == []

    async def test_search_publishes_event(self, mgr_with_bus):
        mgr, bus = mgr_with_bus
        received = []
        async def capture(event):
            received.append(event)
        bus.subscribe("memory.searched", capture)

        await mgr.store("searchable", session_id="s1")
        await mgr.search("searchable")
        assert len(received) == 1
        assert isinstance(received[0], MemorySearchedEvent)


# ─── clear() ──────────────────────────────────────────────────────────────────


class TestMemoryManagerClear:
    async def test_clear_session(self, mgr):
        await mgr.store("s1 content", session_id="s1")
        await mgr.store("s2 content", session_id="s2")
        count = await mgr.clear("s1")
        assert count >= 1
        # s1 episodic should be empty
        assert await mgr._episodic.count("s1") == 0

    async def test_clear_all(self, mgr):
        await mgr.store("a", session_id="s1")
        await mgr.store("b", session_id="s2")
        count = await mgr.clear("")
        assert count >= 2
        assert await mgr._episodic.count() == 0

    async def test_clear_publishes_event(self, mgr_with_bus):
        mgr, bus = mgr_with_bus
        received = []
        async def capture(event):
            received.append(event)
        bus.subscribe("memory.cleared", capture)

        await mgr.store("to clear", session_id="s1")
        await mgr.clear("s1")
        assert len(received) == 1
        assert isinstance(received[0], MemoryClearedEvent)


# ─── store_interaction() ──────────────────────────────────────────────────────


class TestStoreInteraction:
    async def test_stores_combined_content(self, mgr):
        mem_id = await mgr.store_interaction(
            utterance="What is Python?",
            response="Python is a programming language.",
            session_id="s1",
        )
        assert isinstance(mem_id, str)
        results = await mgr._episodic.recall("Python", session_id="s1")
        assert len(results) == 1
        assert "User:" in results[0].content
        assert "Spidy:" in results[0].content

    async def test_intent_becomes_tag(self, mgr):
        await mgr.store_interaction(
            utterance="hey",
            response="hello",
            session_id="s1",
            intent="greeting",
        )
        results = await mgr._episodic.recall("hey", session_id="s1")
        assert any("intent:greeting" in r.tags for r in results)

    async def test_extra_tags_preserved(self, mgr):
        await mgr.store_interaction(
            utterance="test",
            response="ok",
            session_id="s1",
            tags=["custom"],
        )
        results = await mgr._episodic.recall("test", session_id="s1")
        assert any("custom" in r.tags for r in results)


# ─── get_working_context() ────────────────────────────────────────────────────


class TestWorkingContext:
    async def test_working_context_empty(self, mgr):
        result = mgr.get_working_context("no-session")
        assert result == []

    async def test_working_context_returns_entries(self, mgr):
        await mgr.store("msg1", session_id="ctx-s1")
        await mgr.store("msg2", session_id="ctx-s1")
        ctx = mgr.get_working_context("ctx-s1", limit=5)
        assert len(ctx) == 2

    async def test_working_context_respects_limit(self, mgr):
        for i in range(8):
            await mgr.store(f"msg {i}", session_id="limit-s")
        ctx = mgr.get_working_context("limit-s", limit=3)
        assert len(ctx) == 3

    async def test_working_context_absent_returns_empty(self):
        m, _ = make_manager(enable_working=False)
        await m.initialize()
        result = m.get_working_context("s1")
        assert result == []
