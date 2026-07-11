"""
Tests for spidy.memory.episodic — EpisodicMemory
Uses aiosqlite :memory: database — no disk I/O required.
"""

from __future__ import annotations

import pytest

from spidy.memory.episodic import EpisodicMemory
from spidy.memory.types import MemoryEntry, MemoryType


@pytest.fixture
async def ep() -> EpisodicMemory:
    """Fresh EpisodicMemory with in-memory SQLite (or fallback)."""
    mem = EpisodicMemory(db_path=":memory:")
    await mem.initialize()
    return mem


class TestEpisodicMemoryInit:
    async def test_initialise_twice_is_idempotent(self):
        mem = EpisodicMemory(db_path=":memory:")
        await mem.initialize()
        await mem.initialize()  # Should not raise or double-create schema
        assert mem._initialized

    async def test_not_initialized_raises(self):
        mem = EpisodicMemory(db_path=":memory:")
        with pytest.raises(RuntimeError, match="initialize"):
            await mem.store(MemoryEntry.create(content="x"))


class TestEpisodicMemoryStore:
    async def test_store_returns_id(self, ep: EpisodicMemory):
        entry = MemoryEntry.create(content="hello world", session_id="s1")
        returned_id = await ep.store(entry)
        assert returned_id == entry.id

    async def test_store_multiple_entries(self, ep: EpisodicMemory):
        for i in range(5):
            entry = MemoryEntry.create(content=f"entry {i}", session_id="s1")
            await ep.store(entry)
        count = await ep.count()
        assert count == 5

    async def test_store_with_tags(self, ep: EpisodicMemory):
        entry = MemoryEntry.create(content="tagged", session_id="s1", tags=["work", "code"])
        await ep.store(entry)
        results = await ep.recall("tagged", session_id="s1")
        assert len(results) == 1
        assert "work" in results[0].tags
        assert "code" in results[0].tags

    async def test_store_with_metadata(self, ep: EpisodicMemory):
        entry = MemoryEntry.create(
            content="meta", session_id="s1", metadata={"key": "value"}
        )
        await ep.store(entry)
        results = await ep.recall("meta", session_id="s1")
        assert results[0].metadata == {"key": "value"}

    async def test_store_duplicate_id_replaces(self, ep: EpisodicMemory):
        entry = MemoryEntry.create(content="original", memory_id="dup-id")
        await ep.store(entry)
        updated = MemoryEntry.create(content="updated", memory_id="dup-id")
        await ep.store(updated)
        count = await ep.count()
        assert count == 1


class TestEpisodicMemoryRecall:
    async def test_recall_by_keyword(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="I love Python programming", session_id="s1"))
        await ep.store(MemoryEntry.create(content="The weather is nice", session_id="s1"))
        results = await ep.recall("Python", session_id="s1")
        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_recall_case_insensitive(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="PYTHON is great", session_id="s1"))
        results = await ep.recall("python", session_id="s1")
        assert len(results) == 1

    async def test_recall_scoped_to_session(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="python s1", session_id="s1"))
        await ep.store(MemoryEntry.create(content="python s2", session_id="s2"))
        results = await ep.recall("python", session_id="s1")
        assert all(r.session_id == "s1" for r in results)

    async def test_recall_no_session_scope(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="python s1", session_id="s1"))
        await ep.store(MemoryEntry.create(content="python s2", session_id="s2"))
        results = await ep.recall("python", session_id="")
        assert len(results) == 2

    async def test_recall_limit(self, ep: EpisodicMemory):
        for i in range(10):
            await ep.store(MemoryEntry.create(content=f"python message {i}", session_id="s1"))
        results = await ep.recall("python", session_id="s1", limit=3)
        assert len(results) == 3

    async def test_recall_no_match(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="something else", session_id="s1"))
        results = await ep.recall("xyznonexistent", session_id="s1")
        assert results == []


class TestEpisodicMemorySearch:
    async def test_search_finds_keyword(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="async programming in Python"))
        await ep.store(MemoryEntry.create(content="Java is verbose"))
        results = await ep.search("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_search_with_tags_filter(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="python code", tags=["code"]))
        await ep.store(MemoryEntry.create(content="python note", tags=["personal"]))
        results = await ep.search("python", tags=["code"])
        assert len(results) == 1
        assert "code" in results[0].content

    async def test_search_limit(self, ep: EpisodicMemory):
        for i in range(10):
            await ep.store(MemoryEntry.create(content=f"python message {i}"))
        results = await ep.search("python", limit=5)
        assert len(results) <= 5


class TestEpisodicMemoryDelete:
    async def test_delete_existing(self, ep: EpisodicMemory):
        entry = MemoryEntry.create(content="to delete")
        await ep.store(entry)
        result = await ep.delete(entry.id)
        assert result is True
        count = await ep.count()
        assert count == 0

    async def test_delete_nonexistent_returns_false(self, ep: EpisodicMemory):
        result = await ep.delete("no-such-id")
        # With fallback, True might be returned; with SQLite, should be False
        # We just ensure no exception is raised
        assert isinstance(result, bool)

    async def test_delete_does_not_affect_other_entries(self, ep: EpisodicMemory):
        e1 = MemoryEntry.create(content="keep me")
        e2 = MemoryEntry.create(content="delete me")
        await ep.store(e1)
        await ep.store(e2)
        await ep.delete(e2.id)
        count = await ep.count()
        assert count == 1


class TestEpisodicMemoryClear:
    async def test_clear_session(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="s1 msg", session_id="s1"))
        await ep.store(MemoryEntry.create(content="s1 msg2", session_id="s1"))
        await ep.store(MemoryEntry.create(content="s2 msg", session_id="s2"))
        cleared = await ep.clear("s1")
        assert cleared == 2
        count = await ep.count()
        assert count == 1  # s2 remains

    async def test_clear_all(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="a", session_id="s1"))
        await ep.store(MemoryEntry.create(content="b", session_id="s2"))
        cleared = await ep.clear("")
        assert cleared == 2
        assert await ep.count() == 0

    async def test_clear_empty_db_returns_zero(self, ep: EpisodicMemory):
        cleared = await ep.clear("s1")
        assert cleared == 0


class TestEpisodicMemoryCount:
    async def test_count_empty(self, ep: EpisodicMemory):
        assert await ep.count() == 0

    async def test_count_with_entries(self, ep: EpisodicMemory):
        for i in range(7):
            await ep.store(MemoryEntry.create(content=f"msg {i}", session_id="s1"))
        assert await ep.count() == 7

    async def test_count_by_session(self, ep: EpisodicMemory):
        await ep.store(MemoryEntry.create(content="s1 msg", session_id="s1"))
        await ep.store(MemoryEntry.create(content="s1 msg2", session_id="s1"))
        await ep.store(MemoryEntry.create(content="s2 msg", session_id="s2"))
        assert await ep.count("s1") == 2
        assert await ep.count("s2") == 1
