"""
Tests for spidy.memory.types — MemoryEntry, MemoryType, MemorySearchResult
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from spidy.memory.types import MemoryEntry, MemorySearchResult, MemoryType


class TestMemoryType:
    def test_values(self):
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_from_string(self):
        assert MemoryType("working") is MemoryType.WORKING
        assert MemoryType("episodic") is MemoryType.EPISODIC
        assert MemoryType("semantic") is MemoryType.SEMANTIC

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            MemoryType("invalid_tier")


class TestMemoryEntry:
    def test_create_defaults(self):
        entry = MemoryEntry.create(content="hello world")
        assert entry.content == "hello world"
        assert entry.session_id == ""
        assert entry.tags == ()
        assert entry.metadata == {}
        assert isinstance(entry.id, str)
        assert len(entry.id) == 36  # UUID4
        assert entry.memory_type == MemoryType.EPISODIC

    def test_create_with_all_fields(self):
        entry = MemoryEntry.create(
            content="test",
            session_id="s1",
            tags=["work", "code"],
            metadata={"key": "value"},
            memory_type=MemoryType.WORKING,
        )
        assert entry.session_id == "s1"
        assert entry.tags == ("work", "code")
        assert entry.metadata == {"key": "value"}
        assert entry.memory_type == MemoryType.WORKING

    def test_create_with_custom_id(self):
        entry = MemoryEntry.create(content="x", memory_id="custom-id-123")
        assert entry.id == "custom-id-123"

    def test_frozen(self):
        entry = MemoryEntry.create(content="immutable")
        with pytest.raises((AttributeError, TypeError)):
            entry.content = "changed"  # type: ignore[misc]

    def test_timestamp_is_utc(self):
        entry = MemoryEntry.create(content="ts test")
        assert entry.timestamp.tzinfo is not None
        assert entry.timestamp.tzinfo == timezone.utc

    def test_to_dict_keys(self):
        entry = MemoryEntry.create(content="dict test", session_id="s1", tags=["t1"])
        d = entry.to_dict()
        assert "id" in d
        assert "content" in d
        assert "session_id" in d
        assert "tags" in d
        assert "metadata" in d
        assert "timestamp" in d
        assert "memory_type" in d

    def test_to_dict_tags_is_list(self):
        entry = MemoryEntry.create(content="tag test", tags=["a", "b"])
        d = entry.to_dict()
        assert isinstance(d["tags"], list)
        assert d["tags"] == ["a", "b"]

    def test_to_dict_timestamp_is_iso_string(self):
        entry = MemoryEntry.create(content="ts")
        d = entry.to_dict()
        assert isinstance(d["timestamp"], str)
        # Should be parseable
        dt = datetime.fromisoformat(d["timestamp"])
        assert dt.tzinfo is not None

    def test_hashable(self):
        entry = MemoryEntry.create(content="hashable")
        s = {entry}  # must not raise
        assert entry in s

    def test_equality_by_id(self):
        e1 = MemoryEntry.create(content="x", memory_id="same-id")
        e2 = MemoryEntry.create(content="x", memory_id="same-id")
        assert e1 == e2


class TestMemorySearchResult:
    def test_default_score(self):
        entry = MemoryEntry.create(content="test")
        result = MemorySearchResult(entry=entry)
        assert result.score == 1.0

    def test_custom_score(self):
        entry = MemoryEntry.create(content="test")
        result = MemorySearchResult(entry=entry, score=0.75)
        assert result.score == 0.75

    def test_to_dict_includes_score(self):
        entry = MemoryEntry.create(content="scored", session_id="s1")
        result = MemorySearchResult(entry=entry, score=0.88)
        d = result.to_dict()
        assert d["score"] == 0.88
        assert d["content"] == "scored"
        assert d["session_id"] == "s1"

    def test_frozen(self):
        entry = MemoryEntry.create(content="frozen")
        result = MemorySearchResult(entry=entry, score=0.5)
        with pytest.raises((AttributeError, TypeError)):
            result.score = 0.9  # type: ignore[misc]
