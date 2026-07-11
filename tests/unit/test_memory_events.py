"""
Tests for spidy.memory.events — all 6 EventBus event dataclasses
"""

from __future__ import annotations

import pytest

from spidy.memory.events import (
    MemoryClearedEvent,
    MemoryDeletedEvent,
    MemoryErrorEvent,
    MemoryRetrievedEvent,
    MemorySearchedEvent,
    MemoryStoredEvent,
)


class TestMemoryEvents:
    """All memory event dataclasses should be valid Event subclasses."""

    def test_stored_event_topic(self):
        e = MemoryStoredEvent()
        assert e.topic == "memory.stored"

    def test_stored_event_defaults(self):
        e = MemoryStoredEvent()
        assert e.memory_id == ""
        assert e.session_id == ""
        assert e.memory_type == ""
        assert e.content_preview == ""
        assert e.tags == []

    def test_stored_event_fields(self):
        e = MemoryStoredEvent(
            memory_id="abc",
            session_id="s1",
            memory_type="episodic",
            content_preview="hello",
            tags=["work"],
        )
        assert e.memory_id == "abc"
        assert e.tags == ["work"]

    def test_retrieved_event_topic(self):
        e = MemoryRetrievedEvent()
        assert e.topic == "memory.retrieved"

    def test_retrieved_event_fields(self):
        e = MemoryRetrievedEvent(
            session_id="s1",
            query_preview="what did I say",
            result_count=3,
            sources=["working", "episodic"],
        )
        assert e.result_count == 3
        assert "working" in e.sources

    def test_searched_event_topic(self):
        e = MemorySearchedEvent()
        assert e.topic == "memory.searched"

    def test_searched_event_fields(self):
        e = MemorySearchedEvent(
            query_preview="python",
            result_count=5,
            search_type="semantic",
            tags_filter=["code"],
        )
        assert e.search_type == "semantic"
        assert e.result_count == 5

    def test_deleted_event_topic(self):
        e = MemoryDeletedEvent()
        assert e.topic == "memory.deleted"

    def test_deleted_event_fields(self):
        e = MemoryDeletedEvent(memory_id="mem-123", session_id="s1")
        assert e.memory_id == "mem-123"

    def test_cleared_event_topic(self):
        e = MemoryClearedEvent()
        assert e.topic == "memory.cleared"

    def test_cleared_event_fields(self):
        e = MemoryClearedEvent(session_id="s1", count_cleared=42)
        assert e.count_cleared == 42

    def test_error_event_topic(self):
        e = MemoryErrorEvent()
        assert e.topic == "memory.error"

    def test_error_event_fields(self):
        e = MemoryErrorEvent(
            operation="store",
            session_id="s1",
            error="disk full",
        )
        assert e.operation == "store"
        assert e.error == "disk full"

    def test_all_events_are_event_subclasses(self):
        from spidy.core.event_bus import Event
        for cls in [
            MemoryStoredEvent,
            MemoryRetrievedEvent,
            MemorySearchedEvent,
            MemoryDeletedEvent,
            MemoryClearedEvent,
            MemoryErrorEvent,
        ]:
            instance = cls()
            assert isinstance(instance, Event)
