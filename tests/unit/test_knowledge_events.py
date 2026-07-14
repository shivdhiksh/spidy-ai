"""
Unit tests for spidy.knowledge.events
Tests all 6 KnowledgeEvent types.
"""

from __future__ import annotations

import pytest

from spidy.knowledge.events import (
    KnowledgeChunkStoredEvent,
    KnowledgeDocumentIngestedEvent,
    KnowledgeErrorEvent,
    KnowledgeQueriedEvent,
    KnowledgeSourceDeletedEvent,
    KnowledgeWebSearchPerformedEvent,
)
from spidy.core.event_bus import Event


class TestKnowledgeEventTopics:
    """All events have the correct topic strings."""

    def test_document_ingested_topic(self):
        assert KnowledgeDocumentIngestedEvent.topic == "knowledge.document_ingested"

    def test_chunk_stored_topic(self):
        assert KnowledgeChunkStoredEvent.topic == "knowledge.chunk_stored"

    def test_queried_topic(self):
        assert KnowledgeQueriedEvent.topic == "knowledge.queried"

    def test_web_search_performed_topic(self):
        assert KnowledgeWebSearchPerformedEvent.topic == "knowledge.web_search_performed"

    def test_source_deleted_topic(self):
        assert KnowledgeSourceDeletedEvent.topic == "knowledge.source_deleted"

    def test_error_topic(self):
        assert KnowledgeErrorEvent.topic == "knowledge.error"


class TestKnowledgeEventInheritance:
    """All events inherit from the Event base class."""

    def test_document_ingested_is_event(self):
        e = KnowledgeDocumentIngestedEvent()
        assert isinstance(e, Event)

    def test_chunk_stored_is_event(self):
        e = KnowledgeChunkStoredEvent()
        assert isinstance(e, Event)

    def test_queried_is_event(self):
        e = KnowledgeQueriedEvent()
        assert isinstance(e, Event)

    def test_web_search_is_event(self):
        e = KnowledgeWebSearchPerformedEvent()
        assert isinstance(e, Event)

    def test_source_deleted_is_event(self):
        e = KnowledgeSourceDeletedEvent()
        assert isinstance(e, Event)

    def test_error_is_event(self):
        e = KnowledgeErrorEvent()
        assert isinstance(e, Event)


class TestKnowledgeDocumentIngestedEvent:
    """KnowledgeDocumentIngestedEvent defaults and construction."""

    def test_defaults(self):
        e = KnowledgeDocumentIngestedEvent()
        assert e.source == ""
        assert e.source_type == ""
        assert e.chunk_count == 0
        assert e.tags == []

    def test_construction_with_values(self):
        e = KnowledgeDocumentIngestedEvent(
            source="report.pdf",
            source_type="pdf",
            chunk_count=12,
            tags=["finance", "quarterly"],
        )
        assert e.source == "report.pdf"
        assert e.source_type == "pdf"
        assert e.chunk_count == 12
        assert "finance" in e.tags


class TestKnowledgeChunkStoredEvent:
    """KnowledgeChunkStoredEvent defaults and construction."""

    def test_defaults(self):
        e = KnowledgeChunkStoredEvent()
        assert e.chunk_id == ""
        assert e.source == ""
        assert e.chunk_index == 0
        assert e.total_chunks == 0
        assert e.content_preview == ""

    def test_construction_with_values(self):
        e = KnowledgeChunkStoredEvent(
            chunk_id="abc123",
            source="doc.md",
            chunk_index=3,
            total_chunks=10,
            content_preview="This is a preview of chunk content...",
        )
        assert e.chunk_id == "abc123"
        assert e.chunk_index == 3
        assert e.total_chunks == 10
        assert "preview" in e.content_preview


class TestKnowledgeQueriedEvent:
    """KnowledgeQueriedEvent defaults and construction."""

    def test_defaults(self):
        e = KnowledgeQueriedEvent()
        assert e.query_preview == ""
        assert e.result_count == 0
        assert e.used_web_search is False
        assert e.best_score == 0.0

    def test_construction_with_values(self):
        e = KnowledgeQueriedEvent(
            query_preview="What is Python?",
            result_count=5,
            used_web_search=True,
            best_score=0.87,
        )
        assert e.query_preview == "What is Python?"
        assert e.result_count == 5
        assert e.used_web_search is True
        assert e.best_score == 0.87


class TestKnowledgeWebSearchPerformedEvent:
    """KnowledgeWebSearchPerformedEvent defaults and construction."""

    def test_defaults(self):
        e = KnowledgeWebSearchPerformedEvent()
        assert e.query_preview == ""
        assert e.provider == "duckduckgo"
        assert e.result_count == 0

    def test_construction_with_values(self):
        e = KnowledgeWebSearchPerformedEvent(
            query_preview="latest AI news",
            provider="duckduckgo",
            result_count=5,
        )
        assert e.query_preview == "latest AI news"
        assert e.result_count == 5


class TestKnowledgeSourceDeletedEvent:
    """KnowledgeSourceDeletedEvent defaults and construction."""

    def test_defaults(self):
        e = KnowledgeSourceDeletedEvent()
        assert e.source == ""
        assert e.chunks_deleted == 0

    def test_construction_with_values(self):
        e = KnowledgeSourceDeletedEvent(source="old_doc.pdf", chunks_deleted=15)
        assert e.source == "old_doc.pdf"
        assert e.chunks_deleted == 15


class TestKnowledgeErrorEvent:
    """KnowledgeErrorEvent defaults and construction."""

    def test_defaults(self):
        e = KnowledgeErrorEvent()
        assert e.operation == ""
        assert e.source == ""
        assert e.error == ""

    def test_construction_with_values(self):
        e = KnowledgeErrorEvent(
            operation="ingest",
            source="broken.pdf",
            error="File is corrupted",
        )
        assert e.operation == "ingest"
        assert e.source == "broken.pdf"
        assert e.error == "File is corrupted"

    def test_all_operations(self):
        """Ensure event can represent all documented operation types."""
        for op in ("ingest", "query", "delete", "search"):
            e = KnowledgeErrorEvent(operation=op, source="s", error="err")
            assert e.operation == op
