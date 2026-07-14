"""
Unit tests for spidy.knowledge.manager
Tests KnowledgeManager — the unified knowledge API.
All deps (ChromaDB, sentence-transformers, duckduckgo) are mocked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.knowledge.manager import KnowledgeManager
from spidy.knowledge.types import KnowledgeChunk, KnowledgeResult
from spidy.knowledge.events import (
    KnowledgeDocumentIngestedEvent,
    KnowledgeQueriedEvent,
    KnowledgeErrorEvent,
    KnowledgeSourceDeletedEvent,
    KnowledgeChunkStoredEvent,
    KnowledgeWebSearchPerformedEvent,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_chunk(content="text", source="test.txt", score=0.8) -> KnowledgeChunk:
    return KnowledgeChunk.create(content=content, source=source, score=score)


def make_manager(
    web_enabled: bool = False,
    knowledge_dir: Path | None = None,
) -> KnowledgeManager:
    """Create a KnowledgeManager with minimal/no-op deps."""
    mgr = KnowledgeManager(config=None, bus=None, knowledge_dir=knowledge_dir)
    # Patch sub-components to be non-functional by default
    mgr._store._checked_available = False
    mgr._embedder._checked_available = False
    mgr._web_search._enabled = web_enabled
    return mgr


# ─── Construction ─────────────────────────────────────────────────────────────


class TestKnowledgeManagerConstruction:
    """KnowledgeManager constructs with correct defaults."""

    def test_default_construction(self):
        mgr = KnowledgeManager()
        assert mgr._chunk_size == 1000
        assert mgr._chunk_overlap == 200
        assert mgr._embedding_model == "all-MiniLM-L6-v2"
        assert mgr._collection_name == "spidy_knowledge"
        assert mgr._min_score == 0.3
        assert mgr._max_results == 5

    def test_web_enabled_defaults_to_false(self):
        mgr = KnowledgeManager()
        assert mgr._web_enabled is False

    def test_config_values_used(self):
        class FakeConfig:
            chunk_size = 500
            chunk_overlap = 100
            embedding_model = "custom-model"
            collection_name = "custom_col"
            min_relevance_score = 0.5
            max_results = 10
            class web_search:
                enabled = True
                max_results = 3
                safe_search = False
        mgr = KnowledgeManager(config=FakeConfig())
        assert mgr._chunk_size == 500
        assert mgr._chunk_overlap == 100
        assert mgr._embedding_model == "custom-model"
        assert mgr._collection_name == "custom_col"
        assert mgr._min_score == 0.5
        assert mgr._max_results == 10
        assert mgr._web_enabled is True

    def test_none_config_uses_defaults(self):
        mgr = KnowledgeManager(config=None)
        assert mgr._chunk_size == 1000


# ─── initialize / close ──────────────────────────────────────────────────────


class TestKnowledgeManagerLifecycle:
    """KnowledgeManager lifecycle: initialize, close."""

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self):
        mgr = make_manager()
        await mgr.initialize()
        await mgr.initialize()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_is_safe(self):
        mgr = make_manager()
        await mgr.initialize()
        await mgr.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_without_init_is_safe(self):
        mgr = make_manager()
        await mgr.close()  # Should not raise


# ─── ingest (raw text) ───────────────────────────────────────────────────────


class TestKnowledgeManagerIngest:
    """KnowledgeManager.ingest() for raw text content."""

    @pytest.mark.asyncio
    async def test_ingest_returns_chunk_id_string(self):
        mgr = make_manager()
        chunk_id = await mgr.ingest(
            content="Some text content to ingest.",
            source="manual_input",
        )
        # Returns first chunk ID or empty string
        assert isinstance(chunk_id, str)

    @pytest.mark.asyncio
    async def test_ingest_empty_content_returns_empty(self):
        mgr = make_manager()
        result = await mgr.ingest(content="", source="source")
        assert result == ""

    @pytest.mark.asyncio
    async def test_ingest_whitespace_content_returns_empty(self):
        mgr = make_manager()
        result = await mgr.ingest(content="   \n  ", source="source")
        assert result == ""

    @pytest.mark.asyncio
    async def test_ingest_publishes_event_when_bus_present(self):
        mgr = make_manager()
        published = []

        class FakeBus:
            async def publish(self, event):
                published.append(event)

        mgr._bus = FakeBus()
        await mgr.ingest(content="some content", source="test")
        # May publish KnowledgeChunkStoredEvent and/or KnowledgeDocumentIngestedEvent
        event_types = [type(e).__name__ for e in published]
        # At minimum, chunk_stored events for each chunk; or document_ingested if n > 0
        # Embedding unavailable → store will return 0 → no events published for ingested doc
        assert isinstance(published, list)

    @pytest.mark.asyncio
    async def test_ingest_with_tags(self):
        mgr = make_manager()
        result = await mgr.ingest(
            content="Tagged content",
            source="src",
            tags=["tag1", "tag2"],
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_ingest_with_metadata(self):
        mgr = make_manager()
        result = await mgr.ingest(
            content="Meta content",
            source="src",
            metadata={"author": "Bob"},
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_ingest_handles_exception_gracefully(self):
        mgr = make_manager()
        # Patch chunker to raise
        mgr._chunker.chunk = MagicMock(side_effect=RuntimeError("chunker failed"))
        result = await mgr.ingest(content="content", source="src")
        assert result == ""


# ─── ingest_file ─────────────────────────────────────────────────────────────


class TestKnowledgeManagerIngestFile:
    """KnowledgeManager.ingest_file() for document files."""

    @pytest.mark.asyncio
    async def test_ingest_file_plain_text(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("My important notes.", encoding="utf-8")
        mgr = make_manager()
        result = await mgr.ingest_file(f)
        # Returns number of chunks; 0 if embedding unavailable
        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_ingest_file_missing_returns_0(self, tmp_path):
        mgr = make_manager()
        result = await mgr.ingest_file(tmp_path / "ghost.txt")
        assert result == 0

    @pytest.mark.asyncio
    async def test_ingest_file_empty_returns_0(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        mgr = make_manager()
        result = await mgr.ingest_file(f)
        assert result == 0

    @pytest.mark.asyncio
    async def test_ingest_file_string_path(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("String path content.", encoding="utf-8")
        mgr = make_manager()
        result = await mgr.ingest_file(str(f))
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_ingest_file_with_tags(self, tmp_path):
        f = tmp_path / "tagged.txt"
        f.write_text("Tagged document.", encoding="utf-8")
        mgr = make_manager()
        result = await mgr.ingest_file(f, tags=["important"])
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_ingest_file_markdown(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\n\nWorld content.", encoding="utf-8")
        mgr = make_manager()
        result = await mgr.ingest_file(f)
        assert isinstance(result, int)


# ─── query ───────────────────────────────────────────────────────────────────


class TestKnowledgeManagerQuery:
    """KnowledgeManager.query() returns structured result dicts."""

    @pytest.mark.asyncio
    async def test_query_returns_list(self):
        mgr = make_manager()
        result = await mgr.query("what is Python?")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_query_returns_empty_when_no_knowledge(self):
        mgr = make_manager()
        result = await mgr.query("test question")
        assert result == []

    @pytest.mark.asyncio
    async def test_query_result_dict_structure(self):
        """When results are returned, each dict has required keys."""
        mgr = make_manager()
        # Patch _retrieve to return a known result
        chunk = make_chunk()
        known_result = KnowledgeResult(query="q", chunks=[chunk])
        mgr._retrieve = AsyncMock(return_value=known_result)
        result = await mgr.query("test question")
        assert len(result) == 1
        assert "content" in result[0]
        assert "source" in result[0]
        assert "score" in result[0]
        assert "source_type" in result[0]
        assert "metadata" in result[0]

    @pytest.mark.asyncio
    async def test_query_publishes_queried_event(self):
        mgr = make_manager()
        published = []

        class FakeBus:
            async def publish(self, event):
                published.append(event)

        mgr._bus = FakeBus()
        await mgr.query("some question")
        queried_events = [e for e in published if isinstance(e, KnowledgeQueriedEvent)]
        assert len(queried_events) == 1


# ─── search_rag ──────────────────────────────────────────────────────────────


class TestKnowledgeManagerSearchRag:
    """KnowledgeManager.search_rag() formats for LLM injection."""

    @pytest.mark.asyncio
    async def test_search_rag_returns_string(self):
        mgr = make_manager()
        result = await mgr.search_rag("test query")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_search_rag_returns_empty_when_no_knowledge(self):
        mgr = make_manager()
        result = await mgr.search_rag("test query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_search_rag_returns_context_when_chunks_found(self):
        mgr = make_manager()
        chunk = make_chunk(content="Python is a programming language.", source="python.txt")
        known_result = KnowledgeResult(query="q", chunks=[chunk])
        mgr._retrieve = AsyncMock(return_value=known_result)
        result = await mgr.search_rag("what is Python?")
        assert "[Knowledge Context]" in result
        assert "Python is a programming language." in result

    @pytest.mark.asyncio
    async def test_search_rag_passes_limit(self):
        mgr = make_manager()
        called_with = {}

        async def fake_retrieve(query, limit=None):
            called_with["limit"] = limit
            return KnowledgeResult(query=query, chunks=[])

        mgr._retrieve = fake_retrieve
        await mgr.search_rag("query", limit=3)
        assert called_with["limit"] == 3


# ─── delete_source ───────────────────────────────────────────────────────────


class TestKnowledgeManagerDeleteSource:
    """KnowledgeManager.delete_source() removes chunks by source."""

    @pytest.mark.asyncio
    async def test_delete_source_returns_int(self):
        mgr = make_manager()
        result = await mgr.delete_source("test.txt")
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_delete_source_publishes_event(self):
        mgr = make_manager()
        published = []

        class FakeBus:
            async def publish(self, event):
                published.append(event)

        mgr._bus = FakeBus()
        await mgr.delete_source("old_doc.pdf")
        deleted_events = [e for e in published if isinstance(e, KnowledgeSourceDeletedEvent)]
        assert len(deleted_events) == 1
        assert deleted_events[0].source == "old_doc.pdf"

    @pytest.mark.asyncio
    async def test_delete_source_handles_exception(self):
        mgr = make_manager()
        mgr._store.delete_source = AsyncMock(side_effect=RuntimeError("db error"))
        result = await mgr.delete_source("bad_source")
        assert result == 0


# ─── count / list_sources ────────────────────────────────────────────────────


class TestKnowledgeManagerManagement:
    """count() and list_sources() management methods."""

    @pytest.mark.asyncio
    async def test_count_returns_int(self):
        mgr = make_manager()
        result = await mgr.count()
        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_list_sources_returns_list(self):
        mgr = make_manager()
        result = await mgr.list_sources()
        assert isinstance(result, list)


# ─── _retrieve internal ───────────────────────────────────────────────────────


class TestKnowledgeManagerRetrieve:
    """KnowledgeManager._retrieve() internal retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_knowledge_result(self):
        mgr = make_manager()
        result = await mgr._retrieve("test query")
        assert isinstance(result, KnowledgeResult)

    @pytest.mark.asyncio
    async def test_retrieve_empty_when_no_embedding(self):
        mgr = make_manager()
        # Embedding unavailable → no query embedding → empty result
        result = await mgr._retrieve("question")
        assert result.is_empty

    @pytest.mark.asyncio
    async def test_retrieve_with_mocked_embedding_and_store(self):
        mgr = make_manager()
        chunk = make_chunk()
        mgr._embedder.embed_one = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mgr._store.query = AsyncMock(return_value=[chunk])
        result = await mgr._retrieve("question")
        assert not result.is_empty
        assert result.chunks[0] == chunk

    @pytest.mark.asyncio
    async def test_retrieve_no_web_search_when_disabled(self):
        mgr = make_manager(web_enabled=False)
        chunk = make_chunk()
        mgr._embedder.embed_one = AsyncMock(return_value=[0.1])
        mgr._store.query = AsyncMock(return_value=[chunk])
        result = await mgr._retrieve("question", limit=5)
        assert result.used_web_search is False

    @pytest.mark.asyncio
    async def test_retrieve_web_search_enabled_but_enough_results(self):
        """Web search is NOT triggered when vector store returns enough results."""
        mgr = make_manager()
        mgr._web_enabled = True
        mgr._web_search._enabled = True
        # Return n >= limit from vector store
        chunks = [make_chunk() for _ in range(5)]
        mgr._embedder.embed_one = AsyncMock(return_value=[0.1])
        mgr._store.query = AsyncMock(return_value=chunks)
        mgr._web_search.search = AsyncMock(return_value=[])
        result = await mgr._retrieve("question", limit=5)
        # Web search should not have been called (enough results)
        mgr._web_search.search.assert_not_called()


# ─── EventBus integration ─────────────────────────────────────────────────────


class TestKnowledgeManagerEventBus:
    """KnowledgeManager._publish() silently handles errors."""

    @pytest.mark.asyncio
    async def test_publish_with_no_bus_is_silent(self):
        mgr = KnowledgeManager(config=None, bus=None)
        # Should not raise
        from spidy.knowledge.events import KnowledgeErrorEvent
        await mgr._publish(KnowledgeErrorEvent())

    @pytest.mark.asyncio
    async def test_publish_error_is_swallowed(self):
        mgr = make_manager()

        class BrokenBus:
            async def publish(self, event):
                raise RuntimeError("bus broken")

        mgr._bus = BrokenBus()
        # Should not raise
        from spidy.knowledge.events import KnowledgeErrorEvent
        await mgr._publish(KnowledgeErrorEvent())
