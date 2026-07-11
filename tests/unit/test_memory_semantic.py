"""
Tests for spidy.memory.semantic — SemanticMemory
Tests cover:
  1. Graceful degradation when chromadb/sentence-transformers are NOT installed
  2. Functional behaviour with mocked ChromaDB and sentence-transformers
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.memory.semantic import SemanticMemory
from spidy.memory.types import MemoryEntry, MemorySearchResult, MemoryType


def make_entry(content: str, session_id: str = "s1", tags: list[str] | None = None) -> MemoryEntry:
    return MemoryEntry.create(content=content, session_id=session_id, tags=tags)


# ─── Graceful degradation (no optional deps) ──────────────────────────────────


class TestSemanticMemoryDegradation:
    """All ops must be no-ops when chromadb/sentence-transformers are absent."""

    def _make_unavailable(self) -> SemanticMemory:
        """Return a SemanticMemory that has been probed as unavailable."""
        sm = SemanticMemory()
        sm._available = False  # simulate ImportError on probe
        return sm

    async def test_store_returns_id_when_unavailable(self):
        sm = self._make_unavailable()
        entry = make_entry("hello")
        result = await sm.store(entry)
        assert result == entry.id

    async def test_search_returns_empty_when_unavailable(self):
        sm = self._make_unavailable()
        results = await sm.search("query")
        assert results == []

    async def test_delete_returns_false_when_unavailable(self):
        sm = self._make_unavailable()
        result = await sm.delete("some-id")
        assert result is False

    async def test_clear_returns_zero_when_unavailable(self):
        sm = self._make_unavailable()
        result = await sm.clear()
        assert result == 0

    def test_count_returns_zero_when_unavailable(self):
        sm = self._make_unavailable()
        assert sm.count() == 0

    def test_is_available_false(self):
        sm = self._make_unavailable()
        assert sm.is_available is False


# ─── Functional tests with mocked ChromaDB ────────────────────────────────────


class TestSemanticMemoryFunctional:
    """Tests using a fully mocked ChromaDB + encoder."""

    def _make_available(self) -> tuple[SemanticMemory, MagicMock, MagicMock]:
        """Return (SemanticMemory, mock_collection, mock_encoder) with deps mocked."""
        sm = SemanticMemory(collection_name="test_col", persist_dir=":memory:")
        sm._available = True

        # Mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        sm._encoder = mock_encoder

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        sm._collection = mock_collection

        # Mock client
        mock_client = MagicMock()
        sm._client = mock_client

        return sm, mock_collection, mock_encoder

    async def test_store_calls_upsert(self):
        sm, collection, encoder = self._make_available()
        entry = make_entry("embed this")
        await sm.store(entry)
        collection.upsert.assert_called_once()
        call_kwargs = collection.upsert.call_args
        assert entry.id in call_kwargs[1]["ids"]

    async def test_store_encodes_content(self):
        sm, collection, encoder = self._make_available()
        entry = make_entry("encode me please")
        await sm.store(entry)
        encoder.encode.assert_called_once_with("encode me please")

    async def test_store_returns_entry_id(self):
        sm, collection, _ = self._make_available()
        entry = make_entry("return id test")
        result = await sm.store(entry)
        assert result == entry.id

    async def test_search_returns_results(self):
        sm, collection, encoder = self._make_available()
        # Mock the query response
        ts = datetime.now(timezone.utc).timestamp()
        collection.count.return_value = 1
        collection.query.return_value = {
            "ids": [["mem-001"]],
            "documents": [["similar content"]],
            "metadatas": [[{
                "session_id": "s1",
                "tags": "work",
                "timestamp": str(ts),
                "memory_type": "episodic",
            }]],
            "distances": [[0.1]],  # low distance = high similarity
        }
        results = await sm.search("find related content")
        assert len(results) == 1
        assert isinstance(results[0], MemorySearchResult)
        assert results[0].score == pytest.approx(0.9, abs=0.01)
        assert results[0].entry.content == "similar content"

    async def test_search_empty_collection(self):
        sm, collection, encoder = self._make_available()
        collection.count.return_value = 0
        collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        results = await sm.search("anything")
        assert results == []

    async def test_delete_calls_collection_delete(self):
        sm, collection, _ = self._make_available()
        result = await sm.delete("mem-to-delete")
        collection.delete.assert_called_once_with(ids=["mem-to-delete"])
        assert result is True

    async def test_clear_all_resets_collection(self):
        sm, collection, _ = self._make_available()
        collection.count.return_value = 5
        mock_client = sm._client
        mock_client.get_or_create_collection.return_value = MagicMock()
        result = await sm.clear("")
        assert result == 5
        mock_client.delete_collection.assert_called_once()

    def test_is_available_true_when_initialized(self):
        sm, _, _ = self._make_available()
        assert sm.is_available is True
