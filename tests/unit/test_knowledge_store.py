"""
Unit tests for spidy.knowledge.store
Tests VectorStore with ChromaDB mocked via monkeypatching.
All tests run without a real ChromaDB installation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from spidy.knowledge.store import VectorStore, _DEFAULT_COLLECTION
from spidy.knowledge.types import KnowledgeChunk


# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_chunk(
    content: str = "test content",
    source: str = "test.txt",
    score: float = 0.8,
    chunk_id: str | None = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
    source_type: str = "text",
    tags: list[str] | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id or f"chunk_{id(content)}",
        content=content,
        source=source,
        source_type=source_type,  # type: ignore[arg-type]
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        score=score,
        tags=tags or [],
    )


def make_embedding(dim: int = 4) -> list[float]:
    return [0.1] * dim


# ─── Construction ─────────────────────────────────────────────────────────────


class TestVectorStoreConstruction:
    """VectorStore constructs with correct defaults."""

    def test_default_collection_name(self):
        store = VectorStore()
        assert store._collection_name == _DEFAULT_COLLECTION

    def test_custom_collection_name(self):
        store = VectorStore(collection_name="my_collection")
        assert store._collection_name == "my_collection"

    def test_no_client_initially(self):
        store = VectorStore()
        assert store._client is None
        assert store._collection is None

    def test_persist_dir_none_by_default(self):
        store = VectorStore()
        assert store._persist_dir is None

    def test_persist_dir_stored_as_path(self, tmp_path):
        store = VectorStore(persist_dir=tmp_path)
        assert store._persist_dir == tmp_path


# ─── Availability ────────────────────────────────────────────────────────────


class TestVectorStoreAvailability:
    """VectorStore.is_available reflects chromadb import status."""

    def test_is_available_when_chromadb_installed(self):
        store = VectorStore()
        store._checked_available = None
        mock_chromadb = MagicMock()
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            result = store.is_available
        assert result is True

    def test_is_available_false_when_not_installed(self):
        store = VectorStore()
        store._checked_available = False
        assert store.is_available is False

    def test_availability_cached(self):
        store = VectorStore()
        store._checked_available = True
        assert store.is_available is True


# ─── add_sync — unavailable ───────────────────────────────────────────────────


class TestVectorStoreAddSyncUnavailable:
    """add_sync() returns 0 when ChromaDB is unavailable."""

    def test_returns_0_when_unavailable(self):
        store = VectorStore()
        store._checked_available = False
        chunks = [make_chunk()]
        embs = [make_embedding()]
        result = store.add_sync(chunks, embs)
        assert result == 0

    def test_returns_0_for_empty_chunks(self):
        store = VectorStore()
        result = store.add_sync([], [])
        assert result == 0

    def test_returns_0_on_length_mismatch(self):
        store = VectorStore()
        store._checked_available = True
        # Set up collection to bypass _ensure_collection
        mock_collection = MagicMock()
        store._collection = mock_collection
        chunks = [make_chunk(), make_chunk()]
        embs = [make_embedding()]  # length mismatch
        result = store.add_sync(chunks, embs)
        assert result == 0

    def test_returns_0_for_all_empty_embeddings(self):
        store = VectorStore()
        store._checked_available = True
        mock_collection = MagicMock()
        store._collection = mock_collection
        chunks = [make_chunk()]
        embs = [[]]  # empty embedding
        result = store.add_sync(chunks, embs)
        assert result == 0


# ─── add_sync — mocked ChromaDB ──────────────────────────────────────────────


class TestVectorStoreAddSyncMocked:
    """add_sync() with a mocked ChromaDB collection."""

    def _make_store_with_mock_collection(self) -> tuple[VectorStore, MagicMock]:
        store = VectorStore()
        store._checked_available = True
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        store._collection = mock_collection
        return store, mock_collection

    def test_adds_chunks_successfully(self):
        store, mock_col = self._make_store_with_mock_collection()
        chunks = [make_chunk(content="text", chunk_id="c1")]
        embs = [make_embedding(dim=4)]
        result = store.add_sync(chunks, embs)
        assert result == 1
        mock_col.upsert.assert_called_once()

    def test_upsert_called_with_correct_ids(self):
        store, mock_col = self._make_store_with_mock_collection()
        chunks = [make_chunk(chunk_id="my_id")]
        embs = [make_embedding()]
        store.add_sync(chunks, embs)
        call_kwargs = mock_col.upsert.call_args.kwargs
        assert "my_id" in call_kwargs["ids"]

    def test_upsert_called_with_correct_documents(self):
        store, mock_col = self._make_store_with_mock_collection()
        chunks = [make_chunk(content="my content", chunk_id="x")]
        embs = [make_embedding()]
        store.add_sync(chunks, embs)
        call_kwargs = mock_col.upsert.call_args.kwargs
        assert "my content" in call_kwargs["documents"]

    def test_multiple_chunks_added(self):
        store, mock_col = self._make_store_with_mock_collection()
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(3)]
        embs = [make_embedding() for _ in range(3)]
        result = store.add_sync(chunks, embs)
        assert result == 3

    def test_upsert_exception_returns_0(self):
        store, mock_col = self._make_store_with_mock_collection()
        mock_col.upsert.side_effect = RuntimeError("DB error")
        chunks = [make_chunk()]
        embs = [make_embedding()]
        result = store.add_sync(chunks, embs)
        assert result == 0


# ─── query_sync — mocked ChromaDB ────────────────────────────────────────────


class TestVectorStoreQuerySyncMocked:
    """query_sync() with a mocked ChromaDB collection."""

    def _make_store_with_query_results(
        self,
        ids=None, docs=None, metas=None, distances=None
    ) -> VectorStore:
        store = VectorStore()
        store._checked_available = True
        mock_collection = MagicMock()

        ids = ids or ["id1"]
        docs = docs or ["some content"]
        metas = metas or [{"source": "test.txt", "source_type": "text",
                           "chunk_index": "0", "total_chunks": "1", "tags": ""}]
        distances = distances or [0.2]

        mock_collection.count.return_value = len(ids)
        mock_collection.query.return_value = {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
        }
        store._collection = mock_collection
        return store

    def test_returns_empty_for_empty_embedding(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        store._collection = mock_col
        result = store.query_sync(embedding=[])
        assert result == []

    def test_returns_empty_when_unavailable(self):
        store = VectorStore()
        store._checked_available = False
        result = store.query_sync(embedding=[0.1, 0.2])
        assert result == []

    def test_returns_knowledge_chunks(self):
        store = self._make_store_with_query_results()
        results = store.query_sync(embedding=make_embedding())
        assert len(results) == 1
        assert isinstance(results[0], KnowledgeChunk)

    def test_score_computed_from_distance(self):
        # distance=0.2 → score = 1 - 0.2 = 0.8
        store = self._make_store_with_query_results(distances=[0.2])
        results = store.query_sync(embedding=make_embedding())
        assert abs(results[0].score - 0.8) < 1e-6

    def test_min_score_filters_low_scores(self):
        # distance=0.8 → score=0.2, filter at min_score=0.5
        store = self._make_store_with_query_results(distances=[0.8])
        results = store.query_sync(embedding=make_embedding(), min_score=0.5)
        assert results == []

    def test_min_score_passes_high_scores(self):
        # distance=0.1 → score=0.9
        store = self._make_store_with_query_results(distances=[0.1])
        results = store.query_sync(embedding=make_embedding(), min_score=0.5)
        assert len(results) == 1

    def test_query_exception_returns_empty(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.side_effect = RuntimeError("query failed")
        store._collection = mock_col
        result = store.query_sync(embedding=make_embedding())
        assert result == []


# ─── delete_source_sync ───────────────────────────────────────────────────────


class TestVectorStoreDeleteSourceSync:
    """delete_source_sync() removes chunks by source."""

    def test_returns_0_when_unavailable(self):
        store = VectorStore()
        store._checked_available = False
        assert store.delete_source_sync("test.txt") == 0

    def test_deletes_matching_chunks(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["id1", "id2"]}
        store._collection = mock_col
        result = store.delete_source_sync("test.txt")
        assert result == 2
        mock_col.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_returns_0_when_no_matching_chunks(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        store._collection = mock_col
        result = store.delete_source_sync("nonexistent.txt")
        assert result == 0


# ─── count_sync / list_sources_sync ──────────────────────────────────────────


class TestVectorStoreManagement:
    """count_sync() and list_sources_sync() management operations."""

    def test_count_sync_returns_0_when_unavailable(self):
        store = VectorStore()
        store._checked_available = False
        assert store.count_sync() == 0

    def test_count_sync_returns_collection_count(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        mock_col.count.return_value = 42
        store._collection = mock_col
        assert store.count_sync() == 42

    def test_list_sources_returns_empty_when_unavailable(self):
        store = VectorStore()
        store._checked_available = False
        assert store.list_sources_sync() == []

    def test_list_sources_deduplicates(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "doc_a.txt"},
                {"source": "doc_b.txt"},
                {"source": "doc_a.txt"},  # duplicate
            ]
        }
        store._collection = mock_col
        sources = store.list_sources_sync()
        assert len(sources) == 2
        assert "doc_a.txt" in sources
        assert "doc_b.txt" in sources

    def test_list_sources_sorted(self):
        store = VectorStore()
        store._checked_available = True
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "zebra.txt"},
                {"source": "alpha.txt"},
            ]
        }
        store._collection = mock_col
        sources = store.list_sources_sync()
        assert sources == sorted(sources)


# ─── Async wrappers ───────────────────────────────────────────────────────────


class TestVectorStoreAsync:
    """Async wrapper methods delegate to sync implementations."""

    @pytest.mark.asyncio
    async def test_count_async(self):
        store = VectorStore()
        store._checked_available = False
        count = await store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_list_sources_async(self):
        store = VectorStore()
        store._checked_available = False
        sources = await store.list_sources()
        assert sources == []

    @pytest.mark.asyncio
    async def test_add_async(self):
        store = VectorStore()
        store._checked_available = False
        result = await store.add([], [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_query_async(self):
        store = VectorStore()
        store._checked_available = False
        result = await store.query(embedding=[0.1, 0.2])
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_source_async(self):
        store = VectorStore()
        store._checked_available = False
        result = await store.delete_source("source")
        assert result == 0
