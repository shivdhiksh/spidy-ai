"""
Knowledge Engine — Vector Store
=================================
ChromaDB-backed vector store for knowledge chunks.

Architecture
------------
- Uses the ChromaDB ``spidy_knowledge`` collection (separate from
  ``spidy_memories`` used by the M8 Memory Engine).
- Stores chunk content, embeddings, and metadata together.
- Supports metadata-filtered top-k retrieval.
- Graceful degrade: all operations return empty results when ChromaDB
  is absent.  ``is_available`` signals whether the dep is installed.

Dependencies
------------
    chromadb>=0.5.3    (already in pyproject.toml[memory])
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from spidy.knowledge.types import KnowledgeChunk, SourceType
from spidy.logging.logger import get_logger

log = get_logger(__name__)

_DEFAULT_COLLECTION = "spidy_knowledge"


class VectorStore:
    """
    ChromaDB vector store for KnowledgeChunks.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection. Default ``spidy_knowledge``.
    persist_dir:
        Directory for ChromaDB persistent storage.
        Pass ``None`` for in-memory (ephemeral, for tests).
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        persist_dir: Path | str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._client: Any = None
        self._collection: Any = None
        self._checked_available: bool | None = None

    # ── Availability ──────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if chromadb is installed."""
        if self._checked_available is None:
            try:
                import chromadb  # noqa: F401
                self._checked_available = True
            except ImportError:
                self._checked_available = False
        return self._checked_available

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _ensure_collection(self) -> bool:
        """
        Initialise ChromaDB client and collection on first use.

        Returns True on success; False if ChromaDB unavailable or error.
        """
        if self._collection is not None:
            return True
        if not self.is_available:
            return False
        try:
            import chromadb

            if self._persist_dir is not None:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir)
                )
            else:
                self._client = chromadb.EphemeralClient()

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                "VectorStore: collection '{col}' ready ({n} chunks).",
                col=self._collection_name,
                n=self._collection.count(),
            )
            return True
        except Exception as exc:
            log.warning(
                "VectorStore: failed to initialise ChromaDB: {exc}", exc=exc
            )
            return False

    async def initialize(self) -> bool:
        """Async-safe initialisation (runs _ensure_collection in executor)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ensure_collection)

    async def close(self) -> None:
        """Flush and close the ChromaDB client (if persistent)."""
        if self._client is not None:
            try:
                # ChromaDB PersistentClient auto-persists on mutation;
                # explicit close is a best-effort flush.
                if hasattr(self._client, "_server") and hasattr(
                    self._client._server, "stop"
                ):
                    self._client._server.stop()
            except Exception:
                pass
            finally:
                self._client = None
                self._collection = None

    # ── Write operations ──────────────────────────────────────────────────

    def add_sync(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add chunks and their embeddings to the collection synchronously.

        Parameters
        ----------
        chunks:
            KnowledgeChunks to store.
        embeddings:
            Pre-computed embedding vectors; must match ``len(chunks)``.

        Returns
        -------
        int
            Number of chunks actually added (0 if unavailable or error).
        """
        if not chunks or not embeddings:
            return 0
        if len(chunks) != len(embeddings):
            log.warning(
                "VectorStore.add: chunks ({c}) and embeddings ({e}) length mismatch",
                c=len(chunks),
                e=len(embeddings),
            )
            return 0
        if not self._ensure_collection():
            return 0

        # Filter out empty embeddings
        valid_pairs = [
            (c, e) for c, e in zip(chunks, embeddings) if e
        ]
        if not valid_pairs:
            log.warning("VectorStore.add: all embeddings were empty — skipping")
            return 0

        ids = [p[0].chunk_id for p in valid_pairs]
        docs = [p[0].content for p in valid_pairs]
        embs = [p[1] for p in valid_pairs]
        metas = [
            {
                "source": p[0].source,
                "source_type": p[0].source_type,
                "chunk_index": p[0].chunk_index,
                "total_chunks": p[0].total_chunks,
                "tags": ",".join(p[0].tags),
                **{
                    k: str(v)
                    for k, v in p[0].metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            for p in valid_pairs
        ]

        try:
            self._collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embs,
                metadatas=metas,
            )
            log.debug(
                "VectorStore: added {n} chunks to '{col}'",
                n=len(valid_pairs),
                col=self._collection_name,
            )
            return len(valid_pairs)
        except Exception as exc:
            log.warning("VectorStore.add: failed: {exc}", exc=exc)
            return 0

    async def add(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Async version of ``add_sync``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.add_sync, chunks, embeddings)

    # ── Read operations ───────────────────────────────────────────────────

    def query_sync(
        self,
        embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[KnowledgeChunk]:
        """
        Top-k similarity search synchronously.

        Parameters
        ----------
        embedding:
            Query embedding vector.
        n_results:
            Maximum number of results to return.
        where:
            Optional ChromaDB metadata filter dict.
        min_score:
            Minimum cosine similarity score (0.0 = no filter).

        Returns
        -------
        list[KnowledgeChunk]
            Ranked results with ``score`` field populated.
            Empty list on unavailability or error.
        """
        if not embedding or not self._ensure_collection():
            return []

        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": min(n_results, max(1, self._collection.count())),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = self._collection.query(**kwargs)

            chunks: list[KnowledgeChunk] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids_result = results.get("ids", [[]])[0]

            for chunk_id, doc, meta, dist in zip(ids_result, docs, metas, distances):
                # ChromaDB cosine distance → similarity: score = 1 - distance
                score = max(0.0, 1.0 - float(dist))
                if score < min_score:
                    continue
                source_type: SourceType = meta.get("source_type", "unknown")  # type: ignore[assignment]
                tags = [t for t in meta.get("tags", "").split(",") if t]
                chunk = KnowledgeChunk(
                    chunk_id=chunk_id,
                    content=doc,
                    source=meta.get("source", ""),
                    source_type=source_type,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    metadata=dict(meta),
                    tags=tags,
                    score=score,
                )
                chunks.append(chunk)

            return chunks
        except Exception as exc:
            log.warning("VectorStore.query: failed: {exc}", exc=exc)
            return []

    async def query(
        self,
        embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[KnowledgeChunk]:
        """Async version of ``query_sync``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.query_sync, embedding, n_results, where, min_score
        )

    # ── Management operations ──────────────────────────────────────────────

    def delete_source_sync(self, source: str) -> int:
        """
        Delete all chunks with the given source from the collection.

        Returns the number of chunks deleted.
        """
        if not self._ensure_collection():
            return 0
        try:
            results = self._collection.get(
                where={"source": source},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            log.debug(
                "VectorStore: deleted {n} chunks for source '{src}'",
                n=len(ids),
                src=source[:60],
            )
            return len(ids)
        except Exception as exc:
            log.warning(
                "VectorStore.delete_source: failed: {exc}", exc=exc
            )
            return 0

    async def delete_source(self, source: str) -> int:
        """Async version of ``delete_source_sync``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete_source_sync, source)

    def count_sync(self) -> int:
        """Return the number of chunks currently in the collection."""
        if not self._ensure_collection():
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    async def count(self) -> int:
        """Async version of ``count_sync``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.count_sync)

    def list_sources_sync(self) -> list[str]:
        """Return a de-duplicated list of all source identifiers in the store."""
        if not self._ensure_collection():
            return []
        try:
            results = self._collection.get(include=["metadatas"])
            sources: set[str] = set()
            for meta in results.get("metadatas", []):
                src = meta.get("source", "")
                if src:
                    sources.add(src)
            return sorted(sources)
        except Exception as exc:
            log.warning("VectorStore.list_sources: failed: {exc}", exc=exc)
            return []

    async def list_sources(self) -> list[str]:
        """Async version of ``list_sources_sync``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.list_sources_sync)
