"""
Semantic Memory — ChromaDB Vector Store for Similarity Search
=============================================================
Provides embedding-based semantic similarity search across all stored memories.

Architecture
------------
- Vector store: ``chromadb`` (persistent client, EphemeralClient for tests)
- Embeddings: ``sentence-transformers`` (all-MiniLM-L6-v2 by default)
- Persistent path: ``<memory_dir>/chromadb/``

Graceful Degradation
--------------------
If either ``chromadb`` or ``sentence_transformers`` is not installed,
SemanticMemory becomes a **complete no-op**:
- ``store()`` returns the entry ID without embedding
- ``search()`` returns an empty list
- ``delete()`` / ``clear()`` return False / 0 with no side effects
A single warning is logged at startup; no exceptions are raised.

This design lets the rest of the application (including all 914 existing
tests) run normally without any vector dependencies.

Design Decisions
----------------
1. Lazy init — ChromaDB client + embedding model are created on the first
   ``store()`` call, not at import time. This avoids ~2s model load delay
   on startup.
2. EphemeralClient for `:memory:` paths — enables fast in-process tests
   without touching disk.
3. Metadata stored alongside embeddings so we can reconstruct MemoryEntry
   from a ChromaDB result without a second DB query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spidy.logging.logger import get_logger
from spidy.memory.types import MemoryEntry, MemorySearchResult, MemoryType

log = get_logger(__name__)

_WARN_ISSUED = False  # Module-level flag so we warn exactly once


class SemanticMemory:
    """
    ChromaDB-backed semantic (vector similarity) memory.

    Parameters
    ----------
    collection_name:
        ChromaDB collection identifier.
    embedding_model:
        Sentence-Transformers model name for generating embeddings.
    persist_dir:
        Directory where ChromaDB persists data.
        Pass ``:memory:`` to use an ephemeral (in-process) client for tests.
    """

    def __init__(
        self,
        collection_name: str = "spidy_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str | Path = ":memory:",
        device: str = "cpu",
    ) -> None:
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._persist_dir = str(persist_dir)
        self._device = device
        self._available: bool | None = None   # None = not yet probed
        self._client: Any = None
        self._collection: Any = None
        self._encoder: Any = None

    # ── Availability probe (lazy) ──────────────────────────────────────────

    def _probe_availability(self) -> bool:
        """Check once if chromadb + sentence-transformers are importable."""
        global _WARN_ISSUED
        if self._available is not None:
            return self._available

        try:
            import chromadb  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self._available = True
        except ImportError as exc:
            self._available = False
            if not _WARN_ISSUED:
                log.warning(
                    "SemanticMemory: vector search is unavailable — {exc}. "
                    "Install optional dependencies: pip install chromadb sentence-transformers",
                    exc=exc,
                )
                _WARN_ISSUED = True

        return self._available

    def _ensure_initialized(self) -> bool:
        """
        Lazily initialise ChromaDB client + collection + embedding model.

        Returns
        -------
        bool
            True if successfully initialised, False if unavailable.
        """
        if not self._probe_availability():
            return False

        if self._client is not None:
            return True  # Already initialised

        try:
            from spidy.core.chroma import ChromaClientRegistry
            from spidy.core.models import EmbeddingModelRegistry

            self._client = ChromaClientRegistry.get_client(self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._encoder = EmbeddingModelRegistry.get_model(
                self._embedding_model_name,
                device=self._device,
            )
            log.info(
                "SemanticMemory: ChromaDB ready | collection='{col}' | model='{model}'",
                col=self._collection_name,
                model=self._embedding_model_name,
            )
            return True

        except Exception as exc:
            log.error(
                "SemanticMemory: failed to initialise ChromaDB: {exc}",
                exc=exc,
            )
            self._available = False
            return False

    # ── Public API ──────────────────────────────────────────────────────────

    async def store(self, entry: MemoryEntry) -> str:
        """
        Embed and store a memory entry in ChromaDB.

        If ChromaDB is unavailable, returns the entry ID without storing.

        Parameters
        ----------
        entry:
            The memory to embed and store.

        Returns
        -------
        str
            The entry ID.
        """
        if not self._ensure_initialized():
            return entry.id

        try:
            embedding = self._encoder.encode(entry.content).tolist()
            self._collection.upsert(
                ids=[entry.id],
                embeddings=[embedding],
                documents=[entry.content],
                metadatas=[{
                    "session_id": entry.session_id,
                    "tags": ",".join(entry.tags),
                    "timestamp": entry.timestamp.timestamp(),
                    "memory_type": entry.memory_type.value,
                    **{k: str(v) for k, v in entry.metadata.items()},
                }],
            )
            log.debug("SemanticMemory: embedded and stored {id}", id=entry.id[:8])
        except Exception as exc:
            log.error("SemanticMemory: store failed: {exc}", exc=exc)

        return entry.id

    async def search(
        self,
        query: str,
        limit: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """
        Perform cosine-similarity search over all stored embeddings.

        Parameters
        ----------
        query:
            Natural language query to embed and compare against.
        limit:
            Maximum number of results.
        tags:
            If provided, only memories tagged with at least one of these are returned.

        Returns
        -------
        list[MemorySearchResult]
            Results sorted by descending similarity score.
        """
        if not self._ensure_initialized():
            return []

        try:
            query_embedding = self._encoder.encode(query).tolist()
            where_filter: dict | None = None
            if tags:
                # ChromaDB metadata filter — tags stored as comma-separated string
                # We can only filter exact matches on metadata; do post-filter for tags
                pass  # tags post-filtered below

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 3 if tags else limit, max(1, self._collection.count())),
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            output: list[MemorySearchResult] = []
            from datetime import datetime, timezone

            for mem_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB returns cosine distance (0=identical, 2=opposite)
                # Convert to similarity score (1=identical, 0=opposite)
                score = max(0.0, 1.0 - dist)

                tag_list = [t for t in meta.get("tags", "").split(",") if t]

                # Post-filter by tags if requested
                if tags and not any(t in tag_list for t in tags):
                    continue

                ts = float(meta.get("timestamp", 0))
                mem_type_str = meta.get("memory_type", "episodic")

                entry = MemoryEntry(
                    id=mem_id,
                    content=doc,
                    session_id=meta.get("session_id", ""),
                    tags=tuple(tag_list),
                    metadata={
                        k: v for k, v in meta.items()
                        if k not in {"session_id", "tags", "timestamp", "memory_type"}
                    },
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc),
                    memory_type=MemoryType(mem_type_str),
                )
                output.append(MemorySearchResult(entry=entry, score=score))

                if len(output) >= limit:
                    break

            return output

        except Exception as exc:
            log.error("SemanticMemory: search failed: {exc}", exc=exc)
            return []

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory entry from the vector store.

        Returns
        -------
        bool
            True if deleted, False if unavailable or not found.
        """
        if not self._ensure_initialized():
            return False

        try:
            self._collection.delete(ids=[memory_id])
            log.debug("SemanticMemory: deleted {id}", id=memory_id[:8])
            return True
        except Exception as exc:
            log.error("SemanticMemory: delete failed: {exc}", exc=exc)
            return False

    async def clear(self, session_id: str = "") -> int:
        """
        Clear memories. If session_id is given, only that session is cleared.
        If empty, the entire collection is reset.

        Returns
        -------
        int
            Approximate number of entries cleared (0 if unavailable).
        """
        if not self._ensure_initialized():
            return 0

        try:
            if session_id:
                # Query by session_id metadata and delete those IDs
                results = self._collection.get(
                    where={"session_id": session_id},
                    include=[],
                )
                ids = results.get("ids", [])
                if ids:
                    self._collection.delete(ids=ids)
                return len(ids)
            else:
                count = self._collection.count()
                # Delete and recreate collection to clear everything
                self._client.delete_collection(self._collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return count
        except Exception as exc:
            log.error("SemanticMemory: clear failed: {exc}", exc=exc)
            return 0

    def count(self) -> int:
        """Return the number of vectors stored (0 if unavailable)."""
        if not self._ensure_initialized():
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    @property
    def is_available(self) -> bool:
        """True if ChromaDB + sentence-transformers are installed."""
        return self._probe_availability()
