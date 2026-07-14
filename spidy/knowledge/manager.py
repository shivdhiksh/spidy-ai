"""
Knowledge Manager — Unified API for the Knowledge Engine
=========================================================
``KnowledgeManager`` is the single entry point for all knowledge operations.
It implements the ``KnowledgeInterface`` contract defined in M3 and
orchestrates all sub-components behind one clean API.

Architecture
------------
    KnowledgeManager  (KnowledgeInterface)
        ├── DocumentIngestor   PDF / DOCX / Markdown / plain-text
        ├── Chunker            fixed-size + overlap splitting
        ├── EmbeddingEngine    sentence-transformers (lazy init)
        ├── VectorStore        ChromaDB collection "spidy_knowledge"
        └── WebSearchEngine    DuckDuckGo (optional, feature-flagged)

Brain API
---------
The Brain's primary call is ``search_rag(query, limit)`` which:
1. Embeds the query
2. Retrieves top-k chunks from the vector store
3. Optionally augments with web search results (if enabled)
4. Returns a formatted string ready for LLM context injection

EventBus Integration
--------------------
Every public operation publishes a typed ``knowledge.*`` event.
EventBus is optional — tests can run without it.

Graceful Degradation
--------------------
If any optional dep (chromadb, sentence-transformers) is missing:
- ingest_file() still parses and chunks the document; chunks are stored
  as raw text with no embeddings (limited retrieval).
- query() / search_rag() return empty results.
- The Brain continues normally — knowledge slot is not None (the manager
  exists) but returns no context.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spidy.brain.interfaces import KnowledgeInterface
from spidy.knowledge.chunker import Chunker
from spidy.knowledge.embedding import EmbeddingEngine
from spidy.knowledge.events import (
    KnowledgeChunkStoredEvent,
    KnowledgeDocumentIngestedEvent,
    KnowledgeErrorEvent,
    KnowledgeQueriedEvent,
    KnowledgeSourceDeletedEvent,
    KnowledgeWebSearchPerformedEvent,
)
from spidy.knowledge.ingestor import DocumentIngestor
from spidy.knowledge.store import VectorStore
from spidy.knowledge.types import KnowledgeChunk, KnowledgeResult
from spidy.knowledge.web_search import WebSearchEngine
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class KnowledgeManager(KnowledgeInterface):
    """
    Unified knowledge API — implements KnowledgeInterface.

    Parameters
    ----------
    config:
        ``KnowledgeConfig`` from ``SpidyConfig.knowledge``.
        Accepts None (all defaults used).
    bus:
        Optional EventBus for publishing knowledge events.
    knowledge_dir:
        Directory for ChromaDB vector store persistence.
        None = in-memory (ephemeral, for tests).
    """

    def __init__(
        self,
        config: Any | None = None,
        bus: "EventBus | None" = None,
        knowledge_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._knowledge_dir = knowledge_dir

        # ── Pull config values (with fallbacks) ───────────────────────────
        self._chunk_size: int = getattr(config, "chunk_size", 1000)
        self._chunk_overlap: int = getattr(config, "chunk_overlap", 200)
        self._embedding_model: str = getattr(config, "embedding_model", "all-MiniLM-L6-v2")
        self._collection_name: str = getattr(config, "collection_name", "spidy_knowledge")
        self._min_score: float = getattr(config, "min_relevance_score", 0.3)
        self._max_results: int = getattr(config, "max_results", 5)

        web_cfg = getattr(config, "web_search", None)
        self._web_enabled: bool = getattr(web_cfg, "enabled", False)
        self._web_max_results: int = getattr(web_cfg, "max_results", 5)
        self._web_safe_search: bool = getattr(web_cfg, "safe_search", True)

        # ── Sub-components ────────────────────────────────────────────────
        self._ingestor = DocumentIngestor()
        self._chunker = Chunker(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self._embedder = EmbeddingEngine(model_name=self._embedding_model)
        self._store = VectorStore(
            collection_name=self._collection_name,
            persist_dir=self._knowledge_dir,
        )
        self._web_search = WebSearchEngine(
            enabled=self._web_enabled,
            max_results=self._web_max_results,
            safe_search=self._web_safe_search,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialise sub-components.

        Safe to call multiple times (idempotent).
        Heavy I/O (ChromaDB open, model load) is deferred to first use.
        """
        await self._store.initialize()
        log.info(
            "KnowledgeManager ready | "
            "embeddings={emb} | store={store} | web={web}",
            emb=self._embedder.is_available,
            store=self._store.is_available,
            web=self._web_search.is_available,
        )

    async def close(self) -> None:
        """Flush and release resources."""
        await self._store.close()
        log.info("KnowledgeManager closed.")

    # ── KnowledgeInterface implementation ─────────────────────────────────

    async def query(
        self,
        question: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query the knowledge base and return structured result dicts.

        Returns
        -------
        list[dict]
            Each dict: ``{"content": str, "source": str, "score": float,
            "source_type": str, "metadata": dict}``.
        """
        result = await self._retrieve(question, limit=max_results)
        await self._publish(
            KnowledgeQueriedEvent(
                query_preview=question[:80],
                result_count=len(result.chunks),
                used_web_search=result.used_web_search,
                best_score=result.best_score,
            )
        )
        return [
            {
                "content": c.content,
                "source": c.source,
                "score": c.score,
                "source_type": c.source_type,
                "metadata": dict(c.metadata),
            }
            for c in result.chunks
        ]

    async def ingest(
        self,
        content: str,
        source: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add raw text content to the knowledge base.

        Returns the chunk_id of the first stored chunk (or empty string
        on failure).
        """
        try:
            chunks = self._chunker.chunk(
                text=content,
                source=source,
                source_type="text",
                metadata=metadata,
                tags=tags,
            )
            if not chunks:
                return ""

            n = await self._store_chunks(chunks, source=source)

            if n > 0:
                await self._publish(
                    KnowledgeDocumentIngestedEvent(
                        source=source,
                        source_type="text",
                        chunk_count=n,
                        tags=tags or [],
                    )
                )
            return chunks[0].chunk_id
        except Exception as exc:
            log.warning(
                "KnowledgeManager.ingest: error for source '{src}': {exc}",
                src=source,
                exc=exc,
            )
            await self._publish(
                KnowledgeErrorEvent(
                    operation="ingest",
                    source=source,
                    error=str(exc),
                )
            )
            return ""

    async def search_rag(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Retrieve knowledge and format for LLM prompt injection (RAG).

        This is the primary method the Brain calls during ``process()``.

        Returns
        -------
        str
            Formatted context block ready for LLM injection.
            Empty string if no relevant knowledge found.
        """
        result = await self._retrieve(query, limit=limit)
        if result.is_empty:
            return ""
        return result.as_rag_context(max_chunks=limit)

    # ── Document ingestion ─────────────────────────────────────────────────

    async def ingest_file(
        self,
        path: Path | str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest a document file into the knowledge base.

        Parameters
        ----------
        path:
            Path to a PDF, DOCX, Markdown, or plain text file.
        tags:
            Optional tags for retrieval filtering.
        metadata:
            Additional metadata to attach to all chunks.

        Returns
        -------
        int
            Number of chunks ingested. 0 on error or empty file.
        """
        path = Path(path)
        source = str(path)
        source_type = self._ingestor.get_source_type(path)

        # Merge file metadata with caller metadata
        file_meta = self._ingestor.get_metadata(path)
        combined_meta = {**file_meta, **(metadata or {})}

        try:
            pages = self._ingestor.read(path)
            if not pages:
                log.warning(
                    "KnowledgeManager.ingest_file: no text extracted from '{path}'",
                    path=path.name,
                )
                return 0

            chunks = self._chunker.chunk_pages(
                pages=pages,
                source=source,
                source_type=source_type,  # type: ignore[arg-type]
                metadata=combined_meta,
                tags=tags,
            )
            if not chunks:
                return 0

            n = await self._store_chunks(chunks, source=source)

            if n > 0:
                await self._publish(
                    KnowledgeDocumentIngestedEvent(
                        source=source,
                        source_type=source_type,
                        chunk_count=n,
                        tags=tags or [],
                    )
                )
                log.info(
                    "KnowledgeManager: ingested '{path}' → {n} chunks",
                    path=path.name,
                    n=n,
                )
            return n

        except Exception as exc:
            log.warning(
                "KnowledgeManager.ingest_file: failed '{path}': {exc}",
                path=path,
                exc=exc,
            )
            await self._publish(
                KnowledgeErrorEvent(
                    operation="ingest",
                    source=source,
                    error=str(exc),
                )
            )
            return 0

    # ── Management ─────────────────────────────────────────────────────────

    async def delete_source(self, source: str) -> int:
        """
        Remove all chunks for a given source from the knowledge base.

        Returns the number of chunks deleted.
        """
        try:
            n = await self._store.delete_source(source)
            await self._publish(
                KnowledgeSourceDeletedEvent(source=source, chunks_deleted=n)
            )
            return n
        except Exception as exc:
            log.warning(
                "KnowledgeManager.delete_source: failed '{src}': {exc}",
                src=source,
                exc=exc,
            )
            return 0

    async def count(self) -> int:
        """Total number of chunks in the knowledge base."""
        return await self._store.count()

    async def list_sources(self) -> list[str]:
        """All unique source identifiers in the knowledge base."""
        return await self._store.list_sources()

    # ── Private helpers ────────────────────────────────────────────────────

    async def _retrieve(
        self,
        query: str,
        limit: int | None = None,
    ) -> KnowledgeResult:
        """
        Core retrieval pipeline:
        1. Embed the query
        2. Query the vector store
        3. Optionally augment with web search
        4. Merge and rank results
        """
        n = limit or self._max_results
        doc_chunks: list[KnowledgeChunk] = []
        web_chunks: list[KnowledgeChunk] = []
        used_web = False

        # Step 1: embed the query
        query_embedding = await self._embedder.embed_one(query)

        # Step 2: vector store retrieval (if embedding succeeded)
        if query_embedding:
            doc_chunks = await self._store.query(
                embedding=query_embedding,
                n_results=n,
                min_score=self._min_score,
            )

        # Step 3: web search augmentation (optional)
        if self._web_search.is_available and len(doc_chunks) < n:
            remaining = n - len(doc_chunks)
            web_chunks = await self._web_search.search(
                query=query,
                max_results=remaining,
            )
            if web_chunks:
                used_web = True
                await self._publish(
                    KnowledgeWebSearchPerformedEvent(
                        query_preview=query[:80],
                        result_count=len(web_chunks),
                        provider=self._web_search.provider_name,
                    )
                )

        # Merge and rank: doc results first (higher confidence), then web
        all_chunks = doc_chunks + web_chunks
        total = len(all_chunks)

        # Sort by score descending, limit to n
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        all_chunks = all_chunks[:n]

        return KnowledgeResult(
            query=query,
            chunks=all_chunks,
            used_web_search=used_web,
            total_found=total,
        )

    async def _store_chunks(
        self,
        chunks: list[KnowledgeChunk],
        source: str = "",
    ) -> int:
        """
        Embed and store a list of chunks.

        Returns the number of chunks successfully stored.
        """
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed(texts)

        n = await self._store.add(chunks=chunks, embeddings=embeddings)

        # Publish per-chunk events (throttled: only first 5)
        for chunk in chunks[:5]:
            await self._publish(
                KnowledgeChunkStoredEvent(
                    chunk_id=chunk.chunk_id,
                    source=source,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    content_preview=chunk.content_preview,
                )
            )
        return n

    async def _publish(self, event: Any) -> None:
        """Publish an event to the EventBus (silently skip if no bus)."""
        if self._bus is None:
            return
        try:
            await self._bus.publish(event)
        except Exception as exc:
            log.debug(
                "KnowledgeManager: event publish failed (non-fatal): {exc}",
                exc=exc,
            )
