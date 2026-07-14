"""
Knowledge Events — EventBus Events for the Knowledge Engine
============================================================
Topic namespace: ``knowledge.*``

All event types are frozen dataclasses extending ``Event``.
Subscribers can listen to specific operations or the entire namespace.

Topics
------
knowledge.document_ingested  — a document was fully ingested (all chunks stored)
knowledge.chunk_stored       — an individual chunk was stored in the vector store
knowledge.queried            — a knowledge query was executed
knowledge.web_search_performed — a web search was performed
knowledge.source_deleted     — all chunks for a source were removed
knowledge.error              — any knowledge-layer error occurred
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


# ─── Ingestion Events ─────────────────────────────────────────────────────────


@dataclass
class KnowledgeDocumentIngestedEvent(Event):
    """Published when a document has been fully ingested (all chunks stored)."""

    topic = "knowledge.document_ingested"
    source: str = ""                     # file path or URL
    source_type: str = ""                # "pdf" | "docx" | "markdown" | "text" | "url"
    chunk_count: int = 0                 # number of chunks stored
    tags: list[str] = field(default_factory=list)


@dataclass
class KnowledgeChunkStoredEvent(Event):
    """Published when an individual chunk is stored in the vector store."""

    topic = "knowledge.chunk_stored"
    chunk_id: str = ""
    source: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    content_preview: str = ""           # first 80 chars of content


# ─── Query Events ─────────────────────────────────────────────────────────────


@dataclass
class KnowledgeQueriedEvent(Event):
    """Published when a knowledge query is executed."""

    topic = "knowledge.queried"
    query_preview: str = ""             # first 80 chars of the query
    result_count: int = 0
    used_web_search: bool = False
    best_score: float = 0.0


# ─── Web Search Events ────────────────────────────────────────────────────────


@dataclass
class KnowledgeWebSearchPerformedEvent(Event):
    """Published when a web search is performed for knowledge retrieval."""

    topic = "knowledge.web_search_performed"
    query_preview: str = ""
    provider: str = "duckduckgo"
    result_count: int = 0


# ─── Management Events ────────────────────────────────────────────────────────


@dataclass
class KnowledgeSourceDeletedEvent(Event):
    """Published when all chunks for a source are removed from the store."""

    topic = "knowledge.source_deleted"
    source: str = ""
    chunks_deleted: int = 0


# ─── Error Events ─────────────────────────────────────────────────────────────


@dataclass
class KnowledgeErrorEvent(Event):
    """Published when any knowledge-layer error occurs."""

    topic = "knowledge.error"
    operation: str = ""                  # "ingest" | "query" | "delete" | "search"
    source: str = ""
    error: str = ""
