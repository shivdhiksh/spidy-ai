"""
Spidy Knowledge Engine — M10
==============================
Personal knowledge retrieval: document ingestion, chunking, embedding,
semantic retrieval, and optional web search.

Public API
----------
    from spidy.knowledge import KnowledgeManager
    from spidy.knowledge.types import KnowledgeChunk, KnowledgeResult
    from spidy.knowledge.events import (
        KnowledgeDocumentIngestedEvent,
        KnowledgeQueriedEvent,
        KnowledgeErrorEvent,
    )

Architecture
------------
    KnowledgeManager
    ├── DocumentIngestor   PDF / DOCX / Markdown / plain-text
    ├── Chunker            fixed-size + overlap splitting
    ├── EmbeddingEngine    sentence-transformers (lazy init)
    ├── VectorStore        ChromaDB "spidy_knowledge" collection
    └── WebSearchEngine    DuckDuckGo (optional, feature-flagged)

Implements the ``KnowledgeInterface`` contract from ``spidy.brain.interfaces``.
"""

from spidy.knowledge.events import (
    KnowledgeChunkStoredEvent,
    KnowledgeDocumentIngestedEvent,
    KnowledgeErrorEvent,
    KnowledgeQueriedEvent,
    KnowledgeSourceDeletedEvent,
    KnowledgeWebSearchPerformedEvent,
)
from spidy.knowledge.manager import KnowledgeManager
from spidy.knowledge.types import KnowledgeChunk, KnowledgeResult, SourceType

__all__ = [
    # Manager
    "KnowledgeManager",
    # Types
    "KnowledgeChunk",
    "KnowledgeResult",
    "SourceType",
    # Events
    "KnowledgeDocumentIngestedEvent",
    "KnowledgeChunkStoredEvent",
    "KnowledgeQueriedEvent",
    "KnowledgeWebSearchPerformedEvent",
    "KnowledgeSourceDeletedEvent",
    "KnowledgeErrorEvent",
]
