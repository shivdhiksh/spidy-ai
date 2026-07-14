"""
Knowledge Engine — Core Types
==============================
Immutable dataclasses shared across all Knowledge Engine components.

SourceType   — literal string enum for document source classification
KnowledgeChunk — a single piece of text extracted from a document or search
KnowledgeResult — container for a complete query result set
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


# ─── Source Type ──────────────────────────────────────────────────────────────

SourceType = Literal["pdf", "docx", "markdown", "text", "url", "web_search", "unknown"]


# ─── Knowledge Chunk ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeChunk:
    """
    A single piece of knowledge extracted from a document or web search.

    Chunks are the atomic unit of the Knowledge Engine.  Every document is
    split into overlapping chunks before embedding + storing.  Retrieval
    returns ranked chunks with confidence scores.

    Attributes
    ----------
    chunk_id:
        Unique identifier for this chunk (UUID or deterministic hash).
    content:
        The actual text content of this chunk.
    source:
        Human-readable source identifier — file path, URL, or label.
    source_type:
        Classification of the source document format.
    chunk_index:
        Zero-based position of this chunk within its source document.
    total_chunks:
        Total number of chunks extracted from the same source document.
    metadata:
        Arbitrary key-value metadata:
        ``{"title": str, "author": str, "page": int, "date": str, ...}``
    score:
        Relevance / similarity score in [0, 1] range.
        Populated during retrieval; 0.0 for freshly ingested chunks.
    tags:
        Optional tags for metadata-filtered retrieval.
    """

    chunk_id: str
    content: str
    source: str
    source_type: SourceType = "unknown"
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Validate score range
        object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))

    @property
    def content_preview(self) -> str:
        """First 80 characters of content, for logging."""
        return self.content[:80]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for EventBus events and API responses)."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": dict(self.metadata),
            "score": self.score,
            "tags": list(self.tags),
        }

    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        source_type: SourceType = "text",
        chunk_index: int = 0,
        total_chunks: int = 1,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        score: float = 0.0,
    ) -> "KnowledgeChunk":
        """
        Factory method that auto-generates a UUID chunk_id.

        Prefer this over direct construction so tests can still build
        chunks with explicit IDs when needed.
        """
        return cls(
            chunk_id=str(uuid.uuid4()),
            content=content,
            source=source,
            source_type=source_type,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            metadata=metadata or {},
            tags=tags or [],
            score=score,
        )


# ─── Knowledge Result ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeResult:
    """
    The complete result of a knowledge query.

    Bundles all retrieved chunks with query metadata.
    The ``as_rag_context()`` method formats the chunks for LLM injection.

    Attributes
    ----------
    query:
        The original natural language query.
    chunks:
        Ranked list of relevant KnowledgeChunks (highest score first).
    used_web_search:
        True if any chunks came from a web search.
    total_found:
        Total number of matching chunks before limit was applied.
    """

    query: str
    chunks: list[KnowledgeChunk]
    used_web_search: bool = False
    total_found: int = 0

    @property
    def is_empty(self) -> bool:
        """True when no relevant chunks were found."""
        return len(self.chunks) == 0

    @property
    def best_score(self) -> float:
        """Highest relevance score in the result set; 0.0 if empty."""
        if not self.chunks:
            return 0.0
        return max(c.score for c in self.chunks)

    def as_rag_context(
        self,
        max_chunks: int | None = None,
        include_sources: bool = True,
    ) -> str:
        """
        Format chunks as a RAG context block for LLM prompt injection.

        Parameters
        ----------
        max_chunks:
            Limit the number of chunks included. None = all.
        include_sources:
            Whether to include source attribution lines.

        Returns
        -------
        str
            Multi-line text block ready for LLM prompt injection.
            Empty string if no relevant chunks.
        """
        if self.is_empty:
            return ""

        chunks = self.chunks[:max_chunks] if max_chunks else self.chunks
        lines: list[str] = ["[Knowledge Context]"]

        for i, chunk in enumerate(chunks, 1):
            lines.append(f"\n--- Source {i} ---")
            if include_sources:
                src_label = chunk.metadata.get("title") or chunk.source
                lines.append(f"Source: {src_label}")
                if chunk.source_type == "web_search":
                    lines.append(f"URL: {chunk.source}")
            lines.append(chunk.content.strip())

        if self.used_web_search:
            lines.append("\n[Some results from web search]")

        return "\n".join(lines)
