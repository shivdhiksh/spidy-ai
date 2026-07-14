"""
Knowledge Engine — Document Chunker
=====================================
Splits raw text into overlapping fixed-size chunks with metadata preservation.

Design
------
- Fixed character-count chunks with configurable overlap.
- Overlap ensures context is not lost at chunk boundaries.
- Short documents (< chunk_size) are returned as a single chunk.
- Each chunk carries the full source metadata from the parent document.
- Chunk IDs are deterministic: ``{source_hash}_{chunk_index}``.

This is a pure, synchronous, zero-dependency module.
All heavy I/O (embedding, storing) happens in the KnowledgeManager.
"""

from __future__ import annotations

import hashlib
from typing import Any

from spidy.knowledge.types import KnowledgeChunk, SourceType
from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Sensible limits to prevent absurd configs
_MIN_CHUNK_SIZE = 50
_MAX_CHUNK_SIZE = 8000
_MIN_OVERLAP = 0


class Chunker:
    """
    Split raw text into overlapping fixed-size chunks.

    Parameters
    ----------
    chunk_size:
        Target character count per chunk (default 1000).
        Must be at least 50 characters.
    chunk_overlap:
        Number of characters to overlap between adjacent chunks (default 200).
        Must be less than chunk_size.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size < _MIN_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be at least {_MIN_CHUNK_SIZE}, got {chunk_size}"
            )
        if chunk_size > _MAX_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be at most {_MAX_CHUNK_SIZE}, got {chunk_size}"
            )
        if chunk_overlap < _MIN_OVERLAP:
            raise ValueError(
                f"chunk_overlap must be >= 0, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        source: str,
        source_type: SourceType = "text",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> list[KnowledgeChunk]:
        """
        Split ``text`` into overlapping chunks.

        Parameters
        ----------
        text:
            Raw text to split.
        source:
            Source identifier (file path, URL, label).
        source_type:
            Document format classification.
        metadata:
            Metadata to attach to every chunk (author, title, date, page, …).
        tags:
            Tags to attach to every chunk for filtered retrieval.

        Returns
        -------
        list[KnowledgeChunk]
            Ordered list of chunks.  Empty if text is empty.
        """
        text = text.strip()
        if not text:
            log.debug("Chunker: empty text for source '{src}' — skipping", src=source)
            return []

        source_hash = self._hash_source(source)
        meta = dict(metadata or {})
        tag_list = list(tags or [])

        raw_chunks = self._split(text)
        total = len(raw_chunks)

        chunks: list[KnowledgeChunk] = []
        for i, content in enumerate(raw_chunks):
            chunk_id = f"{source_hash}_{i}"
            chunk_meta = {**meta, "chunk_index": i, "total_chunks": total}
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    content=content,
                    source=source,
                    source_type=source_type,
                    chunk_index=i,
                    total_chunks=total,
                    metadata=chunk_meta,
                    tags=tag_list,
                    score=0.0,
                )
            )

        log.debug(
            "Chunker: '{src}' → {n} chunks "
            "(size={sz}, overlap={ov})",
            src=source[:60],
            n=total,
            sz=self._chunk_size,
            ov=self._chunk_overlap,
        )
        return chunks

    def chunk_pages(
        self,
        pages: list[str],
        source: str,
        source_type: SourceType = "pdf",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> list[KnowledgeChunk]:
        """
        Chunk a list of page/section strings (e.g. from PDF).

        Each page is chunked independently, with the page number
        injected into metadata so sources can be accurately attributed.

        Parameters
        ----------
        pages:
            List of raw text strings, one per page or section.
        source, source_type, metadata, tags:
            Same as ``chunk()``.

        Returns
        -------
        list[KnowledgeChunk]
            All chunks from all pages, in order.
        """
        all_chunks: list[KnowledgeChunk] = []
        for page_num, page_text in enumerate(pages):
            if not page_text.strip():
                continue
            page_meta = dict(metadata or {})
            page_meta["page"] = page_num + 1  # 1-indexed for humans
            page_chunks = self.chunk(
                text=page_text,
                source=source,
                source_type=source_type,
                metadata=page_meta,
                tags=tags,
            )
            all_chunks.extend(page_chunks)
        return all_chunks

    # ── Private helpers ───────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """
        Split text into overlapping character-count windows.

        Strategy:
        1. Walk through text with a stride of (chunk_size - chunk_overlap).
        2. Each window is [pos : pos + chunk_size].
        3. Stop when pos >= len(text).
        4. Strip whitespace from each window before yielding.
        """
        stride = self._chunk_size - self._chunk_overlap
        chunks: list[str] = []
        pos = 0
        length = len(text)

        while pos < length:
            end = min(pos + self._chunk_size, length)
            window = text[pos:end].strip()
            if window:
                chunks.append(window)
            pos += stride
            if pos >= length:
                break

        return chunks

    @staticmethod
    def _hash_source(source: str) -> str:
        """Short deterministic hash of the source string for chunk IDs."""
        return hashlib.sha256(source.encode()).hexdigest()[:12]
