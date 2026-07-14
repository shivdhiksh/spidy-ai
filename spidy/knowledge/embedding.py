"""
Knowledge Engine — Embedding Engine
=====================================
Generates dense vector embeddings for text chunks using sentence-transformers.

Design
------
- Lazy init: model loaded on first call, not at import time.
- Graceful degrade: ``is_available`` returns False when deps absent;
  all methods return empty lists.
- Same model as M8 Memory Engine (``all-MiniLM-L6-v2``) by default.
  Both engines maintain their own instances; no sharing is attempted
  (ChromaDB handles its own embedding storage per collection).
- Thread-safe: sentence-transformers models are thread-safe for inference.
- Async-safe: the blocking encode() call runs in a thread executor to
  avoid blocking the asyncio event loop.

Dependencies
------------
    sentence-transformers>=3.0.0   (already in pyproject.toml[memory])
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Wraps sentence-transformers for dense embedding generation.

    Parameters
    ----------
    model_name:
        Name of the sentence-transformers model to use.
        Default is ``all-MiniLM-L6-v2`` (same as M8 Memory Engine).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._checked_available: bool | None = None

    # ── Availability ──────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if sentence-transformers is installed."""
        if self._checked_available is None:
            try:
                import sentence_transformers  # noqa: F401
                self._checked_available = True
            except ImportError:
                self._checked_available = False
        return self._checked_available

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """
        Load the model if not already loaded.

        Returns True on success, False if deps unavailable or load failed.
        """
        if self._model is not None:
            return True
        if not self.is_available:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            log.info(
                "EmbeddingEngine: loading model '{model}'...",
                model=self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
            log.info(
                "EmbeddingEngine: model '{model}' ready.",
                model=self._model_name,
            )
            return True
        except Exception as exc:
            log.warning(
                "EmbeddingEngine: failed to load model '{model}': {exc}",
                model=self._model_name,
                exc=exc,
            )
            return False

    # ── Embedding methods ─────────────────────────────────────────────────

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings synchronously.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per text.  Empty list if unavailable.
        """
        if not texts:
            return []
        if not self._ensure_model():
            log.warning(
                "EmbeddingEngine: cannot embed — model not available."
            )
            return []
        try:
            import numpy as np
            embeddings = self._model.encode(  # type: ignore[union-attr]
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            # Convert numpy array rows to Python lists
            return [
                row.tolist() if hasattr(row, "tolist") else list(row)
                for row in embeddings
            ]
        except Exception as exc:
            log.warning(
                "EmbeddingEngine: embedding failed: {exc}", exc=exc
            )
            return []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings asynchronously (via thread executor).

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per text.  Empty list if unavailable.
        """
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_sync, texts)

    async def embed_one(self, text: str) -> list[float]:
        """
        Generate a single embedding vector asynchronously.

        Returns an empty list if unavailable.
        """
        results = await self.embed([text])
        return results[0] if results else []

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The name of the embedding model."""
        return self._model_name

    @property
    def embedding_dim(self) -> int | None:
        """
        Embedding dimension for the loaded model.
        Returns None if model is not loaded.
        ``all-MiniLM-L6-v2`` produces 384-dim vectors.
        """
        if self._model is None:
            return None
        try:
            return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
        except Exception:
            return None
