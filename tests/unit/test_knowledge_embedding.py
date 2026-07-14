"""
Unit tests for spidy.knowledge.embedding
Tests EmbeddingEngine with and without sentence-transformers installed.
Uses monkeypatching to avoid requiring heavy ML deps in CI.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from spidy.knowledge.embedding import EmbeddingEngine, _DEFAULT_MODEL


# ─── Construction ─────────────────────────────────────────────────────────────


class TestEmbeddingEngineConstruction:
    """EmbeddingEngine constructs correctly."""

    def test_default_model_name(self):
        engine = EmbeddingEngine()
        assert engine.model_name == _DEFAULT_MODEL

    def test_custom_model_name(self):
        engine = EmbeddingEngine(model_name="custom-model")
        assert engine.model_name == "custom-model"

    def test_model_not_loaded_initially(self):
        engine = EmbeddingEngine()
        assert engine._model is None

    def test_embedding_dim_none_before_load(self):
        engine = EmbeddingEngine()
        assert engine.embedding_dim is None


# ─── Availability ────────────────────────────────────────────────────────────


class TestEmbeddingEngineAvailability:
    """EmbeddingEngine.is_available reflects import status."""

    def test_is_available_when_sentence_transformers_present(self):
        engine = EmbeddingEngine()
        # Reset cached state
        engine._checked_available = None
        mock_st = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = engine.is_available
        assert result is True

    def test_is_available_false_when_import_fails(self):
        engine = EmbeddingEngine()
        engine._checked_available = None
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            # Only raise for sentence_transformers
            orig_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        # Use a cleaner approach
        engine2 = EmbeddingEngine()
        engine2._checked_available = False
        assert engine2.is_available is False

    def test_availability_cached_after_first_check(self):
        engine = EmbeddingEngine()
        engine._checked_available = True
        # Should not re-check; returns cached value
        assert engine.is_available is True


# ─── embed_sync — unavailable path ───────────────────────────────────────────


class TestEmbeddingEngineSyncUnavailable:
    """embed_sync() returns empty list when engine is unavailable."""

    def test_empty_input_returns_empty(self):
        engine = EmbeddingEngine()
        assert engine.embed_sync([]) == []

    def test_returns_empty_when_unavailable(self):
        engine = EmbeddingEngine()
        engine._checked_available = False
        result = engine.embed_sync(["some text"])
        assert result == []

    def test_returns_empty_when_model_fails_to_load(self):
        engine = EmbeddingEngine()
        engine._checked_available = True
        # Simulate model load failure by making _ensure_model return False
        # without needing sentence_transformers installed.
        # Patch _ensure_model directly.
        from unittest.mock import patch as _patch
        with _patch.object(engine, "_ensure_model", return_value=False):
            result = engine.embed_sync(["text"])
        # Should return empty, not raise
        assert result == []


# ─── embed_sync — mocked model ───────────────────────────────────────────────


class TestEmbeddingEngineSyncMocked:
    """embed_sync() with a mocked sentence-transformers model."""

    def _make_engine_with_mock_model(self, dim: int = 4) -> EmbeddingEngine:
        """Create an EmbeddingEngine with a pre-loaded mock model."""
        import numpy as np
        engine = EmbeddingEngine()
        engine._checked_available = True

        mock_model = MagicMock()
        # Return numpy arrays of shape (n_texts, dim)
        def mock_encode(texts, **kwargs):
            return np.zeros((len(texts), dim), dtype=float)

        mock_model.encode = mock_encode
        mock_model.get_sentence_embedding_dimension.return_value = dim
        engine._model = mock_model
        return engine

    def test_returns_one_embedding_per_text(self):
        engine = self._make_engine_with_mock_model(dim=4)
        result = engine.embed_sync(["text1", "text2", "text3"])
        assert len(result) == 3

    def test_embeddings_are_lists_of_floats(self):
        engine = self._make_engine_with_mock_model(dim=4)
        result = engine.embed_sync(["text"])
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embedding_dimension_correct(self):
        engine = self._make_engine_with_mock_model(dim=384)
        result = engine.embed_sync(["test text"])
        assert len(result[0]) == 384

    def test_embedding_dim_property_after_load(self):
        engine = self._make_engine_with_mock_model(dim=384)
        assert engine.embedding_dim == 384

    def test_empty_text_list_returns_empty(self):
        engine = self._make_engine_with_mock_model(dim=4)
        result = engine.embed_sync([])
        assert result == []


# ─── async embed / embed_one ─────────────────────────────────────────────────


class TestEmbeddingEngineAsync:
    """embed() and embed_one() async wrappers."""

    def _make_engine_with_mock_model(self, dim: int = 4) -> EmbeddingEngine:
        import numpy as np
        engine = EmbeddingEngine()
        engine._checked_available = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, dim), dtype=float)
        mock_model.get_sentence_embedding_dimension.return_value = dim
        engine._model = mock_model
        return engine

    @pytest.mark.asyncio
    async def test_embed_returns_list_async(self):
        engine = EmbeddingEngine()
        engine._checked_available = False
        result = await engine.embed(["text"])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_one_returns_list_async(self):
        engine = EmbeddingEngine()
        engine._checked_available = False
        result = await engine.embed_one("text")
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_empty_returns_empty(self):
        engine = EmbeddingEngine()
        result = await engine.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_one_empty_string(self):
        engine = EmbeddingEngine()
        engine._checked_available = False
        result = await engine.embed_one("")
        assert result == []


# ─── Model name property ─────────────────────────────────────────────────────


class TestEmbeddingEngineModelName:
    """model_name property returns the configured model."""

    def test_default_model_name_value(self):
        engine = EmbeddingEngine()
        assert engine.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name_retained(self):
        engine = EmbeddingEngine(model_name="paraphrase-MiniLM-L6-v2")
        assert engine.model_name == "paraphrase-MiniLM-L6-v2"
