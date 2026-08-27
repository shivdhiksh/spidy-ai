"""
Tests for GPU-First Performance Optimization
============================================
Verifies:
1. CUDA DLL discovery & registration
2. DeviceManager detection & graceful fallback
3. EmbeddingModelRegistry shares single SentenceTransformer instance
4. ChromaClientRegistry caches client per path
5. FasterWhisperRecognizer device selection & fallback
6. BargeInDetector device configuration & fallback
7. SpidyCore startup diagnostics & model deduplication
8. Memory & Knowledge functionality preserved
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spidy.config.manager import (
    BargeInConfig,
    KnowledgeConfig,
    SemanticMemoryConfig,
    SpidyConfig,
)
from spidy.core.app import SpidyCore
from spidy.core.chroma import ChromaClientRegistry
from spidy.core.device import DeviceInfo, DeviceManager, ensure_cuda_dlls
from spidy.core.models import EmbeddingModelRegistry
from spidy.knowledge.embedding import EmbeddingEngine
from spidy.knowledge.store import VectorStore
from spidy.memory.semantic import SemanticMemory
from spidy.memory.types import MemoryEntry, MemoryType
from spidy.perception.voice.stt.whisper import FasterWhisperRecognizer
from spidy.voice.barge_in import BargeInDetector


class TestCudaAndDeviceManager:
    """Test CUDA DLL registration and DeviceManager detection."""

    def test_ensure_cuda_dlls_runs_safely(self) -> None:
        """ensure_cuda_dlls runs without errors and returns list of paths."""
        res = ensure_cuda_dlls()
        assert isinstance(res, list)

    def test_device_manager_detect_returns_device_info(self) -> None:
        """DeviceManager.detect() returns a valid DeviceInfo instance."""
        dm = DeviceManager()
        info = dm.detect()
        assert isinstance(info, DeviceInfo)
        assert info.device in ("cuda:0", "cpu")
        assert info.provider in ("cuda", "cpu")
        assert DeviceManager.get() == info

    def test_device_manager_cpu_fallback_when_cuda_disabled(self) -> None:
        """DeviceManager cleanly falls back to CPU when torch and ctranslate2 report no CUDA."""
        with patch("spidy.core.device.ensure_cuda_dlls"):
            with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=lambda: False))}):
                with patch("ctranslate2.get_cuda_device_count", return_value=0, create=True):
                    dm = DeviceManager()
                    info = dm._do_detect()
                    assert info.cuda_available is False
                    assert info.device == "cpu"
                    assert info.provider == "cpu"
                    assert info.whisper_compute_type == "int8"


class TestModelAndClientDeduplication:
    """Test shared instances of embedding models and vector store clients."""

    def setup_method(self) -> None:
        EmbeddingModelRegistry.clear_cache()
        ChromaClientRegistry.clear_cache()

    def teardown_method(self) -> None:
        EmbeddingModelRegistry.clear_cache()
        ChromaClientRegistry.clear_cache()

    def test_embedding_model_registry_singleton(self) -> None:
        """EmbeddingModelRegistry returns the exact same object for identical model name & device."""
        mock_model = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_init:
            m1 = EmbeddingModelRegistry.get_model("all-MiniLM-L6-v2", device="cpu")
            m2 = EmbeddingModelRegistry.get_model("all-MiniLM-L6-v2", device="cpu")
            assert m1 is m2
            assert mock_init.call_count == 1

    def test_embedding_engine_and_semantic_memory_share_model(self) -> None:
        """EmbeddingEngine and SemanticMemory reuse the same SentenceTransformer instance."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2]])
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
            engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2", device="cpu")
            sem = SemanticMemory(embedding_model="all-MiniLM-L6-v2", persist_dir=":memory:", device="cpu")

            assert engine._ensure_model() is True
            assert sem._ensure_initialized() is True

            assert engine._model is sem._encoder
            assert mock_st.call_count == 1

    def test_chroma_client_registry_shares_client_for_same_path(self, tmp_path: Path) -> None:
        """ChromaClientRegistry caches client for same persistent directory."""
        mock_client = MagicMock()
        with patch("chromadb.PersistentClient", return_value=mock_client) as mock_p:
            c1 = ChromaClientRegistry.get_client(tmp_path)
            c2 = ChromaClientRegistry.get_client(tmp_path)
            assert c1 is c2
            assert mock_p.call_count == 1


class TestWhisperDeviceSelection:
    """Test device selection and fallback in FasterWhisperRecognizer."""

    def test_whisper_cuda_requested_success(self) -> None:
        """When CUDA requested and available, WhisperModel initializes on CUDA."""
        mock_model = MagicMock()
        with patch("faster_whisper.WhisperModel", return_value=mock_model) as mock_wm:
            recognizer = FasterWhisperRecognizer(model_size="base.en", device="cuda", compute_type="float16")
            recognizer.load()
            assert recognizer._resolved_device == "cuda"
            assert recognizer._resolved_compute_type == "float16"
            mock_wm.assert_called_once_with("base.en", device="cuda", compute_type="float16", download_root=None)

    def test_whisper_cuda_fallback_to_cpu(self) -> None:
        """When CUDA fails at load time, gracefully falls back to CPU int8."""
        def side_effect(model_size, device, compute_type, download_root=None):
            if device == "cuda":
                raise RuntimeError("CUDA cublas error")
            return MagicMock()

        with patch("faster_whisper.WhisperModel", side_effect=side_effect):
            recognizer = FasterWhisperRecognizer(model_size="base.en", device="cuda", compute_type="float16")
            recognizer.load()
            assert recognizer._resolved_device == "cpu"
            assert recognizer._resolved_compute_type == "int8"


class TestBargeInDeviceSelection:
    """Test device configuration in BargeInDetector."""

    def test_barge_in_default_is_cpu(self) -> None:
        """BargeInDetector defaults to CPU int8."""
        handler = MagicMock()
        bus = MagicMock()
        detector = BargeInDetector(interruption_handler=handler, bus=bus)
        assert detector._device == "cpu"
        assert detector._compute_type == "int8"

    def test_barge_in_loads_on_configured_device(self) -> None:
        """BargeInDetector loads with specified device."""
        handler = MagicMock()
        bus = MagicMock()
        detector = BargeInDetector(interruption_handler=handler, bus=bus, device="cpu", compute_type="int8")
        with patch("faster_whisper.WhisperModel") as mock_wm:
            detector.load_model()
            assert detector._model_loaded is True
            mock_wm.assert_called_once_with("tiny.en", device="cpu", compute_type="int8")


class TestConfigAndStartupDiagnostics:
    """Test configuration fields and startup diagnostics."""

    def test_config_models_have_device_fields(self) -> None:
        """Config models contain device and compute_type fields with valid defaults."""
        barge_cfg = BargeInConfig()
        assert barge_cfg.device == "cpu"
        assert barge_cfg.compute_type == "int8"

        sem_cfg = SemanticMemoryConfig()
        assert sem_cfg.device == "auto"

        know_cfg = KnowledgeConfig()
        assert know_cfg.device == "auto"

    def test_spidy_core_startup_logs_ai_summary(self) -> None:
        """SpidyCore._log_ai_runtime_summary logs without errors."""
        core = SpidyCore(text_mode=True)
        core._settings = SpidyConfig()
        # Should execute cleanly
        core._log_ai_runtime_summary()


@pytest.mark.asyncio
class TestMemoryAndKnowledgePreserved:
    """Verify memory storage and vector search functionality remain intact."""

    async def test_semantic_memory_store_and_search(self) -> None:
        """SemanticMemory can store and search memory entries correctly."""
        sem = SemanticMemory(collection_name="test_mem", persist_dir=":memory:")
        entry = MemoryEntry(
            id="m1",
            content="User loves Python and PyTorch",
            session_id="s1",
            tags=("preferences",),
            memory_type=MemoryType.EPISODIC,
        )
        saved_id = await sem.store(entry)
        assert saved_id == entry.id

        results = await sem.search("Python")
        assert isinstance(results, list)

    async def test_vector_store_initialize_and_count(self) -> None:
        """VectorStore initializes and returns count without errors."""
        store = VectorStore(collection_name="test_know", persist_dir=None)
        ok = await store.initialize()
        assert ok is True
        assert await store.count() == 0
