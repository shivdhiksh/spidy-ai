"""
Unit tests for spidy.core.device (DeviceManager)
==================================================
Tests device detection, singleton behavior, and CPU fallback.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from spidy.core.device import DeviceInfo, DeviceManager


class TestDeviceInfo:
    """DeviceInfo dataclass tests."""

    def test_vram_gb_conversion(self):
        """vram_gb should convert bytes to GB correctly."""
        info = DeviceInfo(
            cuda_available=True,
            device="cuda:0",
            provider="cuda",
            gpu_name="RTX 3050",
            vram_bytes=4 * 1024 ** 3,  # 4 GB
            whisper_compute_type="float16",
        )
        assert info.vram_gb == 4.0

    def test_cpu_device_has_zero_vram(self):
        """CPU DeviceInfo should report 0 VRAM."""
        info = DeviceInfo(
            cuda_available=False,
            device="cpu",
            provider="cpu",
            gpu_name="",
            vram_bytes=0,
            whisper_compute_type="int8",
        )
        assert info.vram_gb == 0.0

    def test_is_frozen(self):
        """DeviceInfo is frozen — should not be mutable."""
        info = DeviceInfo(
            cuda_available=False, device="cpu", provider="cpu",
            gpu_name="", vram_bytes=0, whisper_compute_type="int8",
        )
        with pytest.raises((AttributeError, TypeError)):
            info.device = "cuda:0"  # type: ignore


class TestDeviceManagerDetect:
    """DeviceManager.detect() tests."""

    def setup_method(self):
        """Reset singleton before each test."""
        DeviceManager._instance = None

    def test_detect_cpu_when_torch_missing(self):
        """If torch and ctranslate2 are not available, should gracefully fall back to CPU."""
        with patch.dict("sys.modules", {"torch": None, "ctranslate2": None}):
            dm = DeviceManager()
            info = dm.detect()

        assert info.device == "cpu"
        assert info.cuda_available is False
        assert info.whisper_compute_type == "int8"

    def test_detect_cuda_when_available(self):
        """When torch reports CUDA available, DeviceInfo should reflect that."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.name = "NVIDIA GeForce RTX 3050 Laptop GPU"
        mock_props.total_memory = 4 * 1024 ** 3
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}):
            dm = DeviceManager()
            info = dm.detect()

        assert info.cuda_available is True
        assert info.device == "cuda:0"
        assert info.whisper_compute_type == "float16"
        assert "RTX 3050" in info.gpu_name

    def test_detect_cpu_when_cuda_unavailable(self):
        """When torch is present but CUDA unavailable, use CPU."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch, "ctranslate2": None}):
            dm = DeviceManager()
            info = dm.detect()

        assert info.device == "cpu"
        assert info.cuda_available is False

    def test_detect_stores_singleton(self):
        """detect() should store the result for later retrieval via get()."""
        with patch.dict("sys.modules", {"torch": None, "ctranslate2": None}):
            dm = DeviceManager()
            info = dm.detect()

        assert DeviceManager.get() is info

    def test_get_before_detect_raises(self):
        """get() before detect() should raise RuntimeError."""
        DeviceManager._instance = None
        with pytest.raises(RuntimeError, match="detect\\(\\) must be called"):
            DeviceManager.get()

    def test_multiple_detect_calls_are_safe(self):
        """Calling detect() twice should not raise and second call overwrites singleton."""
        with patch.dict("sys.modules", {"torch": None, "ctranslate2": None}):
            dm = DeviceManager()
            info1 = dm.detect()
            info2 = dm.detect()

        # Both should be valid CPU devices
        assert info1.device == "cpu"
        assert info2.device == "cpu"
