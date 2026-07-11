"""
Tests for spidy.vision.screenshot — ScreenshotEngine

All tests mock the optional mss dependency so they run without installing it.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from spidy.vision.screenshot import ScreenshotEngine
from spidy.vision.types import MonitorInfo, ScreenshotResult, VisionDependencyError


# ─── Availability ─────────────────────────────────────────────────────────────


class TestScreenshotEngineAvailability:
    def test_is_available_when_mss_absent(self):
        """When mss is not importable, is_available returns False."""
        with patch("spidy.vision.screenshot._MSS_AVAILABLE", False):
            engine = ScreenshotEngine()
            assert engine.is_available is False

    def test_is_available_when_mss_present(self):
        """When mss is importable, is_available returns True."""
        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            engine = ScreenshotEngine()
            assert engine.is_available is True

    def test_constructor_default_params(self):
        engine = ScreenshotEngine()
        assert engine._default_monitor == 0
        assert engine._capture_format == "PNG"

    def test_constructor_custom_params(self):
        engine = ScreenshotEngine(default_monitor=1, capture_format="jpeg")
        assert engine._default_monitor == 1
        assert engine._capture_format == "JPEG"  # uppercased


# ─── get_monitors() ───────────────────────────────────────────────────────────


class TestGetMonitors:
    def test_returns_synthetic_when_mss_absent(self):
        with patch("spidy.vision.screenshot._MSS_AVAILABLE", False):
            engine = ScreenshotEngine()
            monitors = engine.get_monitors()
            assert len(monitors) == 1
            assert isinstance(monitors[0], MonitorInfo)
            assert monitors[0].is_primary is True
            assert monitors[0].width == 1920

    def test_returns_list_of_monitor_info_when_mss_present(self):
        # Mock mss context manager
        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.monitors = [
            # [0] = virtual combined screen (mss convention)
            {"left": 0, "top": 0, "width": 3840, "height": 1080},
            # [1] = real monitor 0
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            # [2] = real monitor 1
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        ]

        mock_mss_module = MagicMock()
        mock_mss_module.mss.return_value = mock_sct

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                engine = ScreenshotEngine()
                monitors = engine.get_monitors()

        assert len(monitors) == 2
        assert monitors[0].width == 1920
        assert monitors[0].is_primary is True
        assert monitors[1].width == 1920
        assert monitors[1].is_primary is False

    def test_returns_synthetic_on_mss_error(self):
        mock_mss_module = MagicMock()
        mock_mss_module.mss.side_effect = RuntimeError("mss error")

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                engine = ScreenshotEngine()
                monitors = engine.get_monitors()

        assert len(monitors) == 1
        assert isinstance(monitors[0], MonitorInfo)


# ─── capture_fullscreen() ─────────────────────────────────────────────────────


class TestCaptureFullscreen:
    def test_raises_when_mss_absent(self):
        with patch("spidy.vision.screenshot._MSS_AVAILABLE", False):
            engine = ScreenshotEngine()
            with pytest.raises(VisionDependencyError):
                engine.capture_fullscreen()

    def _make_mock_mss(self, width=1920, height=1080, png_bytes=b"\x89PNG"):
        mock_raw = MagicMock()
        mock_raw.width = width
        mock_raw.height = height
        mock_raw.rgb = b"\xff\xff\xff" * (width * height)
        mock_raw.size = (width, height)

        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.monitors = [
            {"left": 0, "top": 0, "width": width, "height": height},
            {"left": 0, "top": 0, "width": width, "height": height},
        ]
        mock_sct.grab.return_value = mock_raw

        mock_tools = MagicMock()
        mock_tools.to_png.return_value = png_bytes

        mock_mss_module = MagicMock()
        mock_mss_module.mss.return_value = mock_sct

        return mock_mss_module, mock_tools, mock_raw

    def test_returns_screenshot_result(self):
        mock_mss_module, mock_tools, mock_raw = self._make_mock_mss()

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                with patch("spidy.vision.screenshot._mss_tools", mock_tools):
                    engine = ScreenshotEngine()
                    result = engine.capture_fullscreen()

        assert isinstance(result, ScreenshotResult)
        assert result.source == "fullscreen"
        assert result.width == 1920
        assert result.height == 1080
        assert result.image_data == b"\x89PNG"
        assert result.region is None
        assert result.monitor_index == 0

    def test_uses_default_monitor(self):
        mock_mss_module, mock_tools, mock_raw = self._make_mock_mss()

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                with patch("spidy.vision.screenshot._mss_tools", mock_tools):
                    engine = ScreenshotEngine(default_monitor=0)
                    result = engine.capture_fullscreen()

        assert result.monitor_index == 0

    def test_timestamp_is_recent(self):
        mock_mss_module, mock_tools, mock_raw = self._make_mock_mss()

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                with patch("spidy.vision.screenshot._mss_tools", mock_tools):
                    engine = ScreenshotEngine()
                    t_before = time.time()
                    result = engine.capture_fullscreen()
                    t_after = time.time()

        assert t_before <= result.timestamp <= t_after

    def test_raises_on_mss_error(self):
        mock_mss_module = MagicMock()
        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.monitors = [{"left": 0, "top": 0, "width": 1920, "height": 1080}]
        mock_sct.grab.side_effect = RuntimeError("grab failed")
        mock_mss_module.mss.return_value = mock_sct

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                engine = ScreenshotEngine()
                with pytest.raises(VisionDependencyError):
                    engine.capture_fullscreen()


# ─── capture_region() ────────────────────────────────────────────────────────


class TestCaptureRegion:
    def test_raises_when_mss_absent(self):
        with patch("spidy.vision.screenshot._MSS_AVAILABLE", False):
            engine = ScreenshotEngine()
            with pytest.raises(VisionDependencyError):
                engine.capture_region(0, 0, 100, 100)

    def test_returns_screenshot_result(self):
        mock_raw = MagicMock()
        mock_raw.width = 100
        mock_raw.height = 100
        mock_raw.rgb = b"\xff" * 30000
        mock_raw.size = (100, 100)

        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.grab.return_value = mock_raw

        mock_tools = MagicMock()
        mock_tools.to_png.return_value = b"\x89PNG_region"

        mock_mss_module = MagicMock()
        mock_mss_module.mss.return_value = mock_sct

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                with patch("spidy.vision.screenshot._mss_tools", mock_tools):
                    engine = ScreenshotEngine()
                    result = engine.capture_region(10, 20, 100, 100, monitor_index=1)

        assert isinstance(result, ScreenshotResult)
        assert result.source == "region"
        assert result.region == (10, 20, 100, 100)
        assert result.monitor_index == 1
        assert result.image_data == b"\x89PNG_region"


# ─── capture_active_window() ─────────────────────────────────────────────────


class TestCaptureActiveWindow:
    def test_raises_when_mss_absent(self):
        with patch("spidy.vision.screenshot._MSS_AVAILABLE", False):
            engine = ScreenshotEngine()
            with pytest.raises(VisionDependencyError):
                engine.capture_active_window()

    def test_falls_back_to_fullscreen_when_win32_absent(self):
        """Without win32gui, should fall back to fullscreen capture."""
        mock_raw = MagicMock()
        mock_raw.width = 1920
        mock_raw.height = 1080
        mock_raw.rgb = b"\xff" * (1920 * 1080 * 3)
        mock_raw.size = (1920, 1080)

        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
        ]
        mock_sct.grab.return_value = mock_raw

        mock_tools = MagicMock()
        mock_tools.to_png.return_value = b"\x89PNG_fullscreen"

        mock_mss_module = MagicMock()
        mock_mss_module.mss.return_value = mock_sct

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._WIN32_AVAILABLE", False):
                with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                    with patch("spidy.vision.screenshot._mss_tools", mock_tools):
                        engine = ScreenshotEngine()
                        result = engine.capture_active_window()

        # Falls back to fullscreen, but source is overridden to "window"
        assert isinstance(result, ScreenshotResult)
        assert result.source == "window"

    def test_uses_win32gui_when_available(self):
        """With win32gui available, uses GetForegroundWindow rect."""
        mock_win32gui = MagicMock()
        mock_win32gui.GetForegroundWindow.return_value = 12345
        mock_win32gui.GetWindowRect.return_value = (100, 200, 900, 800)

        mock_raw = MagicMock()
        mock_raw.width = 800
        mock_raw.height = 600
        mock_raw.rgb = b"\xff" * (800 * 600 * 3)
        mock_raw.size = (800, 600)

        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.grab.return_value = mock_raw

        mock_tools = MagicMock()
        mock_tools.to_png.return_value = b"\x89PNG_window"

        mock_mss_module = MagicMock()
        mock_mss_module.mss.return_value = mock_sct

        with patch("spidy.vision.screenshot._MSS_AVAILABLE", True):
            with patch("spidy.vision.screenshot._WIN32_AVAILABLE", True):
                with patch("spidy.vision.screenshot._win32gui", mock_win32gui):
                    with patch("spidy.vision.screenshot._mss_module", mock_mss_module):
                        with patch("spidy.vision.screenshot._mss_tools", mock_tools):
                            engine = ScreenshotEngine()
                            result = engine.capture_active_window()

        assert isinstance(result, ScreenshotResult)
        assert result.source == "window"
        assert result.image_data == b"\x89PNG_window"
