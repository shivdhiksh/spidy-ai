"""
Tests for spidy.vision.screen_analyzer — ScreenAnalyzer

Mocks win32gui, win32process, psutil, and opencv so tests run cross-platform.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spidy.vision.screen_analyzer import ScreenAnalyzer
from spidy.vision.types import ScreenAnalysis, ScreenshotResult, UIRegion


# ─── Availability ─────────────────────────────────────────────────────────────


class TestScreenAnalyzerAvailability:
    def test_available_when_win32_present(self):
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", False):
                engine = ScreenAnalyzer()
                assert engine.is_available is True

    def test_available_when_psutil_present(self):
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", True):
                engine = ScreenAnalyzer()
                assert engine.is_available is True

    def test_not_available_when_neither_present(self):
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", False):
                engine = ScreenAnalyzer()
                assert engine.is_available is False

    def test_ocv_available_when_cv2_present(self):
        with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", True):
            engine = ScreenAnalyzer()
            assert engine.ocv_available is True

    def test_ocv_not_available_when_cv2_absent(self):
        with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", False):
            engine = ScreenAnalyzer()
            assert engine.ocv_available is False

    def test_constructor_defaults(self):
        engine = ScreenAnalyzer()
        assert engine._min_region_area == 1000
        assert engine._max_regions == 20

    def test_constructor_custom(self):
        engine = ScreenAnalyzer(min_region_area=500, max_regions=10)
        assert engine._min_region_area == 500
        assert engine._max_regions == 10


# ─── get_active_application() ─────────────────────────────────────────────────


class TestGetActiveApplication:
    def test_returns_empty_when_neither_win32_nor_psutil(self):
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", False):
                engine = ScreenAnalyzer()
                result = engine.get_active_application()
                assert result == ""

    def test_uses_win32_when_available(self):
        mock_win32gui = MagicMock()
        mock_win32gui.GetForegroundWindow.return_value = 12345

        mock_win32process = MagicMock()
        mock_win32process.GetWindowThreadProcessId.return_value = (0, 999)

        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_proc.name.return_value = "Code.exe"
        mock_psutil.Process.return_value = mock_proc

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", True):
                with patch("spidy.vision.screen_analyzer._win32gui", mock_win32gui):
                    with patch("spidy.vision.screen_analyzer._win32process", mock_win32process):
                        with patch("spidy.vision.screen_analyzer._psutil", mock_psutil):
                            engine = ScreenAnalyzer()
                            result = engine.get_active_application()

        assert result == "Code.exe"

    def test_returns_empty_when_no_foreground_window(self):
        mock_win32gui = MagicMock()
        mock_win32gui.GetForegroundWindow.return_value = 0  # no foreground window

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", False):
                with patch("spidy.vision.screen_analyzer._win32gui", mock_win32gui):
                    engine = ScreenAnalyzer()
                    result = engine.get_active_application()

        assert result == ""

    def test_psutil_fallback_when_win32_absent(self):
        mock_proc1 = MagicMock()
        mock_proc1.info = {"name": "System", "cpu_percent": 0.1}
        mock_proc2 = MagicMock()
        mock_proc2.info = {"name": "Code.exe", "cpu_percent": 5.0}
        mock_proc3 = MagicMock()
        mock_proc3.info = {"name": "Idle", "cpu_percent": 90.0}

        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = [mock_proc3, mock_proc2, mock_proc1]

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", True):
                with patch("spidy.vision.screen_analyzer._psutil", mock_psutil):
                    engine = ScreenAnalyzer()
                    result = engine.get_active_application()

        # Should skip "Idle" and return "Code.exe"
        assert result == "Code.exe"


# ─── get_active_window_title() ────────────────────────────────────────────────


class TestGetActiveWindowTitle:
    def test_returns_empty_when_win32_absent(self):
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            engine = ScreenAnalyzer()
            result = engine.get_active_window_title()
            assert result == ""

    def test_returns_title(self):
        mock_win32gui = MagicMock()
        mock_win32gui.GetForegroundWindow.return_value = 42
        mock_win32gui.GetWindowText.return_value = "VS Code - main.py"

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._win32gui", mock_win32gui):
                engine = ScreenAnalyzer()
                result = engine.get_active_window_title()

        assert result == "VS Code - main.py"

    def test_returns_empty_when_no_window(self):
        mock_win32gui = MagicMock()
        mock_win32gui.GetForegroundWindow.return_value = 0

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._win32gui", mock_win32gui):
                engine = ScreenAnalyzer()
                result = engine.get_active_window_title()

        assert result == ""


# ─── get_visible_windows() ────────────────────────────────────────────────────


class TestGetVisibleWindows:
    def test_returns_empty_when_win32_absent(self):
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            engine = ScreenAnalyzer()
            result = engine.get_visible_windows()
            assert result == []

    def test_returns_sorted_visible_windows(self):
        # Simulate EnumWindows calling the callback with window handles
        visible_titles = ["Chrome", "VS Code", "Notepad"]
        skip_titles = ["", "Program Manager"]

        def fake_enum_windows(callback, param):
            # Simulate hwnd 1 = Chrome, 2 = VS Code, 3 = empty, 4 = Notepad
            for i, title in enumerate(visible_titles + skip_titles, start=1):
                callback(i, param)

        mock_win32gui = MagicMock()
        mock_win32gui.IsWindowVisible.return_value = True
        side_effect_titles = visible_titles + skip_titles
        title_iter = iter(side_effect_titles)

        def get_text(hwnd):
            try:
                return next(title_iter)
            except StopIteration:
                return ""

        mock_win32gui.GetWindowText.side_effect = get_text
        mock_win32gui.EnumWindows.side_effect = fake_enum_windows

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._win32gui", mock_win32gui):
                engine = ScreenAnalyzer()
                result = engine.get_visible_windows()

        # Visible titles should be returned, skip titles filtered
        assert "Chrome" in result
        assert "VS Code" in result
        assert "Notepad" in result
        assert "" not in result
        assert "Program Manager" not in result
        # Should be sorted
        assert result == sorted(result)

    def test_returns_empty_on_exception(self):
        mock_win32gui = MagicMock()
        mock_win32gui.EnumWindows.side_effect = RuntimeError("EnumWindows failed")

        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._win32gui", mock_win32gui):
                engine = ScreenAnalyzer()
                result = engine.get_visible_windows()

        assert result == []


# ─── detect_ui_regions() ─────────────────────────────────────────────────────


class TestDetectUIRegions:
    def test_returns_empty_when_opencv_absent(self):
        with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", False):
            engine = ScreenAnalyzer()
            result = engine.detect_ui_regions(b"\x89PNG")
            assert result == []

    def test_returns_ui_region_list(self):
        # Mock cv2 to return two contours with areas above threshold
        mock_cv2 = MagicMock()
        mock_np = MagicMock()

        # Mock numpy decode
        mock_img = MagicMock()
        mock_img.shape = [1080, 1920, 3]
        mock_np.frombuffer.return_value = MagicMock()
        mock_cv2.imdecode.return_value = mock_img

        # Mock image processing chain
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.GaussianBlur.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()

        # Two large contours
        contour1 = MagicMock()
        contour2 = MagicMock()
        mock_cv2.findContours.return_value = ([contour1, contour2], None)

        # contourArea returns large areas
        mock_cv2.contourArea.side_effect = [50000, 30000]
        mock_cv2.boundingRect.side_effect = [
            (100, 100, 500, 300),
            (200, 200, 300, 200),
        ]

        with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._cv2", mock_cv2):
                with patch("spidy.vision.screen_analyzer._np", mock_np):
                    engine = ScreenAnalyzer(min_region_area=1000)
                    result = engine.detect_ui_regions(b"\x89PNG")

        assert len(result) == 2
        assert all(isinstance(r, UIRegion) for r in result)
        # Sorted by area (largest first)
        areas = [r.bbox[2] * r.bbox[3] for r in result]
        assert areas == sorted(areas, reverse=True)

    def test_filters_small_regions(self):
        mock_cv2 = MagicMock()
        mock_np = MagicMock()

        mock_img = MagicMock()
        mock_img.shape = [1080, 1920, 3]
        mock_np.frombuffer.return_value = MagicMock()
        mock_cv2.imdecode.return_value = mock_img

        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.GaussianBlur.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()

        contour = MagicMock()
        mock_cv2.findContours.return_value = ([contour], None)
        mock_cv2.contourArea.return_value = 50  # below min_region_area=1000

        with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._cv2", mock_cv2):
                with patch("spidy.vision.screen_analyzer._np", mock_np):
                    engine = ScreenAnalyzer(min_region_area=1000)
                    result = engine.detect_ui_regions(b"\x89PNG")

        assert result == []

    def test_returns_empty_on_exception(self):
        mock_cv2 = MagicMock()
        mock_np = MagicMock()
        mock_np.frombuffer.side_effect = Exception("numpy error")

        with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", True):
            with patch("spidy.vision.screen_analyzer._cv2", mock_cv2):
                with patch("spidy.vision.screen_analyzer._np", mock_np):
                    engine = ScreenAnalyzer()
                    result = engine.detect_ui_regions(b"\x89PNG")

        assert result == []


# ─── analyze_screen() ─────────────────────────────────────────────────────────


class TestAnalyzeScreen:
    def test_returns_screen_analysis(self):
        engine = ScreenAnalyzer()

        with patch.object(engine, "get_active_application", return_value="Code.exe"):
            with patch.object(engine, "get_active_window_title", return_value="main.py"):
                with patch.object(engine, "get_visible_windows", return_value=["Code.exe", "Chrome"]):
                    with patch.object(engine, "detect_ui_regions", return_value=[]):
                        result = engine.analyze_screen()

        assert isinstance(result, ScreenAnalysis)
        assert result.active_app == "Code.exe"
        assert result.active_window == "main.py"
        assert result.window_count == 2

    def test_skips_ui_regions_without_screenshot(self):
        engine = ScreenAnalyzer()

        with patch.object(engine, "get_active_application", return_value=""):
            with patch.object(engine, "get_active_window_title", return_value=""):
                with patch.object(engine, "get_visible_windows", return_value=[]):
                    with patch.object(engine, "detect_ui_regions") as mock_detect:
                        result = engine.analyze_screen(screenshot=None)

        # detect_ui_regions should not be called when no screenshot provided
        mock_detect.assert_not_called()
        assert result.region_count == 0

    def test_uses_screenshot_for_regions(self):
        screenshot = ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080)
        region = UIRegion(name="panel", bbox=(0, 0, 100, 100), confidence=0.8)

        engine = ScreenAnalyzer()

        with patch.object(engine, "get_active_application", return_value="app.exe"):
            with patch.object(engine, "get_active_window_title", return_value=""):
                with patch.object(engine, "get_visible_windows", return_value=[]):
                    with patch.object(engine, "detect_ui_regions", return_value=[region]):
                        result = engine.analyze_screen(screenshot=screenshot)

        assert result.region_count == 1

    def test_timestamp_is_positive(self):
        engine = ScreenAnalyzer()

        with patch.object(engine, "get_active_application", return_value=""):
            with patch.object(engine, "get_active_window_title", return_value=""):
                with patch.object(engine, "get_visible_windows", return_value=[]):
                    result = engine.analyze_screen()

        assert result.timestamp > 0

    def test_returns_screen_analysis_with_defaults_when_everything_absent(self):
        """Analyzer must always return ScreenAnalysis, even with no deps."""
        with patch("spidy.vision.screen_analyzer._WIN32_AVAILABLE", False):
            with patch("spidy.vision.screen_analyzer._PSUTIL_AVAILABLE", False):
                with patch("spidy.vision.screen_analyzer._OPENCV_AVAILABLE", False):
                    engine = ScreenAnalyzer()
                    result = engine.analyze_screen()

        assert isinstance(result, ScreenAnalysis)
        assert result.active_app == ""
        assert result.window_count == 0
