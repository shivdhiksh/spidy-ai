"""
Tests for spidy.vision.manager — VisionManager

Tests VisionManager with mocked sub-engines so no real screen/OCR hardware required.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.vision.events import (
    VisionAnalysisCompletedEvent,
    VisionCaptureCompletedEvent,
    VisionCaptureStartedEvent,
    VisionErrorEvent,
    VisionOCRCompletedEvent,
)
from spidy.vision.manager import VisionManager
from spidy.vision.types import (
    OCRBlock,
    OCRResult,
    ScreenAnalysis,
    ScreenshotResult,
    VisionDependencyError,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_manager(bus=None, config=None) -> VisionManager:
    return VisionManager(config=config, bus=bus)


def make_mock_screenshot_engine():
    engine = MagicMock()
    engine.is_available = True
    engine.capture_fullscreen.return_value = ScreenshotResult(
        image_data=b"\x89PNG", width=1920, height=1080, source="fullscreen"
    )
    engine.capture_region.return_value = ScreenshotResult(
        image_data=b"\x89PNG_region", width=100, height=100,
        region=(0, 0, 100, 100), source="region"
    )
    engine.capture_active_window.return_value = ScreenshotResult(
        image_data=b"\x89PNG_window", width=800, height=600, source="window"
    )
    engine.get_monitors.return_value = []
    return engine


def make_mock_ocr_engine():
    engine = MagicMock()
    engine.is_available = True
    engine.read_image.return_value = OCRResult(
        text="Hello World",
        confidence=0.9,
        blocks=(OCRBlock(text="Hello World", confidence=0.9),),
    )
    return engine


def make_mock_screen_analyzer():
    engine = MagicMock()
    engine.is_available = True
    engine.analyze_screen.return_value = ScreenAnalysis(
        active_app="Code.exe",
        active_window="main.py",
        visible_windows=("Code.exe", "Chrome"),
        ui_regions=(),
        timestamp=1234567890.0,
    )
    return engine


# ─── Lifecycle ────────────────────────────────────────────────────────────────


class TestVisionManagerLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_creates_engines(self):
        manager = make_manager()
        await manager.initialize()
        # After init, engines exist (may be None if disabled)
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        manager = make_manager()
        await manager.initialize()
        await manager.initialize()  # second call is no-op
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_close_resets_initialized(self):
        manager = make_manager()
        await manager.initialize()
        await manager.close()
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_auto_initialize_on_capture(self):
        manager = make_manager()
        assert manager._initialized is False

        # Patch the screenshot engine after init
        with patch.object(manager, "_do_capture", return_value=ScreenshotResult(source="fullscreen")):
            manager._screenshot_engine = make_mock_screenshot_engine()
            manager._initialized = True  # skip real init
            result = await manager.capture()

        assert result.source == "fullscreen"

    @pytest.mark.asyncio
    async def test_constructor_without_bus(self):
        manager = VisionManager(bus=None)
        await manager.initialize()
        # Should not raise
        assert manager._bus is None

    @pytest.mark.asyncio
    async def test_constructor_defaults(self):
        manager = VisionManager()
        assert manager._screenshot_enabled is True
        assert manager._ocr_enabled is True
        assert manager._analysis_enabled is True
        assert manager._default_monitor == 0
        assert manager._ocr_language == "en"


# ─── capture() ───────────────────────────────────────────────────────────────


class TestVisionManagerCapture:
    @pytest.mark.asyncio
    async def test_capture_returns_screenshot_result(self):
        manager = make_manager()
        manager._initialized = True
        manager._screenshot_engine = make_mock_screenshot_engine()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080, source="fullscreen")
            )
            result = await manager.capture(source="fullscreen")

        assert isinstance(result, ScreenshotResult)
        assert result.width == 1920

    @pytest.mark.asyncio
    async def test_capture_returns_empty_when_no_engine(self):
        manager = make_manager()
        manager._initialized = True
        manager._screenshot_engine = None

        result = await manager.capture()
        assert isinstance(result, ScreenshotResult)
        assert result.is_empty

    @pytest.mark.asyncio
    async def test_capture_publishes_started_and_completed_events(self):
        published = []

        async def fake_bus_publish(event):
            published.append(event)
            return 1

        mock_bus = MagicMock()
        mock_bus.publish = fake_bus_publish

        manager = make_manager(bus=mock_bus)
        manager._initialized = True
        manager._screenshot_engine = make_mock_screenshot_engine()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080)
            )
            await manager.capture(source="fullscreen")

        topics = [e.topic for e in published]
        assert "vision.capture_started" in topics
        assert "vision.capture_completed" in topics

    @pytest.mark.asyncio
    async def test_capture_publishes_error_on_dep_error(self):
        published = []

        async def fake_bus_publish(event):
            published.append(event)
            return 1

        mock_bus = MagicMock()
        mock_bus.publish = fake_bus_publish

        manager = make_manager(bus=mock_bus)
        manager._initialized = True
        manager._screenshot_engine = make_mock_screenshot_engine()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=VisionDependencyError("mss not installed")
            )
            result = await manager.capture()

        assert result.is_empty
        topics = [e.topic for e in published]
        assert "vision.error" in topics

    @pytest.mark.asyncio
    async def test_capture_without_bus_does_not_raise(self):
        manager = make_manager(bus=None)
        manager._initialized = True
        manager._screenshot_engine = make_mock_screenshot_engine()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080)
            )
            # Should not raise even without bus
            result = await manager.capture()
        assert isinstance(result, ScreenshotResult)


# ─── _do_capture() dispatch ──────────────────────────────────────────────────


class TestDoCapture:
    def test_dispatch_fullscreen(self):
        manager = VisionManager()
        engine = make_mock_screenshot_engine()
        manager._screenshot_engine = engine

        manager._do_capture("fullscreen", None, 0, 0, 0, 0)
        engine.capture_fullscreen.assert_called_once()

    def test_dispatch_window(self):
        manager = VisionManager()
        engine = make_mock_screenshot_engine()
        manager._screenshot_engine = engine

        manager._do_capture("window", None, 0, 0, 0, 0)
        engine.capture_active_window.assert_called_once()

    def test_dispatch_region(self):
        manager = VisionManager()
        engine = make_mock_screenshot_engine()
        manager._screenshot_engine = engine

        manager._do_capture("region", None, 10, 20, 100, 200)
        engine.capture_region.assert_called_once_with(10, 20, 100, 200, 0)

    def test_dispatch_region_falls_back_to_fullscreen_when_zero_dims(self):
        manager = VisionManager()
        engine = make_mock_screenshot_engine()
        manager._screenshot_engine = engine

        manager._do_capture("region", None, 0, 0, 0, 0)  # width=0, height=0
        # Falls back to fullscreen
        engine.capture_fullscreen.assert_called_once()


# ─── read_screen_text() ───────────────────────────────────────────────────────


class TestReadScreenText:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_ocr_engine(self):
        manager = make_manager()
        manager._initialized = True
        manager._ocr_engine = None

        result = await manager.read_screen_text()
        assert result.available is False
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_publishes_ocr_completed_event(self):
        published = []

        async def fake_publish(event):
            published.append(event)
            return 1

        mock_bus = MagicMock()
        mock_bus.publish = fake_publish

        manager = make_manager(bus=mock_bus)
        manager._initialized = True
        manager._screenshot_engine = make_mock_screenshot_engine()
        manager._ocr_engine = make_mock_ocr_engine()

        with patch("asyncio.get_event_loop") as mock_loop:
            run_executor = AsyncMock()
            # First call = capture, second call = OCR
            run_executor.side_effect = [
                ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080),
                OCRResult(text="Hello", confidence=0.9, blocks=()),
            ]
            mock_loop.return_value.run_in_executor = run_executor
            await manager.read_screen_text()

        topics = [e.topic for e in published]
        assert "vision.ocr_completed" in topics

    @pytest.mark.asyncio
    async def test_returns_empty_when_screenshot_empty(self):
        manager = make_manager()
        manager._initialized = True
        manager._screenshot_engine = make_mock_screenshot_engine()
        manager._ocr_engine = make_mock_ocr_engine()

        # Patch capture to return empty screenshot
        with patch.object(manager, "capture", return_value=ScreenshotResult()):
            result = await manager.read_screen_text()

        assert result.available is False


# ─── analyze_screen() ─────────────────────────────────────────────────────────


class TestAnalyzeScreen:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_analyzer(self):
        manager = make_manager()
        manager._initialized = True
        manager._screen_analyzer = None

        result = await manager.analyze_screen()
        assert isinstance(result, ScreenAnalysis)
        assert result.active_app == ""

    @pytest.mark.asyncio
    async def test_publishes_analysis_completed(self):
        published = []

        async def fake_publish(event):
            published.append(event)
            return 1

        mock_bus = MagicMock()
        mock_bus.publish = fake_publish

        manager = make_manager(bus=mock_bus)
        manager._initialized = True
        manager._screenshot_engine = None  # skip screenshot
        manager._screen_analyzer = make_mock_screen_analyzer()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=ScreenAnalysis(
                    active_app="Code.exe",
                    visible_windows=("Code.exe",),
                )
            )
            await manager.analyze_screen()

        topics = [e.topic for e in published]
        assert "vision.analysis_completed" in topics

    @pytest.mark.asyncio
    async def test_returns_analysis_with_active_app(self):
        manager = make_manager()
        manager._initialized = True
        manager._screenshot_engine = None
        manager._screen_analyzer = make_mock_screen_analyzer()

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=ScreenAnalysis(active_app="Code.exe")
            )
            result = await manager.analyze_screen()

        assert isinstance(result, ScreenAnalysis)


# ─── describe_screen() ───────────────────────────────────────────────────────


class TestDescribeScreen:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        manager = make_manager()
        manager._initialized = True
        manager._screenshot_engine = None
        manager._screen_analyzer = None
        manager._ocr_engine = None

        result = await manager.describe_screen()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_contains_screen_info_when_available(self):
        manager = make_manager()
        manager._initialized = True
        manager._screenshot_engine = None
        manager._ocr_engine = None

        analysis = ScreenAnalysis(
            active_app="TestApp",
            active_window="Test Window",
        )
        with patch.object(manager, "analyze_screen", return_value=analysis):
            with patch.object(manager, "read_screen_text", return_value=OCRResult(text="")):
                result = await manager.describe_screen()

        assert "TestApp" in result or "Test Window" in result

    @pytest.mark.asyncio
    async def test_fallback_message_when_all_fail(self):
        manager = make_manager()
        manager._initialized = True

        with patch.object(manager, "analyze_screen", side_effect=Exception("fail")):
            with patch.object(manager, "read_screen_text", side_effect=Exception("fail")):
                result = await manager.describe_screen()

        assert "could not be determined" in result.lower()


# ─── VisionManager with disabled config ───────────────────────────────────────


class TestVisionManagerDisabled:
    @pytest.mark.asyncio
    async def test_all_disabled_still_initializes(self):
        """Even with all engines disabled, initialize() should not crash."""
        from spidy.config.manager import VisionConfig
        config = VisionConfig(
            enabled=True,
            enable_screenshot=False,
            enable_ocr=False,
            enable_screen_analysis=False,
        )
        manager = VisionManager(config=config)
        await manager.initialize()
        assert manager._screenshot_engine is None
        assert manager._ocr_engine is None
        assert manager._screen_analyzer is None

    @pytest.mark.asyncio
    async def test_capture_with_no_engine_returns_empty(self):
        from spidy.config.manager import VisionConfig
        config = VisionConfig(enable_screenshot=False)
        manager = VisionManager(config=config)
        await manager.initialize()
        result = await manager.capture()
        assert result.is_empty
