"""
Integration tests: VisionManager ↔ Brain

Verifies that:
- Brain accepts vision=VisionManager() param without errors
- brain.vision property returns the manager
- Brain works normally with vision=None (no regression)
- Vision events are published independently of Brain processing
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.brain.brain import Brain
from spidy.brain.interfaces import VisionInterface
from spidy.core.event_bus import EventBus
from spidy.skills.registry import SkillRegistry
from spidy.vision.manager import VisionManager
from spidy.vision.types import OCRResult, ScreenAnalysis, ScreenshotResult, VisionDependencyError


# ─── Minimal Brain setup ──────────────────────────────────────────────────────


def make_brain(bus: EventBus, vision=None) -> Brain:
    """Create a minimal Brain for testing."""
    skill_registry = SkillRegistry()
    brain = Brain(
        bus=bus,
        config=MagicMock(
            intent_classifier="heuristic",
            min_intent_confidence=0.0,
            max_conversation_turns=20,
            decision_mode="auto",
            tool_routing_enabled=True,
        ),
        skill_registry=skill_registry,
        llm_client=None,
        memory=None,
        vision=vision,
        user_name="TestUser",
    )
    return brain


# ─── VisionInterface contract ─────────────────────────────────────────────────


class StubVisionManager(VisionInterface):
    """Minimal stub implementing VisionInterface for testing."""

    async def capture(self, source="fullscreen", **kwargs):
        return ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080)

    async def read_screen_text(self):
        return OCRResult(text="stub text", confidence=0.9, blocks=())

    async def analyze_screen(self):
        return ScreenAnalysis(active_app="stub.exe", visible_windows=("stub.exe",))

    async def describe_screen(self) -> str:
        return "Active application: stub.exe\nVisible text: stub text"


class TestVisionInterfaceContract:
    def test_stub_implements_vision_interface(self):
        stub = StubVisionManager()
        assert isinstance(stub, VisionInterface)

    @pytest.mark.asyncio
    async def test_stub_capture_returns_screenshot(self):
        stub = StubVisionManager()
        result = await stub.capture()
        assert isinstance(result, ScreenshotResult)
        assert not result.is_empty

    @pytest.mark.asyncio
    async def test_stub_read_text_returns_ocr_result(self):
        stub = StubVisionManager()
        result = await stub.read_screen_text()
        assert isinstance(result, OCRResult)
        assert result.text == "stub text"

    @pytest.mark.asyncio
    async def test_stub_analyze_returns_screen_analysis(self):
        stub = StubVisionManager()
        result = await stub.analyze_screen()
        assert isinstance(result, ScreenAnalysis)
        assert result.active_app == "stub.exe"

    @pytest.mark.asyncio
    async def test_stub_describe_returns_string(self):
        stub = StubVisionManager()
        result = await stub.describe_screen()
        assert isinstance(result, str)
        assert "stub.exe" in result


# ─── Brain ↔ Vision integration ───────────────────────────────────────────────


class TestBrainVisionIntegration:
    @pytest.mark.asyncio
    async def test_brain_accepts_vision_param(self):
        """Brain constructor must accept vision= without raising."""
        bus = EventBus()
        stub = StubVisionManager()
        brain = make_brain(bus=bus, vision=stub)
        assert brain.vision is stub

    @pytest.mark.asyncio
    async def test_brain_vision_property_is_none_by_default(self):
        """brain.vision is None when not supplied."""
        bus = EventBus()
        brain = make_brain(bus=bus, vision=None)
        assert brain.vision is None

    @pytest.mark.asyncio
    async def test_brain_starts_with_vision(self):
        """Brain.start() succeeds with vision= provided."""
        bus = EventBus()
        stub = StubVisionManager()
        brain = make_brain(bus=bus, vision=stub)
        await brain.start()
        assert brain.is_running
        await brain.stop()

    @pytest.mark.asyncio
    async def test_brain_starts_without_vision(self):
        """Brain.start() still works with vision=None."""
        bus = EventBus()
        brain = make_brain(bus=bus, vision=None)
        await brain.start()
        assert brain.is_running
        await brain.stop()

    @pytest.mark.asyncio
    async def test_brain_process_does_not_call_vision(self):
        """Brain.process() does not call vision in the standard pipeline."""
        bus = EventBus()
        stub = StubVisionManager()

        # Track calls
        capture_calls = []
        original_capture = stub.capture

        async def tracking_capture(*args, **kwargs):
            capture_calls.append((args, kwargs))
            return await original_capture(*args, **kwargs)

        stub.capture = tracking_capture  # type: ignore[method-assign]

        brain = make_brain(bus=bus, vision=stub)
        await brain.start()

        # Process an utterance — brain.process() should complete without calling vision.capture
        await brain.process("hello")

        # Vision was NOT called during normal Brain.process()
        assert capture_calls == []
        await brain.stop()

    @pytest.mark.asyncio
    async def test_vision_manager_works_standalone(self):
        """VisionManager can initialize and close without Brain."""
        manager = VisionManager(bus=None)
        await manager.initialize()
        assert manager._initialized is True
        await manager.close()
        assert manager._initialized is False


# ─── Config integration ───────────────────────────────────────────────────────


class TestVisionConfigIntegration:
    def test_vision_config_has_defaults(self):
        from spidy.config.manager import VisionConfig
        config = VisionConfig()
        assert config.enabled is True
        assert config.enable_screenshot is True
        assert config.enable_ocr is True
        assert config.enable_screen_analysis is True

    def test_vision_config_sub_models(self):
        from spidy.config.manager import OCRConfig, ScreenshotConfig, VisionConfig
        config = VisionConfig()
        assert isinstance(config.screenshot, ScreenshotConfig)
        assert isinstance(config.ocr, OCRConfig)
        assert config.screenshot.default_monitor == 0
        assert config.ocr.language == "en"
        assert config.ocr.confidence_threshold == 0.5

    def test_spidy_config_has_vision_field(self):
        from spidy.config.manager import SpidyConfig, VisionConfig
        config = SpidyConfig()
        assert hasattr(config, "vision")
        assert isinstance(config.vision, VisionConfig)

    def test_vision_manager_uses_config(self):
        from spidy.config.manager import OCRConfig, ScreenshotConfig, VisionConfig
        config = VisionConfig(
            enable_screenshot=True,
            enable_ocr=False,
            enable_screen_analysis=False,
            screenshot=ScreenshotConfig(default_monitor=1),
            ocr=OCRConfig(language="de", confidence_threshold=0.7),
        )
        manager = VisionManager(config=config)
        assert manager._ocr_enabled is False
        assert manager._analysis_enabled is False
        assert manager._default_monitor == 1
        assert manager._ocr_language == "de"
        assert manager._ocr_confidence == 0.7


# ─── EventBus vision events ───────────────────────────────────────────────────


class TestVisionEventsIntegration:
    @pytest.mark.asyncio
    async def test_vision_events_published_to_bus(self):
        """When capture is called, VisionCaptureStartedEvent is published."""
        collected = []

        async def handler(event):
            collected.append(event)

        bus = EventBus()
        bus.subscribe("vision.capture_started", handler)
        bus.subscribe("vision.capture_completed", handler)

        manager = VisionManager(bus=bus)
        manager._initialized = True
        manager._screenshot_engine = MagicMock()
        manager._screenshot_engine.is_available = True

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=ScreenshotResult(image_data=b"\x89PNG", width=1920, height=1080)
            )
            await manager.capture(source="fullscreen")

        assert len(collected) == 2
        topics = [e.topic for e in collected]
        assert "vision.capture_started" in topics
        assert "vision.capture_completed" in topics

    @pytest.mark.asyncio
    async def test_vision_error_event_on_dep_missing(self):
        """VisionErrorEvent published when capture fails."""
        collected = []

        async def handler(event):
            collected.append(event)

        bus = EventBus()
        bus.subscribe("vision.error", handler)

        manager = VisionManager(bus=bus)
        manager._initialized = True
        manager._screenshot_engine = MagicMock()
        manager._screenshot_engine.is_available = True

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=VisionDependencyError("mss missing")
            )
            await manager.capture()

        assert len(collected) == 1
        assert collected[0].topic == "vision.error"
        assert "mss" in collected[0].error
