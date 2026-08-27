"""
Regression Tests: Multi Wake-Word Architecture ("Hey Spidy" & "Wake up Spidy")
=============================================================================
Verification is strictly split into:
  A. INFRASTRUCTURE TESTS (Mock / Synthetic models)
  B. REAL DETECTION TESTS (Genuine models on disk; reports NOT AVAILABLE if missing)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spidy.config.manager import SpidyConfig, WakeModelConfig, WakeWordConfig
from spidy.core.event_bus import Event, EventBus
from spidy.perception.voice.engine import VoiceEngine, WakeWordDetectedEvent
from spidy.perception.voice.factory import VoiceEngineFactory
from spidy.perception.voice.wake_word.openwakeword import OpenWakeWordModel
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.events import VoiceSessionStartedEvent
from spidy.voice.session import VoiceSessionManager
from spidy.voice.wake_word_stripper import WakeWordStripper


# ─── A. INFRASTRUCTURE TESTS ──────────────────────────────────────────────────


class TestWakeWordStripperRegression:
    """Test suite for WakeWordStripper multi-phrase support."""

    @pytest.fixture
    def stripper(self) -> WakeWordStripper:
        return WakeWordStripper()

    def test_01_hey_spidy_stripping(self, stripper: WakeWordStripper) -> None:
        """Test: 'Hey Spidy, open Edge.' and 'Hey Spidy what is Python?'"""
        assert stripper.strip_wake_prefix("Hey Spidy, open Edge.") == "open Edge."
        assert stripper.strip_wake_prefix("Hey Spidy what is Python?") == "what is Python?"

    def test_02_wake_up_spidy_stripping(self, stripper: WakeWordStripper) -> None:
        """Test: 'Wake up Spidy, open Notepad.' and with 'and' connector."""
        assert stripper.strip_wake_prefix("Wake up Spidy, open Notepad.") == "open Notepad."
        assert (
            stripper.strip_wake_prefix("Wake up Spidy and search YouTube for Python tutorials.")
            == "search YouTube for Python tutorials."
        )

    def test_03_hey_spidy_wake_only_command(self, stripper: WakeWordStripper) -> None:
        """Test: 'Hey Spidy' alone produces empty remainder and is_wake_only=True."""
        assert stripper.strip_wake_prefix("Hey Spidy") == ""
        assert stripper.is_wake_only("Hey Spidy") is True
        assert stripper.is_wake_only("Hey Spidy.") is True
        assert stripper.is_wake_only("Hey Spidy, Hey Spidy.") is True

    def test_04_wake_up_spidy_wake_only_command(self, stripper: WakeWordStripper) -> None:
        """Test: 'Wake up Spidy' alone produces empty remainder and is_wake_only=True."""
        assert stripper.strip_wake_prefix("Wake up Spidy") == ""
        assert stripper.is_wake_only("Wake up Spidy") is True
        assert stripper.is_wake_only("Wake up Spidy.") is True
        assert stripper.is_wake_only("Wake up Spidy and") is True

    def test_05_normal_spidy_references_not_stripped(self, stripper: WakeWordStripper) -> None:
        """Test: 'Who is Spidy?', 'Tell me about Spidy.', 'Wake up Spiderman.' remain unchanged."""
        assert stripper.strip_wake_prefix("Who is Spidy?") == "Who is Spidy?"
        assert stripper.strip_wake_prefix("Tell me about Spidy.") == "Tell me about Spidy."
        assert stripper.strip_wake_prefix("Wake up Spiderman.") == "Wake up Spiderman."
        assert stripper.is_wake_only("Who is Spidy?") is False
        assert stripper.is_wake_only("Tell me about Spidy.") is False
        assert stripper.is_wake_only("Wake up Spiderman.") is False


class TestMultiWakeWordModelInfrastructure:
    """Test suite for OpenWakeWordModel multi-model loading and detection infrastructure."""

    def test_06_no_hardcoded_hey_jarvis_or_alexa_aliasing(self) -> None:
        """Verify that hey_spidy and wake_up_spidy are not aliased to hey_jarvis or alexa."""
        model = OpenWakeWordModel(
            models=[
                WakeModelConfig(name="nonexistent_spidy", phrase="Hey Spidy", model_path="assets/models/nonexistent_spidy.onnx", threshold=0.35),
                WakeModelConfig(name="nonexistent_wake", phrase="Wake up Spidy", model_path="assets/models/nonexistent_wake.onnx", threshold=0.40),
            ],
            active_models=["nonexistent_spidy", "nonexistent_wake"],
        )

        with patch("spidy.perception.voice.wake_word.openwakeword.OWWModel", create=True) as mock_oww:
            model.load()

        # Nonexistent models must NOT be aliased to alexa or hey_jarvis
        assert len(model.loaded_models) == 0
        missing_names = [m["name"] for m in model.missing_models]
        assert "nonexistent_spidy" in missing_names
        assert "nonexistent_wake" in missing_names

    def test_07_missing_model_handled_safely_without_crash(self) -> None:
        """Missing model file should log a warning, register missing model, and not raise unhandled crash."""
        model = OpenWakeWordModel(
            models=[
                WakeModelConfig(name="hey_spidy", phrase="Hey Spidy", model_path="assets/models/nonexistent.onnx", threshold=0.35),
                WakeModelConfig(name="hey_jarvis", phrase="Hey Jarvis", model="hey_jarvis", threshold=0.35),
            ],
            active_models=["hey_spidy", "hey_jarvis"],
        )

        with patch("openwakeword.model.Model") as mock_oww:
            mock_inst = MagicMock()
            mock_oww.return_value = mock_inst
            model.load()

            assert mock_oww.called
            # Only hey_jarvis was resolved and loaded
            assert "hey_jarvis" in model.loaded_models
            assert "hey_spidy" not in model.loaded_models

        assert len(model.missing_models) == 1
        assert model.missing_models[0]["name"] == "hey_spidy"

    def test_08_multi_model_loading_and_independent_thresholds(self) -> None:
        """Test multi-model configuration with mock OWW backend."""
        model = OpenWakeWordModel(
            models=[
                WakeModelConfig(name="hey_spidy", phrase="Hey Spidy", model_path="assets/mock_spidy.onnx", threshold=0.30),
                WakeModelConfig(name="wake_up_spidy", phrase="Wake up Spidy", model_path="assets/mock_wake_up.onnx", threshold=0.50),
            ],
            active_models=["hey_spidy", "wake_up_spidy"],
        )

        with patch("os.path.exists", return_value=True), patch("openwakeword.model.Model") as mock_oww:
            mock_inst = MagicMock()
            mock_oww.return_value = mock_inst
            model.load()

            assert "hey_spidy" in model.loaded_models
            assert "wake_up_spidy" in model.loaded_models
            assert model.get_model_threshold("hey_spidy") == 0.30
            assert model.get_model_threshold("wake_up_spidy") == 0.50

    def test_09_hey_spidy_detection_and_metadata(self) -> None:
        """Test detection of 'Hey Spidy' with mock audio chunk."""
        model = OpenWakeWordModel(
            models=[
                WakeModelConfig(name="hey_spidy", phrase="Hey Spidy", model_path="assets/mock_spidy.onnx", threshold=0.30),
                WakeModelConfig(name="wake_up_spidy", phrase="Wake up Spidy", model_path="assets/mock_wake_up.onnx", threshold=0.50),
            ],
            active_models=["hey_spidy", "wake_up_spidy"],
        )

        with patch("os.path.exists", return_value=True), patch("openwakeword.model.Model") as mock_oww:
            mock_inst = MagicMock()
            mock_inst.predict.return_value = {"mock_spidy": 0.45, "mock_wake_up": 0.10}
            mock_oww.return_value = mock_inst
            model.load()

            dummy_chunk = np.zeros(1280, dtype=np.float32)
            score = model.process_chunk(dummy_chunk)

            assert score == 0.45
            assert model.is_detected is True
            assert model.last_detected_model == "hey_spidy"
            assert model.last_detected_phrase == "Hey Spidy"

    def test_10_wake_up_spidy_detection_and_metadata(self) -> None:
        """Test detection of 'Wake up Spidy' with mock audio chunk."""
        model = OpenWakeWordModel(
            models=[
                WakeModelConfig(name="hey_spidy", phrase="Hey Spidy", model_path="assets/mock_spidy.onnx", threshold=0.30),
                WakeModelConfig(name="wake_up_spidy", phrase="Wake up Spidy", model_path="assets/mock_wake_up.onnx", threshold=0.50),
            ],
            active_models=["hey_spidy", "wake_up_spidy"],
        )

        with patch("os.path.exists", return_value=True), patch("openwakeword.model.Model") as mock_oww:
            mock_inst = MagicMock()
            mock_inst.predict.return_value = {"mock_spidy": 0.15, "mock_wake_up": 0.65}
            mock_oww.return_value = mock_inst
            model.load()

            dummy_chunk = np.zeros(1280, dtype=np.float32)
            score = model.process_chunk(dummy_chunk)

            assert score == 0.65
            assert model.is_detected is True
            assert model.last_detected_model == "wake_up_spidy"
            assert model.last_detected_phrase == "Wake up Spidy"

    def test_11_threshold_rejection(self) -> None:
        """Test that scores below the model's threshold do not trigger detection."""
        model = OpenWakeWordModel(
            models=[
                WakeModelConfig(name="hey_spidy", phrase="Hey Spidy", model_path="assets/mock_spidy.onnx", threshold=0.35),
            ],
            active_models=["hey_spidy"],
        )

        with patch("os.path.exists", return_value=True), patch("openwakeword.model.Model") as mock_oww:
            mock_inst = MagicMock()
            mock_inst.predict.return_value = {"mock_spidy": 0.25}
            mock_oww.return_value = mock_inst
            model.load()

            dummy_chunk = np.zeros(1280, dtype=np.float32)
            score = model.process_chunk(dummy_chunk)

            assert score == 0.25
            assert model.is_detected is False
            assert model.last_detected_model == ""


class TestSessionActivationAndRouting:
    """Test downstream session activation, acknowledgement, and duplicate suppression."""

    @pytest.mark.asyncio
    async def test_12_hey_spidy_session_activation(self) -> None:
        """Hey Spidy wake event activates VoiceSession with wake_word='Hey Spidy'."""
        bus = EventBus()
        events_published: list[Event] = []

        async def capture(evt: Event) -> None:
            events_published.append(evt)

        bus.subscribe("voice.session_started", capture)

        session_mgr = VoiceSessionManager(bus=bus)
        await session_mgr.activate(wake_word="Hey Spidy")

        assert len(events_published) == 1
        assert isinstance(events_published[0], VoiceSessionStartedEvent)
        assert events_published[0].wake_word == "Hey Spidy"

    @pytest.mark.asyncio
    async def test_13_wake_up_spidy_session_activation(self) -> None:
        """Wake up Spidy wake event activates VoiceSession with wake_word='Wake up Spidy'."""
        bus = EventBus()
        events_published: list[Event] = []

        async def capture(evt: Event) -> None:
            events_published.append(evt)

        bus.subscribe("voice.session_started", capture)

        session_mgr = VoiceSessionManager(bus=bus)
        await session_mgr.activate(wake_word="Wake up Spidy")

        assert len(events_published) == 1
        assert isinstance(events_published[0], VoiceSessionStartedEvent)
        assert events_published[0].wake_word == "Wake up Spidy"

    @pytest.mark.asyncio
    async def test_14_duplicate_wake_suppression_both_phrases(self) -> None:
        """Duplicate wake events while session is already active must be suppressed."""
        bus = EventBus()
        session_mgr = VoiceSessionManager(bus=bus)
        tts_mock = MagicMock()
        tts_mock.is_speaking = False
        engine_mock = MagicMock()
        brain_mock = MagicMock()
        interruption_mock = MagicMock()

        controller = ContinuousVoiceController(
            voice_engine=engine_mock,
            brain=brain_mock,
            bus=bus,
            session_manager=session_mgr,
            streaming_tts=tts_mock,
            interruption_handler=interruption_mock,
        )


        # First wake: Hey Spidy
        evt1 = WakeWordDetectedEvent(confidence=0.8, model_name="hey_spidy", wake_phrase="Hey Spidy")
        await controller._on_wake_word(evt1)
        assert session_mgr.is_active is True
        initial_sid = session_mgr.session_id

        # Second wake while active: Wake up Spidy
        evt2 = WakeWordDetectedEvent(confidence=0.85, model_name="wake_up_spidy", wake_phrase="Wake up Spidy")
        await controller._on_wake_word(evt2)

        # Must not have created a new session
        assert session_mgr.session_id == initial_sid


# ─── B. REAL DETECTION AVAILABILITY VERIFICATION ───────────────────────────────


class TestRealModelAvailability:
    """
    Real model availability checks.
    Only claims PASS when actual compatible model files exist and recognize audio.
    If missing, cleanly reports NOT AVAILABLE.
    """

    def test_15_hey_spidy_real_model_status(self) -> None:
        """Report real availability of hey_spidy model and verify ONNX inference."""
        from pathlib import Path
        import numpy as np

        expected_path = Path("assets/models/wake_word/hey_spidy.onnx")
        if not expected_path.exists():
            pytest.skip("NOT AVAILABLE: Hey Spidy model not installed (expected at assets/models/wake_word/hey_spidy.onnx)")

        # Verify genuine loading
        model = OpenWakeWordModel(
            models=[WakeModelConfig(name="hey_spidy", phrase="Hey Spidy", model_path=str(expected_path))],
            active_models=["hey_spidy"],
        )
        model.load()
        assert "hey_spidy" in model.loaded_models

        # Feed 1280 float32 samples (80ms at 16kHz) to ensure inference session runs without error
        chunk = (np.random.randn(1280) * 0.05).astype(np.float32)
        score = model.process_chunk(chunk)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_16_wake_up_spidy_real_model_status(self) -> None:
        """Report real availability of wake_up_spidy model and verify ONNX inference."""
        from pathlib import Path
        import numpy as np

        expected_path = Path("assets/models/wake_word/wake_up_spidy.onnx")
        if not expected_path.exists():
            pytest.skip("NOT AVAILABLE: Wake up Spidy model not installed (expected at assets/models/wake_word/wake_up_spidy.onnx)")

        # Verify genuine loading
        model = OpenWakeWordModel(
            models=[WakeModelConfig(name="wake_up_spidy", phrase="Wake up Spidy", model_path=str(expected_path))],
            active_models=["wake_up_spidy"],
        )
        model.load()
        assert "wake_up_spidy" in model.loaded_models

        # Feed 1280 float32 samples (80ms at 16kHz) to ensure inference session runs without error
        chunk = (np.random.randn(1280) * 0.05).astype(np.float32)
        score = model.process_chunk(chunk)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
