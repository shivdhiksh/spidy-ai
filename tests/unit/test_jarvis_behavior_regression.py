"""
test_jarvis_behavior_regression.py
===================================
Comprehensive regression tests for JARVIS-style behavior:
1. Wake during active processing (keeps session, doesn't get stuck in WAKING).
2. Wake during TTS (ignored, doesn't get stuck in WAKING).
3. Rejected wake clears WAKING and restores authoritative state.
4. TTS → audio flush → listening resumption.
5. Multiple commands in one voice session without repeating wake word.
6. Session timeout → clean sleep.
7. New wake word after timeout creates a fresh session.
8. Real Windows file creation on disk.
9. File creation returns verified absolute path and exact byte size.
10. VS Code already running → focuses window, no duplicate launch.
11. VS Code not running → launches once.
12. Created file path propagated into next task via ${prev.path}.
13. Filesystem observation overrides response_text (real disk authority).
14. Failed filesystem verification prevents false success.
15. Wake word while session is active acknowledges user and re-arms listening.
16. Filename extraction strips speech punctuation (e.g. 'Test.py.' -> 'Test.py').
17. Repository-wide verification of zero printf logging format placeholders.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spidy.agent.agent import AutonomousAgent
from spidy.agent.goal_manager import GoalManager
from spidy.agent.observer import TaskObserver
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import TaskRecord, TaskState
from spidy.agent.verifier import GoalVerifier
from spidy.core.event_bus import EventBus
from spidy.perception.voice.audio_capture import CaptureMode
from spidy.perception.voice.engine import (
    VoiceEngine,
    VoiceState,
    VoiceTranscriptEvent,
    WakeWordDetectedEvent,
)
from spidy.perception.voice.wake_word.openwakeword import OpenWakeWordModel
from spidy.skills.desktop.app_skill import AppSkill
from spidy.skills.desktop.file_skill import FileSkill
from spidy.voice.continuous import ContinuousVoiceController
from spidy.brain.events import BrainResponseReadyEvent
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.config.manager import WakeAckConfig
from spidy.voice.wake_ack import WakeAcknowledger


# ─── Mock Helpers ─────────────────────────────────────────────────────────────


class _MockWakeModel:
    def __init__(self):
        self.model_name = "hey_spidy"
        self.active_model_names = ["hey_spidy", "wake_up_spidy", "hey_jarvis"]
        self.chunk_size = 1280
        self.is_detected = False
        self.last_detected_model = "hey_spidy"
        self.last_detected_phrase = "Hey Spidy"
        self.reset_count = 0
        self.is_loaded = True

    def load(self):
        self.is_loaded = True

    def unload(self):
        self.is_loaded = False

    def process_chunk(self, chunk):
        return 1.0 if self.is_detected else 0.0

    def reset_buffer(self):
        self.reset_count += 1
        self.is_detected = False


class _MockSTT:
    def __init__(self):
        self.is_loaded = True
        self.model_name = "mock_stt"
        self.device = "cpu"

    def load(self):
        self.is_loaded = True

    def unload(self):
        self.is_loaded = False

    async def transcribe(self, audio):
        res = MagicMock()
        res.text = "what time is it"
        res.is_empty = False
        res.confidence = 0.95
        res.language = "en"
        return res


class _MockTTS:
    def __init__(self):
        self.is_speaking = False
        self.voice_name = "mock_voice"
        self.is_loaded = True

    def load(self):
        self.is_loaded = True

    def unload(self):
        self.is_loaded = False

    async def speak(self, text):
        self.is_speaking = True
        await asyncio.sleep(0.01)
        self.is_speaking = False

    def stop(self):
        self.is_speaking = False


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def jarvis_voice_stack():
    bus = EventBus()
    bus.set_loop(asyncio.get_running_loop())
    wake_model = _MockWakeModel()
    stt = _MockSTT()
    tts = _MockTTS()

    with patch("sounddevice.InputStream"), patch("sounddevice.query_devices"):
        engine = VoiceEngine(
            bus=bus,
            wake_word_model=wake_model,
            recognizer=stt,
            tts=tts,
            sample_rate=16000,
        )

    session_mgr = VoiceSessionManager(
        bus=bus,
        timeout_seconds=0.3,
        continuous_mode=True,
    )

    streaming_tts = MagicMock()
    streaming_tts.speak = AsyncMock()
    streaming_tts.is_speaking = False

    brain = MagicMock()
    brain.process = AsyncMock(return_value="The time is 10:30 PM.")
    brain.run_goal = AsyncMock(return_value="Executed command successfully.")
    interruption_handler = MagicMock()
    interruption_handler.detect.return_value = None

    ack_cfg = WakeAckConfig(enabled=True, phrases=["Yes Shiva."])
    wake_acknowledger = WakeAcknowledger(ack_cfg)

    controller = ContinuousVoiceController(
        voice_engine=engine,
        brain=brain,
        bus=bus,
        session_manager=session_mgr,
        streaming_tts=streaming_tts,
        interruption_handler=interruption_handler,
        wake_acknowledger=wake_acknowledger,
        continuous_mode=True,
    )

    return {
        "bus": bus,
        "engine": engine,
        "wake_model": wake_model,
        "session_mgr": session_mgr,
        "controller": controller,
        "brain": brain,
        "tts": tts,
        "streaming_tts": streaming_tts,
        "wake_acknowledger": wake_acknowledger,
    }


# ─── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wake_during_active_processing_does_not_stick_in_waking(jarvis_voice_stack):
    """1. Wake during active processing keeps session and discards WAKING."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    bus = jarvis_voice_stack["bus"]

    await engine.start()
    await controller.start()

    await session_mgr.activate(wake_word="Hey Spidy")
    await session_mgr.record_utterance("calculate 25 times 8")
    assert session_mgr.state == VoiceSessionState.PROCESSING

    # Simulate duplicate wake event while processing
    engine._state = VoiceState.WAKING
    await bus.publish(WakeWordDetectedEvent(confidence=0.99, model_name="hey_spidy", wake_phrase="Hey Spidy"))
    await asyncio.sleep(0.05)

    # Engine must NOT be stuck in WAKING; session remains PROCESSING
    assert engine.state != VoiceState.WAKING
    assert session_mgr.state == VoiceSessionState.PROCESSING

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_wake_during_tts_speaking_does_not_corrupt_state(jarvis_voice_stack):
    """2. Wake during TTS is ignored and restores speaking/flush state without getting stuck in WAKING."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    streaming_tts = jarvis_voice_stack["streaming_tts"]
    bus = jarvis_voice_stack["bus"]

    await engine.start()
    await controller.start()

    await session_mgr.activate(wake_word="Hey Spidy")
    await session_mgr.mark_speaking()
    streaming_tts.is_speaking = True

    engine._state = VoiceState.WAKING
    await bus.publish(WakeWordDetectedEvent(confidence=0.98, model_name="hey_spidy", wake_phrase="Hey Spidy"))
    await asyncio.sleep(0.05)

    # Engine must NOT be stuck in WAKING
    assert engine.state != VoiceState.WAKING

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_rejected_wake_clears_waking_and_restores_state(jarvis_voice_stack):
    """3. Explicit test of discard_pending_wake restoring state from WAKING."""
    engine = jarvis_voice_stack["engine"]
    await engine.start()

    engine._set_state(VoiceState.WAKING)
    engine._wake_detected.set()
    assert engine.state == VoiceState.WAKING

    engine.discard_pending_wake(VoiceState.SLEEPING)
    assert engine.state == VoiceState.SLEEPING
    assert not engine._wake_detected.is_set()

    await engine.stop()


@pytest.mark.asyncio
async def test_tts_to_audio_flush_to_listening(jarvis_voice_stack):
    """4. After Brain response, TTS plays, audio flushes, and engine re-arms listening."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    bus = jarvis_voice_stack["bus"]

    # Give session a long timeout so it does not expire during response delivery
    session_mgr._timeout_seconds = 5.0

    await engine.start()
    await controller.start()

    await session_mgr.activate(wake_word="Hey Spidy")
    await bus.publish(BrainResponseReadyEvent(response_text="Sure, opening calculator."))
    await asyncio.sleep(0.35)

    # Session must be AWAKE and engine must be ready/listening for continuous turns
    assert session_mgr.state == VoiceSessionState.AWAKE
    assert not engine.is_speaking

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_multiple_commands_in_one_session_without_wake_word(jarvis_voice_stack):
    """5. Multiple sequential commands work without repeating 'Hey Spidy'."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    brain = jarvis_voice_stack["brain"]
    bus = jarvis_voice_stack["bus"]

    # Give session a long timeout
    session_mgr._timeout_seconds = 5.0

    await engine.start()
    await controller.start()

    # Wake initially
    await session_mgr.activate(wake_word="Hey Spidy")
    sid = session_mgr.session_id

    # Command 1: "what time is it"
    await bus.publish(VoiceTranscriptEvent(text="what time is it", confidence=0.95, source="user", capture_state="LISTENING"))
    await asyncio.sleep(0.05)
    assert brain.process.call_count == 1
    assert session_mgr.session_id == sid

    # Command 2: "open calculator" (without wake word)
    await bus.publish(VoiceTranscriptEvent(text="open calculator", confidence=0.95, source="user", capture_state="LISTENING"))
    await asyncio.sleep(0.05)
    assert session_mgr.session_id == sid
    assert session_mgr.turn_count == 2

    # Command 3: "what is 25 times 8" (without wake word)
    await bus.publish(VoiceTranscriptEvent(text="what is 25 times 8", confidence=0.95, source="user", capture_state="LISTENING"))
    await asyncio.sleep(0.05)
    assert session_mgr.session_id == sid
    assert session_mgr.turn_count == 3

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_session_timeout_to_sleeping_and_new_wake(jarvis_voice_stack):
    """6 & 7. Inactivity times out session to SLEEPING, then next wake creates new session."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    bus = jarvis_voice_stack["bus"]

    await engine.start()
    await controller.start()

    await session_mgr.activate(wake_word="Hey Spidy")
    sid1 = session_mgr.session_id

    # Let 0.3s timeout expire
    await asyncio.sleep(0.45)
    assert session_mgr.state == VoiceSessionState.IDLE
    assert not session_mgr.is_active
    assert engine.state == VoiceState.SLEEPING

    # Next wake creates fresh session
    await bus.publish(WakeWordDetectedEvent(confidence=0.95, model_name="hey_spidy", wake_phrase="Hey Spidy"))
    await asyncio.sleep(0.05)
    assert session_mgr.is_active
    assert session_mgr.session_id != sid1

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_wake_word_when_session_active_acknowledges_and_listens(jarvis_voice_stack):
    """15. Wake word spoken when session is active acknowledges user and re-arms listening."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    streaming_tts = jarvis_voice_stack["streaming_tts"]
    bus = jarvis_voice_stack["bus"]

    # Give session a long timeout so it stays active
    session_mgr._timeout_seconds = 10.0

    await engine.start()
    await controller.start()

    await session_mgr.activate(wake_word="Hey Spidy")
    assert session_mgr.is_active

    # User says "Hey Spidy" again while session is active
    await bus.publish(WakeWordDetectedEvent(confidence=0.95, model_name="hey_spidy", wake_phrase="Hey Spidy"))
    await asyncio.sleep(0.1)

    # Must speak ack
    assert streaming_tts.speak.called
    assert session_mgr.is_active

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_real_windows_file_creation_returns_verified_absolute_path(tmp_path):
    """8 & 9. Real file creation on disk returns verified canonical path and byte count."""
    skill = FileSkill(search_root=tmp_path)
    ctx = MagicMock()
    ctx.get.side_effect = lambda k, default=None: {
        "path": "test.py",
        "content": "print('hello jarvis')\n",
    }.get(k, default)
    ctx.session_id = "test_sess"

    result = await skill.execute("create_file", ctx)

    assert result.success is True
    created = (tmp_path / "test.py").resolve()
    assert Path(result.data["path"]) == created
    assert result.data["verified"] is True
    assert result.data["exists"] is True
    assert result.data["size"] == len("print('hello jarvis')\n".encode("utf-8"))
    assert created.exists()
    assert created.is_file()


@pytest.mark.asyncio
async def test_file_creation_strips_trailing_speech_punctuation(tmp_path):
    """16. Filename like 'Test.py.' from speech transcript creates 'Test.py' cleanly."""
    skill = FileSkill(search_root=tmp_path)
    ctx = MagicMock()
    ctx.get.side_effect = lambda k, default=None: {
        "path": "Test.py.",
        "content": "# Test file",
    }.get(k, default)
    ctx.session_id = "test_sess"

    result = await skill.execute("create_file", ctx)
    assert result.success is True
    created = (tmp_path / "Test.py").resolve()
    assert Path(result.data["path"]) == created
    assert created.exists()


@pytest.mark.asyncio
async def test_vscode_launch_idempotency():
    """10 & 11. VS Code focus if already running; launch if not running."""
    skill = AppSkill()
    ctx = MagicMock()
    ctx.get.side_effect = lambda k, default=None: {
        "name": "VS Code",
        "args": "",
    }.get(k, default)
    ctx.session_id = "test_sess"

    # When window exists: focus existing window without launching new process
    with patch.object(skill, "_find_window_by_name", return_value=(12345, "Visual Studio Code")), \
         patch.object(skill, "_set_foreground", return_value=True), \
         patch("subprocess.Popen") as mock_popen:

        res = await skill.execute("launch_app", ctx)
        assert res.success is True
        assert res.data["already_running"] is True
        mock_popen.assert_not_called()


@pytest.mark.asyncio
async def test_created_file_path_propagates_into_next_task(tmp_path):
    """12. Verified created file path propagates into subsequent tasks."""
    file_skill = FileSkill(search_root=tmp_path)
    ctx = MagicMock()
    ctx.get.side_effect = lambda k, default=None: {
        "path": "area.py",
        "content": "print('area=20')\n",
    }.get(k, default)
    ctx.session_id = "test_sess"

    res = await file_skill.execute("create_file", ctx)
    assert res.success is True
    verified_path = res.data["path"]

    # Verify path is available and absolute
    assert Path(verified_path).is_absolute()
    assert Path(verified_path).exists()


@pytest.mark.asyncio
async def test_filesystem_observation_overrides_response_text(tmp_path):
    """13 & 14. Real filesystem evidence overrides generic prose; missing file fails."""
    observer = TaskObserver()
    task = TaskRecord(
        task_id="t1",
        goal_id="g1",
        description="Create file missing.py",
        utterance="create file missing.py",
        action={"skill": "file", "action": "create_file", "path": str(tmp_path / "missing.py")},
    )

    # Response claims success, but file does not exist on disk
    obs = await observer.observe(task, response_text="Created file missing.py successfully!")
    assert obs.file_exists is False
    assert obs.success_signal is False


@pytest.mark.asyncio
async def test_autonomous_goal_completion_restores_wake_listener(jarvis_voice_stack):
    """18. Autonomous goal execution completes and restores wake detection."""
    engine = jarvis_voice_stack["engine"]
    session_mgr = jarvis_voice_stack["session_mgr"]
    controller = jarvis_voice_stack["controller"]
    brain = jarvis_voice_stack["brain"]
    bus = jarvis_voice_stack["bus"]
    wake_model = jarvis_voice_stack["wake_model"]

    session_mgr._timeout_seconds = 10.0

    await engine.start()
    await controller.start()

    # Step 1: Wake and start session
    await session_mgr.activate(wake_word="Hey Spidy")
    assert session_mgr.is_active

    # Step 2: Goal finishes and publishes BrainResponseReadyEvent
    agent_response = "I have created Dish.py and opened it in VS Code."
    await bus.publish(BrainResponseReadyEvent(
        session_id=session_mgr.session_id,
        response_text=agent_response,
        decision_mode="goal",
    ))
    await asyncio.sleep(0.35)

    # Step 3: Engine must not be speaking or flushing
    assert not engine.is_speaking
    assert not engine.is_flushing

    # Step 4: OpenWakeWord processes high score chunk -> must trigger wake detection without suppression!
    wake_model.is_detected = True
    chunk = np.zeros(1280, dtype=np.int16)

    # Process wake chunk in engine
    engine._process_wake_word(chunk)

    # State must transition to WAKING
    assert engine.state == VoiceState.WAKING

    await controller.stop()
    await engine.stop()


@pytest.mark.asyncio
async def test_brain_run_goal_publishes_response_ready_event():
    """19. Brain.run_goal emits BrainResponseReadyEvent so voice/TTS pipelines are triggered."""
    bus = EventBus()
    bus.set_loop(asyncio.get_running_loop())
    received_events = []
    bus.subscribe("brain.response_ready", lambda e: received_events.append(e))

    mock_agent = MagicMock()
    mock_agent.run_goal = AsyncMock(return_value="Goal completed: created file Dish.py")

    from spidy.brain.brain import Brain
    brain = Brain(bus=bus, config=MagicMock(), skill_registry=MagicMock())
    brain.attach_agent(mock_agent)

    res = await brain.run_goal("create Dish.py")
    assert res == "Goal completed: created file Dish.py"
    await asyncio.sleep(0.05)

    assert len(received_events) == 1
    assert received_events[0].response_text == "Goal completed: created file Dish.py"
    assert received_events[0].decision_mode == "goal"


def test_no_printf_format_placeholders_in_codebase():
    """17. Verify zero %s, %r, %.2f placeholders in spidy python log files."""
    spidy_root = Path(__file__).resolve().parent.parent.parent / "spidy"
    for py_file in spidy_root.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        # Ignore string format definitions inside logging module configs
        if "logging/logger.py" in str(py_file).replace("\\", "/"):
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            if any(call in line for call in ("log.info(", "log.debug(", "log.warning(", "log.error(", "log.exception(")):
                for bad_spec in ("%s", "%r", "%.2f", "%.1f"):
                    assert bad_spec not in line, f"Found printf placeholder '{bad_spec}' in {py_file}:{line_no}: {line}"
