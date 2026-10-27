"""
verify_wake_timeout_recovery_live.py
====================================
Live diagnostic script verifying that VoiceEngine and all 3 OpenWakeWord models
survive a VoiceSession timeout and allow subsequent wake phrases to activate new
conversational sessions and execute commands.
"""

import asyncio
import sys
import time

from spidy.config import ConfigManager
from spidy.core.event_bus import EventBus
from spidy.perception.voice.audio_capture import AudioCaptureEngine, CaptureMode
from spidy.perception.voice.engine import VoiceEngine, VoiceState, VoiceTranscriptEvent, WakeWordDetectedEvent
from spidy.perception.voice.wake_word.openwakeword import OpenWakeWordModel, _MODEL_MAP
from spidy.perception.voice.stt.whisper import FasterWhisperRecognizer
from spidy.perception.voice.tts.piper import PiperTTSEngine
from spidy.voice.session import VoiceSessionManager, VoiceSessionState
from spidy.voice.continuous import ContinuousVoiceController
from spidy.voice.streaming_tts import StreamingTTSWrapper
from spidy.voice.interruption import InterruptionHandler


async def main() -> int:
    print("==================================================")
    print("SPIDY WAKE TIMEOUT RECOVERY TEST")
    print("==================================================")

    cfg = ConfigManager().load()
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)

    # 1. Initialize OpenWakeWord with all 3 configured models
    wake_model = OpenWakeWordModel(
        model_name="hey_spidy",
        threshold=0.35,
        models=cfg.voice.wake_word.models,
        active_models=cfg.voice.wake_word.active_models,
    )
    wake_model.load()

    active_models = wake_model.active_model_names

    # 2. Build VoiceEngine
    engine = VoiceEngine(
        bus=bus,
        wake_word_model=wake_model,
        recognizer=FasterWhisperRecognizer(
            model_size=cfg.voice.stt.model,
            device="cpu",
            compute_type="int8",
        ),
        tts=PiperTTSEngine(
            voice=cfg.voice.tts.voice,
            speed=cfg.voice.tts.speed,
            volume=cfg.voice.tts.volume,
        ),
        sample_rate=cfg.voice.audio.sample_rate,
        wake_threshold=cfg.voice.wake_word.threshold,
    )
    await engine.start()

    session_mgr = VoiceSessionManager(
        bus=bus,
        timeout_seconds=0.5,  # Short timeout for diagnostic test
        continuous_mode=True,
    )

    streaming_tts = StreamingTTSWrapper(engine=engine._tts)
    interruption_handler = InterruptionHandler(
        bus=bus,
        brain=None,
        tts=engine._tts,
        session=session_mgr,
    )

    controller = ContinuousVoiceController(
        voice_engine=engine,
        brain=None,
        bus=bus,
        session_manager=session_mgr,
        streaming_tts=streaming_tts,
        interruption_handler=interruption_handler,
        continuous_mode=True,
    )
    await controller.start()

    # Step 1: Confirm Initial State
    task_alive = engine._processing_task is not None and not engine._processing_task.done()
    capture_alive = engine._capture_engine.is_running
    wake_listener_active = (engine.state == VoiceState.SLEEPING and engine._capture_engine.mode == CaptureMode.DETECTING)

    print("\nInitial:")
    print(f"engine_running={not engine._stop_event.is_set()}")
    print(f"capture_alive={capture_alive}")
    print(f"wake_listener={wake_listener_active}")
    print(f"models={active_models}")

    assert task_alive and capture_alive and wake_listener_active
    assert len(active_models) >= 3

    # Step 2: First Wake (Hey Spidy)
    print("\nFirst wake:")
    phrase1 = "Hey Spidy"
    model1 = "hey_spidy"
    print(f"phrase={phrase1}")

    await bus.publish(WakeWordDetectedEvent(
        confidence=0.95,
        model_name=model1,
        wake_phrase=phrase1,
    ))
    await asyncio.sleep(0.10)
    session_created1 = (session_mgr.state == VoiceSessionState.AWAKE)
    sid1 = session_mgr.session_id
    print(f"session_created={session_created1}")
    assert session_created1 and sid1

    # Step 3: Timeout
    print("\nTimeout:")
    await asyncio.sleep(0.65)  # Wait for 0.5s timeout to trigger
    session_active_after_timeout = session_mgr.is_active
    engine_running_after = not engine._stop_event.is_set()
    capture_alive_after = engine._capture_engine.is_running
    wake_listener_after = (engine.state == VoiceState.SLEEPING and engine._capture_engine.mode == CaptureMode.DETECTING)

    print(f"session_active={session_active_after_timeout}")
    print(f"engine_running={engine_running_after}")
    print(f"capture_alive={capture_alive_after}")
    print(f"wake_listener={wake_listener_after}")

    assert not session_active_after_timeout
    assert engine_running_after and capture_alive_after and wake_listener_after

    # Step 4: Second Wake (Wake up Spidy)
    print("\nSecond wake:")
    phrase2 = "Wake up Spidy"
    model2 = "wake_up_spidy"
    print(f"phrase={phrase2}")

    await bus.publish(WakeWordDetectedEvent(
        confidence=0.92,
        model_name=model2,
        wake_phrase=phrase2,
    ))
    await asyncio.sleep(0.10)
    session_created2 = (session_mgr.state == VoiceSessionState.AWAKE)
    sid2 = session_mgr.session_id
    print(f"session_created={session_created2}")
    assert session_created2 and sid2 and (sid2 != sid1)

    # Step 5: Issue Command
    print("\nCommand:")
    await session_mgr.record_utterance("open notepad")
    cmd_success = (session_mgr.turn_count == 1 and session_mgr.current_goal == "open notepad")
    print(f"success={cmd_success}")
    assert cmd_success

    # Cleanup
    await controller.stop()
    await engine.stop()

    print("\nRESULT: PASS")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
