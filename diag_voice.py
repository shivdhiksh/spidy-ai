"""
Voice startup diagnostic script.
Exercises the exact same startup path as SpidyCore._run()
without launching the full app, Qt UI, Brain, etc.

Run:  py -3 diag_voice.py
Ctrl+C to stop after ~30 seconds of microphone listening.
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from spidy.config.manager import ConfigManager
from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.logging import configure as configure_logging

configure_logging(level="INFO", colorize=False)
log = get_logger("diag_voice")


async def main() -> None:
    log.info("=== Voice Diagnostic Started ===")

    # 1. Load config
    cfg_mgr = ConfigManager()
    settings = cfg_mgr.load()

    log.info("[1] Config loaded. wake_word.enabled={v} | model={m} | threshold={t}",
             v=settings.voice.wake_word.enabled,
             m=settings.voice.wake_word.model,
             t=settings.voice.wake_word.threshold)

    if not settings.voice.wake_word.enabled:
        log.error("[1] wake_word.enabled is FALSE — voice will never start!")
        return

    # 2. EventBus
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)
    log.info("[2] EventBus created.")

    # 3. Build VoiceEngine
    log.info("[3] Calling VoiceEngineFactory.build() ...")
    try:
        from spidy.perception.voice.factory import VoiceEngineFactory
        engine = VoiceEngineFactory.build(settings, bus)
        log.info("[3] VoiceEngine created: {cls}", cls=type(engine).__name__)
    except Exception as exc:
        log.error("[3] VoiceEngineFactory.build() FAILED: {exc}", exc=exc)
        raise

    # 4. Start VoiceEngine (opens mic, starts processing loop)
    log.info("[4] Calling engine.start() ...")
    try:
        await engine.start()
        log.info("[4] engine.start() returned OK. state={s}", s=engine.state.name)
    except Exception as exc:
        log.error("[4] engine.start() FAILED: {exc}", exc=exc)
        raise

    # 5. Check AudioCaptureEngine thread
    time.sleep(0.3)
    capture_alive = engine._capture_engine.is_running
    log.info("[5] AudioCaptureEngine.is_running={v}", v=capture_alive)
    if not capture_alive:
        log.error("[5] Capture thread is NOT alive — microphone did NOT open!")
    else:
        log.info("[5] Microphone is open and streaming audio chunks.")

    # 6. Listen for 40 seconds then report
    log.info("")
    log.info(">>> Listening for 90 s. Say 'hey jarvis' clearly into the microphone.")
    log.info(">>> Watch for [VOICE DIAG-10], [VOICE DIAG-11] and 'Wake word detected!'")
    log.info("")

    detected_event = asyncio.Event()

    async def _on_wake(event):
        log.info(">>> WAKE WORD EVENT RECEIVED: confidence={c} model={m}",
                 c=event.confidence, m=event.model_name)
        detected_event.set()

    bus.subscribe("wake_word.detected", _on_wake)

    try:
        await asyncio.wait_for(detected_event.wait(), timeout=90.0)
        log.info("=== SUCCESS: Wake word pipeline is WORKING ===")
    except asyncio.TimeoutError:
        log.warning("=== TIMEOUT: No wake word detected in 90 s ===")
        log.warning("    Check DIAG-10 logs (audio frames) and DIAG-11 logs (score > 0.1).")
        log.warning("    wake_chunk_counter={n}", n=engine._wake_chunk_counter)

    await engine.stop()
    log.info("=== Diagnostic complete ===")


asyncio.run(main())
