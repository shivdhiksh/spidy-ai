"""Closing the window (X) stops speech and tasks but must not end Spidy: "Hey Spidy" has to work afterwards."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from spidy.core.app import SpidyCore


def _core() -> SpidyCore:
    core = object.__new__(SpidyCore)
    core._voice_controller = MagicMock()
    core._voice_controller._session.deactivate = AsyncMock()
    core._brain = MagicMock()
    core._brain.cancel_goal = AsyncMock(return_value=True)
    core._bus = MagicMock()
    core._shutdown_event = asyncio.Event() if False else MagicMock()
    return core


def test_closing_the_window_halts_speech_and_tasks_and_ends_the_session():
    core = _core()
    asyncio.run(core._on_ui_closed(MagicMock()))
    core._voice_controller.halt.assert_called_once()                       # stops talking now
    core._brain.cancel_goal.assert_awaited_once()                          # running task dropped
    core._voice_controller._session.deactivate.assert_awaited_once()       # conversation over
    core._voice_controller._engine.return_to_sleeping.assert_called_once() # wake-word listening restored


def test_closing_the_window_does_not_shut_spidy_down():
    core = _core()
    asyncio.run(core._on_ui_closed(MagicMock()))
    core._bus.publish.assert_not_called()                                  # no system.shutting_down
    core._shutdown_event.set.assert_not_called()


def test_close_handler_is_subscribed_to_the_ui_closed_topic():
    import inspect
    import spidy.core.app as app
    src = inspect.getsource(app)
    assert 'subscribe("ui.closed", self._on_ui_closed)' in src


# ── always available ──────────────────────────────────────────────────────────

def test_supervisor_restarts_after_a_crash_but_not_after_a_quit():
    from spidy.main import supervise
    codes = iter([3221226356, 1, 0])           # heap corruption crash, another crash, then a deliberate quit
    runs: list[int] = []

    def run():
        runs.append(1)
        return next(codes)
    assert supervise([], run=run, sleep=lambda s: None) == 0
    assert len(runs) == 3


def test_supervisor_gives_up_on_a_crash_loop():
    from spidy.main import supervise
    runs: list[int] = []
    assert supervise([], max_crashes=3, run=lambda: runs.append(1) or 7, sleep=lambda s: None) == 7
    assert len(runs) == 4


def test_autostart_uses_the_supervisor():
    from spidy.desktop.autostart import get_default_autostart_command
    assert "-m spidy.main --supervise --background" in get_default_autostart_command(background=True)


def test_dead_microphone_listener_is_restarted():
    from unittest.mock import MagicMock
    from spidy.perception.voice.engine import VoiceEngine

    class _Wake:
        chunk_size = 1280
        model_name = "hey_spidy"
        def process_chunk(self, c): return 0.0
        def reset_buffer(self): pass
        def unload(self): pass

    eng = VoiceEngine(bus=MagicMock(), wake_word_model=_Wake(), recognizer=MagicMock(spec=["unload"]),
                      tts=MagicMock(is_speaking=False))
    cap = MagicMock(is_running=False, chunks_read=0)
    eng._capture_engine = cap
    eng.ensure_processing_task = lambda: True

    async def go():
        task = asyncio.create_task(eng._listener_health(every=0.01))
        await asyncio.sleep(0.05)
        eng._stop_event.set()
        await task
    asyncio.run(go())
    cap.start.assert_called()                    # reopened the microphone
