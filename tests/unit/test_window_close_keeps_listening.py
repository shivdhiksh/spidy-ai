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
