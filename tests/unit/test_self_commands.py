"""Spidy must recognise 'close yourself' style commands, and nothing else."""

from __future__ import annotations

import pytest

from spidy.brain.self_commands import is_quit_request


@pytest.mark.parametrize(
    "text",
    [
        "close yourself",
        "Thank you. Okay, smarty close yourself.",   # real transcript from a session
        "quit Spidy",
        "exit spidey",
        "shut yourself down",
        "turn yourself off",
        "Spidy, shut down",
        "Spidy please quit",
        "terminate yourself",
    ],
)
def test_quit_phrases(text):
    assert is_quit_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "close chrome",
        "close the window",
        "quit the app",
        "exit full screen",
        "shut down the computer",
        "how are you",
        "open vs code",
        "stop the music",
        "Spidy stop",          # barge-in phrase: interrupts speech only
        "stop Spidy",
        "stop",
        "",
    ],
)
def test_other_commands_are_not_quit(text):
    assert not is_quit_request(text)
