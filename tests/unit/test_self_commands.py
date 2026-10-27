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


# ── "Who developed you?" always gets the same answer ─────────────────────────

from spidy.brain.self_commands import CREATOR_REPLY, is_creator_question  # noqa: E402


@pytest.mark.parametrize("text", [
    "who developed you", "Who developed you?", "who created you", "who made you", "who built you",
    "Who developed Spidy", "who is your developer", "who's your creator", "who is your boss",
    "who are your developers", "who designed you", "which company made you", "what company developed you",
    "who is behind you", "are you made by NVIDIA", "who programmed you", "Hey Spidy who created you please",
    "who exactly developed you", "who owns you", "who trained you",
])
def test_creator_questions(text):
    assert is_creator_question(text)


@pytest.mark.parametrize("text", [
    "who developed python", "who created the iphone", "open vs code", "who is the prime minister",
    "what is your name", "who made this song", "what can you do", "build me a website", "who wrote hamlet",
])
def test_other_questions_are_not_creator_questions(text):
    assert not is_creator_question(text)


def test_the_reply_names_shiva_dhikshith_and_not_nvidia_as_the_developer():
    assert CREATOR_REPLY.startswith("I'm SPIDY, an AI desktop assistant developed by my boss Shiva Dhikshith.")
    assert "Shiva Dhikshith Sir is my creator and developer." in CREATOR_REPLY
    assert "NVIDIA to power some of my intelligence" in CREATOR_REPLY


@pytest.mark.asyncio
async def test_brain_answers_with_the_fixed_reply_without_calling_the_model():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus

    bus = EventBus()
    bus.set_loop(asyncio.get_running_loop())
    got = []

    async def collect(e):
        got.append(e)
    bus.subscribe("brain.response_ready", collect)
    brain = Brain(bus=bus, config=MagicMock(), skill_registry=MagicMock())
    brain._llm = MagicMock(complete=AsyncMock(side_effect=AssertionError("the model must not be asked")))
    for q in ("who developed you?", "Who created you"):
        assert await brain.process(q) == CREATOR_REPLY
    assert [e.response_text for e in got] == [CREATOR_REPLY, CREATOR_REPLY]
