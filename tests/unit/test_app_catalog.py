"""Finding apps by spoken name, and routing 'open <app>' to the real launcher."""

from __future__ import annotations

import asyncio

import pytest

from spidy.brain.intent_classifier import IntentClassifier
from spidy.skills.desktop.app_catalog import StartApp, find_start_app, normalize_app_name

APPS = [
    StartApp("Visual Studio Code", "Microsoft.VisualStudioCode"),
    StartApp("WhatsApp", "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"),
    StartApp("Google Chrome", "Chrome"),
    StartApp("Notepad", "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"),
    StartApp("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
    StartApp("Word", "Microsoft.Office.WINWORD.EXE.15"),
]


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("VS Core", "vs code"),            # real Whisper mistake from a session
        ("VS Code.", "vs code"),
        ("whats app", "whatsapp"),
        ("the WhatsApp app please", "whatsapp"),
        ("Google Chrome", "chrome"),
        ("  Notepad  ", "notepad"),
    ],
)
def test_normalize(spoken, expected):
    assert normalize_app_name(spoken) == expected


@pytest.mark.parametrize(
    "spoken, found",
    [
        ("VS Core", "Visual Studio Code"),
        ("vs code", "Visual Studio Code"),
        ("whatsapp", "WhatsApp"),
        ("chrome", "Google Chrome"),
        ("calc", "Calculator"),            # prefix of a Start menu name
        ("telegram", None),                # not installed
        ("", None),
    ],
)
def test_find_start_app(spoken, found):
    app = find_start_app(spoken, APPS)
    assert (app.name if app else None) == found


@pytest.mark.parametrize(
    "utterance, name",
    [
        ("Alright, can you open VS Core?", "vs core"),
        ("Open WhatsApp.", "whatsapp"),
        ("launch telegram", "telegram"),
        ("start spotify please", "spotify"),
        ("can you open visual studio code for me", "visual studio code"),
    ],
)
def test_open_anything_routes_to_the_launcher(utterance, name):
    intent = asyncio.run(IntentClassifier().classify(utterance))
    assert intent.action == "launch_app"
    assert {e.name: e.value for e in intent.entities}["name"] == name


@pytest.mark.parametrize("utterance", ["open it", "open a new tab", "how are you", "open the downloads folder"])
def test_non_app_requests_are_not_launches(utterance):
    intent = asyncio.run(IntentClassifier().classify(utterance))
    assert intent.action != "launch_app"


def test_two_apps_in_one_sentence_become_two_steps():
    intent = asyncio.run(IntentClassifier().classify("open vs code and open whatsapp"))
    assert intent.action == "compound"
    assert len(intent.entities) == 2
