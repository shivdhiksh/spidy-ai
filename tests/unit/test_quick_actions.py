"""Everyday requests are answered directly and correctly (browser, network, files and screen are faked)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from spidy.brain import quick_actions as qa
from spidy.brain.intent_classifier import IntentClassifier


@pytest.fixture
def env(monkeypatch, tmp_path):
    opened, started, popen = [], [], []
    monkeypatch.setattr(qa.webbrowser, "open_new_tab", lambda url: opened.append(url))
    monkeypatch.setattr(qa.os, "startfile", lambda p: started.append(str(p)), raising=False)
    monkeypatch.setattr(qa.subprocess, "Popen", lambda args, **kw: popen.append(args))
    monkeypatch.setattr(qa, "FACTS_FILE", tmp_path / "facts.json")
    monkeypatch.setattr(qa, "DATA_DIR", tmp_path)
    return {"opened": opened, "started": started, "popen": popen, "tmp": tmp_path}


def run(text, announce=None):
    return asyncio.run(qa.QuickActions(announce).try_handle(text))


@pytest.mark.parametrize("text, url, reply", [
    ("open google", "https://www.google.com", "Opened Google."),
    ("Open YouTube.", "https://www.youtube.com", "Opened YouTube."),
    ("hey spidy can you open chatgpt please", "https://chatgpt.com", "Opened ChatGPT."),
    ("Open today's top news", "https://news.google.com", "Opened Google News."),
    ("open gmail in browser", "https://mail.google.com", "Opened Gmail."),
    ("Open a new browser tab", "https://www.google.com", "Opened a new tab."),
])
def test_sites_open_in_the_users_browser(env, text, url, reply):
    assert run(text).reply == reply
    assert env["opened"] == [url]


def test_search_opens_google_results(env):
    r = run("search for best biryani in Hyderabad")
    assert r.reply == "Here are the results for best biryani in Hyderabad."
    assert env["opened"] == ["https://www.google.com/search?q=best+biryani+in+Hyderabad"]


def test_youtube_searches_and_apps_are_left_to_the_existing_skills(env):
    for text in ("play relaxing music on youtube", "open vs code", "open notepad", "what can you do for me"):
        assert run(text) is None, text


def test_folders_and_file_explorer(env):
    assert run("open file explorer").reply == "Opened File Explorer."
    assert env["popen"] == [["explorer.exe"]]
    assert run("open my downloads folder").reply == "Opened your Downloads folder."
    assert env["started"][0].lower().endswith("downloads")


def test_find_file_shows_it_in_explorer(env, monkeypatch):
    f = env["tmp"] / "README.md"
    f.write_text("x")
    monkeypatch.setattr(qa, "find_files", lambda q: [f])
    r = run("find my file README")
    assert r.reply.startswith("Found it! README.md is in")
    assert env["popen"] == [["explorer.exe", f"/select,{f}"]]
    assert run("find files on my computer").reply == "Sure! What's the file called?"


def test_find_files_matches_every_word(monkeypatch, tmp_path):
    (tmp_path / "Documents" / "uni").mkdir(parents=True)
    (tmp_path / "Documents" / "uni" / "DBMS_assignment_final.docx").write_text("x")
    (tmp_path / "Documents" / "DBMS_notes.txt").write_text("x")
    monkeypatch.setattr(qa, "user_folder", lambda n: tmp_path / n.capitalize())
    monkeypatch.setattr(qa, "HOME", tmp_path)
    assert [p.name for p in qa.find_files("DBMS assignment")] == ["DBMS_assignment_final.docx"]


def test_weather_and_briefing(env, monkeypatch):
    monkeypatch.setattr(qa.QuickActions, "_weather_now", lambda self: "It's 24 degrees and clear in Hyderabad.")
    monkeypatch.setattr(qa.QuickActions, "_headlines", staticmethod(lambda n=3: ["Headline one", "Headline two"]))
    assert run("what's the weather today").reply == "It's 24 degrees and clear in Hyderabad."
    r = run("Give me a quick briefing: the time, the weather and the top news").reply
    assert r.startswith("Here's your briefing. It's ") and "24 degrees" in r and "Headline one. Headline two." in r


def test_timer_rings_later():
    said: list[str] = []

    async def announce(text):
        said.append(text)

    async def go():
        r = await qa.QuickActions(announce).try_handle("set a timer for 0.05 seconds")
        await asyncio.sleep(0.2)
        return r
    r = asyncio.run(go())
    assert r.reply == "Timer set for 0.05 seconds. I'll tell you when it's done!"
    assert said == ["Time's up! Your 0.05 seconds timer is done."]
    assert run("Set a timer for 25 minutes").reply == "Timer set for 25 minutes. I'll tell you when it's done!"


def test_clipboard_code_is_sent_to_the_model(env, monkeypatch):
    monkeypatch.setattr(qa, "read_clipboard", lambda: "print(1/0)")
    r = run("Fix the error I copied")
    assert r.reply is None and r.rewrite.endswith("print(1/0)") and r.rewrite.startswith("Find the problem")
    monkeypatch.setattr(qa, "read_clipboard", lambda: "")
    assert "Copy the code" in run("Explain the code I copied").reply


def test_screen_questions_use_a_screenshot_and_the_vision_model(env, monkeypatch):
    shot = env["tmp"] / "s.png"
    shot.write_bytes(b"png")
    asked: list[str] = []
    monkeypatch.setattr(qa, "take_screenshot", lambda: shot)
    monkeypatch.setattr(qa, "ask_vision", lambda p, q: asked.append(q) or "A code editor with Python open.")
    monkeypatch.setattr(qa, "active_app", lambda: "main.py - Visual Studio Code")
    assert run("What is on my screen right now?").reply == "A code editor with Python open."
    assert run("Read the text on my screen").reply == "A code editor with Python open."
    assert "Read out" in asked[1]
    assert run("Which app am I using right now?").reply == "You're using main.py - Visual Studio Code."
    assert run("take a screenshot").reply.startswith("Done! Screenshot saved")
    monkeypatch.setattr(qa, "ask_vision", lambda p, q: None)
    assert "vision model isn't reachable" in run("Summarise what I am looking at").reply


def test_memory_persists_and_feeds_the_model(env):
    assert run("remember that my favourite editor is VS Code").reply == \
        "Got it! I'll remember that your favourite editor is VS Code."
    assert run("What do you remember about me?").reply == "Here's what I remember: your favourite editor is VS Code."
    assert "my favourite editor is VS Code" in qa.facts_context()
    assert run("forget that my favourite editor is VS Code").reply == "Okay, I've forgotten that."
    assert qa.facts_context() == ""


@pytest.mark.parametrize("text, action", [
    ("hello", "greet"), ("hey spidy good morning", "greet"), ("hi spidy how are you", "greet"),
    ("Say hello and tell me one fun fact.", "chat"),
])
def test_greeting_only_when_it_is_just_a_greeting(text, action):
    assert asyncio.run(IntentClassifier().classify(text)).action == action


def test_hello_world_program_is_not_a_greeting():
    assert asyncio.run(IntentClassifier().classify("Create a python program that prints hello world")).action != "greet"


@pytest.mark.parametrize("text, code", [
    # the whole request goes to the code writer, so nothing the user asked for is lost
    ("Create a python program that prints hello world", ("python", "Create a python program that prints hello world")),
    ("write a python program: prints the first 10 prime numbers",
     ("python", "write a python program: prints the first 10 prime numbers")),
    ("make a game where you guess a number", ("python", "make a game where you guess a number")),
    ("create an html page", ("html", "create an html page")),
    ("create a file called notes.txt", None),
    ("write a python function to sort a list of names", ("python", "write a python function to sort a list of names")),
])
def test_program_requests_become_real_files(env, text, code):
    r = run(text)
    assert (r.code if r else None) == code


def test_brain_writes_the_program_to_a_file(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus
    from spidy.llm.client import LLMResponse

    async def go():
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=LLMResponse(text="```python\nprint('Hello World')\n```"))
        brain = Brain(bus=bus, config=MagicMock(), skill_registry=MagicMock(), llm_client=llm)
        monkeypatch.setattr(qa, "user_folder", lambda n: tmp_path)
        monkeypatch.setattr("subprocess.Popen", lambda *a, **k: None)
        return await brain.process("Create a python program that prints hello world")
    reply = asyncio.run(go())
    f = tmp_path / "Spidy Code" / "prints_hello_world.py"
    assert f.read_text() == "print('Hello World')\n"
    assert reply == "Done! I wrote prints_hello_world.py in your Documents, in the Spidy Code folder, and opened it in VS Code."


def test_cut_off_website_is_continued_until_complete(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from spidy.brain.brain import Brain
    from spidy.core.event_bus import EventBus
    from spidy.llm.client import LLMResponse

    parts = [LLMResponse(text="```html\n<!DOCTYPE html>\n<html><body>\n<script>\nconst a = 1;\n"),   # cut off
             LLMResponse(text="const b = 2;\n</script>\n</body></html>\n```")]

    async def go():
        bus = EventBus()
        bus.set_loop(asyncio.get_running_loop())
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=parts)
        brain = Brain(bus=bus, config=MagicMock(), skill_registry=MagicMock(), llm_client=llm)
        monkeypatch.setattr(qa, "user_folder", lambda n: tmp_path)
        monkeypatch.setattr("subprocess.Popen", lambda *a, **k: None)
        monkeypatch.setattr("webbrowser.open_new_tab", lambda *a, **k: True)
        return await brain.process("Make a calculator website.")
    reply = asyncio.run(go())
    f = tmp_path / "Spidy Code" / "calculator.html"
    code = f.read_text()
    assert code.startswith("<!DOCTYPE html>") and code.rstrip().endswith("</body></html>")
    assert "const a = 1;" in code and "const b = 2;" in code
    assert "browser" in reply
