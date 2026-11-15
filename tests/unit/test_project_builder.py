"""Live code writing: files appear in VS Code as they are written; projects are planned and built file by file."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from spidy.brain.project_builder import (
    ProjectBuilder, _CodeStripper, is_project_request, parse_plan,
)
from spidy.llm.client import LLMResponse


class _FakeLLM:
    """Streams each answer in small pieces; records every flush the builder makes to disk."""

    def __init__(self, plan: dict | None, files: dict[str, str]):
        self.plan, self.files, self.prompts = plan, files, []

    async def complete(self, msgs, **kw):
        return LLMResponse(text=json.dumps(self.plan) if self.plan else "no plan")

    async def stream(self, msgs, **kw):
        self.prompts.append(msgs[-1].content)
        target = next((p for p in self.files if f"`{p}`" in msgs[-1].content), None) or next(iter(self.files))
        answer = f"```\n{self.files[target]}\n```\nextra words the model added"
        for i in range(0, len(answer), 7):
            await asyncio.sleep(0.06)
            yield answer[i:i + 7]


def _run(builder, *a, **k):
    return asyncio.run(builder.build(*a, **k))


def test_project_requests_are_recognised():
    assert is_project_request("build a website with frontend and backend")
    assert is_project_request("make a full stack todo app")
    assert is_project_request("create a REST API with a database")
    assert not is_project_request("write a python program that prints hello world")
    assert not is_project_request("make a calculator website")


def test_single_file_is_streamed_into_the_editor_live(tmp_path):
    code = "\n".join(f"print({i})" for i in range(60))
    llm = _FakeLLM(None, {"x": code})
    seen_sizes, opened = [], []
    target = tmp_path / "hello.py"

    def opener(p, new):
        opened.append((p, new))

    b = ProjectBuilder(llm, tmp_path, open_in_editor=opener)

    async def watch():
        for _ in range(80):
            await asyncio.sleep(0.05)
            if target.exists():
                seen_sizes.append(len(target.read_text()))

    async def go():
        w = asyncio.create_task(watch())
        reply = await b.build("write a python program that counts", language="python", ext=".py", stem="hello")
        w.cancel()
        return reply
    reply = asyncio.run(go())
    assert target.read_text() == code + "\n"                     # fences and chatter stripped
    assert len({s for s in seen_sizes if 0 < s < len(code)}) >= 2  # grew in visible steps = live
    assert opened and opened[0][0] == target and opened[0][1] is False
    assert "hello.py" in reply


def test_project_is_planned_and_every_file_written(tmp_path):
    plan = {"name": "todo_app", "summary": "A todo app.", "run": "npm install, then node server.js",
            "files": [{"path": "server.js", "purpose": "Express API"},
                      {"path": "public/index.html", "purpose": "page"},
                      {"path": "public/app.js", "purpose": "frontend logic"},
                      {"path": "../evil.txt", "purpose": "must be dropped"}]}
    files = {"server.js": "const express = require('express');", "public/index.html": "<html></html>",
             "public/app.js": "fetch('/api/todos')"}
    llm = _FakeLLM(plan, files)
    opened = []
    b = ProjectBuilder(llm, tmp_path, open_in_editor=lambda p, new: opened.append((p, new)))
    reply = _run(b, "build a todo website with frontend and backend")
    root = tmp_path / "todo_app"
    for rel, content in files.items():
        assert (root / rel).read_text().strip() == content
    assert not (tmp_path / "evil.txt").exists()                  # never writes outside the project
    assert (root / "README.md").exists()                          # run instructions added
    assert opened[0] == (root, True)                              # project folder opened in a new window first
    assert "server.js" in llm.prompts[1] and "const express" in llm.prompts[1]   # later files see earlier ones
    assert "3 files" in reply and "npm install" in reply


def test_bad_plan_falls_back_to_one_file(tmp_path):
    llm = _FakeLLM(None, {"x": "print('hi')"})
    b = ProjectBuilder(llm, tmp_path, open_in_editor=lambda p, n: None)
    _run(b, "build a full stack app", language="python", ext=".py", stem="app")
    assert (tmp_path / "app.py").read_text() == "print('hi')\n"


def test_plan_parser_rejects_unsafe_paths():
    plan = parse_plan('here: {"name": "x", "files": [{"path": "C:/Windows/a.txt"}, {"path": "ok/b.py"}]}', "x")
    assert [f.path for f in plan.files] == ["ok/b.py"]


def test_code_stripper_handles_fences_while_streaming():
    s = _CodeStripper()
    s.raw = "```py"
    assert s.content() == ""
    s.raw = "```python\nprint(1)\n"
    assert s.content() == "print(1)\n"
    s.raw += "```\nThat's it!"
    assert s.content() == "print(1)\n" and s.done
