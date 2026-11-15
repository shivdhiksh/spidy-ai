"""
Live project builder: Spidy writes code *on screen*, file by file, in VS Code.

* A single program: VS Code opens the (empty) file at once and the code appears in it as the model
  writes it (streamed to disk a few times a second; VS Code reloads the open file live).
* A project ("a website with a frontend and a backend", "full-stack todo app", ...): the model first
  plans the files (JSON), the project folder opens in VS Code, then every file is written live in
  order -- each one is given the files already written so the pieces fit together -- and it ends with
  a README holding the run instructions.

Progress goes to the HUD (status line) and, sparingly, to Spidy's voice.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from spidy.logging.logger import get_logger

log = get_logger(__name__)

_PROJECT_RE = re.compile(
    r"\b(front\s*-?\s*end|back\s*-?\s*end|full\s*-?\s*stack|frontend|backend|server\s+and|with\s+(?:a\s+)?"
    r"(?:server|database|api|backend|login)|rest\s+api|project|multi[-\s]?file|whole\s+app|complete\s+app|"
    r"entire\s+(?:app|website|project)|web\s+app\s+with|django|flask\s+app|express|next\.?js|mern|crud)\b",
    re.IGNORECASE)
_MAX_FILES = 12
_CONTEXT_CHARS = 14000           # earlier files handed to the model when writing the next one


def is_project_request(text: str) -> bool:
    """True when the user wants several files (frontend + backend, a full app, a project)."""
    return bool(_PROJECT_RE.search(text or ""))


@dataclass
class PlannedFile:
    path: str
    purpose: str = ""


@dataclass
class Plan:
    name: str
    summary: str = ""
    files: list[PlannedFile] = field(default_factory=list)
    run: str = ""


def slug(text: str, default: str = "project") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:40].strip("_") or default


def _safe_rel(path: str) -> str | None:
    """Project-relative path with no escaping the folder (no drive letters, no '..')."""
    p = path.replace("\\", "/").strip().lstrip("/")
    if not p or ":" in p or any(part in ("..", "") for part in p.split("/")):
        return None
    return p


def parse_plan(text: str, fallback_name: str) -> Plan | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    files = []
    for f in data.get("files", [])[:_MAX_FILES]:
        rel = _safe_rel(str(f.get("path", "")) if isinstance(f, dict) else str(f))
        if rel and rel not in {x.path for x in files}:
            files.append(PlannedFile(rel, str(f.get("purpose", "")) if isinstance(f, dict) else ""))
    if not files:
        return None
    return Plan(name=slug(data.get("name") or fallback_name), summary=str(data.get("summary", "")),
                files=files, run=str(data.get("run", "")))


class _CodeStripper:
    """Turns a streamed model answer into raw file content: drops the opening ```lang line and
    everything from the closing fence on, without waiting for the whole answer."""

    def __init__(self) -> None:
        self.raw = ""
        self.done = False

    def content(self) -> str:
        text = self.raw
        if text.lstrip().startswith("```"):
            text = text.lstrip()
            nl = text.find("\n")
            if nl < 0:
                return ""                    # still inside the opening fence line
            text = text[nl + 1:]
        end = text.find("\n```")
        if end >= 0:
            self.done = True
            text = text[:end + 1]
        return text


class ProjectBuilder:
    def __init__(self, llm: Any, root: Path,
                 say: Callable[[str], Awaitable[None]] | None = None,
                 status: Callable[[str], Awaitable[None] | None] | None = None,
                 open_in_editor: Callable[[Path, bool], None] | None = None) -> None:
        self._llm = llm
        self._root = root
        self._say = say
        self._status = status
        self._open = open_in_editor or open_in_vscode

    # ── public ────────────────────────────────────────────────────────────
    async def build(self, request: str, language: str = "python", ext: str = ".py",
                    project: bool | None = None, stem: str = "program") -> str:
        project = is_project_request(request) if project is None else project
        if project:
            await self._speak("On it! Let me plan the project first, then I'll open VS Code so you can watch me build it.")
            plan = await self._plan(request, language)
        else:
            plan = None
        if plan is None:
            plan = Plan(name=stem, files=[PlannedFile(f"{stem}{ext}", request)])
            folder = self._root
            await self._speak(f"On it! Opening VS Code, watch me write your {'website' if language == 'html' else language + ' code'}.")
        else:
            folder = self._unique(self._root / plan.name)
            await self._speak(f"Here's the plan: {len(plan.files)} files. Opening the project in VS Code now.")
        folder.mkdir(parents=True, exist_ok=True)

        multi = len(plan.files) > 1
        paths: list[Path] = []
        for f in plan.files:                                 # the whole file tree appears at once
            p = folder / f.path if multi else self._unique(folder / f.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("", encoding="utf-8")
            paths.append(p)
        if multi:
            self._open(folder, True)                         # new VS Code window on the project folder
            await asyncio.sleep(1.5)

        written: list[tuple[str, str]] = []
        for i, (f, p) in enumerate(zip(plan.files, paths), 1):
            await self._show(f"Writing {f.path} ({i} of {len(plan.files)})" if multi else f"Writing {p.name}")
            self._open(p, False)
            code = await self._write_live(request, plan, f, p, written, language, multi)
            written.append((f.path, code))
            if multi and i == len(plan.files) // 2 and len(plan.files) >= 4:
                await self._speak("Halfway there!")
        if multi and not any(Path(f.path).name.lower() == "readme.md" for f in plan.files):
            readme = folder / "README.md"
            readme.write_text(f"# {plan.name.replace('_', ' ').title()}\n\n{plan.summary}\n\n## Run it\n\n{plan.run}\n",
                              encoding="utf-8")
        if language == "html" and not multi:
            try:
                import webbrowser
                webbrowser.open_new_tab(paths[0].as_uri())
            except Exception:  # noqa: BLE001
                pass
        await self._show("")
        lines = sum(c.count("\n") for _, c in written)
        log.info("ProjectBuilder: wrote {n} file(s), {l} lines in {f}", n=len(written), l=lines, f=folder)
        if multi:
            run = f" To run it: {plan.run}" if plan.run else " The README explains how to run it."
            return (f"Done! I built {plan.name.replace('_', ' ')} with {len(written)} files in your Documents, "
                    f"Spidy Code folder, and it's open in VS Code.{run}")
        where = "opened it in VS Code" + (" and your browser" if language == "html" else "")
        return f"Done! I wrote {paths[0].name} in your Documents, in the Spidy Code folder, and {where}."

    # ── steps ─────────────────────────────────────────────────────────────
    async def _plan(self, request: str, language: str) -> Plan | None:
        from spidy.llm.client import LLMMessage
        msgs = [LLMMessage("system", (
            "You are a senior software architect. Plan a small but complete, working project for the request. "
            "Reply with ONLY JSON: {\"name\": short_snake_case_name, \"summary\": one sentence, "
            "\"files\": [{\"path\": relative/path.ext, \"purpose\": what this file contains}], "
            "\"run\": short plain-English steps to install and run it}. "
            f"At most {_MAX_FILES - 2} files, order them so dependencies come first, use simple popular tech "
            "(e.g. HTML/CSS/JS frontend; Node.js Express or Python Flask backend), include a README.md last.")),
            LLMMessage("user", request)]
        resp = await self._complete(msgs, max_tokens=1500, timeout=150)
        plan = parse_plan(resp.text if resp and resp.success else "", fallback_name=slug(request))
        if plan is None:
            log.warning("ProjectBuilder: no usable plan; falling back to a single file.")
        return plan

    async def _write_live(self, request: str, plan: Plan, f: PlannedFile, path: Path,
                          written: list[tuple[str, str]], language: str, multi: bool) -> str:
        from spidy.llm.client import LLMMessage
        if multi:
            tree = "\n".join(f"- {x.path}: {x.purpose}" for x in plan.files)
            ctx, budget = [], _CONTEXT_CHARS
            for p, c in reversed(written):                   # most recent files first
                snippet = c[:max(0, budget)]
                if not snippet:
                    break
                ctx.append(f"### {p}\n{snippet}")
                budget -= len(snippet)
            system = ("You are an expert full-stack developer writing one file of a project. Reply with ONLY the "
                      "complete content of the requested file in one fenced code block -- no explanation. It must "
                      "work with the other files exactly as planned (same names, routes, ports and IDs).")
            user = (f"Project request: {request}\n\nPlanned files:\n{tree}\n\n"
                    + (("Files already written:\n" + "\n\n".join(reversed(ctx)) + "\n\n") if ctx else "")
                    + f"Now write `{f.path}` -- {f.purpose}")
        else:
            if language == "html":
                system = ("You are an expert web developer. Build exactly what the user asks as ONE complete, "
                          "self-contained HTML file: modern, beautiful, responsive, all CSS in <style> and all JS in "
                          "<script>, every feature working. Keep it focused (under about 350 lines).")
            else:
                system = (f"You are an expert {language} developer. Write a complete, runnable {language} program that "
                          "does exactly what the user asks, with short helpful comments. Single file, standard "
                          "library where possible. Keep it focused (under about 350 lines).")
            system += " Reply with ONLY the code in one fenced code block. No explanation."
            user = request
        msgs = [LLMMessage("system", system), LLMMessage("user", user)]
        return await self._stream_to_file(msgs, path)

    async def _stream_to_file(self, msgs, path: Path) -> str:
        """Stream the model's answer into ``path`` a few times a second so it appears live in VS Code."""
        strip = _CodeStripper()
        last_flush, flushed = 0.0, ""
        stream = getattr(self._llm, "stream", None)
        got_any = False
        if callable(stream):
            try:
                async for tok in stream(msgs, temperature=0.3, max_tokens=14000):
                    got_any = True
                    strip.raw += tok
                    now = time.monotonic()
                    if now - last_flush > 0.35:
                        content = strip.content()
                        if content != flushed:
                            path.write_text(content, encoding="utf-8")
                            flushed = content
                        last_flush = now
                    if strip.done:
                        break
            except Exception as exc:  # noqa: BLE001
                log.warning("ProjectBuilder: streaming failed ({e}); using one-shot completion.", e=exc)
        if not got_any:                                      # no streaming available: write it in one go
            resp = await self._complete(msgs, max_tokens=14000, timeout=360)
            strip.raw = resp.text if resp and resp.success else ""
        content = strip.content().rstrip() + "\n"
        path.write_text(content, encoding="utf-8")
        return content

    # ── helpers ───────────────────────────────────────────────────────────
    async def _complete(self, msgs, max_tokens: int, timeout: float):
        try:
            return await self._llm.complete(msgs, temperature=0.3, max_tokens=max_tokens, timeout=timeout)
        except TypeError:
            return await self._llm.complete(msgs, temperature=0.3, max_tokens=max_tokens)

    async def _speak(self, text: str) -> None:
        if self._say:
            await self._say(text)

    async def _show(self, text: str) -> None:
        if self._status:
            r = self._status(text)
            if asyncio.iscoroutine(r):
                await r

    @staticmethod
    def _unique(path: Path) -> Path:
        if not path.exists():
            return path
        stem, suf, n = path.stem, path.suffix, 2
        while (path.parent / f"{stem}_{n}{suf}").exists():
            n += 1
        return path.parent / f"{stem}_{n}{suf}"


def open_in_vscode(path: Path, new_window: bool = False) -> None:
    """Open a folder (new window) or a file (reusing the window) in VS Code."""
    try:
        args = ["cmd", "/c", "code", "-n" if new_window else "-r", str(path)]
        subprocess.Popen(args, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not open VS Code: {e}", e=exc)
