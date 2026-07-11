"""
NoteSkill — Take, read, search, and manage notes.
Permission tiers: T0 for reading, T1 for writing.

Notes are stored as JSONL (one JSON object per line) in a configurable file.
Each note has: id, content, timestamp, tags.

Actions
-------
take_note   — Write a new note            (T1)
read_notes  — Read recent notes           (T0)
find_note   — Search notes by keyword     (T0)
clear_notes — Delete all notes            (T2)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spidy.skills.base import BaseSkill, SkillCapability, ParamSchema, SkillContext, SkillResult


class NoteSkill(BaseSkill):
    """
    Manages a personal notes file (JSONL format).

    Thread-safe: uses a lock for all file access.
    """

    name = "note_skill"
    version = "1.0.0"

    def __init__(self, notes_file: str = "notes.jsonl") -> None:
        self._notes_file = Path(notes_file)
        self._lock = threading.Lock()

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="take_note",
                description="Write a new note.",
                permission_tier="T1",
                params=[ParamSchema("content", "string", required=True,
                                    description="The note text.")],
                examples=["take a note", "note this", "remember that", "write down"],
            ),
            SkillCapability(
                action="read_notes",
                description="Read recent notes.",
                permission_tier="T0",
                params=[ParamSchema("limit", "int", required=False,
                                    default=5, description="Number of notes to show.")],
                examples=["read my notes", "show notes", "what did I note"],
            ),
            SkillCapability(
                action="find_note",
                description="Search notes by keyword.",
                permission_tier="T0",
                params=[ParamSchema("query", "string", required=True,
                                    description="Search keyword.")],
                examples=["find note about", "search notes for"],
            ),
            SkillCapability(
                action="clear_notes",
                description="Delete all notes permanently.",
                permission_tier="T2",
                examples=["clear all notes", "delete my notes"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "take_note":
            return await self._take_note(context)
        if action == "read_notes":
            return await self._read_notes(context)
        if action == "find_note":
            return await self._find_note(context)
        if action == "clear_notes":
            return await self._clear_notes(context)
        return SkillResult.fail(f"NoteSkill: unknown action '{action}'.")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _take_note(self, context: SkillContext) -> SkillResult:
        content = context.get("content", "").strip()
        if not content:
            return SkillResult.fail("Please provide content for the note.")

        note: dict[str, Any] = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": context.session_id,
        }
        try:
            with self._lock:
                self._notes_file.parent.mkdir(parents=True, exist_ok=True)
                with self._notes_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(note) + "\n")
            return SkillResult.ok(
                f"Note saved: \"{content[:60]}{'...' if len(content) > 60 else ''}\"",
                data=note,
                action_taken="take_note",
            )
        except Exception as exc:
            return SkillResult.fail(f"Failed to save note: {exc}", error=exc)

    async def _read_notes(self, context: SkillContext) -> SkillResult:
        limit = int(context.get("limit", 5))
        notes = self._load_notes()
        if not notes:
            return SkillResult.ok("You have no notes yet.", action_taken="read_notes")

        recent = notes[-limit:]
        lines = [
            f"{i + 1}. [{n['timestamp'][:10]}] {n['content']}"
            for i, n in enumerate(recent)
        ]
        msg = f"Your last {len(recent)} note(s):\n" + "\n".join(lines)
        return SkillResult.ok(msg, data=recent, action_taken="read_notes")

    async def _find_note(self, context: SkillContext) -> SkillResult:
        query = context.get("query", "").strip().lower()
        if not query:
            return SkillResult.fail("Please provide a search keyword.")

        notes = self._load_notes()
        matches = [n for n in notes if query in n.get("content", "").lower()]
        if not matches:
            return SkillResult.ok(
                f"No notes found containing \"{query}\".",
                action_taken="find_note",
            )

        lines = [
            f"{i + 1}. [{m['timestamp'][:10]}] {m['content']}"
            for i, m in enumerate(matches[:10])
        ]
        msg = f"Found {len(matches)} note(s) matching \"{query}\":\n" + "\n".join(lines)
        return SkillResult.ok(msg, data=matches, action_taken="find_note")

    async def _clear_notes(self, context: SkillContext) -> SkillResult:
        try:
            with self._lock:
                if self._notes_file.exists():
                    self._notes_file.unlink()
            return SkillResult.ok("All notes cleared.", action_taken="clear_notes")
        except Exception as exc:
            return SkillResult.fail(f"Failed to clear notes: {exc}", error=exc)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_notes(self) -> list[dict]:
        if not self._notes_file.exists():
            return []
        notes = []
        try:
            with self._lock:
                with self._notes_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                notes.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception:  # noqa: BLE001
            pass
        return notes
