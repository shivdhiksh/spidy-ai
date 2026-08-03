"""
ClipboardSkill — System Clipboard Access
=========================================
Read from and write to the Windows system clipboard.

Permission Tiers
----------------
T0 — Read clipboard contents (no system modification)
T1 — Write text to clipboard

Actions
-------
get_clipboard    — Read the current clipboard text               (T0)
set_clipboard    — Write text to the clipboard                   (T1)

Architecture
------------
Primary: pyperclip (cross-platform, installed in requirements)
Fallback: Windows ctypes (win32clipboard via ctypes.windll)
All failures return informative SkillResult.fail() — never raise.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.skills.base import BaseSkill, ParamSchema, SkillCapability, SkillContext, SkillResult
from spidy.logging.logger import get_logger

log = get_logger(__name__)


class ClipboardSkill(BaseSkill):
    """
    Clipboard read/write skill.

    Uses pyperclip as the primary backend with a ctypes fallback
    for Windows environments where pyperclip is unavailable.
    """

    name = "clipboard_skill"
    version = "1.0.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="get_clipboard",
                description="Read and return the current clipboard contents.",
                permission_tier="T0",
                params=[],
                examples=[
                    "what's in the clipboard",
                    "show clipboard",
                    "read clipboard",
                    "what did i copy",
                    "paste contents",
                    "clipboard contents",
                ],
            ),
            SkillCapability(
                action="set_clipboard",
                description="Copy text to the clipboard.",
                permission_tier="T1",
                params=[
                    ParamSchema(
                        "text",
                        "string",
                        required=True,
                        description="The text to copy to the clipboard.",
                    ),
                ],
                examples=[
                    "copy hello world to clipboard",
                    "put this text in the clipboard",
                    "copy this to clipboard",
                ],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "get_clipboard":
            return await self._get_clipboard(context)
        if action == "set_clipboard":
            return await self._set_clipboard(context)
        return SkillResult.fail(f"ClipboardSkill: unknown action '{action}'.")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _get_clipboard(self, _context: SkillContext) -> SkillResult:
        try:
            text = await asyncio.to_thread(self._read_clipboard)
        except RuntimeError as exc:
            return SkillResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"ClipboardSkill: could not read clipboard: {exc}", error=exc
            )

        if not text:
            return SkillResult.ok(
                "The clipboard is empty.",
                data={"text": "", "empty": True},
                action_taken="get_clipboard",
            )

        preview = text[:200] + ("…" if len(text) > 200 else "")
        return SkillResult.ok(
            f"Clipboard contains ({len(text)} chars):\n{preview}",
            data={"text": text, "length": len(text)},
            action_taken="get_clipboard",
        )

    async def _set_clipboard(self, context: SkillContext) -> SkillResult:
        text: str = context.get("text", "") or ""
        if not text:
            return SkillResult.fail(
                "ClipboardSkill: 'text' parameter is required for set_clipboard."
            )
        try:
            await asyncio.to_thread(self._write_clipboard, text)
        except RuntimeError as exc:
            return SkillResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"ClipboardSkill: could not write to clipboard: {exc}", error=exc
            )

        preview = text[:80] + ("…" if len(text) > 80 else "")
        return SkillResult.ok(
            f"Copied to clipboard: \"{preview}\"",
            data={"text": text, "length": len(text)},
            action_taken="set_clipboard",
        )

    # ── Internal clipboard I/O ────────────────────────────────────────────

    @staticmethod
    def _read_clipboard() -> str:
        """Read clipboard text. Tries pyperclip, then Windows ctypes."""
        try:
            import pyperclip  # type: ignore[import-untyped]
            return pyperclip.paste() or ""
        except ImportError:
            pass

        # Fallback: Windows ctypes
        try:
            import ctypes
            import ctypes.wintypes
            CF_UNICODETEXT = 13
            ctypes.windll.user32.OpenClipboard(0)
            try:
                handle = ctypes.windll.user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return ""
                ptr = ctypes.windll.kernel32.GlobalLock(handle)
                try:
                    return ctypes.wstring_at(ptr) if ptr else ""
                finally:
                    ctypes.windll.kernel32.GlobalUnlock(handle)
            finally:
                ctypes.windll.user32.CloseClipboard()
        except AttributeError:
            raise RuntimeError(
                "ClipboardSkill: clipboard access requires pyperclip or Windows. "
                "Install with: pip install pyperclip"
            )

    @staticmethod
    def _write_clipboard(text: str) -> None:
        """Write text to clipboard. Tries pyperclip, then Windows ctypes."""
        try:
            import pyperclip  # type: ignore[import-untyped]
            pyperclip.copy(text)
            return
        except ImportError:
            pass

        # Fallback: Windows ctypes
        try:
            import ctypes
            import ctypes.wintypes
            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002

            data = text.encode("utf-16-le") + b"\x00\x00"
            handle = ctypes.windll.kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            ptr = ctypes.windll.kernel32.GlobalLock(handle)
            ctypes.memmove(ptr, data, len(data))
            ctypes.windll.kernel32.GlobalUnlock(handle)

            ctypes.windll.user32.OpenClipboard(0)
            try:
                ctypes.windll.user32.EmptyClipboard()
                ctypes.windll.user32.SetClipboardData(CF_UNICODETEXT, handle)
            finally:
                ctypes.windll.user32.CloseClipboard()
        except AttributeError:
            raise RuntimeError(
                "ClipboardSkill: clipboard write requires pyperclip or Windows. "
                "Install with: pip install pyperclip"
            )
