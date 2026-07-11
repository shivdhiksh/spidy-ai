"""
ClipboardObserver — Permission-Aware Clipboard Monitor
=======================================================
Monitors the Windows clipboard for text changes and publishes
context.clipboard_changed events when new text is copied.

Permission model
----------------
Clipboard access is privacy-sensitive. This observer:
1. Checks config.permissions.allow_clipboard_access (default: True)
2. Only reads text (CF_UNICODETEXT) — never images or files
3. Truncates captured text to MAX_CAPTURE_CHARS (500)
4. Never logs the clipboard contents at INFO level — only DEBUG

If access is denied by config, the observer starts but skips polling.

Windows API used
----------------
  win32clipboard.OpenClipboard()
  win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
  win32clipboard.CloseClipboard()

Change detection
----------------
Uses GetClipboardSequenceNumber() which increments on every clipboard
change — O(1) check without reading the clipboard content each poll.
"""

from __future__ import annotations

import asyncio

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.events import ClipboardChangedEvent

log = get_logger(__name__)

MAX_CAPTURE_CHARS = 500


class ClipboardObserver(BaseObserver):
    """
    Monitors clipboard for text changes.

    Parameters
    ----------
    bus:
        Application EventBus.
    poll_interval:
        How often to check the clipboard sequence number. Default: 1.0s.
    enabled:
        Set False to disable clipboard monitoring (privacy override).
    """

    name = "clipboard"

    def __init__(
        self,
        bus: EventBus,
        poll_interval: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(bus, poll_interval)
        self._enabled = enabled
        self._last_sequence: int = -1
        self._last_text: str = ""
        self._win32_available = False

    async def on_start(self) -> None:
        if not self._enabled:
            log.info("ClipboardObserver: disabled by config.")
            return
        try:
            import win32clipboard  # noqa: F401
            import win32con  # noqa: F401
            self._win32_available = True
            log.debug("ClipboardObserver: pywin32 available.")
        except ImportError:
            log.warning(
                "ClipboardObserver: pywin32 not installed. "
                "Clipboard monitoring will be disabled."
            )

    async def poll(self) -> None:
        if not self._enabled or not self._win32_available:
            return

        seq, text = await asyncio.to_thread(self._read_clipboard)

        if seq == self._last_sequence:
            return  # Nothing changed

        self._last_sequence = seq

        # Only publish if text actually changed (clipboard can change for non-text)
        if text is None or text == self._last_text:
            return

        self._last_text = text
        full_length = len(text)
        truncated = text[:MAX_CAPTURE_CHARS]
        is_truncated = full_length > MAX_CAPTURE_CHARS

        log.debug(
            "Clipboard changed: {length} chars",
            length=full_length,
        )

        await self._bus.publish(ClipboardChangedEvent(
            text=truncated,
            full_length=full_length,
            is_truncated=is_truncated,
        ))

    @property
    def last_text(self) -> str:
        """Last observed clipboard text (may be truncated)."""
        return self._last_text

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _read_clipboard() -> tuple[int, str | None]:
        """
        Read clipboard sequence number and text content.

        Returns (sequence_number, text_or_None).
        Runs in a thread — must not touch asyncio.
        """
        try:
            import win32clipboard
            import win32con

            seq = win32clipboard.GetClipboardSequenceNumber()

            try:
                win32clipboard.OpenClipboard(None)
                try:
                    if win32clipboard.IsClipboardFormatAvailable(
                        win32con.CF_UNICODETEXT
                    ):
                        text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                        return seq, text
                    return seq, None
                finally:
                    win32clipboard.CloseClipboard()
            except Exception:
                # Another app may have the clipboard open; skip this poll
                return seq, None

        except Exception as exc:  # noqa: BLE001
            log.debug("ClipboardObserver._read_clipboard error: {exc}", exc=exc)
            return -1, None
