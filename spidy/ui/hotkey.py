"""
Global Hotkey Manager — Milestone 5
=====================================
Registers a system-wide hotkey that toggles the Spidy overlay
regardless of which application has focus.

Implementation
--------------
Uses the ``keyboard`` library (pure Python + ctypes on Windows).
The hotkey callback fires in a background thread; events are published
via EventBus.publish_threadsafe() so Qt and asyncio code stays safe.

Graceful degradation
---------------------
If the ``keyboard`` library is not installed or the hotkey cannot be
registered (e.g. insufficient permissions, CI environment), the manager
logs a warning and continues without a hotkey.  The overlay can still
be shown programmatically via UIShowEvent.

Thread safety
-------------
``keyboard`` fires its callback in its own OS hook thread.
We bridge to asyncio via ``asyncio.run_coroutine_threadsafe``.

Wake-word readiness
-------------------
This module is also the registration point for future wake-word triggered
show logic.  When the VoiceEngine detects "Hey Spidy" it will publish
``UIShowEvent(source="wake_word")``.  The GlobalHotkeyManager is already
prepared to handle that event pattern (see ``register_ui_events``).
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

from spidy.logging.logger import get_logger
from spidy.ui.events import UIHotkeyPressedEvent, UIShowEvent, UIHideEvent

log = get_logger(__name__)


class GlobalHotkeyManager:
    """
    Registers a global hotkey that publishes UIHotkeyPressedEvent.

    The UI layer subscribes to UIHotkeyPressedEvent and toggles
    the overlay visibility accordingly.

    Parameters
    ----------
    bus:
        The application EventBus.
    hotkey:
        Hotkey string in ``keyboard`` format, e.g. ``"ctrl+space"``.
    loop:
        The asyncio event loop running in the main thread (or wherever
        the EventBus lives).  Required for thread-safe publishing.
    """

    def __init__(
        self,
        bus: "EventBus",
        hotkey: str = "ctrl+space",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._bus = bus
        self._hotkey = hotkey
        self._loop = loop
        self._registered = False
        self._keyboard_available = False

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def hotkey(self) -> str:
        return self._hotkey

    @property
    def is_registered(self) -> bool:
        return self._registered

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> bool:
        """
        Register the global hotkey.

        Parameters
        ----------
        loop:
            Asyncio event loop. If None, uses the loop passed in __init__
            or tries to get the running loop.

        Returns
        -------
        bool
            True if the hotkey was registered successfully.
        """
        if loop is not None:
            self._loop = loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                log.warning("No event loop — hotkey will not be registered.")
                return False

        try:
            import keyboard  # type: ignore[import-untyped]
            self._keyboard_available = True
        except ImportError:
            log.warning(
                "The 'keyboard' package is not installed. "
                "Global hotkey will not be available. "
                "Install with: pip install keyboard"
            )
            return False

        try:
            import keyboard
            keyboard.add_hotkey(self._hotkey, self._on_hotkey_fired, suppress=False)
            self._registered = True
            log.info(f"Global hotkey registered: {self._hotkey!r}")
            return True
        except Exception as exc:
            log.warning(
                f"Failed to register global hotkey {self._hotkey!r}: {exc}. "
                "This is expected in CI or restricted environments."
            )
            return False

    def stop(self) -> None:
        """Unregister the global hotkey."""
        if not self._registered:
            return
        try:
            import keyboard
            keyboard.remove_hotkey(self._hotkey)
            log.info(f"Global hotkey unregistered: {self._hotkey!r}")
        except Exception as exc:
            log.debug(f"Failed to unregister hotkey: {exc}")
        finally:
            self._registered = False

    def update_hotkey(self, new_hotkey: str) -> bool:
        """
        Change the registered hotkey at runtime.

        Stops the old hotkey and registers the new one.
        """
        was_registered = self._registered
        self.stop()
        self._hotkey = new_hotkey
        if was_registered:
            return self.start()
        return False

    @staticmethod
    def parse_hotkey(text: str) -> str:
        """
        Normalise a hotkey string to keyboard library format.

        Examples
        --------
        "Ctrl+Space"    → "ctrl+space"
        "CTRL+SHIFT+S"  → "ctrl+shift+s"
        "ctrl + space"  → "ctrl+space"
        """
        return "+".join(
            part.strip().lower()
            for part in text.split("+")
        )

    @staticmethod
    def validate_hotkey(text: str) -> bool:
        """
        Basic validation — checks that the hotkey string is non-empty
        and contains at least one modifier key.
        """
        if not text or not text.strip():
            return False
        parts = [p.strip().lower() for p in text.split("+")]
        modifiers = {"ctrl", "shift", "alt", "win", "control", "windows"}
        has_modifier = any(p in modifiers for p in parts)
        has_key = any(p not in modifiers for p in parts)
        return has_modifier and has_key

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_hotkey_fired(self) -> None:
        """Called from keyboard's OS hook thread."""
        if self._loop is None or self._loop.is_closed():
            return
        event = UIHotkeyPressedEvent(hotkey=self._hotkey)
        try:
            asyncio.run_coroutine_threadsafe(
                self._bus.publish(event),
                self._loop,
            )
        except Exception as exc:
            log.debug(f"Hotkey event dispatch failed: {exc}")

    def __repr__(self) -> str:
        return (
            f"GlobalHotkeyManager(hotkey={self._hotkey!r}, "
            f"registered={self._registered})"
        )
