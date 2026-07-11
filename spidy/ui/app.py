"""
SpidyApp — Qt application lifecycle manager
============================================
Owns the QApplication, the OverlayWindow, the SystemTrayManager,
and the GlobalHotkeyManager.  Bridges the EventBus to Qt signals.

Usage
-----
    # In your main entry point:
    from spidy.ui.app import SpidyApp

    app = SpidyApp(bus=event_bus, config=spidy_config)
    app.start()          # Creates QApplication + overlay, shows tray icon
    app.run()            # Enters Qt event loop (blocks until exit)

    # To show/hide from asyncio:
    await bus.publish(UIShowEvent())
    await bus.publish(UIHideEvent())

    # To update state from asyncio:
    await bus.publish(UIStateChangeEvent(state="listening"))

Thread model
------------
SpidyApp.run() blocks the calling thread in the Qt event loop.
asyncio runs in a separate thread (started before calling run()).
The UISignalBridge forwards asyncio events to Qt safely.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

if TYPE_CHECKING:
    from spidy.config.manager import SpidyConfig
    from spidy.core.event_bus import EventBus

from spidy.logging.logger import get_logger
from spidy.ui.events import (
    UIHideEvent, UIMessageEvent, UINotifyEvent, UIReadyEvent,
    UIShowEvent, UIStateChangeEvent, UIThemeChangeEvent,
    UIWaveformDataEvent,
)
from spidy.ui.hotkey import GlobalHotkeyManager
from spidy.ui.overlay import OverlayWindow, UISignalBridge
from spidy.ui.state import UIStateMachine, UIState
from spidy.ui.themes import build_default_theme_manager
from spidy.ui.tray import SystemTrayManager

log = get_logger(__name__)


class SpidyApp:
    """
    Top-level Qt application manager for Spidy's overlay UI.

    Parameters
    ----------
    bus:
        The application EventBus.
    config:
        The loaded SpidyConfig (only ``ui`` section is used here).
    loop:
        The asyncio event loop running in the background thread.
        Required for thread-safe EventBus publishing from Qt callbacks.
    """

    def __init__(
        self,
        bus: "EventBus",
        config: "SpidyConfig",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._bus = bus
        self._config = config
        self._loop = loop

        self._qt_app: QApplication | None = None
        self._overlay: OverlayWindow | None = None
        self._tray: SystemTrayManager | None = None
        self._hotkey: GlobalHotkeyManager | None = None
        self._bridge: UISignalBridge | None = None
        self._state_machine = UIStateMachine()
        self._theme_mgr = build_default_theme_manager(config.ui.theme)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Initialise Qt and create all UI components.

        Must be called from the thread that will run the Qt event loop.
        """
        if not QApplication.instance():
            self._qt_app = QApplication(sys.argv)
            self._qt_app.setApplicationName("Spidy")
            self._qt_app.setOrganizationName("SpidyAI")
            self._qt_app.setQuitOnLastWindowClosed(False)

        self._setup_overlay()
        self._setup_tray()
        self._setup_hotkey()
        self._setup_event_subscriptions()

        # Show initial overlay
        if self._config.ui.enabled:
            assert self._overlay is not None
            self._overlay.show_animated()
            self._publish_sync(UIReadyEvent())
            log.info("Spidy overlay UI started.")

    def run(self) -> int:
        """Enter the Qt event loop. Blocks until the application exits."""
        if self._qt_app is None:
            raise RuntimeError("Call start() before run().")
        code = self._qt_app.exec()
        self._shutdown()
        return code

    def quit(self) -> None:
        """Gracefully exit the Qt application."""
        if self._qt_app:
            self._qt_app.quit()

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_overlay(self) -> None:
        cfg = self._config.ui
        theme = self._theme_mgr.current

        self._overlay = OverlayWindow(
            theme=theme,
            width=cfg.width,
            height=cfg.height,
            edge=cfg.position,
            always_on_top=cfg.always_on_top,
            animate=cfg.animate,
        )

        # Signal bridge
        self._bridge = UISignalBridge()
        self._bridge.state_change_requested.connect(self._overlay.set_state)
        self._bridge.message_received.connect(self._overlay.add_message)
        self._bridge.notification_requested.connect(
            self._overlay.show_notification
        )
        self._bridge.waveform_data_received.connect(self._overlay.update_waveform)
        self._bridge.theme_change_requested.connect(self._on_theme_change)

        # Overlay outbound signals → EventBus
        self._overlay.mic_button_clicked.connect(self._on_mic_clicked)
        self._overlay.overlay_closed.connect(self._on_overlay_closed)

        # Theme change propagation
        self._theme_mgr.add_listener(self._overlay.apply_theme)

    def _setup_tray(self) -> None:
        self._tray = SystemTrayManager()
        self._tray.show_overlay_requested.connect(self._on_show_requested)
        self._tray.hide_overlay_requested.connect(self._on_hide_requested)
        self._tray.theme_change_requested.connect(self._on_theme_change)
        self._tray.settings_requested.connect(self._on_settings_requested)
        self._tray.quit_requested.connect(self.quit)
        self._tray.apply_theme(self._theme_mgr.current)
        self._tray.start()

        # Keep tray in sync with overlay visibility
        if self._overlay:
            self._overlay.overlay_closed.connect(
                lambda: self._tray.set_overlay_visible(False)
            )

    def _setup_hotkey(self) -> None:
        self._hotkey = GlobalHotkeyManager(
            bus=self._bus,
            hotkey=self._config.ui.hotkey,
            loop=self._loop,
        )
        self._hotkey.start(self._loop)

    def _setup_event_subscriptions(self) -> None:
        """Subscribe the bridge to relevant EventBus topics."""
        b = self._bus
        bridge = self._bridge

        b.subscribe("ui.show", self._handle_show)
        b.subscribe("ui.hide", self._handle_hide)
        b.subscribe("ui.state_change", self._handle_state_change)
        b.subscribe("ui.message", self._handle_message)
        b.subscribe("ui.notify", self._handle_notify)
        b.subscribe("ui.waveform_data", self._handle_waveform)
        b.subscribe("ui.theme_change", self._handle_theme_change)
        b.subscribe("ui.hotkey_pressed", self._handle_hotkey)

        # Listen for voice pipeline events (future integration)
        b.subscribe("voice.listening_started", self._handle_listening_started)
        b.subscribe("voice.listening_stopped", self._handle_listening_stopped)
        b.subscribe("voice.speaking_started", self._handle_speaking_started)
        b.subscribe("voice.speaking_stopped", self._handle_speaking_stopped)

        log.debug("UI EventBus subscriptions registered.")

    # ── EventBus handlers (async — called from asyncio loop) ─────────────

    async def _handle_show(self, event: UIShowEvent) -> None:
        if self._overlay:
            self._overlay.show_animated()
            if self._tray:
                self._tray.set_overlay_visible(True)

    async def _handle_hide(self, event: UIHideEvent) -> None:
        if self._overlay:
            self._overlay.hide_animated()
            if self._tray:
                self._tray.set_overlay_visible(False)

    async def _handle_state_change(self, event: UIStateChangeEvent) -> None:
        if self._bridge:
            self._bridge.request_state_change(event.state)

    async def _handle_message(self, event: UIMessageEvent) -> None:
        if self._bridge:
            self._bridge.request_message(event.role, event.text)

    async def _handle_notify(self, event: UINotifyEvent) -> None:
        if self._bridge:
            self._bridge.request_notification(
                event.title, event.body, event.level, event.duration_ms
            )

    async def _handle_waveform(self, event: UIWaveformDataEvent) -> None:
        if self._bridge:
            self._bridge.request_waveform(event.amplitudes)

    async def _handle_theme_change(self, event: UIThemeChangeEvent) -> None:
        if self._bridge:
            self._bridge.request_theme_change(event.theme)

    async def _handle_hotkey(self, event) -> None:
        """Toggle overlay on hotkey press."""
        if self._overlay and self._overlay.isVisible():
            self._overlay.hide_animated()
            if self._tray:
                self._tray.set_overlay_visible(False)
        else:
            if self._overlay:
                self._overlay.show_animated()
            if self._tray:
                self._tray.set_overlay_visible(True)

    # Voice pipeline forward-compat handlers
    async def _handle_listening_started(self, event) -> None:
        if self._bridge:
            self._bridge.request_state_change("listening")

    async def _handle_listening_stopped(self, event) -> None:
        if self._bridge:
            self._bridge.request_state_change("thinking")

    async def _handle_speaking_started(self, event) -> None:
        if self._bridge:
            self._bridge.request_state_change("speaking")

    async def _handle_speaking_stopped(self, event) -> None:
        if self._bridge:
            self._bridge.request_state_change("idle")

    # ── Qt slot handlers (called from Qt main thread) ─────────────────────

    def _on_mic_clicked(self) -> None:
        from spidy.ui.events import UIMicButtonClickedEvent
        self._publish_sync(UIMicButtonClickedEvent())

    def _on_overlay_closed(self) -> None:
        from spidy.ui.events import UIClosedEvent
        self._publish_sync(UIClosedEvent())

    def _on_show_requested(self) -> None:
        if self._overlay:
            self._overlay.show_animated()

    def _on_hide_requested(self) -> None:
        if self._overlay:
            self._overlay.hide_animated()

    def _on_settings_requested(self) -> None:
        from spidy.ui.events import UISettingsOpenedEvent
        self._publish_sync(UISettingsOpenedEvent())

    def _on_theme_change(self, name: str) -> None:
        try:
            theme = self._theme_mgr.set_theme(name)
            if self._tray:
                self._tray.apply_theme(theme)
        except KeyError:
            log.warning(f"Unknown theme: {name!r}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _publish_sync(self, event) -> None:
        """Publish an event from the Qt thread to the asyncio EventBus."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._bus.publish(event), self._loop
            )

    def _shutdown(self) -> None:
        if self._hotkey:
            self._hotkey.stop()
        if self._tray:
            self._tray.stop()
        log.info("Spidy UI shutdown complete.")
