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
    UIConfirmationGrantedEvent, UIConfirmationDeniedEvent,
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

        # M17: task progress + confirmation wiring
        self._bridge.task_progress_received.connect(self._overlay.update_task_progress)
        self._bridge.confirmation_show.connect(self._overlay.show_confirmation)
        self._bridge.confirmation_hide.connect(self._overlay.hide_confirmation)

        # Thread-safe show / hide / toggle wiring (queued connection across threads)
        self._bridge.show_hud_requested.connect(self._overlay.show_hud)
        self._bridge.hide_hud_requested.connect(self._overlay.hide_hud)
        self._bridge.toggle_hud_requested.connect(self._overlay.toggle_hud)

        # Confirmation card signals -> EventBus (publish back to agent layer)
        self._overlay._confirmation_card.confirmed.connect(self._on_confirmation_granted)
        self._overlay._confirmation_card.cancelled.connect(self._on_confirmation_denied)

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

        # Voice pipeline events
        b.subscribe("voice.listening_started",  self._handle_listening_started)
        b.subscribe("voice.listening_stopped",  self._handle_listening_stopped)
        b.subscribe("voice.speaking_started",   self._handle_speaking_started)
        b.subscribe("voice.speaking_stopped",   self._handle_speaking_stopped)

        # Wake word
        b.subscribe("wake_word.detected", self._handle_wake_detected)

        # Agent events (M17 -- autonomous task progress + confirmation)
        b.subscribe("agent.goal_created",          self._handle_agent_goal_created)
        b.subscribe("agent.goal_started",          self._handle_agent_goal_started)
        b.subscribe("agent.goal_completed",        self._handle_agent_goal_completed)
        b.subscribe("agent.goal_failed",           self._handle_agent_goal_failed)
        b.subscribe("agent.task_started",          self._handle_agent_task_started)
        b.subscribe("agent.task_completed",        self._handle_agent_task_completed)
        b.subscribe("agent.task_failed",           self._handle_agent_task_failed)
        b.subscribe("agent.progress",              self._handle_agent_progress)
        b.subscribe("agent.confirmation_required", self._handle_agent_confirmation_required)

        log.debug("UI EventBus subscriptions registered.")

    # ── EventBus handlers (async — called from asyncio loop) ─────────────

    async def _handle_show(self, event: UIShowEvent) -> None:
        log.info("[HUD LIFECYCLE] show_hud requested via EventBus ui.show")
        if self._bridge:
            self._bridge.request_show_hud()
        if self._tray:
            self._tray.set_overlay_visible(True)

    async def _handle_hide(self, event: UIHideEvent) -> None:
        log.info("[HUD LIFECYCLE] hide_hud requested via EventBus ui.hide")
        if self._bridge:
            self._bridge.request_hide_hud()
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
                event.title, event.body, event.level
            )

    async def _handle_waveform(self, event: UIWaveformDataEvent) -> None:
        if self._bridge:
            self._bridge.request_waveform(event.amplitudes)

    async def _handle_theme_change(self, event: UIThemeChangeEvent) -> None:
        if self._bridge:
            self._bridge.request_theme_change(event.theme)

    async def _handle_hotkey(self, event) -> None:
        """Toggle overlay on hotkey press thread-safely."""
        log.info("[HUD LIFECYCLE] toggle_hud requested via hotkey")
        if self._bridge:
            self._bridge.request_toggle_hud()

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

    async def _handle_wake_detected(self, event) -> None:
        """Wake word detected → thread-safely request show and transition to LISTENING."""
        wake_phrase = getattr(event, "wake_phrase", None) or getattr(event, "model_name", "Hey Spidy")
        vis_before = self._overlay.isVisible() and self._overlay.windowOpacity() > 0.05 if self._overlay else False
        log.info(
            "[HUD LIFECYCLE] wake_received phrase=\"{phrase}\" visible_before={vis}",
            phrase=wake_phrase,
            vis=vis_before,
        )
        if self._bridge:
            self._bridge.request_show_hud()
            self._bridge.request_state_change("listening")
        if self._tray:
            self._tray.set_overlay_visible(True)

    # ── Public HUD Lifecycle API ──────────────────────────────────────────

    def show_hud(self) -> None:
        """Show HUD overlay window thread-safely."""
        if self._bridge:
            self._bridge.request_show_hud()
        elif self._overlay:
            self._overlay.show_hud()
        if self._tray:
            self._tray.set_overlay_visible(True)

    def hide_hud(self) -> None:
        """Hide HUD overlay window thread-safely."""
        if self._bridge:
            self._bridge.request_hide_hud()
        elif self._overlay:
            self._overlay.hide_hud()
        if self._tray:
            self._tray.set_overlay_visible(False)

    def toggle_hud(self) -> None:
        """Toggle HUD overlay window visibility thread-safely."""
        if self._bridge:
            self._bridge.request_toggle_hud()
        elif self._overlay:
            self._overlay.toggle_hud()

    # ── Qt slot handlers (called from Qt main thread) ─────────────────────

    def _on_mic_clicked(self) -> None:
        from spidy.ui.events import UIMicButtonClickedEvent
        self._publish_sync(UIMicButtonClickedEvent())

    def _on_overlay_closed(self) -> None:
        from spidy.ui.events import UIClosedEvent
        self._publish_sync(UIClosedEvent())

    def _on_show_requested(self) -> None:
        self.show_hud()

    def _on_hide_requested(self) -> None:
        self.hide_hud()

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

    def _on_confirmation_granted(self, task_id: str, goal_id: str) -> None:
        """User pressed CONFIRM -- publish back to agent layer."""
        from spidy.ui.events import UIConfirmationGrantedEvent
        self._publish_sync(UIConfirmationGrantedEvent(goal_id=goal_id, task_id=task_id))

    def _on_confirmation_denied(self, task_id: str, goal_id: str) -> None:
        """User pressed CANCEL -- publish back to agent layer."""
        from spidy.ui.events import UIConfirmationDeniedEvent
        self._publish_sync(UIConfirmationDeniedEvent(goal_id=goal_id, task_id=task_id))

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


    # ==========================================================================
    # Agent event handlers (M17)
    # ==========================================================================

    # Internal state for tracking task progress
    _agent_goal_desc: str = ""
    _agent_steps: list   = []

    async def _handle_agent_goal_created(self, event) -> None:
        desc = getattr(event, "description", "")
        self._agent_goal_desc = desc
        self._agent_steps = []

    async def _handle_agent_goal_started(self, event) -> None:
        desc = getattr(event, "description", "")
        self._agent_goal_desc = desc
        if self._bridge:
            self._bridge.request_state_change("working")
            self._bridge.request_task_progress(desc, self._agent_steps)

    async def _handle_agent_goal_completed(self, event) -> None:
        summary = getattr(event, "summary", "Goal completed.")
        if self._bridge:
            self._bridge.request_state_change("idle")
            self._bridge.request_notification("Goal Complete", summary, "success")
        self._agent_steps = []

    async def _handle_agent_goal_failed(self, event) -> None:
        error = getattr(event, "error", "Goal failed.")
        if self._bridge:
            self._bridge.request_state_change("error")
            self._bridge.request_notification("Goal Failed", error, "error")
        self._agent_steps = []

    async def _handle_agent_task_started(self, event) -> None:
        desc  = getattr(event, "description", "")
        index = getattr(event, "task_index", len(self._agent_steps))
        total = getattr(event, "total_tasks", 0)
        # Mark all prior steps done, this one running
        steps = []
        for i, s in enumerate(self._agent_steps):
            if isinstance(s, dict) and s.get("status") == "running":
                steps.append({"description": s["description"], "status": "done"})
            else:
                steps.append(s)
        steps.append({"description": desc, "status": "running"})
        self._agent_steps = steps
        if self._bridge:
            self._bridge.request_task_progress(self._agent_goal_desc, self._agent_steps)

    async def _handle_agent_task_completed(self, event) -> None:
        # Mark the last running step as done
        steps = []
        for s in self._agent_steps:
            if isinstance(s, dict) and s.get("status") == "running":
                steps.append({"description": s["description"], "status": "done"})
            else:
                steps.append(s)
        self._agent_steps = steps
        if self._bridge:
            self._bridge.request_task_progress(self._agent_goal_desc, self._agent_steps)

    async def _handle_agent_task_failed(self, event) -> None:
        steps = []
        for s in self._agent_steps:
            if isinstance(s, dict) and s.get("status") == "running":
                steps.append({"description": s["description"], "status": "failed"})
            else:
                steps.append(s)
        self._agent_steps = steps
        if self._bridge:
            self._bridge.request_task_progress(self._agent_goal_desc, self._agent_steps)

    async def _handle_agent_progress(self, event) -> None:
        msg = getattr(event, "message", "")
        if self._bridge and msg:
            self._bridge.request_state_change("working")

    async def _handle_agent_confirmation_required(self, event) -> None:
        task_id  = getattr(event, "task_id",          "")
        goal_id  = getattr(event, "goal_id",          "")
        desc     = getattr(event, "task_description", "")
        prompt   = getattr(event, "prompt",           "")
        if not prompt:
            prompt = "This action requires your confirmation before proceeding."
        if self._bridge:
            self._bridge.request_confirmation_show(task_id, goal_id, desc, prompt)
