"""
ObserverManager — Context Observer Orchestrator
================================================
The ObserverManager owns all Context Observers and provides a single
lifecycle endpoint: start() / stop().

It also:
- Aggregates observer data into a DesktopStateSnapshot
- Publishes context.snapshot_updated on a regular cadence
- Exposes the latest snapshot for synchronous reads

This is the only class SpidyCore needs to know about for Milestone 2.
All individual observers are internal to this module.

Lifecycle
---------
    mgr = ObserverManager(config, bus)
    await mgr.start()          # starts all observers as asyncio tasks
    ...
    snapshot = mgr.snapshot    # read current state at any time
    ...
    await mgr.stop()           # stops all observers gracefully

Configuration
-------------
All observers read their settings from ContextConfig (from SpidyConfig).
Observers can be individually enabled/disabled via config.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.clipboard import ClipboardObserver
from spidy.perception.context.downloads import DownloadObserver
from spidy.perception.context.events import SnapshotUpdatedEvent
from spidy.perception.context.notifications import NotificationObserver
from spidy.perception.context.process import ProcessObserver
from spidy.perception.context.resources import SystemResourceObserver
from spidy.perception.context.snapshot import (
    DesktopStateSnapshot,
    ProcessInfo,
    ResourceInfo,
    WindowInfo,
)
from spidy.perception.context.window import ActiveWindowObserver

if TYPE_CHECKING:
    from spidy.config.manager import ContextConfig

log = get_logger(__name__)


class ObserverManager:
    """
    Orchestrates all context observers and maintains the desktop snapshot.

    Parameters
    ----------
    bus:
        The application EventBus.
    config:
        ContextConfig section from SpidyConfig.
    """

    def __init__(self, bus: EventBus, config: "ContextConfig") -> None:
        self._bus = bus
        self._config = config
        self._observers: list[BaseObserver] = []
        self._snapshot_task: asyncio.Task | None = None
        self._running = False

        # Named observer references for direct access
        self._window_obs: ActiveWindowObserver | None = None
        self._clipboard_obs: ClipboardObserver | None = None
        self._process_obs: ProcessObserver | None = None
        self._resource_obs: SystemResourceObserver | None = None

        # Latest snapshot
        self._snapshot = DesktopStateSnapshot()

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Build and start all enabled observers."""
        if self._running:
            return

        self._running = True
        cfg = self._config

        log.info("ObserverManager starting...")

        # ── Active Window (always enabled — core feature)
        self._window_obs = ActiveWindowObserver(
            bus=self._bus,
            poll_interval=cfg.window_poll_interval,
        )
        self._observers.append(self._window_obs)

        # ── Clipboard
        if cfg.clipboard_enabled:
            self._clipboard_obs = ClipboardObserver(
                bus=self._bus,
                poll_interval=cfg.clipboard_poll_interval,
                enabled=cfg.clipboard_enabled,
            )
            self._observers.append(self._clipboard_obs)

        # ── Process monitor
        if cfg.process_monitor_enabled:
            self._process_obs = ProcessObserver(
                bus=self._bus,
                poll_interval=cfg.process_poll_interval,
            )
            self._observers.append(self._process_obs)

        # ── System resources
        if cfg.resource_monitor_enabled:
            self._resource_obs = SystemResourceObserver(
                bus=self._bus,
                poll_interval=cfg.resource_poll_interval,
                cpu_alert_threshold=cfg.cpu_alert_threshold,
                ram_alert_threshold=cfg.ram_alert_threshold,
                battery_alert_threshold=cfg.battery_alert_threshold,
            )
            self._observers.append(self._resource_obs)

        # ── Downloads
        if cfg.download_monitor_enabled:
            self._observers.append(DownloadObserver(bus=self._bus))

        # ── Notifications (stub)
        if cfg.notification_monitor_enabled:
            self._observers.append(NotificationObserver(bus=self._bus))

        # Start all observers
        for obs in self._observers:
            await obs.start()

        # Start snapshot aggregation task
        self._snapshot_task = asyncio.create_task(
            self._snapshot_loop(),
            name="observer-snapshot",
        )

        log.info(
            "ObserverManager running | {n} observers active",
            n=len(self._observers),
        )

    async def stop(self) -> None:
        """Stop all observers and the snapshot loop."""
        self._running = False

        if self._snapshot_task and not self._snapshot_task.done():
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass

        for obs in reversed(self._observers):
            await obs.stop()

        self._observers.clear()
        log.info("ObserverManager stopped.")

    @property
    def snapshot(self) -> DesktopStateSnapshot:
        """Return the most recent desktop state snapshot."""
        return self._snapshot

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Snapshot aggregation ───────────────────────────────────────────────

    async def _snapshot_loop(self) -> None:
        """Periodically build and publish a DesktopStateSnapshot."""
        interval = self._config.snapshot_interval

        while self._running:
            await asyncio.sleep(interval)
            try:
                snapshot = self._build_snapshot()
                self._snapshot = snapshot
                await self._bus.publish(SnapshotUpdatedEvent(
                    active_window_title=snapshot.window.title,
                    active_app_name=snapshot.window.app_name,
                    active_pid=snapshot.window.pid,
                    cpu_pct=snapshot.resources.cpu_pct,
                    ram_pct=snapshot.resources.ram_pct,
                    battery_pct=snapshot.resources.battery_pct,
                    battery_plugged=snapshot.resources.battery_plugged,
                    clipboard_preview=snapshot.clipboard_text[:100],
                ))
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                log.debug("Snapshot loop error: {exc}", exc=exc)

    def _build_snapshot(self) -> DesktopStateSnapshot:
        """Aggregate observer data into an immutable DesktopStateSnapshot."""
        # Window info from window observer
        window_info: WindowInfo = (
            self._window_obs.current_window()
            if self._window_obs else WindowInfo()
        )

        # Resource info from resource observer
        resource_info: ResourceInfo = (
            self._resource_obs.latest
            if self._resource_obs else ResourceInfo()
        )

        # Clipboard
        clipboard_text = (
            self._clipboard_obs.last_text
            if self._clipboard_obs else ""
        )

        # Recent process starts
        recent: list[ProcessInfo] = []
        if self._process_obs:
            recent = [
                ProcessInfo(name=r.name, pid=r.pid, exe=r.exe)
                for r in self._process_obs.known_processes[:5]
            ]

        return DesktopStateSnapshot(
            window=window_info,
            resources=resource_info,
            clipboard_text=clipboard_text[:500],
            clipboard_length=len(clipboard_text),
            recent_starts=tuple(recent),
        )
