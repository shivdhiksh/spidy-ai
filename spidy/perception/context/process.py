"""
ProcessObserver — New & Exited Process Monitor
===============================================
Tracks the set of running processes and publishes events when
a new process starts or an existing one exits.

Design
------
- On start: snapshot all current PIDs into _known_pids
- Each poll: compare current PIDs with _known_pids
  - New PIDs → ProcessStartedEvent
  - Gone PIDs → ProcessEndedEvent
- Uses psutil.process_iter() — the most reliable cross-platform approach
- Stores minimal info per PID (name, exe) to keep memory low

Performance
-----------
psutil.process_iter() on Windows with 200 processes takes ~5-15ms.
We call it every 3 seconds (default) so CPU impact is < 1%.

Filtering
---------
Many system processes are short-lived (svchost.exe instances, etc).
We skip processes that started and ended within one poll interval
to avoid flooding the event bus.

Privacy
-------
Command-line args are collected but never logged above DEBUG level.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import psutil

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.events import ProcessEndedEvent, ProcessStartedEvent

log = get_logger(__name__)


@dataclass
class _ProcessRecord:
    name: str
    pid: int
    exe: str


# Processes to filter from notifications (too noisy)
_NOISE_PROCESSES = frozenset({
    "svchost.exe", "conhost.exe", "RuntimeBroker.exe",
    "SearchProtocolHost.exe", "SearchFilterHost.exe",
    "WmiPrvSE.exe", "dllhost.exe",
})


class ProcessObserver(BaseObserver):
    """
    Monitors running processes for new starts and exits.

    Parameters
    ----------
    bus:
        Application EventBus.
    poll_interval:
        How often to refresh the process list. Default: 3.0s.
    filter_noise:
        Whether to suppress events for common high-frequency system processes.
    track_cmdline:
        Whether to collect process command lines (may require elevation).
    """

    name = "process"

    def __init__(
        self,
        bus: EventBus,
        poll_interval: float = 3.0,
        filter_noise: bool = True,
        track_cmdline: bool = False,
    ) -> None:
        super().__init__(bus, poll_interval)
        self._filter_noise = filter_noise
        self._track_cmdline = track_cmdline
        self._known: dict[int, _ProcessRecord] = {}  # pid → record

    async def on_start(self) -> None:
        """Snapshot existing processes so we don't fire events for them."""
        self._known = await asyncio.to_thread(self._snapshot_all)
        log.debug(
            "ProcessObserver: tracking {n} existing processes.",
            n=len(self._known),
        )

    async def poll(self) -> None:
        current = await asyncio.to_thread(self._snapshot_all)
        current_pids = set(current.keys())
        known_pids = set(self._known.keys())

        new_pids = current_pids - known_pids
        gone_pids = known_pids - current_pids

        # Publish events for new processes
        for pid in new_pids:
            rec = current[pid]
            if self._filter_noise and rec.name in _NOISE_PROCESSES:
                continue
            log.debug("Process started: {name} (pid={pid})", name=rec.name, pid=pid)
            await self._bus.publish(ProcessStartedEvent(
                name=rec.name,
                pid=pid,
                exe=rec.exe,
            ))

        # Publish events for ended processes
        for pid in gone_pids:
            rec = self._known[pid]
            if self._filter_noise and rec.name in _NOISE_PROCESSES:
                continue
            log.debug("Process ended: {name} (pid={pid})", name=rec.name, pid=pid)
            await self._bus.publish(ProcessEndedEvent(
                name=rec.name,
                pid=pid,
            ))

        self._known = current

    @property
    def known_processes(self) -> list[_ProcessRecord]:
        """Snapshot of the currently known process list."""
        return list(self._known.values())

    # ── Internal ──────────────────────────────────────────────────────────

    def _snapshot_all(self) -> dict[int, _ProcessRecord]:
        """
        Capture all running processes as a pid→record dict.
        Runs in a thread — psutil.process_iter() is blocking.
        """
        result: dict[int, _ProcessRecord] = {}
        attrs = ["pid", "name", "exe"]

        for proc in psutil.process_iter(attrs=attrs):
            try:
                info = proc.info
                pid = info["pid"]
                name = info.get("name") or ""
                exe = info.get("exe") or ""

                result[pid] = _ProcessRecord(
                    name=name,
                    pid=pid,
                    exe=exe,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        return result
