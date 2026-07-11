"""
DownloadObserver — Downloads Folder File Watcher
=================================================
Monitors the user's Downloads folder for new files and publishes
context.download_added events.

Implementation
--------------
Uses watchdog's Observer + FileSystemEventHandler running in a
background thread, bridged to the asyncio EventBus via
publish_threadsafe().

This is more efficient than polling the folder contents each cycle:
watchdog uses OS-native filesystem events (ReadDirectoryChangesW on
Windows) with near-zero CPU overhead.

Behaviour
---------
- Watches %USERPROFILE%\\Downloads by default
- Fires on file creation AND on file moved-to (drag/drop)
- Skips .tmp, .part, .crdownload (partial download files)
- After a file is created, waits SETTLE_SECONDS before publishing
  to ensure the download is complete (file handle released)
- Recursion depth: 1 (immediate children only, not subfolders)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.events import DownloadAddedEvent, DownloadModifiedEvent

log = get_logger(__name__)

# Extensions that indicate an incomplete download — skip events for these
_PARTIAL_EXTENSIONS = frozenset({
    ".tmp", ".part", ".crdownload", ".download",
    ".partial", ".opdownload",
})

# Wait this many seconds after creation before publishing (file settle time)
SETTLE_SECONDS = 2.0


class DownloadObserver(BaseObserver):
    """
    Watches the Downloads folder for new files.

    Parameters
    ----------
    bus:
        Application EventBus.
    poll_interval:
        Not used for watchdog-based watching; only used in fallback poll mode.
    downloads_path:
        Absolute path to monitor. Defaults to ~\\Downloads.
    """

    name = "downloads"

    def __init__(
        self,
        bus: EventBus,
        poll_interval: float = 2.0,
        downloads_path: Path | None = None,
    ) -> None:
        super().__init__(bus, poll_interval)
        self._downloads_path = downloads_path or (
            Path.home() / "Downloads"
        )
        self._watchdog_observer: object | None = None
        self._watchdog_available = False

    async def on_start(self) -> None:
        try:
            from watchdog.observers import Observer  # noqa: F401
            self._watchdog_available = True
        except ImportError:
            log.warning(
                "DownloadObserver: watchdog not installed. "
                "Install with: pip install watchdog"
            )
            return

        if not self._downloads_path.exists():
            log.warning(
                "DownloadObserver: downloads folder not found: {path}",
                path=self._downloads_path,
            )
            return

        self._start_watchdog()
        log.info(
            "DownloadObserver: watching {path}",
            path=self._downloads_path,
        )

    async def on_stop(self) -> None:
        if self._watchdog_observer is not None:
            try:
                self._watchdog_observer.stop()
                self._watchdog_observer.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            self._watchdog_observer = None
        log.debug("DownloadObserver stopped.")

    async def poll(self) -> None:
        # watchdog handles events — poll() is a no-op in normal operation
        # It runs as a heartbeat to keep the BaseObserver loop alive
        if not self._watchdog_available or self._downloads_path is None:
            return

    def _start_watchdog(self) -> None:
        """Start the watchdog Observer in a background thread."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        bus = self._bus

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if path.suffix.lower() in _PARTIAL_EXTENSIONS:
                    return
                # Settle before publishing
                time.sleep(SETTLE_SECONDS)
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                log.info(
                    "Download detected: {name} ({size} bytes)",
                    name=path.name,
                    size=size,
                )
                bus.publish_threadsafe(DownloadAddedEvent(
                    filename=path.name,
                    path=str(path),
                    size_bytes=size,
                    extension=path.suffix.lower(),
                ))

            def on_moved(self, event):
                """Handle files moved into the downloads folder."""
                if event.is_directory:
                    return
                dest = Path(event.dest_path)
                if dest.suffix.lower() in _PARTIAL_EXTENSIONS:
                    return
                try:
                    size = dest.stat().st_size
                except OSError:
                    size = 0
                bus.publish_threadsafe(DownloadAddedEvent(
                    filename=dest.name,
                    path=str(dest),
                    size_bytes=size,
                    extension=dest.suffix.lower(),
                ))

            def on_modified(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if path.suffix.lower() in _PARTIAL_EXTENSIONS:
                    return
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                bus.publish_threadsafe(DownloadModifiedEvent(
                    filename=path.name,
                    path=str(path),
                    size_bytes=size,
                ))

        observer = Observer()
        observer.schedule(_Handler(), str(self._downloads_path), recursive=False)
        observer.daemon = True
        observer.start()
        self._watchdog_observer = observer
