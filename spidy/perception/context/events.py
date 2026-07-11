"""
Context Observer Events
========================
All EventBus events published by the Milestone 2 context observers
are defined here in one place.

Design
------
- Every event is a frozen dataclass that extends Event
- Fields use Python primitive types (str, int, float, bool)
  so they can be JSON-serialised without extra work
- Optional fields use default=None / default_factory
- topic is a class attribute (not an instance field)

Topic namespace: "context.*"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from spidy.core.event_bus import Event


# ─── Window / Application ─────────────────────────────────────────────────────

@dataclass
class WindowChangedEvent(Event):
    """Emitted when the foreground window title changes."""
    topic = "context.window_changed"
    title: str = ""
    hwnd: int = 0
    app_name: str = ""      # e.g. "Code.exe"
    pid: int = 0


@dataclass
class AppChangedEvent(Event):
    """Emitted when the foreground application (exe) changes."""
    topic = "context.app_changed"
    app_name: str = ""      # e.g. "Code.exe"
    exe_path: str = ""      # Full path to executable
    pid: int = 0


# ─── Clipboard ────────────────────────────────────────────────────────────────

@dataclass
class ClipboardChangedEvent(Event):
    """Emitted when clipboard text content changes."""
    topic = "context.clipboard_changed"
    text: str = ""          # First 500 chars
    full_length: int = 0    # Total length of clipboard text
    is_truncated: bool = False


# ─── Processes ────────────────────────────────────────────────────────────────

@dataclass
class ProcessStartedEvent(Event):
    """Emitted when a new process starts."""
    topic = "context.process_started"
    name: str = ""
    pid: int = 0
    exe: str = ""           # Executable path (empty if inaccessible)
    cmdline: str = ""       # Command line (empty if inaccessible)


@dataclass
class ProcessEndedEvent(Event):
    """Emitted when a tracked process exits."""
    topic = "context.process_ended"
    name: str = ""
    pid: int = 0


# ─── System Resources ─────────────────────────────────────────────────────────

@dataclass
class ResourceUpdateEvent(Event):
    """Periodic system resource snapshot."""
    topic = "context.resource_update"
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_pct: float = 0.0        # 0.0 if GPU unavailable
    gpu_vram_used_gb: float = 0.0
    battery_pct: float = -1.0   # -1.0 if no battery
    battery_plugged: bool = False
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0


@dataclass
class ResourceAlertEvent(Event):
    """
    Emitted when a resource crosses a configured threshold.
    Only fires once until the value drops back below threshold.
    """
    topic = "context.resource_alert"
    resource: str = ""      # "cpu" | "ram" | "battery" | "gpu"
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""


# ─── Downloads ────────────────────────────────────────────────────────────────

@dataclass
class DownloadAddedEvent(Event):
    """Emitted when a new file appears in the Downloads folder."""
    topic = "context.download_added"
    filename: str = ""
    path: str = ""
    size_bytes: int = 0
    extension: str = ""


@dataclass
class DownloadModifiedEvent(Event):
    """Emitted when a file in Downloads is modified (e.g. download in progress)."""
    topic = "context.download_modified"
    filename: str = ""
    path: str = ""
    size_bytes: int = 0


# ─── Notifications ────────────────────────────────────────────────────────────

@dataclass
class NotificationEvent(Event):
    """Emitted when a Windows notification is detected."""
    topic = "context.notification"
    app: str = ""
    title: str = ""
    body: str = ""


# ─── Desktop Snapshot ─────────────────────────────────────────────────────────

@dataclass
class SnapshotUpdatedEvent(Event):
    """
    Emitted periodically with the full desktop state.
    The Brain uses this as context for intent resolution.
    """
    topic = "context.snapshot_updated"
    active_window_title: str = ""
    active_app_name: str = ""
    active_pid: int = 0
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    battery_pct: float = -1.0
    battery_plugged: bool = False
    clipboard_preview: str = ""   # First 100 chars
    recent_process_starts: list = field(default_factory=list)


# ─── Observer errors ──────────────────────────────────────────────────────────

@dataclass
class ObserverErrorEvent(Event):
    """Emitted when an observer self-suspends after too many errors."""
    topic = "context.observer_error"
    observer_name: str = ""
    message: str = ""
