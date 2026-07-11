"""
DesktopStateSnapshot — Aggregate Desktop Context
=================================================
A single immutable object that captures the complete state of the
Windows desktop at a point in time.

The Brain reads this snapshot when resolving user intent:
  "What is the user doing right now?"
  "What app is in focus?"
  "Is the battery about to die?"

Usage
-----
    snapshot = DesktopStateSnapshot.current()  # Build from live data
    brain.update_context(snapshot)
    print(snapshot.summary())

Design
------
- Immutable (frozen=True) — thread-safe by definition
- All fields have sensible defaults — never raises on construction
- Built by ObserverManager.take_snapshot() which calls each observer
- Published as SnapshotUpdatedEvent every N seconds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ProcessInfo:
    """Lightweight representation of a running process."""
    name: str
    pid: int
    exe: str = ""

    def __str__(self) -> str:
        return f"{self.name}(pid={self.pid})"


@dataclass(frozen=True)
class WindowInfo:
    """State of the foreground window."""
    title: str = ""
    app_name: str = ""          # e.g. "Code.exe"
    exe_path: str = ""          # Full path to executable
    pid: int = 0
    hwnd: int = 0

    @property
    def is_valid(self) -> bool:
        return bool(self.title or self.app_name)

    def __str__(self) -> str:
        if self.title:
            return f"'{self.title}' ({self.app_name})"
        return self.app_name or "Unknown"


@dataclass(frozen=True)
class ResourceInfo:
    """Current system resource utilisation."""
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_pct: float = 0.0
    gpu_vram_used_gb: float = 0.0
    battery_pct: float = -1.0   # -1 if no battery
    battery_plugged: bool = False
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0

    @property
    def has_battery(self) -> bool:
        return self.battery_pct >= 0

    @property
    def is_low_battery(self) -> bool:
        return self.has_battery and not self.battery_plugged and self.battery_pct < 20.0

    @property
    def is_high_cpu(self) -> bool:
        return self.cpu_pct > 85.0

    @property
    def is_high_ram(self) -> bool:
        return self.ram_pct > 85.0

    def __str__(self) -> str:
        parts = [f"CPU={self.cpu_pct:.0f}%", f"RAM={self.ram_pct:.0f}%"]
        if self.has_battery:
            plug = "⚡" if self.battery_plugged else "🔋"
            parts.append(f"Bat={self.battery_pct:.0f}%{plug}")
        return " | ".join(parts)


@dataclass(frozen=True)
class DesktopStateSnapshot:
    """
    Complete point-in-time desktop state.

    Built by ObserverManager and published as SnapshotUpdatedEvent.
    """
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Window / Application
    window: WindowInfo = field(default_factory=WindowInfo)

    # Resources
    resources: ResourceInfo = field(default_factory=ResourceInfo)

    # Clipboard
    clipboard_text: str = ""          # First 500 chars of clipboard
    clipboard_length: int = 0

    # Recent process activity (last N started)
    recent_starts: tuple[ProcessInfo, ...] = field(default_factory=tuple)

    # Raw metadata dict for future extensibility
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self, max_len: int = 120) -> str:
        """
        One-line human-readable summary. Used by the Brain as context prefix.

        Example:
            "VS Code — 'main.py' | CPU=12% | RAM=68% | Bat=82%🔋"
        """
        parts: list[str] = []

        if self.window.is_valid:
            parts.append(str(self.window))

        parts.append(str(self.resources))

        if self.clipboard_length > 0:
            preview = self.clipboard_text[:40]
            if self.clipboard_length > 40:
                preview += "..."
            parts.append(f"Clipboard: '{preview}'")

        result = " | ".join(parts)
        return result[:max_len] if len(result) > max_len else result

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "window": {
                "title": self.window.title,
                "app_name": self.window.app_name,
                "exe_path": self.window.exe_path,
                "pid": self.window.pid,
            },
            "resources": {
                "cpu_pct": self.resources.cpu_pct,
                "ram_pct": self.resources.ram_pct,
                "ram_used_gb": self.resources.ram_used_gb,
                "ram_total_gb": self.resources.ram_total_gb,
                "gpu_pct": self.resources.gpu_pct,
                "battery_pct": self.resources.battery_pct,
                "battery_plugged": self.resources.battery_plugged,
            },
            "clipboard_length": self.clipboard_length,
            "recent_starts": [
                {"name": p.name, "pid": p.pid} for p in self.recent_starts
            ],
        }
