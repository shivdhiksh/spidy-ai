# spidy.perception.context — Milestone 2: Context Observer Layer
"""
Context Observer Layer — Public API
=====================================
Import from this package to access context observers without
knowing the internal module structure.

Usage
-----
    from spidy.perception.context import ObserverManager
    from spidy.perception.context.events import SnapshotUpdatedEvent
    from spidy.perception.context.snapshot import DesktopStateSnapshot
"""

from spidy.perception.context.observer_manager import ObserverManager
from spidy.perception.context.snapshot import DesktopStateSnapshot, WindowInfo, ResourceInfo

__all__ = [
    "ObserverManager",
    "DesktopStateSnapshot",
    "WindowInfo",
    "ResourceInfo",
]
