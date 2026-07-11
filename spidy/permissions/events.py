"""
Permission Events — EventBus events for the permission system
=============================================================
Topic namespace: ``permission.*``
"""

from __future__ import annotations

from dataclasses import dataclass

from spidy.core.event_bus import Event


@dataclass
class PermissionGrantedEvent(Event):
    """Emitted when an action is permitted."""
    topic = "permission.granted"
    action: str = ""
    tier: str = ""
    session_id: str = ""


@dataclass
class PermissionDeniedEvent(Event):
    """Emitted when an action is blocked by the permission system."""
    topic = "permission.denied"
    action: str = ""
    tier: str = ""
    reason: str = ""
    session_id: str = ""


@dataclass
class PermissionRequestedEvent(Event):
    """
    Emitted when a T1/T2 action needs user confirmation.

    The UI subscribes to this to show a confirmation dialog.
    The response is sent back via a PermissionResponseEvent.
    """
    topic = "permission.requested"
    action: str = ""
    tier: str = ""
    description: str = ""
    session_id: str = ""
