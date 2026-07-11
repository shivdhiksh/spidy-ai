"""
NotificationObserver — Windows Notification Monitor
=====================================================
Monitors Windows toast notifications and Action Center entries.

Current status: STUB (Milestone 2)
-----------------------------------
True Windows notification interception requires either:
  A) Windows.UI.Notifications COM API (requires Python 3.12+ winrt package)
  B) Accessibility API (UI Automation) to read the Action Center
  C) App-specific log parsing (Discord, Slack, etc.)

None of these are trivial to implement reliably without additional
system permissions. This stub publishes placeholder events and
provides the correct interface for a future implementation.

Planned approach (Milestone 5+)
---------------------------------
  1. Use the `winrt` package (Windows.UI.Notifications.Management)
     to subscribe to notification feed changes
  2. Or: Hook the UIA tree to watch for new toasts
  3. Fall back to: Watch specific app log files for notable messages

For now this observer:
- Runs the polling loop so it participates in the observer lifecycle
- Logs that it is in stub mode
- Can be subclassed or replaced with a real implementation
"""

from __future__ import annotations

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver

log = get_logger(__name__)


class NotificationObserver(BaseObserver):
    """
    Stub notification observer.

    Future: subscribe to Windows.UI.Notifications.Management.UserNotificationListener
    """

    name = "notifications"

    def __init__(self, bus: EventBus, poll_interval: float = 10.0) -> None:
        super().__init__(bus, poll_interval)
        self._stub_warned = False

    async def on_start(self) -> None:
        log.info(
            "NotificationObserver: running in stub mode. "
            "Full Windows notification support is planned for a future milestone."
        )

    async def poll(self) -> None:
        """No-op stub. Real implementation goes here."""
