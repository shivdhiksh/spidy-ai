"""
Unit tests — Context Observer Layer (Milestone 2)
==================================================
Tests all individual observer components with mocked I/O.

No real Windows API calls, no real psutil calls, no real file system.
Everything is mocked or uses lightweight fakes.

Run with:
    pytest tests/unit/test_context_observers.py -v
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import time

import pytest

from spidy.core.event_bus import EventBus
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.clipboard import ClipboardObserver
from spidy.perception.context.downloads import DownloadObserver
from spidy.perception.context.events import (
    AppChangedEvent,
    ClipboardChangedEvent,
    DownloadAddedEvent,
    ObserverErrorEvent,
    ProcessEndedEvent,
    ProcessStartedEvent,
    ResourceAlertEvent,
    ResourceUpdateEvent,
    SnapshotUpdatedEvent,
    WindowChangedEvent,
)
from spidy.perception.context.notifications import NotificationObserver
from spidy.perception.context.process import ProcessObserver, _ProcessRecord
from spidy.perception.context.resources import SystemResourceObserver
from spidy.perception.context.snapshot import (
    DesktopStateSnapshot,
    ProcessInfo,
    ResourceInfo,
    WindowInfo,
)
from spidy.perception.context.window import ActiveWindowObserver


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_bus() -> EventBus:
    bus = EventBus()
    bus.set_loop(asyncio.get_running_loop())
    return bus


# ─── BaseObserver tests ───────────────────────────────────────────────────────


class TestBaseObserver:
    """Tests for the BaseObserver ABC contract."""

    @pytest.mark.asyncio
    async def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseObserver(bus=MagicMock())

    @pytest.mark.asyncio
    async def test_concrete_observer_starts_and_stops(self):
        """A minimal concrete observer can start and stop cleanly."""
        bus = make_bus()
        poll_calls: list[int] = []

        class _Counter(BaseObserver):
            name = "counter"

            async def poll(self) -> None:
                poll_calls.append(1)

        obs = _Counter(bus, poll_interval=0.05)
        await obs.start()
        assert obs.is_running
        await asyncio.sleep(0.2)
        await obs.stop()
        assert not obs.is_running
        assert len(poll_calls) >= 2

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        """Calling start() twice should not create two tasks."""
        bus = make_bus()

        class _Noop(BaseObserver):
            name = "noop"
            async def poll(self): pass

        obs = _Noop(bus, poll_interval=0.1)
        await obs.start()
        task1 = obs._task
        await obs.start()  # second call — should be ignored
        assert obs._task is task1
        await obs.stop()

    @pytest.mark.asyncio
    async def test_poll_errors_are_isolated(self):
        """An exception in poll() should not crash the observer loop."""
        bus = make_bus()
        count = {"calls": 0}

        class _Raiser(BaseObserver):
            name = "raiser"
            async def poll(self) -> None:
                count["calls"] += 1
                raise ValueError("expected test error")

        obs = _Raiser(bus, poll_interval=0.05)
        await obs.start()
        await asyncio.sleep(0.25)  # 5 polls at 50ms
        await obs.stop()
        # Should have polled multiple times despite errors
        assert count["calls"] >= 3

    @pytest.mark.asyncio
    async def test_self_suspension_after_max_errors(self):
        """After N consecutive errors, observer self-suspends.

        We patch _MAX_CONSECUTIVE_ERRORS to 5 so the observer reaches
        suspension within the 0.5 s window (exponential backoff from
        poll_interval=0.01 would take ~1.1 s to accumulate 10 errors).
        """
        from spidy.perception.context import base as base_module

        bus = make_bus()
        suspended_events: list[ObserverErrorEvent] = []
        bus.subscribe("context.observer_error", lambda e: suspended_events.append(e))

        class _AlwaysFail(BaseObserver):
            name = "always_fail"
            async def poll(self) -> None:
                raise RuntimeError("always fails")

        with patch.object(base_module, "_MAX_CONSECUTIVE_ERRORS", 5):
            obs = _AlwaysFail(bus, poll_interval=0.01)
            await obs.start()
            await asyncio.sleep(0.5)  # give it time to fail 5 times + publish event
            # Observer should have stopped itself
            # (task done after max_consecutive_errors reached)
            await asyncio.sleep(0.1)
        assert obs._task is None or obs._task.done()


# ─── DesktopStateSnapshot tests ───────────────────────────────────────────────


class TestDesktopStateSnapshot:
    def test_default_snapshot_is_valid(self):
        snap = DesktopStateSnapshot()
        assert snap.window.title == ""
        assert snap.resources.cpu_pct == 0.0
        assert snap.clipboard_text == ""

    def test_summary_with_no_data(self):
        snap = DesktopStateSnapshot()
        summary = snap.summary()
        assert isinstance(summary, str)

    def test_summary_includes_window_title(self):
        snap = DesktopStateSnapshot(
            window=WindowInfo(title="VS Code — main.py", app_name="Code.exe")
        )
        assert "VS Code" in snap.summary()

    def test_summary_includes_cpu(self):
        snap = DesktopStateSnapshot(
            resources=ResourceInfo(cpu_pct=42.5, ram_pct=60.0)
        )
        assert "42" in snap.summary()

    def test_to_dict_is_serialisable(self):
        import json
        snap = DesktopStateSnapshot(
            window=WindowInfo(title="Test", app_name="test.exe"),
            resources=ResourceInfo(cpu_pct=10.0, ram_pct=50.0),
        )
        d = snap.to_dict()
        # Should be JSON serialisable
        json.dumps(d)

    def test_window_info_is_valid_when_populated(self):
        w = WindowInfo(title="Hello", app_name="app.exe")
        assert w.is_valid
        assert "Hello" in str(w)

    def test_window_info_not_valid_when_empty(self):
        assert not WindowInfo().is_valid

    def test_resource_info_battery_flags(self):
        r = ResourceInfo(battery_pct=15.0, battery_plugged=False)
        assert r.has_battery
        assert r.is_low_battery

    def test_resource_info_no_battery(self):
        r = ResourceInfo(battery_pct=-1.0)
        assert not r.has_battery
        assert not r.is_low_battery

    def test_resource_info_high_cpu(self):
        r = ResourceInfo(cpu_pct=91.0)
        assert r.is_high_cpu

    def test_resource_snapshot_str(self):
        r = ResourceInfo(cpu_pct=45.0, ram_pct=60.0, battery_pct=80.0)
        s = str(r)
        assert "45" in s
        assert "60" in s


# ─── ActiveWindowObserver tests ───────────────────────────────────────────────


class TestActiveWindowObserver:
    @pytest.mark.asyncio
    async def test_initial_state(self):
        bus = make_bus()
        obs = ActiveWindowObserver(bus)
        assert obs.current_window() == WindowInfo()

    @pytest.mark.asyncio
    async def test_poll_fires_window_changed_event(self):
        bus = make_bus()
        events: list[WindowChangedEvent] = []
        bus.subscribe("context.window_changed", lambda e: events.append(e))

        obs = ActiveWindowObserver(bus)
        obs._win32_available = True

        fake_window = WindowInfo(
            title="My Window",
            app_name="test.exe",
            pid=1234,
            hwnd=99,
        )

        with patch.object(
            ActiveWindowObserver, "_get_foreground_window",
            staticmethod(lambda: fake_window),
        ):
            await obs.poll()

        assert len(events) == 1
        assert events[0].title == "My Window"

    @pytest.mark.asyncio
    async def test_no_event_if_window_unchanged(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.window_changed", lambda e: events.append(e))

        obs = ActiveWindowObserver(bus)
        obs._win32_available = True
        obs._last_window = WindowInfo(title="Same", app_name="same.exe")

        with patch.object(
            ActiveWindowObserver, "_get_foreground_window",
            staticmethod(lambda: WindowInfo(title="Same", app_name="same.exe")),
        ):
            await obs.poll()

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_app_changed_event_on_exe_change(self):
        bus = make_bus()
        app_events: list[AppChangedEvent] = []
        bus.subscribe("context.app_changed", lambda e: app_events.append(e))

        obs = ActiveWindowObserver(bus)
        obs._win32_available = True
        obs._last_window = WindowInfo(title="Old title", app_name="old.exe")

        new_window = WindowInfo(title="New title", app_name="new.exe", pid=999)

        with patch.object(
            ActiveWindowObserver, "_get_foreground_window",
            staticmethod(lambda: new_window),
        ):
            await obs.poll()

        assert len(app_events) == 1
        assert app_events[0].app_name == "new.exe"

    @pytest.mark.asyncio
    async def test_no_app_event_on_title_only_change(self):
        bus = make_bus()
        app_events: list = []
        bus.subscribe("context.app_changed", lambda e: app_events.append(e))

        obs = ActiveWindowObserver(bus)
        obs._win32_available = True
        obs._last_window = WindowInfo(title="Old title", app_name="same.exe")

        with patch.object(
            ActiveWindowObserver, "_get_foreground_window",
            staticmethod(lambda: WindowInfo(title="New title", app_name="same.exe")),
        ):
            await obs.poll()

        assert len(app_events) == 0  # Same EXE, no app_changed

    @pytest.mark.asyncio
    async def test_win32_unavailable_skips_poll(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.window_changed", lambda e: events.append(e))

        obs = ActiveWindowObserver(bus)
        obs._win32_available = False  # pretend pywin32 not installed
        await obs.poll()

        assert len(events) == 0


# ─── ClipboardObserver tests ──────────────────────────────────────────────────


class TestClipboardObserver:
    @pytest.mark.asyncio
    async def test_disabled_skips_poll(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.clipboard_changed", lambda e: events.append(e))

        obs = ClipboardObserver(bus, enabled=False)
        await obs.poll()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_fires_event_on_text_change(self):
        bus = make_bus()
        events: list[ClipboardChangedEvent] = []
        bus.subscribe("context.clipboard_changed", lambda e: events.append(e))

        obs = ClipboardObserver(bus, enabled=True)
        obs._win32_available = True
        obs._last_sequence = 0

        with patch.object(
            ClipboardObserver, "_read_clipboard",
            staticmethod(lambda: (1, "Hello clipboard")),
        ):
            await obs.poll()

        assert len(events) == 1
        assert events[0].text == "Hello clipboard"
        assert events[0].full_length == 15

    @pytest.mark.asyncio
    async def test_no_event_if_sequence_unchanged(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.clipboard_changed", lambda e: events.append(e))

        obs = ClipboardObserver(bus, enabled=True)
        obs._win32_available = True
        obs._last_sequence = 5

        with patch.object(
            ClipboardObserver, "_read_clipboard",
            staticmethod(lambda: (5, "same text")),  # same sequence
        ):
            await obs.poll()

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_no_event_if_text_unchanged(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.clipboard_changed", lambda e: events.append(e))

        obs = ClipboardObserver(bus, enabled=True)
        obs._win32_available = True
        obs._last_sequence = 0
        obs._last_text = "same"

        with patch.object(
            ClipboardObserver, "_read_clipboard",
            staticmethod(lambda: (1, "same")),  # seq changed, text same
        ):
            await obs.poll()

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_long_text_is_truncated(self):
        bus = make_bus()
        events: list[ClipboardChangedEvent] = []
        bus.subscribe("context.clipboard_changed", lambda e: events.append(e))

        obs = ClipboardObserver(bus, enabled=True)
        obs._win32_available = True
        obs._last_sequence = 0

        long_text = "x" * 1000

        with patch.object(
            ClipboardObserver, "_read_clipboard",
            staticmethod(lambda: (1, long_text)),
        ):
            await obs.poll()

        assert events[0].full_length == 1000
        assert len(events[0].text) == 500
        assert events[0].is_truncated is True


# ─── ProcessObserver tests ────────────────────────────────────────────────────


class TestProcessObserver:
    def _make_record(self, name: str, pid: int) -> _ProcessRecord:
        return _ProcessRecord(name=name, pid=pid, exe=f"C:\\fake\\{name}")

    @pytest.mark.asyncio
    async def test_new_process_fires_started_event(self):
        bus = make_bus()
        events: list[ProcessStartedEvent] = []
        bus.subscribe("context.process_started", lambda e: events.append(e))

        obs = ProcessObserver(bus, filter_noise=False)
        obs._known = {100: self._make_record("old.exe", 100)}

        # Simulate: old.exe still there + new.exe appeared
        new_snapshot = {
            100: self._make_record("old.exe", 100),
            200: self._make_record("new.exe", 200),
        }
        with patch.object(obs, "_snapshot_all", return_value=new_snapshot):
            await obs.poll()

        assert len(events) == 1
        assert events[0].name == "new.exe"
        assert events[0].pid == 200

    @pytest.mark.asyncio
    async def test_ended_process_fires_ended_event(self):
        bus = make_bus()
        events: list[ProcessEndedEvent] = []
        bus.subscribe("context.process_ended", lambda e: events.append(e))

        obs = ProcessObserver(bus, filter_noise=False)
        obs._known = {
            100: self._make_record("app.exe", 100),
            200: self._make_record("dying.exe", 200),
        }

        new_snapshot = {100: self._make_record("app.exe", 100)}  # 200 gone
        with patch.object(obs, "_snapshot_all", return_value=new_snapshot):
            await obs.poll()

        assert len(events) == 1
        assert events[0].name == "dying.exe"
        assert events[0].pid == 200

    @pytest.mark.asyncio
    async def test_noise_filtering(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.process_started", lambda e: events.append(e))

        obs = ProcessObserver(bus, filter_noise=True)  # filtering ON
        obs._known = {}

        new_snapshot = {
            100: self._make_record("svchost.exe", 100),  # noise — filtered
            200: self._make_record("myapp.exe", 200),    # signal — kept
        }
        with patch.object(obs, "_snapshot_all", return_value=new_snapshot):
            await obs.poll()

        assert len(events) == 1
        assert events[0].name == "myapp.exe"

    @pytest.mark.asyncio
    async def test_no_events_when_nothing_changes(self):
        bus = make_bus()
        events: list = []
        bus.subscribe("context.process_started", lambda e: events.append(e))
        bus.subscribe("context.process_ended", lambda e: events.append(e))

        obs = ProcessObserver(bus)
        obs._known = {100: self._make_record("stable.exe", 100)}

        with patch.object(obs, "_snapshot_all",
                          return_value={100: self._make_record("stable.exe", 100)}):
            await obs.poll()

        assert len(events) == 0


# ─── SystemResourceObserver tests ─────────────────────────────────────────────


class TestSystemResourceObserver:
    @pytest.mark.asyncio
    async def test_publishes_resource_update(self):
        bus = make_bus()
        updates: list[ResourceUpdateEvent] = []
        bus.subscribe("context.resource_update", lambda e: updates.append(e))

        obs = SystemResourceObserver(bus, poll_interval=1.0)

        from spidy.perception.context.snapshot import ResourceInfo
        fake_info = ResourceInfo(cpu_pct=45.0, ram_pct=60.0, battery_pct=80.0)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()

        assert len(updates) == 1
        assert updates[0].cpu_pct == 45.0
        assert updates[0].ram_pct == 60.0

    @pytest.mark.asyncio
    async def test_cpu_alert_fires_when_exceeded(self):
        bus = make_bus()
        alerts: list[ResourceAlertEvent] = []
        bus.subscribe("context.resource_alert", lambda e: alerts.append(e))

        obs = SystemResourceObserver(bus, cpu_alert_threshold=80.0)
        fake_info = ResourceInfo(cpu_pct=91.0)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()

        assert any(a.resource == "cpu" for a in alerts)

    @pytest.mark.asyncio
    async def test_cpu_alert_does_not_repeat(self):
        """Alert should fire ONCE, not on every poll while above threshold."""
        bus = make_bus()
        alerts: list[ResourceAlertEvent] = []
        bus.subscribe("context.resource_alert", lambda e: alerts.append(e))

        obs = SystemResourceObserver(bus, cpu_alert_threshold=80.0)
        fake_info = ResourceInfo(cpu_pct=91.0)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()  # First poll — alert fires
            await obs.poll()  # Second poll — should NOT fire again

        assert sum(1 for a in alerts if a.resource == "cpu") == 1

    @pytest.mark.asyncio
    async def test_battery_alert_fires_when_low(self):
        bus = make_bus()
        alerts: list[ResourceAlertEvent] = []
        bus.subscribe("context.resource_alert", lambda e: alerts.append(e))

        obs = SystemResourceObserver(bus, battery_alert_threshold=20.0)
        fake_info = ResourceInfo(battery_pct=15.0, battery_plugged=False)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()

        assert any(a.resource == "battery" for a in alerts)

    @pytest.mark.asyncio
    async def test_battery_alert_does_not_fire_when_plugged(self):
        bus = make_bus()
        alerts: list[ResourceAlertEvent] = []
        bus.subscribe("context.resource_alert", lambda e: alerts.append(e))

        obs = SystemResourceObserver(bus, battery_alert_threshold=20.0)
        fake_info = ResourceInfo(battery_pct=10.0, battery_plugged=True)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()

        # No battery alert when plugged in
        assert not any(a.resource == "battery" for a in alerts)

    @pytest.mark.asyncio
    async def test_no_alert_when_below_threshold(self):
        bus = make_bus()
        alerts: list = []
        bus.subscribe("context.resource_alert", lambda e: alerts.append(e))

        obs = SystemResourceObserver(bus, cpu_alert_threshold=90.0)
        fake_info = ResourceInfo(cpu_pct=50.0)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_latest_is_updated_after_poll(self):
        bus = make_bus()
        obs = SystemResourceObserver(bus)
        fake_info = ResourceInfo(cpu_pct=77.0, ram_pct=55.0)

        with patch.object(obs, "_collect", return_value=fake_info):
            await obs.poll()

        assert obs.latest.cpu_pct == 77.0
        assert obs.latest.ram_pct == 55.0


# ─── DownloadObserver tests ───────────────────────────────────────────────────


class TestDownloadObserver:
    @pytest.mark.asyncio
    async def test_poll_is_noop_when_watchdog_unavailable(self):
        bus = make_bus()
        obs = DownloadObserver(bus)
        obs._watchdog_available = False
        # Should not raise
        await obs.poll()

    @pytest.mark.asyncio
    async def test_on_start_sets_watchdog_unavailable_if_path_missing(self):
        bus = make_bus()
        obs = DownloadObserver(bus, downloads_path=Path("/nonexistent/path"))
        obs._watchdog_available = True  # Pretend watchdog is installed

        # Should handle missing path gracefully
        with patch("spidy.perception.context.downloads.Path.exists", return_value=False):
            await obs.on_start()
        # Should not have started watchdog
        assert obs._watchdog_observer is None

    @pytest.mark.asyncio
    async def test_partial_extensions_filtered(self):
        from spidy.perception.context.downloads import _PARTIAL_EXTENSIONS
        assert ".crdownload" in _PARTIAL_EXTENSIONS
        assert ".part" in _PARTIAL_EXTENSIONS
        assert ".tmp" in _PARTIAL_EXTENSIONS
        assert ".pdf" not in _PARTIAL_EXTENSIONS


# ─── NotificationObserver tests ───────────────────────────────────────────────


class TestNotificationObserver:
    @pytest.mark.asyncio
    async def test_is_noop_stub(self):
        bus = make_bus()
        obs = NotificationObserver(bus)
        assert obs.name == "notifications"
        # poll does nothing — should not raise
        await obs.poll()

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        bus = make_bus()
        obs = NotificationObserver(bus, poll_interval=0.05)
        await obs.start()
        assert obs.is_running
        await asyncio.sleep(0.1)
        await obs.stop()
        assert not obs.is_running
