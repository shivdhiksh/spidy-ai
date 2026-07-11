"""
Tests for Milestone 6 — Desktop Events
========================================
Unit tests for all desktop event dataclasses.
Verifies correct topic routing and field defaults.
"""

from __future__ import annotations

import pytest

from spidy.skills.desktop.events import (
    AppClosedEvent,
    AppForegroundedEvent,
    AppLaunchedEvent,
    BrightnessChangedEvent,
    FileOpenedEvent,
    FileSearchResultEvent,
    FolderOpenedEvent,
    RecentFilesResultEvent,
    RecycleBinEmptiedEvent,
    RunningAppsResultEvent,
    SystemRestartInitiatedEvent,
    SystemShutdownInitiatedEvent,
    SystemSleepInitiatedEvent,
    VolumeChangedEvent,
    WorkstationLockedEvent,
)
from spidy.core.event_bus import Event


# ──────────────────────────────────────────────────────────────────────────────
# 1. Event topic routing
# ──────────────────────────────────────────────────────────────────────────────


class TestDesktopEventTopics:
    """Verify all events have correct topic strings and are Event subclasses."""

    EVENT_TOPICS = [
        (FileSearchResultEvent, "desktop.file.search_result"),
        (FileOpenedEvent, "desktop.file.opened"),
        (FolderOpenedEvent, "desktop.file.folder_opened"),
        (RecentFilesResultEvent, "desktop.file.recent_result"),
        (RunningAppsResultEvent, "desktop.app.running_result"),
        (AppLaunchedEvent, "desktop.app.launched"),
        (AppForegroundedEvent, "desktop.app.foregrounded"),
        (AppClosedEvent, "desktop.app.closed"),
        (VolumeChangedEvent, "desktop.system.volume_changed"),
        (BrightnessChangedEvent, "desktop.system.brightness_changed"),
        (WorkstationLockedEvent, "desktop.system.workstation_locked"),
        (SystemSleepInitiatedEvent, "desktop.system.sleep_initiated"),
        (SystemShutdownInitiatedEvent, "desktop.system.shutdown_initiated"),
        (SystemRestartInitiatedEvent, "desktop.system.restart_initiated"),
        (RecycleBinEmptiedEvent, "desktop.system.recycle_bin_emptied"),
    ]

    @pytest.mark.parametrize("event_cls,expected_topic", EVENT_TOPICS)
    def test_event_topic(self, event_cls, expected_topic):
        assert event_cls.topic == expected_topic

    @pytest.mark.parametrize("event_cls,_", EVENT_TOPICS)
    def test_is_event_subclass(self, event_cls, _):
        assert issubclass(event_cls, Event)

    @pytest.mark.parametrize("event_cls,expected_topic", EVENT_TOPICS)
    def test_instance_has_correct_topic(self, event_cls, expected_topic):
        # All events should be instantiable with no args (all fields have defaults)
        instance = event_cls()
        assert instance.topic == expected_topic


# ──────────────────────────────────────────────────────────────────────────────
# 2. Event field defaults
# ──────────────────────────────────────────────────────────────────────────────


class TestDesktopEventDefaults:
    def test_file_search_result_defaults(self):
        e = FileSearchResultEvent()
        assert e.query == ""
        assert e.results == []
        assert e.total_found == 0
        assert e.session_id == ""

    def test_file_search_result_with_data(self):
        e = FileSearchResultEvent(
            query="resume.pdf",
            results=["/home/user/resume.pdf"],
            total_found=1,
            search_root="/home/user",
            session_id="sess1",
        )
        assert e.query == "resume.pdf"
        assert len(e.results) == 1
        assert e.total_found == 1
        assert e.session_id == "sess1"

    def test_file_opened_defaults(self):
        e = FileOpenedEvent()
        assert e.path == ""
        assert e.session_id == ""

    def test_folder_opened_defaults(self):
        e = FolderOpenedEvent()
        assert e.path == ""
        assert e.reveal_mode is False

    def test_folder_opened_reveal_mode(self):
        e = FolderOpenedEvent(path="/path/to/file.txt", reveal_mode=True)
        assert e.reveal_mode is True

    def test_recent_files_result_defaults(self):
        e = RecentFilesResultEvent()
        assert e.files == []

    def test_running_apps_result_defaults(self):
        e = RunningAppsResultEvent()
        assert e.apps == []

    def test_app_launched_defaults(self):
        e = AppLaunchedEvent()
        assert e.app_name == ""
        assert e.pid == 0

    def test_app_launched_with_data(self):
        e = AppLaunchedEvent(app_name="notepad.exe", pid=1234, session_id="s1")
        assert e.app_name == "notepad.exe"
        assert e.pid == 1234

    def test_app_foregrounded_defaults(self):
        e = AppForegroundedEvent()
        assert e.hwnd == 0

    def test_app_closed_defaults(self):
        e = AppClosedEvent()
        assert e.forced is False

    def test_app_closed_forced(self):
        e = AppClosedEvent(app_name="chrome.exe", pid=9999, forced=True)
        assert e.forced is True

    def test_volume_changed_defaults(self):
        e = VolumeChangedEvent()
        assert e.level == 0
        assert e.muted is False

    def test_volume_changed_muted(self):
        e = VolumeChangedEvent(level=50, muted=True)
        assert e.muted is True

    def test_brightness_changed_defaults(self):
        e = BrightnessChangedEvent()
        assert e.level == 0

    def test_workstation_locked_defaults(self):
        e = WorkstationLockedEvent()
        assert e.session_id == ""

    def test_sleep_initiated_defaults(self):
        e = SystemSleepInitiatedEvent()
        assert e.session_id == ""

    def test_shutdown_initiated_defaults(self):
        e = SystemShutdownInitiatedEvent()
        assert e.delay_seconds == 0

    def test_shutdown_initiated_with_delay(self):
        e = SystemShutdownInitiatedEvent(delay_seconds=60, session_id="sess")
        assert e.delay_seconds == 60

    def test_restart_initiated_defaults(self):
        e = SystemRestartInitiatedEvent()
        assert e.delay_seconds == 0

    def test_recycle_bin_emptied_defaults(self):
        e = RecycleBinEmptiedEvent()
        assert e.session_id == ""


# ──────────────────────────────────────────────────────────────────────────────
# 3. EventBus roundtrip
# ──────────────────────────────────────────────────────────────────────────────


class TestDesktopEventsEventBusRoundtrip:
    @pytest.mark.asyncio
    async def test_file_search_event_roundtrip(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("desktop.file.search_result", handler)
        event = FileSearchResultEvent(query="test", results=["a.txt"], total_found=1)
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].query == "test"

    @pytest.mark.asyncio
    async def test_app_launched_event_roundtrip(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("desktop.app.launched", handler)
        event = AppLaunchedEvent(app_name="notepad.exe", pid=1234)
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].pid == 1234

    @pytest.mark.asyncio
    async def test_volume_changed_event_roundtrip(self):
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("desktop.system.volume_changed", handler)
        event = VolumeChangedEvent(level=75, muted=False)
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].level == 75

    @pytest.mark.asyncio
    async def test_all_events_publishable(self):
        """Verify all desktop events can be published without error."""
        from spidy.core.event_bus import EventBus
        bus = EventBus()
        events_to_publish = [
            FileSearchResultEvent(), FileOpenedEvent(), FolderOpenedEvent(),
            RecentFilesResultEvent(), RunningAppsResultEvent(), AppLaunchedEvent(),
            AppForegroundedEvent(), AppClosedEvent(), VolumeChangedEvent(),
            BrightnessChangedEvent(), WorkstationLockedEvent(),
            SystemSleepInitiatedEvent(), SystemShutdownInitiatedEvent(),
            SystemRestartInitiatedEvent(), RecycleBinEmptiedEvent(),
        ]
        for event in events_to_publish:
            count = await bus.publish(event)
            assert count == 0  # No subscribers, but no errors
