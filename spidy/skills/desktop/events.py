"""
Desktop Agent Events — EventBus events for the Desktop & File Agent skills
==========================================================================
Topic namespace:
  ``desktop.file.*``    — FileSkill events
  ``desktop.app.*``     — AppSkill events
  ``desktop.system.*``  — SystemControlSkill events

All desktop skills communicate exclusively through these typed events.
The Brain and UI subscribe to these; skills publish them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spidy.core.event_bus import Event


# ─── FileSkill Events ─────────────────────────────────────────────────────────


@dataclass
class FileSearchResultEvent(Event):
    """
    Emitted after a file or folder search completes.

    Published by: FileSkill (search_files, search_folders)
    """
    topic = "desktop.file.search_result"
    query: str = ""
    results: list[str] = field(default_factory=list)
    total_found: int = 0
    search_root: str = ""
    session_id: str = ""


@dataclass
class FileOpenedEvent(Event):
    """
    Emitted when a file is opened with its default application.

    Published by: FileSkill (open_file)
    """
    topic = "desktop.file.opened"
    path: str = ""
    session_id: str = ""


@dataclass
class FolderOpenedEvent(Event):
    """
    Emitted when a folder is opened in Windows Explorer.

    Published by: FileSkill (open_folder, reveal_in_explorer)
    """
    topic = "desktop.file.folder_opened"
    path: str = ""
    reveal_mode: bool = False   # True = reveal in Explorer (select file)
    session_id: str = ""


@dataclass
class RecentFilesResultEvent(Event):
    """
    Emitted after the recent files list is retrieved.

    Published by: FileSkill (list_recent_files)
    """
    topic = "desktop.file.recent_result"
    files: list[str] = field(default_factory=list)
    session_id: str = ""


# ─── AppSkill Events ──────────────────────────────────────────────────────────


@dataclass
class RunningAppsResultEvent(Event):
    """
    Emitted after the list of running apps is retrieved.

    Published by: AppSkill (detect_running_apps)
    """
    topic = "desktop.app.running_result"
    apps: list[dict] = field(default_factory=list)  # [{name, pid, status}]
    session_id: str = ""


@dataclass
class AppLaunchedEvent(Event):
    """
    Emitted when an application is successfully launched.

    Published by: AppSkill (launch_app)
    """
    topic = "desktop.app.launched"
    app_name: str = ""
    pid: int = 0
    session_id: str = ""


@dataclass
class AppForegroundedEvent(Event):
    """
    Emitted when an application is brought to the foreground.

    Published by: AppSkill (bring_app_to_foreground)
    """
    topic = "desktop.app.foregrounded"
    app_name: str = ""
    hwnd: int = 0
    session_id: str = ""


@dataclass
class AppClosedEvent(Event):
    """
    Emitted when an application process is terminated.

    Published by: AppSkill (close_app)
    """
    topic = "desktop.app.closed"
    app_name: str = ""
    pid: int = 0
    forced: bool = False
    session_id: str = ""


# ─── SystemControlSkill Events ────────────────────────────────────────────────


@dataclass
class VolumeChangedEvent(Event):
    """
    Emitted when system volume is changed.

    Published by: SystemControlSkill (set_volume)
    """
    topic = "desktop.system.volume_changed"
    level: int = 0          # 0–100
    muted: bool = False
    session_id: str = ""


@dataclass
class BrightnessChangedEvent(Event):
    """
    Emitted when screen brightness is changed.

    Published by: SystemControlSkill (set_brightness)
    """
    topic = "desktop.system.brightness_changed"
    level: int = 0          # 0–100
    session_id: str = ""


@dataclass
class WorkstationLockedEvent(Event):
    """
    Emitted when the workstation is locked.

    Published by: SystemControlSkill (lock_workstation)
    """
    topic = "desktop.system.workstation_locked"
    session_id: str = ""


@dataclass
class SystemSleepInitiatedEvent(Event):
    """
    Emitted just before the system enters sleep mode.

    Published by: SystemControlSkill (sleep_system)
    """
    topic = "desktop.system.sleep_initiated"
    session_id: str = ""


@dataclass
class SystemShutdownInitiatedEvent(Event):
    """
    Emitted when a shutdown command is dispatched.

    Published by: SystemControlSkill (shutdown_system)
    """
    topic = "desktop.system.shutdown_initiated"
    delay_seconds: int = 0
    session_id: str = ""


@dataclass
class SystemRestartInitiatedEvent(Event):
    """
    Emitted when a restart command is dispatched.

    Published by: SystemControlSkill (restart_system)
    """
    topic = "desktop.system.restart_initiated"
    delay_seconds: int = 0
    session_id: str = ""


@dataclass
class RecycleBinEmptiedEvent(Event):
    """
    Emitted when the recycle bin is emptied.

    Published by: SystemControlSkill (empty_recycle_bin)
    """
    topic = "desktop.system.recycle_bin_emptied"
    session_id: str = ""
