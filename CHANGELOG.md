# Changelog

All notable changes to **Spidy** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Planned — Milestone 3: Reasoning & Conversation
- LLM client abstraction (Ollama local / cloud fallback)
- Intent resolution pipeline
- Conversation context manager with rolling window
- Response routing to skills

---

## [0.2.0] — 2026-07-11 — Milestone 2: Context Observer Layer

### Added

**`spidy.perception.context` package — 11 new files**

- `base.py` — `BaseObserver` ABC:
  - Abstract `poll()` and `name` interface
  - Async polling loop (`asyncio.Task`) with configurable `poll_interval`
  - Error isolation: exceptions in `poll()` never crash the observer
  - Exponential backoff on consecutive errors (base 1.5, max 30 s)
  - Self-suspension after `_MAX_CONSECUTIVE_ERRORS` (10) failures
  - Publishes `context.observer_error` on self-suspension
  - Optional `on_start()` / `on_stop()` lifecycle hooks

- `events.py` — 10 typed event dataclasses:
  - `WindowChangedEvent` — foreground window title changed
  - `AppChangedEvent` — foreground application (exe) changed
  - `ClipboardChangedEvent` — clipboard text content changed
  - `ProcessStartedEvent` — new process detected
  - `ProcessEndedEvent` — tracked process exited
  - `ResourceUpdateEvent` — periodic CPU/RAM/battery/GPU/disk snapshot
  - `ResourceAlertEvent` — resource crossed a configured threshold
  - `DownloadAddedEvent` — new file in Downloads folder
  - `DownloadModifiedEvent` — file in Downloads modified (in-progress download)
  - `SnapshotUpdatedEvent` — full desktop state snapshot published on cadence

- `snapshot.py` — Immutable desktop state model:
  - `WindowInfo(frozen)` — title, app_name, exe_path, pid, hwnd
  - `ResourceInfo(frozen)` — CPU/RAM/GPU/battery/disk fields + `has_battery`, `is_low_battery`, `is_high_cpu`, `is_high_ram` properties
  - `ProcessInfo(frozen)` — lightweight pid + name + exe
  - `DesktopStateSnapshot(frozen)` — aggregate of all the above, with `summary()` and `to_dict()`

- `window.py` — `ActiveWindowObserver`:
  - Polls foreground window via `win32gui` + `psutil`
  - Publishes `WindowChangedEvent` on title change
  - Publishes `AppChangedEvent` on exe change (app switch)
  - Gracefully disables if `pywin32` not installed

- `clipboard.py` — `ClipboardObserver`:
  - Uses `GetClipboardSequenceNumber()` for O(1) change detection (no content read each poll)
  - Only reads `CF_UNICODETEXT` (no images/files)
  - Truncates captured text to 500 chars; sets `is_truncated` flag
  - Can be disabled via config for privacy

- `process.py` — `ProcessObserver`:
  - Delta-tracks running processes via `psutil.process_iter()`
  - Publishes `ProcessStartedEvent` / `ProcessEndedEvent`
  - Configurable noise filter (`svchost.exe`, `conhost.exe`, etc.)
  - Snapshots existing processes on start (avoids spurious started events)

- `resources.py` — `SystemResourceObserver`:
  - Collects CPU, RAM, battery, GPU (via torch.cuda if available), disk I/O delta
  - Always publishes `ResourceUpdateEvent` every poll
  - Conditionally publishes `ResourceAlertEvent` with hysteresis (5%) to prevent alert storms
  - Configurable thresholds: cpu=90%, ram=85%, battery=20%, gpu=95%

- `downloads.py` — `DownloadObserver`:
  - Uses `watchdog` (`ReadDirectoryChangesW` under the hood) for near-zero CPU overhead
  - Skips `.tmp`, `.part`, `.crdownload`, `.download`, `.partial`, `.opdownload`
  - Waits `SETTLE_SECONDS=2.0` before publishing (file handle settle time)
  - Handles moved files (`on_moved`) in addition to created files
  - Gracefully disables if `watchdog` not installed or Downloads folder missing

- `notifications.py` — `NotificationObserver` (stub):
  - Implements `BaseObserver` interface
  - No-op `poll()` — full WinRT/UIA implementation planned for Milestone 5+
  - Logs stub mode warning on start

- `observer_manager.py` — `ObserverManager`:
  - Single lifecycle entry point (`start()` / `stop()`)
  - Builds all enabled observers from `ContextConfig`
  - Runs a `_snapshot_loop` task that aggregates observer data into `DesktopStateSnapshot` on a configurable cadence
  - Publishes `SnapshotUpdatedEvent` each cycle
  - `snapshot` property for synchronous reads by the Brain (Milestone 3+)

- `__init__.py` — Public re-exports: `ObserverManager`, `DesktopStateSnapshot`, `WindowInfo`, `ResourceInfo`

**Tests**

- `tests/unit/test_context_observers.py` — 43 unit tests:
  - `TestBaseObserver` (5 tests)
  - `TestDesktopStateSnapshot` (8 tests)
  - `TestActiveWindowObserver` (5 tests)
  - `TestClipboardObserver` (5 tests)
  - `TestProcessObserver` (4 tests)
  - `TestSystemResourceObserver` (7 tests)
  - `TestDownloadObserver` (3 tests)
  - `TestNotificationObserver` (2 tests)

### Fixed

- `test_self_suspension_after_max_errors` — test was not giving the observer enough time to reach 10 consecutive errors due to exponential backoff. Fixed by using `patch.object(base_module, "_MAX_CONSECUTIVE_ERRORS", 5)` within the test, allowing the observer to self-suspend well within the existing 0.5 s window.

---

## [0.1.0] — 2026-07-10 — Milestone 1: Voice Pipeline

### Added

- `spidy.perception.voice` — Complete voice pipeline
  - `AudioCaptureEngine` — pyaudio-based capture with mode-routing callbacks
  - `VoiceEngine` — async state machine (sleeping / detecting / waking / listening / processing / speaking)
  - `WakeWordModel` (ABC) — pluggable wake-word backend
  - `SpeechRecognizer` (ABC) — pluggable STT backend
  - `TTSEngine` (ABC) — pluggable TTS backend
  - `AudioBuffer`, `TranscriptResult` — core data types

- `spidy.skills.registry` — `SkillRegistry`:
  - Register/unregister `BaseSkill` subclasses
  - Action routing with capability introspection
  - Thread-safe; duplicate registration prevented
  - Warning on conflicting action names

- Tests: 43 unit tests across `test_voice_engine.py`, `test_voice_pipeline.py`, `test_skill_registry.py`

---

## [0.0.1] — 2026-07-09 — Milestone 0: Foundation

### Added

- `spidy.core.event_bus` — `EventBus` with async pub/sub and threadsafe publishing
- `spidy.config` — YAML configuration with Pydantic models (`SpidyConfig`, `ContextConfig`, etc.)
- `spidy.logging` — Structured `loguru` logger factory
- `spidy.core.app` — `SpidyCore` bootstrap with `initialise()` / `start()` / `stop()` lifecycle
- `spidy.core.device` — Device manager stub
- Project scaffolding: `pyproject.toml`, `pytest.ini`, `Makefile`, `requirements*.txt`, `README.md`, CI workflow
- Tests: 59 unit tests across `test_config.py`, `test_core.py`, `test_event_bus.py`, `test_logger.py`, `test_device_manager.py`
