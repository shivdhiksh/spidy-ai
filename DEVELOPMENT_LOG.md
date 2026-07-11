# SPIDY — Development Log

> Engineering decisions, milestone status, and session notes.

---

## Milestone 0 — Foundation ✅

**Status:** Complete  
**Tests:** All passing (unit coverage via `test_config.py`, `test_core.py`, `test_event_bus.py`, `test_logger.py`, `test_device_manager.py`)

### What was built

- **`spidy.core.event_bus`** — Async pub/sub `EventBus` with:
  - Typed `Event` dataclass hierarchy
  - `publish()` — awaits all subscribers sequentially; isolates exceptions
  - `publish_threadsafe()` — `call_soon_threadsafe` bridge for non-async threads
  - Built-in system events: `SpidyStartedEvent`, `SpidyShuttingDownEvent`, `ErrorEvent`
  - Observable stats (`published`, `errors`, `topics`)

- **`spidy.config`** — YAML-based configuration with Pydantic validation
  - `SpidyConfig` root model with nested sections
  - `ContextConfig` sub-model (used by Milestone 2)
  - Sensible defaults; user file is optional

- **`spidy.logging`** — `loguru`-based structured logger
  - Per-module `get_logger(__name__)` factory
  - JSON-friendly format for log analysis

- **`spidy.core.app`** (`SpidyCore`) — Application bootstrap with async lifecycle
  - `initialise()` / `start()` / `stop()` lifecycle phases
  - Wires EventBus, Config, all modules

- **`spidy.core.device`** — Device manager stub

---

## Milestone 1 — Voice Pipeline ✅

**Status:** Complete  
**Tests:** All passing (`test_voice_engine.py` 13 tests, `test_voice_pipeline.py` 17 tests, `test_skill_registry.py` 13 tests)

### What was built

- **`spidy.perception.voice`** — Wake word → STT → intent → TTS pipeline
  - `AudioCaptureEngine` — pyaudio capture with callback dispatch
  - `VoiceEngine` — state machine (sleeping → detecting → waking → listening → processing → speaking)
  - `WakeWordModel` / `SpeechRecognizer` / `TTSEngine` — ABCs for swappable backends
  - `AudioBuffer`, `TranscriptResult`, `SkillContext`, `SkillResult`

- **`spidy.skills.registry`** — `SkillRegistry` with action routing, capability queries, thread-safe register/unregister

---

## Milestone 2 — Context Observer Layer ✅

**Status:** Complete and fully verified  
**Tests:** 43 tests, all passing (as of 2026-07-11)

### What was built

The Context Observer Layer gives Spidy ambient awareness of the Windows desktop.
It runs as a background system, publishing typed events to the EventBus.

#### Architecture

```
ObserverManager
├── ActiveWindowObserver    → context.window_changed, context.app_changed
├── ClipboardObserver       → context.clipboard_changed
├── ProcessObserver         → context.process_started, context.process_ended
├── SystemResourceObserver  → context.resource_update, context.resource_alert
├── DownloadObserver        → context.download_added, context.download_modified
└── NotificationObserver    → (stub, Milestone 5+)
```

#### Files added

| File | Purpose |
|------|---------|
| `spidy/perception/context/__init__.py` | Public API (`ObserverManager`, `DesktopStateSnapshot`, etc.) |
| `spidy/perception/context/base.py` | `BaseObserver` ABC with poll loop, error isolation, exponential backoff |
| `spidy/perception/context/events.py` | All context event dataclasses (10 event types) |
| `spidy/perception/context/snapshot.py` | `DesktopStateSnapshot`, `WindowInfo`, `ResourceInfo`, `ProcessInfo` |
| `spidy/perception/context/observer_manager.py` | `ObserverManager` — lifecycle orchestrator and snapshot aggregator |
| `spidy/perception/context/window.py` | `ActiveWindowObserver` — win32gui foreground window tracking |
| `spidy/perception/context/clipboard.py` | `ClipboardObserver` — sequence-number-based change detection |
| `spidy/perception/context/process.py` | `ProcessObserver` — psutil delta tracking |
| `spidy/perception/context/resources.py` | `SystemResourceObserver` — CPU/RAM/battery/GPU/disk with hysteresis alerts |
| `spidy/perception/context/downloads.py` | `DownloadObserver` — watchdog filesystem events |
| `spidy/perception/context/notifications.py` | `NotificationObserver` — stub (future WinRT implementation) |
| `tests/unit/test_context_observers.py` | 43 unit tests (no real OS calls, all mocked) |

#### Key design decisions

1. **EventBus-only output** — Observers never return data. All output is published events.
2. **Async tasks** — Each observer runs as an `asyncio.Task`. No threads needed (except where win32 APIs block — those use `asyncio.to_thread()`).
3. **Graceful degradation** — If `pywin32` or `watchdog` is not installed, observers log a warning and silently skip polling. Spidy still works.
4. **Exponential backoff** — Observer poll errors increase the sleep interval up to `_MAX_BACKOFF_SECONDS=30`. After `_MAX_CONSECUTIVE_ERRORS=10` failures, the observer self-suspends and publishes `context.observer_error`.
5. **Hysteresis on alerts** — Resource alerts fire once and are suppressed until the value recovers past `threshold ± hysteresis` (5%). Prevents alert storms.
6. **Clipboard privacy** — Only CF_UNICODETEXT; never images/files. Never logged above DEBUG. Truncated to 500 chars. Can be disabled via config.

#### Bug fixed (2026-07-11 session)

- `test_self_suspension_after_max_errors` was failing because the test waited 0.5 s for the observer to accumulate 10 errors, but with exponential backoff starting at `poll_interval=0.01 s` the cumulative sleep time to 10 errors is ~1.14 s. Fixed by patching `_MAX_CONSECUTIVE_ERRORS` to 5 in the test (using `patch.object`), allowing the observer to self-suspend within the existing 0.5 s window.

---

## Next: Milestone 3 — Reasoning & Conversation

> **Awaiting user approval before starting.**

Planned scope (subject to approval):
- LLM client abstraction (local Ollama + cloud fallback)
- Intent resolution pipeline
- Conversation context manager
- Response routing

---

*Last updated: 2026-07-11*
