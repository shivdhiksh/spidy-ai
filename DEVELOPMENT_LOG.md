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

## Milestone 3 — Brain Core ✅

**Status:** Complete and fully verified  
**Tests:** 106 tests, all passing (2026-07-11)

### Context: Lifelong AI Companion Architecture

This milestone introduced a critical product reframe: Spidy is not just a desktop assistant — it is a **Lifelong AI Companion**. The Brain was designed from scratch to support this:

- Long-term memory (M6), knowledge graph + RAG (M8+), continuous learning (M9+)
- Multi-LLM provider support (Ollama today; OpenAI/Claude/Gemini planned M4+)
- All future companion systems slot in as interface implementations with no Brain restructuring

### Architecture

```
Brain (orchestrator)
├── IntentClassifier      utterance → Intent (heuristic M3, LLM-based M4+)
├── ConversationManager   rolling context window + session lifecycle
├── DecisionEngine        Intent → Decision (SKILL / LLM_DIRECT / CLARIFY / REJECT)
├── Planner               Decision → Plan (ordered PlanSteps)
├── ToolRouter            Plan → ToolResult(s) via SkillRegistry or LLM
│
├── MemoryInterface       [PLACEHOLDER] long-term recall, M6
├── KnowledgeInterface    [PLACEHOLDER] document RAG + knowledge graph, M8+
└── LearningInterface     [PLACEHOLDER] preferences + habits + feedback, M9+
```

### Files added

| File | Purpose |
|------|---------|
| `spidy/brain/types.py` | Shared data types: Intent, Entity, Decision, Plan, PlanStep, ToolResult, ConversationTurn |
| `spidy/brain/events.py` | EventBus events: brain.processing_started, brain.response_ready, session/tool events |
| `spidy/brain/interfaces.py` | **Three forward-declared ABCs** for Lifelong AI Companion integration |
| `spidy/brain/intent_classifier.py` | Heuristic IntentClassifier (15+ action categories, entity extraction) |
| `spidy/brain/conversation_manager.py` | ConversationManager (rolling window, session lifecycle, LLM message builder) |
| `spidy/brain/decision_engine.py` | DecisionEngine (intent → SKILL/LLM_DIRECT/CLARIFY/REJECT via registry lookup) |
| `spidy/brain/planner.py` | Planner (Decision → single-step Plan; multi-step ready for M4+) |
| `spidy/brain/tool_router.py` | ToolRouter (executes Plans via SkillRegistry or LLM; emits tool events) |
| `spidy/brain/brain.py` | Brain top-level orchestrator; subscribes to voice.user_spoke events |
| `spidy/brain/__init__.py` | Public API: `from spidy.brain import Brain` |
| `spidy/llm/client.py` | Multi-backend LLM abstraction: LLMMessage, LLMResponse, BaseLLMClient, LLMClientFactory |
| `spidy/llm/backends/__init__.py` | Package init |
| `spidy/llm/backends/ollama.py` | OllamaClient (stdlib urllib, zero extra deps, graceful fail if Ollama not running) |
| `tests/unit/test_brain.py` | 106 unit tests across all 9 test classes |

### Files modified

| File | Change |
|------|--------|
| `spidy/config/manager.py` | Added `BrainConfig` with companion feature flags; added to `SpidyConfig` |
| `spidy/core/app.py` | Wired Brain into `_run()` and `_stop()` lifecycle |

### Key design decisions

1. **Companion interfaces as permanent contracts** — `MemoryInterface`, `KnowledgeInterface`, `LearningInterface` are not stubs to be deleted later. They are the permanent ABCs that future milestones fill in via subclassing.

2. **Brain accepts None for all companion slots** — No features break when interfaces are absent. The Brain degrades gracefully: context window only, no RAG, no personalisation.

3. **Heuristic IntentClassifier as strategy** — The M3 classifier uses keyword rules. It's designed as a replaceable strategy; an LLM-based classifier (M4+) subclasses and overrides `classify()`.

4. **OllamaClient uses zero extra dependencies** — Python stdlib `urllib` + `asyncio.to_thread()`. No httpx or openai package needed. Future providers are added in M4.

5. **BrainConfig separate from ReasoningConfig** — `ReasoningConfig` owns LLM connection settings; `BrainConfig` owns Brain behavior and companion feature flags. Both live in `SpidyConfig`.

6. **EventBus is the only coupling** — Brain → EventBus for voice.user_spoke events. VoiceEngine, UI, and Memory systems are all decoupled.

---

## Milestone 4 — Multi-LLM + Skills Runtime ✅

**Status:** Complete and fully verified  
**Tests:** 134 new tests (385 total, all passing) — 2026-07-11

### What was built

**Multi-LLM Backend Layer (`spidy/llm/`)**

| File | Purpose |
|------|---------|
| `backends/openai.py` | OpenAIClient — stdlib urllib, no openai package required |
| `backends/claude.py` | ClaudeClient — Anthropic Messages API, system-message extraction |
| `backends/gemini.py` | GeminiClient — Google Generative Language API, role mapping |
| `router.py` | LLMRouter — ordered failover, EventBus events, failure threshold |
| `events.py` | LLM EventBus events: request_started, response_ready, provider_failed, provider_switched |
| `client.py` | Updated — `health_check()`, `provider_name`, `build_router()`, `build_from_provider_config()` |

**All four backends use Python stdlib `urllib` only — zero cloud package dependencies.**

**Permission Manager (`spidy/permissions/`)**

| File | Purpose |
|------|---------|
| `manager.py` | `PermissionTier` enum (T0–T3), `PermissionManager` with tier evaluation, audit log, session tracking |
| `events.py` | `PermissionGrantedEvent`, `PermissionDeniedEvent`, `PermissionRequestedEvent` |

Tier semantics:
- T0 — Always allowed (time, help, greet)
- T1 — Auto-approved per session (take_note, set_volume)
- T2 — Auto-denied in M4 (confirmation UI is M5+)
- T3 — Locked until `unlock_t3()` is called

**Skill Executor (`spidy/execution/`)**

| File | Purpose |
|------|---------|
| `executor.py` | `SkillExecutor` — permission gate, retry loop, timeout, EventBus events |
| `events.py` | `SkillExecutionStartedEvent`, `SkillExecutionCompletedEvent`, `SkillExecutionFailedEvent`, `SkillRetryEvent` |

**Built-in Skills (`spidy/skills/builtin/`)**

| Skill | Actions | Tiers |
|-------|---------|-------|
| `HelpSkill` | `show_help` | T0 |
| `TimeSkill` | `get_time`, `get_date`, `get_day`, `get_datetime` | T0 |
| `GreetSkill` | `greet`, `farewell`, `introduce` | T0 |
| `NoteSkill` | `take_note` (T1), `read_notes`, `find_note` (T0), `clear_notes` (T2) | Mixed |
| `TimerSkill` | `set_timer`, `list_timers`, `cancel_timer` | T0 |
| `SystemSkill` | `get_system_info` (T0), `set_volume`, `lock_screen` (T1, stubs) | Mixed |

**Config Extensions (`spidy/config/manager.py`)**

| Model | Fields |
|-------|--------|
| `LLMProviderConfig` | name, enabled, model, base_url, api_key, temperature, max_tokens, timeout |
| `MultiLLMConfig` | providers list, failure_threshold, auto_fallback |
| `SkillsConfig` | per-skill enable flags, notes_file |
| `ExecutorConfig` | max_retries, retry_delay_seconds, skill_timeout_seconds, permission_checking_enabled |

All added to `SpidyConfig` as `llm`, `skills`, `executor` fields.

**Tests**

| File | Tests | Coverage |
|------|-------|---------|
| `test_llm_backends.py` | 52 | Data types, OpenAI, Claude, Gemini clients, LLMRouter, LLMClientFactory, events |
| `test_permissions.py` | 24 | Tier enum, T0/T1/T2/T3 evaluation, policy override, EventBus events |
| `test_executor.py` | 16 | Basic execution, retry, timeout, permission gate, event publishing |
| `test_builtin_skills.py` | 44 | All 6 built-in skills + register_builtin_skills() |
| **Total new** | **136** | |
| **Grand total** | **385** | All passing |

### Key design decisions

1. **Zero cloud dependencies** — All four LLM backends use Python stdlib `urllib` + `asyncio.to_thread()`. OpenAI, Claude, and Gemini are fully implemented without their official SDKs.

2. **LLMRouter is a BaseLLMClient** — The router itself implements the same interface as individual backends. Any code that accepts `BaseLLMClient` works with single providers or the router transparently.

3. **Failure threshold per provider** — Once a provider exceeds N consecutive failures it is skipped for the session. `reset_failures()` re-enables it.

4. **PermissionManager is injected, not global** — SkillExecutor receives `PermissionManager | None`. With None, all actions are allowed (useful in tests and for the Brain's internal tool routing).

5. **T2 auto-denied in M4** — Confirmation dialogs require a UI (M5+). T2 actions are blocked in M4 to prevent unconfirmed destructive operations.

6. **Built-in skills are self-contained** — Each skill only imports Python stdlib. `register_builtin_skills()` is individually guarded so one registration failure doesn't prevent the rest from loading.

---

## Milestone 5 — Desktop Overlay UI ✅

**Status:** Complete  
**Tests:** 568 passed, 0 failed (385 previous + 183 new M5 tests)  
**New dependencies:** `PySide6>=6.7.0`, `keyboard>=0.13.5`

### What was built

#### `spidy/ui/events.py` — UI EventBus Events
- 12 event types: `UIShowEvent`, `UIHideEvent`, `UIStateChangeEvent`, `UIMessageEvent`, `UINotifyEvent`, `UIWaveformDataEvent`, `UIThemeChangeEvent`, `UIReadyEvent`, `UIClosedEvent`, `UIHotkeyPressedEvent`, `UIMicButtonClickedEvent`, `UISettingsOpenedEvent`
- UI communicates **exclusively** through the EventBus — no direct imports from Brain or Voice
- `UIShowEvent.source` field prepared for `"wake_word"` trigger in M6

#### `spidy/ui/state.py` — UI State Machine
- `UIState` enum: `IDLE`, `WAKE_READY`, `LISTENING`, `THINKING`, `SPEAKING`, `ERROR`
- `UIStateMachine`: enforced transition graph, strict/lenient modes, listener callbacks
- `WAKE_READY` state pre-wired for M6 wake-word integration (no activation logic yet)

#### `spidy/ui/themes/` — Theme System
- `ThemeColors` (31 color tokens), `ThemeFonts`, `ThemeGeometry` — all frozen dataclasses
- `DARK_THEME`: Deep navy + electric violet (#6C63FF) + teal — glassmorphism palette
- `LIGHT_THEME`: Pearl white + frosted glass — adjusted accent for contrast
- `ThemeManager`: registry, set/toggle, listener callbacks
- `build_default_theme_manager()` convenience factory

#### `spidy/ui/hotkey.py` — Global Hotkey Manager
- Uses `keyboard` library (pure Python ctypes on Windows)
- Thread-safe: fires callback in OS hook thread → `asyncio.run_coroutine_threadsafe`
- Graceful degradation: logs warning if `keyboard` unavailable (CI/restricted envs)
- `parse_hotkey()` / `validate_hotkey()` utilities for hotkey string normalization

#### `spidy/ui/widgets/` — Qt Widgets
- **`WaveformWidget`**: 20-bar animated visualiser; simulated sine-wave mode + real amplitude mode; 20fps QTimer, exponential smoothing
- **`MicrophoneButton`**: Animated round button; idle (purple), listening (red pulsing ring), thinking (spinner arc), speaking (wave icon); 25fps
- **`SpeakingIndicator`**: Three-dot wave animation; staggered phase offsets; 25fps
- **`ChatView` + `ChatBubble`**: Scrollable conversation with fade-in bubbles, auto-scroll, 80-message cap, theme-aware

#### `spidy/ui/notifications.py` — Toast Notification System
- `ToastWidget`: fade-in/out animation, left accent stripe, info/success/warning/error levels
- `NotificationManager`: max-stack enforcement, vertical stacking above overlay, theme-aware

#### `spidy/ui/tray.py` — System Tray Integration
- `SystemTrayManager`: vector-drawn Spidy 'S' tray icon (no file assets needed)
- Context menu: Show/Hide, Dark/Light mode, Settings, Exit Spidy
- Native OS balloon notifications via `QSystemTrayIcon.showMessage()`

#### `spidy/ui/overlay.py` — Main Overlay Window
- `OverlayWindow(QWidget)`: frameless, always-on-top, `Qt.WindowType.Tool` (no taskbar)
- **Windows acrylic blur** via `SetWindowCompositionAttribute` (ACCENT_ENABLE_ACRYLICBLURBEHIND), falls back silently on unsupported configs
- Slide-in animation from screen edge + fade-in via `QPropertyAnimation`
- State-driven widget show/hide (waveform, indicator, mic button state, status labels)
- Edge glow animation (animated border) in `WAKE_READY` state
- `show_for_wake_word()` slot prepared for M6 connection
- `UISignalBridge(QObject)`: thread-safe bridge — asyncio → Qt signals via auto-queued PySide6 emissions

#### `spidy/ui/app.py` — Qt Application Lifecycle
- `SpidyApp`: owns `QApplication`, `OverlayWindow`, `SystemTrayManager`, `GlobalHotkeyManager`, `UISignalBridge`
- EventBus subscriptions for all `ui.*` topics
- Forward-compatible handlers: `voice.listening_started/stopped`, `voice.speaking_started/stopped`
- `_publish_sync()` bridge: Qt callbacks → `asyncio.run_coroutine_threadsafe` → EventBus

### Architecture decisions

1. **EventBus-first UI** — The overlay never imports Brain, Skills, or Voice modules. All communication is through the EventBus, making the UI fully replaceable.

2. **UISignalBridge pattern** — PySide6 auto-queues cross-thread signal emissions. The bridge is instantiated in the Qt main thread; asyncio code calls `bridge.request_*()` methods which emit signals that Qt safely delivers on the main thread.

3. **Wake-word readiness** — `UIState.WAKE_READY`, `UIShowEvent.source`, `show_for_wake_word()` slot, and edge glow animation are all implemented but intentionally disconnected from any voice pipeline. In M6, a single line connects `VoiceEngine.wake_word_detected → overlay.show_for_wake_word`.

4. **Windows acrylic blur** — Applied via `user32.SetWindowCompositionAttribute` (ABGR GradientColor `0xCC0D0E1A` ≈ 80% transparent deep navy). Falls back to custom `paintEvent` rounded-rect background on failure. No exception propagation.

5. **GC safety** — PySide6 child widgets must have an explicit Python-side parent to avoid the C++ object being collected by Python GC. All widgets created in `_make_bubble` and `_build_ui` have parents set.

6. **Test isolation** — Qt tests use `QT_QPA_PLATFORM=offscreen` env var. `isHidden()` is used instead of `isVisible()` for child widget tests since Qt's `isVisible()` includes ancestry chain.

---

---

## Milestone 6 — Desktop & File Agent ✅

**Status:** Complete and fully verified  
**Tests:** 765 tests total, 0 failed (568 previous + 180 new M6 tests + 17 new permission tests)  
**New dependencies:** `pycaw>=20240418`, `winshell>=0.6`, `screen-brightness-control>=0.23.0`

### What was built

Three fully-integrated Windows desktop agent skills, wired into the EventBus and permission system.

#### Architecture

```
register_desktop_skills()
├── FileSkill              → desktop.file.{search_result,opened,folder_opened,recent_result}
├── AppSkill               → desktop.app.{running_result,launched,foregrounded,closed}
└── SystemControlSkill     → desktop.system.{volume_changed,brightness_changed,
                                              workstation_locked,sleep_initiated,
                                              shutdown_initiated,restart_initiated,
                                              recycle_bin_emptied}
```

All skills flow through `SkillExecutor` → `PermissionManager` → `EventBus`.

#### Files added

| File | Purpose |
|------|---------|
| `spidy/skills/desktop/__init__.py` | `register_desktop_skills()` loader; per-skill enable flags |
| `spidy/skills/desktop/events.py` | 15 typed EventBus event dataclasses (3 namespaces) |
| `spidy/skills/desktop/file_skill.py` | FileSkill — search, open, reveal (pathlib, T0/T1) |
| `spidy/skills/desktop/app_skill.py` | AppSkill — detect, launch, focus, close (psutil + ctypes, T0/T1/T2) |
| `spidy/skills/desktop/system_control_skill.py` | SystemControlSkill — volume, brightness, lock, sleep, shutdown, restart, recycle bin (T1/T2/T3) |
| `tests/unit/test_desktop_events.py` | 41 tests — all 15 event types |
| `tests/unit/test_file_skill.py` | 54 tests — all FileSkill actions |
| `tests/unit/test_app_skill.py` | 44 tests — all AppSkill actions |
| `tests/unit/test_system_control_skill.py` | 45 tests — all SystemControlSkill actions |
| `tests/integration/test_desktop_skills_integration.py` | 17 integration tests |

#### Files modified

| File | Change |
|------|--------|
| `spidy/permissions/events.py` | Added `PermissionResponseEvent` for T2 UI round-trip |
| `spidy/permissions/manager.py` | T2 is now interactive: `asyncio.Future` + `PermissionResponseEvent` |
| `spidy/config/manager.py` | `SkillsConfig` + `PermissionsConfig` M6 fields |
| `requirements.txt` | pycaw, winshell, screen-brightness-control |
| `config/spidy_config.yaml` | M6 skill enable flags and search defaults |
| `tests/unit/test_permissions.py` | +11 tests for T2 interactive confirmation workflow |

### Key design decisions

1. **EventBus-only output** — All 3 desktop skills publish typed events after every action. Brain and UI never poll skills directly.

2. **Graceful Windows degradation** — All Windows-specific imports (`pycaw`, `winshell`, `ctypes.windll`) are guarded with `try/except`. On import failure the action returns a friendly error string instead of crashing.

3. **Pure pathlib for file search** — `FileSkill.search_files()` uses `pathlib.Path.rglob()` with `fnmatch` glob patterns. No `win32api` required.

4. **Subprocess-only app launch** — `AppSkill.launch_app()` uses `subprocess.Popen()` + PATH + alias dict. No registry parsing. Covers 30+ common apps.

5. **T2 confirmation is now interactive** — `PermissionManager` now awaits an `asyncio.Future` instead of instantly denying T2 actions (M4 behavior). The UI publishes a `PermissionResponseEvent` (or calls `respond_to_confirmation()` directly) to resolve the future. A configurable timeout defaults to 30 s → deny.

6. **Tier assignments are explicit and conservative:**
   - T0: Read-only (search, detect, list)
   - T1: Non-destructive opens/launches (open_file, launch_app, lock_workstation, set_volume)
   - T2: Session-disruptive (close_app, sleep, empty_recycle_bin)
   - T3: Irreversible (shutdown, restart)

7. **AppSkill close_app is T2** — Even though closing an app is "mild", it can cause data loss if unsaved. T2 requires explicit user confirmation dialog.

---

*Last updated: 2026-07-11 — Milestone 6 complete (765 tests passing)*
