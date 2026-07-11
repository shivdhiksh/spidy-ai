# Changelog

All notable changes to **Spidy** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Planned — Milestone 5: Desktop Agent + Vision
- App launcher, file manager, screenshot, window management
- Vision pipeline (screenshot → LLM vision analysis)
- T2 permission confirmation dialogs (UI)
- Context-aware system state injection

---

## [0.4.0] — 2026-07-11 · Milestone 4: Multi-LLM + Skills Runtime

### Added

**Multi-LLM Backend Layer**
- `spidy/llm/backends/openai.py` — `OpenAIClient` using stdlib `urllib` (no `openai` package)
- `spidy/llm/backends/claude.py` — `ClaudeClient` for Anthropic Messages API with system-message extraction and consecutive-role merging
- `spidy/llm/backends/gemini.py` — `GeminiClient` for Google Generative Language API with OpenAI→Gemini role mapping
- `spidy/llm/router.py` — `LLMRouter` implementing `BaseLLMClient` with ordered failover, per-provider failure threshold, `reset_failures()`, and streaming support
- `spidy/llm/events.py` — `LLMRequestStartedEvent`, `LLMResponseReadyEvent`, `LLMProviderFailedEvent`, `LLMProviderSwitchedEvent`

**Permission Manager**
- `spidy/permissions/manager.py` — `PermissionTier` enum (T0–T3), `PermissionManager` with tier evaluation, policy override via config, per-session T1 tracking, T3 unlock, audit log (JSONL)
- `spidy/permissions/events.py` — `PermissionGrantedEvent`, `PermissionDeniedEvent`, `PermissionRequestedEvent`

**Skill Executor**
- `spidy/execution/executor.py` — `SkillExecutor` with permission gate, configurable retry loop, per-skill timeout, EventBus lifecycle events
- `spidy/execution/events.py` — `SkillExecutionStartedEvent`, `SkillExecutionCompletedEvent`, `SkillExecutionFailedEvent`, `SkillRetryEvent`

**Built-in Skills (6)**
- `HelpSkill` — Lists all registered capabilities from the SkillRegistry (T0)
- `TimeSkill` — Current time/date/day/datetime (T0, stdlib only)
- `GreetSkill` — Personalised greetings, farewells, and self-introduction with time-of-day awareness (T0)
- `NoteSkill` — JSONL notes file: take (T1), read (T0), find (T0), clear (T2); thread-safe
- `TimerSkill` — In-memory asyncio countdown timers with human duration parsing (T0)
- `SystemSkill` — Live CPU/RAM/battery info via psutil; volume/lock stubs for M7 (T0/T1)
- `register_builtin_skills()` — Convenience loader with per-skill enable flags from `SkillsConfig`

**Config Extensions**
- `LLMProviderConfig` — Per-provider config with name, enabled, model, base_url, api_key, temperature, max_tokens, timeout
- `MultiLLMConfig` — Ordered providers list, failure_threshold, auto_fallback
- `SkillsConfig` — Per-skill enable flags, notes_file path
- `ExecutorConfig` — max_retries, retry_delay_seconds, skill_timeout_seconds, permission_checking_enabled
- All new config models wired into `SpidyConfig` as `llm`, `skills`, `executor` fields

**Tests**
- `tests/unit/test_llm_backends.py` — 52 tests (data types, OpenAI, Claude, Gemini, LLMRouter, factory, events)
- `tests/unit/test_permissions.py` — 24 tests (tier enum, T0–T3, policy override, EventBus)
- `tests/unit/test_executor.py` — 16 tests (execution, retry, timeout, permissions, events)
- `tests/unit/test_builtin_skills.py` — 44 tests (all 6 skills, loader)
- **Total: 136 new tests; 385 total — all passing**

### Changed
- `spidy/llm/client.py` — Added `provider_name` class attribute, `health_check()` method, `build_from_provider_config()` and `build_router()` factory methods; expanded provider dispatch to all four backends
- `spidy/llm/backends/ollama.py` — Added `provider_name = "ollama"`, `health_check()` implementation
- `spidy/permissions/__init__.py` — Exports `PermissionManager`, `PermissionTier`

---

## [0.3.0] — 2026-07-11 — Milestone 3: Brain Core (Lifelong AI Companion Kernel)

### Product Reframe

Spidy is now designed as a **Lifelong AI Companion**, not just a desktop assistant.
The Brain architecture was designed from day one to support:
long-term memory, preference learning, habit learning, workflow learning,
personal knowledge graph, multi-LLM, optional web search, document knowledge base,
and continuous learning from user feedback.

### Added

**`spidy/brain/` package — 9 new files**

- `types.py` — All shared Brain data types:
  - `Intent` — classified action + entities + confidence + source
  - `Entity` — named entity (name + value)
  - `ConversationTurn` — single conversation exchange (role, text, intent, timestamp)
  - `Decision` — what the Brain decided (mode + skill_name + rationale)
  - `DecisionMode` — enum: SKILL / LLM_DIRECT / CLARIFY / REJECT
  - `TurnRole` — enum: USER / ASSISTANT / SYSTEM
  - `Plan` + `PlanStep` — ordered execution plan
  - `ToolResult` — outcome of executing a plan step

- `events.py` — 6 new EventBus events:
  - `BrainProcessingStartedEvent` (brain.processing_started)
  - `BrainResponseReadyEvent` (brain.response_ready)
  - `BrainSessionStartedEvent` (brain.session_started)
  - `BrainSessionEndedEvent` (brain.session_ended)
  - `BrainToolCalledEvent` (brain.tool_called)
  - `BrainToolResultEvent` (brain.tool_result)

- `interfaces.py` — **Three forward-declared Lifelong AI Companion ABCs**:
  - `MemoryInterface` — 4 methods: store, recall, search, clear → M6
  - `KnowledgeInterface` — 3 methods: query, ingest, search_rag → M8+
  - `LearningInterface` — 3 methods: record_feedback, get_preference, get_habit → M9+
  - All raise `NotImplementedError` in M3 — permanent contracts, not stubs
  - Brain accepts `None` for all; companion features degrade gracefully

- `intent_classifier.py` — `IntentClassifier` (heuristic strategy, M3):
  - 15+ action categories: file_*, app_*, web_*, system_*, note_*, timer_*, etc.
  - Keyword pattern matching with per-rule confidence levels
  - Entity extraction for query, app_name, filename, note_content, timer_duration
  - Falls back to `action="chat"` for open-ended conversational input
  - Designed as a strategy — LLM-based classifier (M4+) replaces via subclass

- `conversation_manager.py` — `ConversationManager`:
  - Rolling context window (deque, max_turns configurable)
  - O(1) append + oldest-turn eviction
  - `get_llm_messages()` — builds OpenAI-format message list with system prompt
  - `get_summary()` — one-line recent context for DecisionEngine
  - Session lifecycle: `start_session()` / `end_session()`
  - Publishes `brain.session_started` / `brain.session_ended` events

- `decision_engine.py` — `DecisionEngine`:
  - SKILL: action in SkillRegistry → dispatch to skill
  - LLM_DIRECT: `chat` or unknown action → answer via LLM
  - CLARIFY: confidence < 0.6 → ask for clarification
  - REJECT: empty utterance or system-reserved action
  - Accepts optional `MemoryInterface` for future context enrichment (M6)

- `planner.py` — `Planner`:
  - Converts Decision → Plan (single-step in M3)
  - SKILL decision: extracts entity params from Intent
  - LLM decision: creates llm step with conversation context
  - CLARIFY: creates clarify step with question text
  - REJECT: creates noop step
  - Accepts optional `KnowledgeInterface` for future RAG injection (M8+)

- `tool_router.py` — `ToolRouter`:
  - Routes skill steps to `SkillRegistry.find_skill_for_action()` → `skill.execute()`
  - Routes llm steps to `BaseLLMClient.complete()`
  - Handles clarify and noop steps directly
  - All errors caught → `ToolResult.fail()` (never raises to Brain)
  - Publishes `brain.tool_called` + `brain.tool_result` events per step
  - Accepts optional `LearningInterface` for future feedback collection (M9+)

- `brain.py` — `Brain` (top-level orchestrator):
  - `process(utterance)` — full pipeline: classify → decide → plan → route
  - `start()` / `stop()` lifecycle (subscribes to `voice.user_spoke`)
  - `_on_user_spoke()` — EventBus handler for voice integration
  - Accepts memory, knowledge, learning companion slots (all None by default)
  - `_compose_response()` — aggregates ToolResults into final response text

**`spidy/llm/` package — 3 new files**

- `client.py` — Multi-backend LLM abstraction:
  - `LLMMessage` (frozen dataclass: role + content)
  - `LLMUsage` (token counts)
  - `LLMResponse` (text, model, usage, finish_reason, success, error_message)
  - `BaseLLMClient` (ABC: complete, stream, close)
  - `LLMClientFactory.build(config)` — dispatches to correct backend

- `backends/__init__.py` — Package init

- `backends/ollama.py` — `OllamaClient`:
  - Uses stdlib `urllib` + `asyncio.to_thread()` — zero extra dependencies
  - `complete()` → POST /api/chat (stream=false)
  - `stream()` → POST /api/chat (stream=true) → async generator
  - Returns `LLMResponse.failure()` if Ollama is not running (never raises)

**Config**

- `BrainConfig` (new Pydantic model in `spidy/config/manager.py`):
  - `intent_classifier` — strategy selection ("heuristic" / "llm")
  - `min_intent_confidence` — clarify threshold (default 0.6)
  - `max_conversation_turns` — rolling window size (default 20)
  - `decision_mode` — auto / skill_only / llm_only
  - `tool_routing_enabled` — enable skill dispatch
  - `enable_memory` / `enable_knowledge` / `enable_learning` — companion feature flags
  - `enable_web_search` — optional web search (M4+)
- Added `brain: BrainConfig` field to `SpidyConfig`

**SpidyCore**

- `spidy/core/app.py` — Brain wired into `_run()` + `_stop()` lifecycle
  - Creates `SkillRegistry`, builds `OllamaClient`, instantiates `Brain`
  - Non-fatal: logs warning + continues without Brain if construction fails

**Tests**

- `tests/unit/test_brain.py` — **106 unit tests** across 9 test classes:
  - `TestBrainConfig` (4 tests)
  - `TestIntentClassifier` (14 tests)
  - `TestConversationManager` (15 tests)
  - `TestDecisionEngine` (7 tests)
  - `TestPlanner` (7 tests)
  - `TestToolRouter` (10 tests)
  - `TestBrainPipeline` (12 tests)
  - `TestBrainInterfaces` (13 tests)
  - `TestLLMClient` (7 tests)
  - `TestBrainEvents` (6 tests)
  - `TestBrainTypes` (11 tests)

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
