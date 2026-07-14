# Changelog

All notable changes to **Spidy** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

_No unreleased changes._

---

## [0.11.0] — 2026-07-12 · Milestone 10: Knowledge Engine

### Added

- **`spidy.knowledge` package** — complete five-component Knowledge Engine:
  - `knowledge/types.py` — frozen dataclasses: `KnowledgeChunk`, `KnowledgeResult`;
    `SourceType` literal alias (pdf / docx / markdown / text / url / web_search / unknown)
  - `knowledge/events.py` — 6 typed EventBus events under `knowledge.*` namespace:
    `KnowledgeDocumentIngestedEvent`, `KnowledgeChunkStoredEvent`, `KnowledgeQueriedEvent`,
    `KnowledgeWebSearchPerformedEvent`, `KnowledgeSourceDeletedEvent`, `KnowledgeErrorEvent`
  - `knowledge/chunker.py` — `Chunker`: fixed-size + overlap text splitting;
    deterministic chunk IDs (`SHA-256[:12]_index`); `chunk()` and `chunk_pages()` APIs
  - `knowledge/embedding.py` — `EmbeddingEngine`: lazy `sentence-transformers` model init;
    `embed_sync()` + `embed()` (async) + `embed_one()`; graceful degrade when dep absent
  - `knowledge/ingestor.py` — `DocumentIngestor` unified dispatch +
    `PDFIngestor` (pypdf), `DocxIngestor` (python-docx), `MarkdownIngestor` (stdlib),
    `PlainTextIngestor` (stdlib); all format deps are optional and gracefully absent
  - `knowledge/store.py` — `VectorStore`: ChromaDB `spidy_knowledge` collection;
    `add_sync/add`, `query_sync/query`, `delete_source_sync/delete_source`,
    `count_sync/count`, `list_sources_sync/list_sources`; ephemeral (tests) and persistent modes
  - `knowledge/web_search.py` — `WebSearchEngine` + `DuckDuckGoProvider`:
    feature-flagged DuckDuckGo search (disabled by default, no API key required);
    results as `KnowledgeChunk` objects with rank-decay scores
  - `knowledge/manager.py` — `KnowledgeManager` implementing `KnowledgeInterface`:
    orchestrates all sub-components; `query()`, `ingest()`, `search_rag()`,
    `ingest_file()`, `delete_source()`, `count()`, `list_sources()`
  - `knowledge/__init__.py` — clean public API re-exports

- **`spidy.config.manager`** — Two new Pydantic models:
  - `KnowledgeConfig` — chunk_size, chunk_overlap, embedding_model, collection_name,
    min_relevance_score, max_results, web_search nested config
  - `WebSearchConfig` — enabled flag, provider, max_results, safe_search
  - Both added to `SpidyConfig` root model as `knowledge: KnowledgeConfig`

- **`spidy.core.app` — SpidyCore Knowledge Engine integration:**
  - `_knowledge_mgr: KnowledgeManager | None` instance variable
  - `knowledge` property — exposes the active `KnowledgeManager`
  - Step 8 in `_initialise()` — initialises `KnowledgeManager` (non-fatal; running
    without knowledge engine if deps are absent)
  - `_stop()` — closes `KnowledgeManager` before stopping other subsystems
  - `Brain()` constructor call updated to pass `knowledge=self._knowledge_mgr`
  - Module docstring updated to list `KnowledgeManager` as dependency step 7

- **`tests/unit/test_knowledge_types.py`** — 36 tests:
  `KnowledgeChunk` construction, immutability, score clamping, `to_dict()`, `create()`;
  `KnowledgeResult` construction, `is_empty`, `best_score`, `as_rag_context()`

- **`tests/unit/test_knowledge_events.py`** — 28 tests:
  All 6 event types; topic strings, Event inheritance, field defaults, construction

- **`tests/unit/test_knowledge_chunker.py`** — 34 tests:
  Constructor validation, `chunk()`, `chunk_pages()`, `_split()`, `_hash_source()`

- **`tests/unit/test_knowledge_ingestor.py`** — 42 tests:
  `PlainTextIngestor`, `MarkdownIngestor` (Markdown stripping), `PDFIngestor` (graceful degrade),
  `DocxIngestor` (graceful degrade), `DocumentIngestor` (dispatch, source_type, metadata)

- **`tests/unit/test_knowledge_embedding.py`** — 22 tests:
  Construction, availability, `embed_sync()` unavailable/mocked, async wrappers,
  `embedding_dim` property; no real sentence-transformers required

- **`tests/unit/test_knowledge_store.py`** — 34 tests:
  Construction, availability, `add_sync()`, `query_sync()` (score computation, min_score
  filtering), `delete_source_sync()`, `count_sync()`, `list_sources_sync()`;
  all with mocked ChromaDB — no real ChromaDB required

- **`tests/unit/test_knowledge_web_search.py`** — 30 tests:
  `BaseSearchProvider`, `DuckDuckGoProvider` availability + search,
  `WebSearchEngine` availability gate, `search_sync()`, `_result_to_chunk()`, async wrappers

- **`tests/unit/test_knowledge_manager.py`** — 38 tests:
  Construction, lifecycle, `ingest()`, `ingest_file()`, `query()`, `search_rag()`,
  `delete_source()`, `count()`, `list_sources()`, `_retrieve()` pipeline,
  EventBus integration (silent on missing bus, swallows publish errors)

### Architecture

```
KnowledgeManager  (KnowledgeInterface)
    ├── DocumentIngestor   PDF / DOCX / Markdown / plain-text (all optional deps)
    ├── Chunker            fixed-size + overlap splitting (zero deps)
    ├── EmbeddingEngine    sentence-transformers (lazy init, graceful degrade)
    ├── VectorStore        ChromaDB "spidy_knowledge" collection (graceful degrade)
    └── WebSearchEngine    DuckDuckGo (disabled by default, no API key required)
```

### Graceful Degradation

- `sentence-transformers` absent → embeddings return `[]`; ingest still chunks and stores;
  retrieval returns empty results
- `chromadb` absent → store operations are no-ops; retrieval returns empty
- `duckduckgo-search` absent OR `web_search.enabled: false` → web search silently disabled
- All deps absent → `KnowledgeManager` still initialises; Brain continues normally

---

## [0.10.0-alpha] — 2026-07-12 · Spidy Alpha Integration

### Added

- **`spidy.main`** — Enhanced production entry point:
  - `--text-mode` CLI flag: starts an interactive stdin REPL, forwarding input
    directly to the Brain. Useful for testing without audio hardware.
  - Full argument parser with clear error messages for unknown flags and missing
    config files.

- **`spidy.core.app` — SpidyCore integration enhancements:**
  - `text_mode: bool` parameter on `SpidyCore.__init__()`.
  - `_run_text_repl()` — asyncio-safe stdin REPL using `run_in_executor(None, input)`
    (correct Windows-compatible pattern; avoids broken `connect_read_pipe`).
  - `_start_ui_thread()` — starts `SpidyApp` (Qt overlay) in a `daemon=True` thread
    named `SpidyQtUI`, passing the asyncio loop for thread-safe event publishing.
  - `core.brain` property — exposes the active `Brain` instance.
  - `core.memory` property — exposes the active `MemoryManager` instance.
  - `core.vision` property — exposes the active `VisionManager` instance.
  - Module docstring updated to reflect the complete M0–M9 module dependency order.

- **`tests/integration/test_alpha_integration.py`** — 34 new end-to-end integration tests:
  - `TestAlphaCoreLifecycle` (5) — CREATED→RUNNING→STOPPED; property access guards.
  - `TestAlphaTextCommandFlow` (5) — Brain.process() for greeting/time/help/unknown/empty.
  - `TestAlphaDesktopSkillRegistration` (2) — All 3 desktop skills; config flag gating.
  - `TestAlphaBrowserSkillRegistration` (2) — Browser skill; disabled flag.
  - `TestAlphaMemoryStorage` (6) — Episodic count, recall, store/recall failure resilience.
  - `TestAlphaUIEvents` (4) — Show/hide/state/message events; Brain→UI event chain.
  - `TestAlphaErrorRecovery` (3) — Non-fatal memory failure; no-LLM brain; idempotent shutdown.
  - `TestAlphaCompleteExecutionFlow` (3) — Full pipeline + memory; all skills; 5-turn sequence.
  - `TestAlphaTextModeRepl` (3) — text_mode init; REPL exits on quit; REPL processes command.

### Fixed

- **Integration gap**: Qt overlay UI (`SpidyApp`) was never started despite being fully
  implemented since Milestone 5. Now started in a daemon thread from `SpidyCore._run()`.
- **Integration gap**: No mechanism existed to send text commands to the Brain without
  voice hardware. Text-mode REPL fills this gap.
- **Stale docstring** in `core/app.py`: removed `[Future] Brain, MemoryEngine, UI, Skills...`
  comment; replaced with accurate M0–M9 dependency order.

### Changed

- `SpidyCore.__init__()` signature: added `text_mode: bool = False` (backward-compatible).
- `_stop()`: now calls `SpidyApp.quit()` before stopping other subsystems.

---

## [0.9.0] — 2026-07-12 · Milestone 9: Vision Engine

### Added

- **`spidy.vision` package** — complete three-engine Vision system:
  - `vision/types.py` — frozen dataclasses: `ScreenshotResult`, `OCRBlock`, `OCRResult`,
    `ScreenAnalysis`, `UIRegion`, `MonitorInfo`, `VisionDependencyError`
  - `vision/events.py` — 5 typed EventBus events under `vision.*` namespace:
    `VisionCaptureStartedEvent`, `VisionCaptureCompletedEvent`, `VisionOCRCompletedEvent`,
    `VisionAnalysisCompletedEvent`, `VisionErrorEvent`
  - `vision/screenshot.py` — `ScreenshotEngine`: ultra-fast `mss`-based capture;
    full-screen, active-window (`win32gui`), and region modes; multi-monitor enumeration
  - `vision/ocr.py` — `OCREngine`: `easyocr` text extraction with lazy reader init,
    confidence thresholding, bbox conversion, Pillow/numpy decode path
  - `vision/screen_analyzer.py` — `ScreenAnalyzer`: active-app + window-title detection
    (`win32gui`/`psutil`), `EnumWindows` visible-window list, OpenCV contour-based UI region detection
  - `vision/manager.py` — `VisionManager` implementing `VisionInterface`: unified async
    API, thread-executor wrapping for sync engines, EventBus publishing, `describe_screen()`
    context-injection method

- **Brain integration** (`spidy.brain`)
  - `brain/interfaces.py` — `VisionInterface` ABC added as M9 companion slot
  - `brain/brain.py` — `vision=` parameter in `Brain.__init__()`, `Brain.vision` property,
    startup log includes vision status

- **Config** (`spidy.config.manager`)
  - `ScreenshotConfig`, `OCRConfig`, `VisionConfig` Pydantic models added
  - `SpidyConfig.vision: VisionConfig` field added

- **SpidyCore** (`spidy.core.app`)
  - Step 7 in `_initialise()`: `VisionManager` constructed and initialized
  - `vision=self._vision_mgr` passed to `Brain`
  - `_vision_mgr.close()` called in `_stop()`

- **Configuration** (`config/spidy_config.yaml`)
  - `vision:` section with per-engine enable flags, screenshot and OCR sub-config

- **Tests** — 180 new tests (1229 total, 0 failures):
  - `tests/unit/test_vision_types.py` — 40 tests
  - `tests/unit/test_vision_events.py` — 28 tests
  - `tests/unit/test_vision_screenshot.py` — 28 tests
  - `tests/unit/test_vision_ocr.py` — 26 tests
  - `tests/unit/test_vision_screen_analyzer.py` — 30 tests
  - `tests/unit/test_vision_manager.py` — 21 tests
  - `tests/integration/test_vision_brain_integration.py` — 17 tests

### Design

- All three vision dependencies (`mss`, `easyocr`, `opencv-python`) are optional
- Every engine has `is_available` and gracefully returns empty results when deps absent
- `VisionDependencyError` is the single exception type for dep-missing conditions
- `VisionManager._publish()` silently skips events when no bus provided (unit test friendly)
- Thread-executor pattern used for synchronous capture/OCR to avoid blocking asyncio loop

---

## [0.8.0] — 2026-07-11 · Milestone 8: Memory Engine

### Added

- **`spidy.memory` package** — complete three-tier memory engine:
  - `memory/types.py` — `MemoryEntry` (frozen dataclass), `MemoryType` (enum), `MemorySearchResult`
  - `memory/events.py` — 6 typed EventBus events: `MemoryStoredEvent`, `MemoryRetrievedEvent`, `MemorySearchedEvent`, `MemoryDeletedEvent`, `MemoryClearedEvent`, `MemoryErrorEvent`
  - `memory/working.py` — `WorkingMemory`: per-session rolling buffer, configurable capacity (default 20 messages), keyword find, session isolation
  - `memory/episodic.py` — `EpisodicMemory`: async SQLite via `aiosqlite` with graceful in-process fallback, indexed schema, keyword LIKE search
  - `memory/semantic.py` — `SemanticMemory`: ChromaDB vector store, cosine similarity, lazy init; complete no-op when optional deps absent
  - `memory/manager.py` — `MemoryManager` implementing `MemoryInterface`: unified Brain API, three-tier coordination, EventBus publishing, `store_interaction()` + `get_working_context()` convenience methods

- **Brain integration** (`spidy.brain.brain`)
  - `Brain.process()` now recalls relevant memories before planning (step 5)
  - `Brain.process()` stores each interaction after responding (step 11)
  - Both calls are failure-isolated (`try/except` with `log.warning`)
  - `Brain` constructor accepts `memory: MemoryInterface | None` (backward compatible)

- **SpidyCore integration** (`spidy.core.app`)
  - `MemoryManager` initialised in `_initialise()` step 6, before Brain
  - Passed to `Brain` as `memory=self._memory_mgr`
  - Gracefully closed in `_stop()`

- **Config** (`spidy.config.manager`, `config/spidy_config.yaml`)
  - `MemoryConfig.enabled` — master switch
  - `MemoryConfig.enable_working` / `enable_episodic` / `enable_semantic` — per-tier flags

- **Dependency**: `aiosqlite>=0.20.0` added to `requirements.txt`

- **Tests**: 135 new tests (1049 total)
  - `tests/unit/test_memory_types.py` (14 tests)
  - `tests/unit/test_memory_events.py` (14 tests)
  - `tests/unit/test_memory_working.py` (26 tests)
  - `tests/unit/test_memory_episodic.py` (27 tests)
  - `tests/unit/test_memory_semantic.py` (14 tests)
  - `tests/unit/test_memory_manager.py` (32 tests)
  - `tests/integration/test_memory_brain_integration.py` (8 tests)

### Changed

- `spidy/memory/__init__.py` — previously a stub; now exports full public API
- `spidy/config/manager.py` — `MemoryConfig` gains 4 new optional boolean fields
- `config/spidy_config.yaml` — `memory:` section gains enable/disable flags

---

## [0.7.0] — 2026-07-11 · Milestone 7: Browser Agent

### Added

**Browser Agent Core** (`spidy/browser/`)

**`BrowserAgent`** (`browser/agent.py`) — Owns the browser session lifecycle; provides a clean async API to BrowserSkill. Auto-starts on first use. Supports:
- `start()` / `stop()` with CDP attach before launching a new window
- `open_url(url, new_tab)` — navigate to any URL, optionally in a new tab
- `close_tab(tab_id)` — close a specific tab by ID
- `get_current_page_info()` — returns `PageInfo` (title, URL, tab_id, load_time_ms)
- `get_all_tabs()` — returns list of `TabInfo` (all open tabs)
- `navigate_back()` / `navigate_forward()` / `refresh()`
- `search_google(query)` / `search_youtube(query)` — open search results pages
- `read_page_text(max_chars)` — extract visible body text
- `download_file(url, dest_dir)` — returns `DownloadResult`
- `get_browser_history(limit)` — reads Chrome/Edge SQLite history (best-effort)

**`BrowserBackend` ABC** (`browser/backends/base.py`) — Stable async contract for any browser automation backend. Mirrors `BaseLLMClient` pattern — makes Playwright swappable.

**`PlaywrightBackend`** (`browser/backends/playwright_backend.py`) — Playwright async API implementation:
- Chromium / Firefox / WebKit engine selection
- CDP attach to existing Chrome session (`--remote-debugging-port=9222`) before launching new window
- Headed mode by default (user sees the browser)
- Download handler with configurable destination directory
- `get_browser_history()` via Chrome/Edge SQLite copy (non-destructive, temp file)
- Graceful degradation if `playwright` is not installed (RuntimeError → caught by BrowserAgent)

**Browser Types** (`browser/types.py`) — Five frozen dataclasses:
- `PageInfo(title, url, tab_id, load_time_ms)`
- `TabInfo(tab_id, title, url, is_active)`
- `DownloadResult(filename, path, size_bytes, success, error)`
- `HistoryEntry(title, url, visit_time)`
- `SearchResult(title, url, snippet)` — reserved for future result-scraping

---

**Browser Skills** (`spidy/skills/browser/`)

**`BrowserSkill`** (`browser_skill.py`) — 14 capabilities (T0/T1/T2):

| Action | Tier | Description |
|--------|------|-------------|
| `get_page_info` | T0 | Read current page title + URL |
| `list_tabs` | T0 | List all open tabs |
| `read_page` | T0 | Extract visible text from page |
| `open_browser` | T1 | Launch browser or ensure it's open |
| `open_url` | T1 | Navigate to a URL |
| `open_new_tab` | T1 | Open a URL in a new tab |
| `navigate_back` | T1 | Go back in browser history |
| `navigate_forward` | T1 | Go forward in browser history |
| `refresh_page` | T1 | Reload current page |
| `search_google` | T1 | Open Google search results |
| `search_youtube` | T1 | Open YouTube search results |
| `close_browser` | T2 | Close browser + all tabs (confirmation required) |
| `close_tab` | T2 | Close a specific tab (confirmation required) |
| `download_file` | T2 | Download a file to disk (confirmation required) |

**`register_browser_skills(registry, config, bus)`** — Skill loader; follows the same pattern as `register_desktop_skills()`.

---

**Browser Events** (`spidy/skills/browser/events.py`) — 12 typed EventBus events (`browser.*`):

`BrowserLaunchedEvent`, `BrowserClosedEvent`, `PageNavigatedEvent`, `TabClosedEvent`, `PageReadEvent`, `SearchResultsEvent`, `DownloadStartedEvent`, `DownloadCompletedEvent`, `NavigationBackEvent`, `NavigationForwardEvent`, `PageRefreshedEvent`, `BrowserErrorEvent`

---

**Config Extensions**

- `BrowserConfig` Pydantic model added to `spidy/config/manager.py`
- `browser:` section added to `config/spidy_config.yaml`
- `browser_skill_enabled` + 9 browser configuration fields added to `SkillsConfig`
- `browser: BrowserConfig` field added to `SpidyConfig`
- `playwright>=1.44.0` added to `requirements.txt`

**Core Integration**

- `spidy/core/app.py`: `register_browser_skills()` called after `register_desktop_skills()` in the startup sequence

---

### Tests Added

| File | Tests |
|------|-------|
| `tests/unit/test_browser_types.py` | 15 |
| `tests/unit/test_browser_events.py` | 37 |
| `tests/unit/test_browser_agent.py` | 40 |
| `tests/unit/test_browser_skill.py` | 62 |
| `tests/integration/test_browser_skill_integration.py` | 22 |
| **Total new** | **176** |
| **Grand total (all milestones)** | **914** |

### Architecture Notes

- **BrowserSkill → BrowserAgent → BrowserBackend ← PlaywrightBackend** — Brain/BrowserSkill never import Playwright directly.
- **Lazy init** — Browser does not launch at startup; first T1/T2 action triggers `BrowserAgent.start()`.
- **CDP attach** — If Chrome is already open with `--remote-debugging-port=9222`, Spidy attaches to that session (no new window). Falls back to launching a new browser.
- **Headed by default** — User sees the browser. Set `browser.headless: true` in config for background operation.
- **T2 confirmation** — `close_browser`, `close_tab`, and `download_file` require interactive T2 confirmation through the existing `PermissionManager` framework.

---

## [0.6.0] — 2026-07-11 · Milestone 6: Desktop & File Agent

### Added

**Desktop Agent Skills** (`spidy/skills/desktop/`)

**`FileSkill`** (`file_skill.py`) — Windows File & Folder Agent (T0/T1)
- `search_files` — Recursive file search by name pattern; `fnmatch` glob support (T0)
- `search_folders` — Recursive folder search by name (T0)
- `open_file` — Opens file with default application via `os.startfile()` (T1)
- `open_folder` — Opens folder in Windows Explorer via `subprocess` (T1)
- `reveal_in_explorer` — Reveals and selects a file in Explorer (T1)
- `list_recent_files` — Lists recently accessed files from shell recent folder (T0)
- Configurable `search_root` and `max_results` (default: home dir, 50 results)
- Pure `pathlib` search — no Windows API required for file enumeration
- Publishes `FileSearchResultEvent`, `FileOpenedEvent`, `FolderOpenedEvent`, `RecentFilesResultEvent`

**`AppSkill`** (`app_skill.py`) — Windows Application Agent (T0/T1/T2)
- `detect_running_apps` — Lists running processes via `psutil` (T0)
- `launch_app` — Launches application by name or alias via `subprocess.Popen()` (T1)
- `bring_app_to_foreground` — Focuses running app window via `ctypes.windll.user32` (T1)
- `close_app` — Terminates process: graceful `.terminate()` then `.kill()` fallback (T2)
- 30+ built-in app aliases (`notepad`, `chrome`, `vscode`, `spotify`, `discord`, etc.)
- Publishes `RunningAppsResultEvent`, `AppLaunchedEvent`, `AppForegroundedEvent`, `AppClosedEvent`

**`SystemControlSkill`** (`system_control_skill.py`) — Windows System Control Agent (T1/T2/T3)
- `set_volume` — System volume via `pycaw` COM API; graceful stub if unavailable (T1)
- `set_brightness` — Screen brightness via `screen-brightness-control`; graceful fallback (T1)
- `lock_workstation` — Lock session via `ctypes.windll.user32.LockWorkStation()` (T1)
- `sleep_system` — Sleep via `ctypes.windll.powrprof.SetSuspendState()` (T2)
- `shutdown_system` — Shutdown via `subprocess` + `shutdown.exe` with delay parameter (T3)
- `restart_system` — Restart via `subprocess` + `shutdown.exe /r` (T3)
- `empty_recycle_bin` — Empty recycle bin via `winshell`; graceful stub if unavailable (T2)
- All Windows imports guarded with `try/except`; skill works (degraded) on non-Windows
- Publishes `VolumeChangedEvent`, `BrightnessChangedEvent`, `WorkstationLockedEvent`,
  `SystemSleepInitiatedEvent`, `SystemShutdownInitiatedEvent`, `SystemRestartInitiatedEvent`, `RecycleBinEmptiedEvent`

**Desktop EventBus Events** (`spidy/skills/desktop/events.py`)
- 15 typed event dataclasses across 3 namespaces:
  - `desktop.file.*` — FileSearchResultEvent, FileOpenedEvent, FolderOpenedEvent, RecentFilesResultEvent
  - `desktop.app.*` — RunningAppsResultEvent, AppLaunchedEvent, AppForegroundedEvent, AppClosedEvent
  - `desktop.system.*` — VolumeChangedEvent, BrightnessChangedEvent, WorkstationLockedEvent,
    SystemSleepInitiatedEvent, SystemShutdownInitiatedEvent, SystemRestartInitiatedEvent, RecycleBinEmptiedEvent

**Desktop Skills Loader** (`spidy/skills/desktop/__init__.py`)
- `register_desktop_skills(registry, config, bus)` — registers all 3 desktop skills
- Per-skill enable flags (`file_skill_enabled`, `app_skill_enabled`, `system_control_skill_enabled`)
- Individually guarded: one registration failure does not block others

**Permission Manager Extended** (`spidy/permissions/`)
- Added `PermissionResponseEvent` — UI publishes this to resolve pending T2 Future
- T2 confirmation is now **interactive**: `PermissionManager` publishes `PermissionRequestedEvent`,
  waits on an `asyncio.Future`, and resolves when the UI (or test) calls `respond_to_confirmation()`
  or publishes `PermissionResponseEvent` on the bus
- `t2_confirmation_timeout_seconds` config field (default 30 s); times out → deny
- `_on_permission_response()` EventBus subscriber wires the round-trip automatically

**Config Extensions** (`spidy/config/manager.py`)
- `SkillsConfig` extended with 5 new M6 fields:
  `file_skill_enabled`, `app_skill_enabled`, `system_control_skill_enabled`,
  `desktop_file_search_max_results`, `desktop_file_search_root`
- `PermissionsConfig` extended with `t2_confirmation_timeout_seconds`

### Dependencies
- Added `pycaw>=20240418` (Windows Audio Session API, volume control)
- Added `winshell>=0.6` (Windows shell utilities, recycle bin)
- Added `screen-brightness-control>=0.23.0` (screen brightness)
- All M6 deps guarded — Spidy works without them (actions produce friendly errors)

### Tests
- **180 new tests** across 5 new test files (17 additions in test_permissions.py)
  - `test_desktop_events.py`: 41 tests — all 15 event types, field defaults, EventBus roundtrip
  - `test_file_skill.py`: 54 tests — capabilities, search, open, recent, EventBus integration
  - `test_app_skill.py`: 44 tests — capabilities, detect, launch, foreground, close, aliases
  - `test_system_control_skill.py`: 45 tests — all 7 actions, permission tiers, event publishing
  - `tests/integration/test_desktop_skills_integration.py`: 17 tests — registry, executor, full pipeline
  - `test_permissions.py`: +11 new tests for T2 confirmation workflow (interactive approval/denial via bus)
- **Total: 765 tests passing, 0 failed**

---

## [0.5.0] — 2026-07-11 · Milestone 5: Desktop Overlay UI

### Added

**UI EventBus Events** (`spidy/ui/events.py`)
- 12 event types covering inbound (show/hide/state/message/notify/waveform/theme) and outbound (ready/closed/hotkey/mic/settings) UI events
- `UIShowEvent.source` field: prepared for `"wake_word"` trigger (M6)
- `UIHotkeyPressedEvent`: published when global hotkey fires

**UI State Machine** (`spidy/ui/state.py`)
- `UIState` enum: `IDLE`, `WAKE_READY`, `LISTENING`, `THINKING`, `SPEAKING`, `ERROR`
- `UIStateMachine`: enforced allowed-transition graph, strict/lenient mode, listener callbacks, reset
- `WAKE_READY` state pre-wired for M6 wake-word (no voice activation yet)

**Theme System** (`spidy/ui/themes/`)
- `ThemeColors`: 55 frozen color tokens per theme
- `DARK_THEME`: deep navy + electric violet (#6C63FF) glassmorphism palette
- `LIGHT_THEME`: pearl white + frosted glass with adjusted accent
- `ThemeManager`: registry, set/toggle, listener callbacks, `build_default_theme_manager()`

**Global Hotkey Manager** (`spidy/ui/hotkey.py`)
- System-wide hotkey via `keyboard` library (Ctrl+Space default)
- Thread-safe EventBus bridging via `asyncio.run_coroutine_threadsafe`
- Graceful degradation if `keyboard` unavailable
- `parse_hotkey()` / `validate_hotkey()` utilities

**Qt Widgets** (`spidy/ui/widgets/`)
- `WaveformWidget`: 20-bar animated audio visualiser (simulated + real-data modes, 20fps)
- `MicrophoneButton`: animated round button with idle/listening/thinking/speaking states (25fps)
- `SpeakingIndicator`: three-dot wave animation with staggered phase (25fps)
- `ChatView` + `ChatBubble`: scrollable conversation display, fade-in bubbles, 80-message cap

**Notification System** (`spidy/ui/notifications.py`)
- `ToastWidget`: fade-in/out with left accent stripe and 4 severity levels
- `NotificationManager`: vertical stacking, max-stack enforcement, auto-dismiss

**System Tray** (`spidy/ui/tray.py`)
- `SystemTrayManager`: vector-drawn 'S' tray icon (no file assets)
- Context menu: Show/Hide, Dark/Light mode, Settings, Exit
- Native OS balloon notifications

**Main Overlay Window** (`spidy/ui/overlay.py`)
- `OverlayWindow`: frameless, always-on-top `Qt.WindowType.Tool` PySide6 widget
- **Windows acrylic blur** via `SetWindowCompositionAttribute` with fallback
- Slide-from-edge + fade-in via `QPropertyAnimation`
- State-driven widget visibility (all 6 states fully wired)
- Edge glow animation in `WAKE_READY` state (prepared for M6)
- `show_for_wake_word()` slot (ready for M6 voice connection)
- `UISignalBridge(QObject)`: thread-safe async→Qt bridge using PySide6 signal auto-queuing

**Qt Application Lifecycle** (`spidy/ui/app.py`)
- `SpidyApp`: owns QApplication, OverlayWindow, SystemTrayManager, GlobalHotkeyManager
- Subscribes to `ui.*` + `voice.*` EventBus topics
- Bridges Qt callbacks to asyncio EventBus via `run_coroutine_threadsafe`

**UIConfig Extended** (`spidy/config/manager.py`)
- 9 new M5 fields: `animation_speed_ms`, `notification_duration_ms`, `edge_margin`, `compact_mode`, `glassmorphism_enabled`, `accent_color`, `click_through_when_idle`, `wake_ready_glow`, `drag_to_reposition`
- New field validators for `theme`, `position`, `opacity`

### Changed
- `UIConfig.width` default: 420 → 400; `height`: 600 → 580 (matches overlay layout)
- `spidy/ui/__init__.py`: now exports all public UI symbols (events, state, themes, hotkey)
- `spidy/ui/widgets/__init__.py`: exports all widget classes
- `spidy/ui/themes/__init__.py`: exports all theme types + factory

### Dependencies
- Added `PySide6>=6.7.0` (Qt6 Python bindings)
- Added `keyboard>=0.13.5` (global hotkey registration)

### Tests
- **183 new tests** across 4 test files
- `test_ui_events.py`: 40 tests — all 12 event types
- `test_ui_state.py`: 50 tests — state machine transitions, listeners, properties
- `test_ui_theme.py`: 45 tests — hex color validation, ThemeManager, toggle
- `test_ui_hotkey_config.py`: 22 tests — hotkey parse/validate, UIConfig validators
- `test_ui_overlay.py`: 50 tests — Qt widgets and OverlayWindow (offscreen platform)
- **Total: 568 tests passing**

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
