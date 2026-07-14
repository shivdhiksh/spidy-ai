# Spidy — Product Roadmap

> **Spidy is a Lifelong AI Companion for Windows.**  
> Not just an assistant. A system that knows you, grows with you, and lives beside you.

---

## Current Status

| Milestone | Title | Status | Tests |
|-----------|-------|--------|-----------|
| **M0** | Foundation (config, logging, event bus, core) | ✅ Complete | 59 |
| **M1** | Voice Pipeline (wake word, STT, TTS) | ✅ Complete | 43 |
| **M2** | Context Observer Layer | ✅ Complete | 43 |
| **M3** | Brain Core (intent, conversation, decision, planning) | ✅ Complete | 106 |
| **M4** | Multi-LLM + Skills Runtime | ✅ Complete | 112 |
| **M5** | Desktop Overlay UI | ✅ Complete | 57 |
| **M6** | Desktop & File Agents | ✅ Complete | 236 |
| **M7** | Browser Agent (Playwright) | ✅ Complete | 914 |
| **M8** | Memory Engine (Working/Episodic/Semantic) | ✅ Complete | 1049 |
| **M9** | Vision & Screen Understanding | ✅ Complete | 1229 |
| **M10** | Knowledge Engine | ✅ Complete | 1531 |
| **M11** | Learning Engine | ✅ Complete | 141 |
| **M12** | Plugin Marketplace | 🔜 Planned | — |
| **M13** | Developer Mode | 🔜 Planned | — |
| **M14** | Deployment & Distribution | 🔜 Planned | — |
| **v1.0** | Spidy v1.0 — Public Release | 🎯 Target | — |
| **v2.0** | Spidy v2.0 — Full Companion | 🎯 Target | — |

---

## Milestone Detail

---

### ✅ M0 — Foundation
**Complete**

- `EventBus` — async pub/sub with thread-safe publishing
- `ConfigManager` — YAML + Pydantic validation + env overrides
- `LoggingManager` — structured loguru logger
- `SpidyCore` — application lifecycle (CREATED → RUNNING → STOPPED)
- `DeviceManager` — CUDA/GPU detection

---

### ✅ M1 — Voice Pipeline
**Complete**

- Wake word detection (`openwakeword`)
- Speech-to-Text (`faster-whisper`)
- Text-to-Speech (`piper-tts`)
- `VoiceEngine` state machine (sleeping → detecting → listening → processing → speaking)
- `AudioCaptureEngine` with pyaudio callback dispatch
- `SkillRegistry` — capability catalog for all skills

---

### ✅ M2 — Context Observer Layer
**Complete**

- `ActiveWindowObserver` — foreground app/window tracking (win32gui)
- `ClipboardObserver` — clipboard change detection (privacy-safe)
- `ProcessObserver` — process lifecycle monitoring (psutil)
- `SystemResourceObserver` — CPU/RAM/battery/GPU/disk with hysteresis alerts
- `DownloadObserver` — watchdog filesystem monitoring
- `NotificationObserver` — stub (activated M8+)
- `ObserverManager` — lifecycle orchestrator + snapshot aggregator
- `DesktopStateSnapshot` — immutable aggregate of all observer states

---

### ✅ M3 — Brain Core
**Complete** · Lifelong AI Companion kernel established

- `IntentClassifier` — heuristic classifier (15+ categories, entity extraction)
- `ConversationManager` — rolling context window, session lifecycle
- `DecisionEngine` — intent → SKILL / LLM_DIRECT / CLARIFY / REJECT
- `Planner` — decision → ordered plan
- `ToolRouter` — plan execution via SkillRegistry or LLM
- `OllamaClient` — stdlib-based Ollama backend (zero extra dependencies)
- `BrainConfig` — companion feature flags for all future systems
- **Three Lifelong AI Companion interface contracts** (ABCs):
  - `MemoryInterface` (M6)
  - `KnowledgeInterface` (M9)
  - `LearningInterface` (M10)

---

### 🔜 M4 — Multi-LLM + Skills Runtime
**Next milestone**

**Goal:** Replace heuristic intelligence with real LLM reasoning and build the first useful skill set.

**Multi-LLM Backends**
- `OpenAIClient` — OpenAI API (GPT-4o, GPT-4o-mini)
- `ClaudeClient` — Anthropic Claude API (claude-3.5-sonnet, claude-3-haiku)
- `GeminiClient` — Google Gemini API (gemini-1.5-flash, gemini-1.5-pro)
- `OllamaClient` — enhanced with streaming and model management
- `LLMRouter` — automatic failover across providers

**LLM-Based Intent Classifier**
- Replace heuristic classifier with LLM-based classification
- Structured output (JSON schema) for entity extraction
- Confidence scoring from model logprobs

**Built-in Skills (Tier 0 — No Permission Required)**
- `HelpSkill` — list capabilities
- `TimeSkill` — current time and date
- `GreetSkill` — personalised greetings
- `RepeatSkill` — repeat last response

**Built-in Skills (Tier 1 — User Confirmation)**
- `AppLaunchSkill` — open applications by name
- `NoteSkill` — take, read, and search notes
- `TimerSkill` — set timers and alarms
- `VolumeSkill` — system volume control
- `ClipboardSkill` — read/write clipboard

**Executor + Verifier**
- Multi-step skill chaining
- Skill result verification and retry logic
- Permission tier enforcement

---

### 🔜 M5 — Desktop Overlay UI
**Goal:** A beautiful, always-available floating interface.

- PySide6 Qt6 overlay window (always-on-top, transparent)
- Glassmorphism design with dark/light theme
- Animated conversation bubbles
- Hotkey activation (`Ctrl+Space` default, configurable)
- Waveform visualiser during listening
- Skill progress indicators
- Settings panel (theme, hotkey, LLM provider, privacy)
- System tray integration with context menu
- Smooth fade-in / fade-out animations

---

### 🔜 M6 — Memory Engine
**Goal:** Spidy remembers. Every session. Forever.

This milestone implements the `MemoryInterface` contract established in M3.

**Short-Term Memory**
- Rolling conversation buffer (in-memory, fast)
- Session scoping and isolation
- Automatic expiry

**Long-Term Episodic Memory**
- SQLite-backed store (`spidy_memory.db`)
- Every interaction logged with timestamp, intent, entities, result
- Tag-based retrieval
- Memory pruning and consolidation

**Semantic Memory (Vector Search)**
- ChromaDB vector store
- Sentence-Transformers embeddings (`all-MiniLM-L6-v2`)
- Similarity search over all past interactions
- Memory injection into LLM context window

**Memory-Augmented Brain**
- `recall()` called during `DecisionEngine.decide()` for context enrichment
- Most relevant past interactions surfaced per query
- "Remember when you said..." capability

---

### 🔜 M7 — Desktop & File Agents
**Goal:** Spidy acts on your computer.

**File Agent**
- Open, read, rename, move, delete files
- File search (filename, content, type, recency)
- Directory listing and navigation
- Archive operations (zip, extract)

**App Agent**
- Launch any installed application
- Focus running application windows
- Close applications (with confirmation)
- List running processes

**System Agent**
- Volume, brightness, display controls
- Power management (sleep, lock, restart — T2 permission)
- Clipboard read/write
- Screenshot capture

**Permission Tiers**
- T0 — No confirmation (read, list, non-destructive)
- T1 — One-time confirmation (write, launch)
- T2 — Always confirm (delete, shutdown, send)
- T3 — Explicit unlock required (admin operations)

---

### 🔜 M8 — Vision & Screen Understanding
**Goal:** Spidy sees your screen.

- Screenshot capture (`mss`) — ultra-fast full-screen and region capture
- OCR — on-screen text extraction (`easyocr`)
- Object detection — UI element identification (`opencv-python`)
- Active window content extraction
- "What's on my screen?" natural language query
- Screen-based context injection into Brain
- `NotificationObserver` — full WinRT implementation for Windows toast notifications

---

### ✅ M9 — Vision & Screen Understanding
**Complete**

- `ScreenshotEngine` — `mss`-based ultra-fast capture: full-screen, active-window, region, multi-monitor
- `OCREngine` — on-screen text extraction (`easyocr`) with confidence scoring and lazy model init
- `ScreenAnalyzer` — active app + visible window detection (`win32gui`/`psutil`), OpenCV UI region detection
- `VisionManager` — unified async Vision API implementing `VisionInterface`
- `VisionInterface` ABC added to `brain/interfaces.py`
- `Brain.vision` companion slot wired in SpidyCore step 7
- All deps (`mss`, `easyocr`, `opencv-python`) optional — graceful degrade when absent
- `describe_screen()` — human-readable screen state for LLM context injection
- 5 EventBus events under `vision.*` namespace
- `VisionConfig` + `SpidyConfig.vision` field + `config/spidy_config.yaml` `vision:` section
- **180 new tests** (1229 total)

---

### 🔜 M10 — Knowledge Engine
**Goal:** Spidy knows your documents and the web.

This milestone implements the `KnowledgeInterface` contract established in M3.

**Personal Document Knowledge Base**
- PDF ingestion and chunking (`pypdf`)
- Word document ingestion (`python-docx`)
- Markdown and plain text ingestion
- Web page ingestion via URL
- Automatic re-indexing on file change

**Knowledge Graph**
- Entity extraction from documents
- Relationship mapping between entities
- "What do my documents say about X?" queries
- Source attribution for every answer

**RAG Pipeline**
- Chunk → Embed → Store → Retrieve → Inject
- Retrieved context injected into LLM prompt
- Hallucination reduction through grounded retrieval
- Configurable retrieval depth (top-k chunks)

**Optional Web Search**
- DuckDuckGo / Brave Search API integration
- User-controlled (disabled by default)
- Results summarised and attributed
- Privacy mode — no search history retained

---

### ✅ M11 — Learning Engine
**Complete** · Spidy learns who you are and what you like.

This milestone implements the `LearningInterface` contract established in M3.

**Preference Learning**
- Implicit signals: app-usage patterns automatically decoded to preference keys
- Explicit feedback: confidence boosted/decayed via FeedbackProcessor
- Preference model per category (browser, IDE, media, etc.)
- Preferences stored locally in SQLite (`spidy_learning.db`) or in-memory fallback
- Privacy-preserving: never leaves the device

**Habit Detection**
- `HabitDetector` — trigger/action pair counting with configurable threshold
- Time-of-day and app-context-aware habit keys
- Promoted to stable `Habit` objects once `min_observations` is reached
- Persistent via SQLite with in-memory fallback
- EventBus events: `learning.habit_detected`, `learning.habit_suggested`

**Workflow Learning**
- `WorkflowLearner` — multi-step action sequence learning with sliding window
- `_extract_subsequences()` — enumerates all (min_len..max_steps) subsequences
- Sequences promoted to stable `Workflow` objects after `min_observations`
- `suggest_next_step()` — proactively suggests the next action in a known workflow
- EventBus events: `learning.workflow_detected`, `learning.workflow_suggested`

**Feedback Processing**
- `FeedbackProcessor` — explicit and implicit signal ingestion
- Positive feedback → boosts confidence on tagged preferences
- Negative feedback → decays confidence on tagged preferences
- `sentiment_by_tag()` — aggregate sentiment query for any tag
- EventBus event: `learning.feedback_recorded`

**Data Types (all immutable dataclasses)**
- `Preference` — key/value with confidence, source, and metadata
- `Habit` — trigger+action with observed_count and stable hash ID
- `Workflow` — step sequence with trigger_count and stable hash ID
- `FeedbackSignal` — utterance/response/rating with tag classification
- `LearningContext` — ambient context snapshot with `time_bucket` property

**Configuration & Integration**
- `LearningConfig` — full Pydantic config model in `SpidyConfig.learning`
- `LearningManager` — unified async API implementing `LearningInterface`
- Wired as **step 9** in `SpidyCore._initialise()`, after KnowledgeManager
- Passed to `Brain` as `learning=` companion slot
- Graceful degradation: disabled at runtime if initialization fails
- **141 new tests** (1672 total)

---

### 🔜 M11 — Browser Agent
**Goal:** Spidy controls your browser.

- Playwright-based browser automation
- Navigate to URLs, click elements, fill forms
- Extract page content for Knowledge Engine ingestion
- "Open this link", "scroll down", "click submit"
- Web scraping for structured data extraction
- Session management (login state preservation)
- Permission-gated: T1 for navigation, T2 for form submission

---

### 🔜 M12 — Plugin Marketplace
**Goal:** Anyone can extend Spidy.

**Plugin Architecture**
- Plugin discovery via `pluggy` or custom entry-points
- Sandboxed execution (restricted imports, timeouts)
- Plugin manifest (`plugin.yaml`) — name, version, author, permissions required
- Hot-load/unload without restart

**Built-in Plugin Types**
- `SkillPlugin` — adds new actions to the SkillRegistry
- `ObserverPlugin` — adds new context observers
- `LLMPlugin` — adds new LLM backends
- `UIPlugin` — adds panels or widgets to the overlay

**Marketplace**
- Plugin registry (GitHub-based, curated)
- One-command install: `spidy plugin install <name>`
- Version pinning and update notifications
- Community ratings and reviews (roadmap)

**Developer SDK**
- `spidy-sdk` Python package
- Scaffold generator: `spidy plugin new <name>`
- Local plugin testing harness
- Plugin documentation generator

---

### 🔜 M13 — Developer Mode
**Goal:** Spidy becomes a developer's power tool.

**Coding Assistant**
- Active file awareness (editor integration via Language Server Protocol)
- Code explanation, review, and improvement suggestions
- Docstring and comment generation
- Test generation for selected functions
- Refactoring suggestions

**Git Integration**
- Commit message generation from staged diff
- PR description generation
- Branch management commands
- "What changed in the last commit?" query

**Terminal Integration**
- Command suggestion and explanation
- Error diagnosis ("why did this command fail?")
- Shell script generation
- Command history analysis

**IDE Extensions** *(roadmap)*
- VS Code extension
- JetBrains plugin

---

### 🔜 M14 — Deployment & Distribution
**Goal:** Spidy ships to real users.

**Windows Installer**
- WiX / NSIS installer with all dependencies bundled
- Auto-start on Windows login (optional)
- Silent update mechanism
- Uninstall with complete cleanup

**Model Management**
- Bundled `llama3.2:3b` (default, runs on any machine)
- Optional larger models via `spidy models install <name>`
- GPU detection and model routing (CPU vs CUDA)
- Model storage in `%APPDATA%/Spidy/models`

**Crash Reporting** *(opt-in)*
- Sentry integration (opt-in, anonymised)
- Local crash log always generated
- "Send report" dialog on crash

**Auto-Update**
- GitHub Releases-based update channel
- `stable`, `beta`, and `nightly` channels
- Staged rollout support

**Performance**
- Cold start < 3 seconds (model pre-warming)
- < 2% CPU at idle
- < 200 MB RAM baseline
- < 4 GB RAM with full model loaded

**Privacy Audit**
- Zero network calls without explicit user action
- All AI inference runs locally by default
- Data residency audit: all files in `%APPDATA%/Spidy`
- Compliance documentation (GDPR baseline)

---

## Version Targets

---

### 🎯 Spidy v1.0 — "The Companion"
**Target:** After M10 (Learning Engine)  
**Theme:** A genuinely useful, personalized AI that runs 100% locally.

**What v1.0 delivers:**
- Wake word → full voice pipeline
- Brain with heuristic + LLM-based classification
- Desktop and file control (M7 Skills)
- Conversation memory across sessions (M6)
- Personal document knowledge base and RAG (M9)
- Habit and preference learning (M10)
- Beautiful desktop overlay (M5)
- Ollama (local) + optional cloud LLM (M4)
- Windows installer with auto-update (subset of M14)

**v1.0 is the first version recommended for daily use.**  
It is private, local-first, and genuinely learns who you are.

---

### 🎯 Spidy v2.0 — "The Platform"
**Target:** After M14 (Deployment)  
**Theme:** An extensible AI platform. Build anything on Spidy.

**What v2.0 adds over v1.0:**
- Vision and screen understanding (M8)
- Browser agent (M11)
- Developer Mode — coding assistant, Git, terminal (M13)
- Plugin Marketplace — community-extensible (M12)
- Multi-LLM with intelligent routing and failover (M4 enhanced)
- Full deployment stack — installer, auto-update, crash reporting (M14)
- Windows 10 and 11 certified
- Plugin SDK published to PyPI

**v2.0 is the platform release.**  
Third-party developers can build and publish Spidy plugins.  
Spidy becomes whatever the user needs it to be.

---

## Architecture at v2.0

```
Spidy v2.0 Architecture
─────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │  Input Layer                                                │
  │  VoiceEngine (wake word → STT)    HotkeyEngine (Ctrl+Space) │
  │  UI Overlay (text input)          Browser Extension (M11)   │
  └──────────────────────┬──────────────────────────────────────┘
                         │ utterance / command
  ┌──────────────────────▼──────────────────────────────────────┐
  │  Brain Core (M3)                                            │
  │  IntentClassifier → DecisionEngine → Planner → ToolRouter  │
  │                                                             │
  │  Companion Slots:                                           │
  │  ┌──────────────┐ ┌─────────────────┐ ┌──────────────────┐ │
  │  │ MemoryEngine │ │ KnowledgeEngine │ │ LearningEngine   │ │
  │  │     (M6)     │ │      (M9)       │ │      (M10)       │ │
  │  └──────────────┘ └─────────────────┘ └──────────────────┘ │
  └──────────────────────┬──────────────────────────────────────┘
                         │ plan steps
  ┌──────────────────────▼──────────────────────────────────────┐
  │  Skills & Agents (M4, M7, M11, M12, M13)                   │
  │  FileAgent   AppAgent   BrowserAgent   VisionAgent          │
  │  CodeAgent   PluginSkills...                                │
  └──────────────────────┬──────────────────────────────────────┘
                         │ skill results
  ┌──────────────────────▼──────────────────────────────────────┐
  │  LLM Layer (M4)                                             │
  │  OllamaClient  OpenAIClient  ClaudeClient  GeminiClient     │
  │  LLMRouter (failover + cost routing)                        │
  └──────────────────────┬──────────────────────────────────────┘
                         │ response text
  ┌──────────────────────▼──────────────────────────────────────┐
  │  Output Layer                                               │
  │  TTSEngine (piper)    UI Overlay (PySide6)                  │
  │  BrainResponseReadyEvent → all subscribers                  │
  └─────────────────────────────────────────────────────────────┘
                         │ (always underneath everything)
  ┌─────────────────────────────────────────────────────────────┐
  │  EventBus — the nervous system                              │
  │  ContextObserverLayer — ambient awareness                   │
  │  PermissionManager — safety gating                          │
  └─────────────────────────────────────────────────────────────┘
```

---

## Design Principles (All Milestones)

1. **Local-First** — All AI inference runs locally by default. Cloud providers are opt-in.
2. **Privacy by Design** — No data leaves the device without explicit user action.
3. **Graceful Degradation** — Missing optional components never crash Spidy.
4. **EventBus Decoupling** — Every module communicates only through typed events.
5. **Interface-First** — Future systems are defined as ABCs before they are built.
6. **Testable Everything** — No test requires a real OS call, LLM, or hardware.
7. **Permission-Gated** — Every destructive or network action requires the correct tier.
8. **Zero Surprise Installs** — Core Spidy runs with only `pydantic`, `loguru`, `psutil`.

---

## Non-Goals (Intentionally Out of Scope)

- **Mobile** — Windows only. macOS/Linux are not in scope for v1.0 or v2.0.
- **Cloud Sync** — User data stays local. No account required, ever.
- **Training User Data** — Spidy's models are never trained on user data externally.
- **Browser as Primary UI** — Spidy is a native Windows app, not a web app.
- **Autonomous Agent** — Spidy asks before acting. No unsupervised background tasks.

---

*Last updated: 2026-07-11 · Milestone 3 complete · Next: Milestone 4*
