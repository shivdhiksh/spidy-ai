# Spidy — Your AI Desktop Companion

> Say **"Hey Spidy"**. She wakes up, listens, talks back, and does things on your Windows PC.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-informational.svg)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

Spidy is an always-on, voice-first desktop assistant for Windows. A tiny wake-word model listens in the
background; when you say "Hey Spidy" (or "Wake up Spidy") the dashboard appears, she answers "Yes boss.", and
you can talk to her or type. She opens apps and sites, finds your files, writes code, looks at your screen and
plans multi-step tasks, and she speaks her answers in a natural female American voice.

---

## Features

| Area | What Spidy does |
|---|---|
| **Voice** | Wake word ("Hey Spidy" / "Wake up Spidy"), speech-to-text, spoken replies, hands-free conversation that stays open between questions |
| **Wake-word accuracy** | Two-stage detection: a small ONNX model fires, then Whisper confirms the name "Spidy" was really said, so TV and background voices are ignored |
| **Apps & websites** | "Open VS Code and WhatsApp", "open YouTube": launches the real app and tells you honestly if it could not |
| **Files** | Find and open files by name, recent-files list, quick folders |
| **Browser** | Open sites, search, play YouTube, browser automation |
| **Code assistant** | Write, explain and fix code; open your editor and terminal |
| **Screen vision** | "What's on my screen?", read text, summarise what you are looking at |
| **Autonomous tasks** | Plans multi-step goals, executes them, verifies the result, and asks before anything risky |
| **Memory** | Remembers things you teach her and what you talked about |
| **Dashboard** | A dark HUD: rotating light-sphere face that reacts to her state, live CPU/GPU/RAM/network, weather, and a working screen for every menu item |
| **Always on, light on resources** | Starts with Windows, lives in the tray, loads the heavy speech models only when the wake word starts to sound and frees them when idle |
| **Safe** | Permission tiers, confirmation for dangerous actions, no fake "done" messages |

---

## How it works

```
 microphone ──► wake-word model ──► Whisper confirms "Spidy" ──► dashboard appears, "Yes boss."
                                                                      │
                  speech-to-text (faster-whisper) ◄───────────────────┘
                              │
                              ▼
        intent classifier ──► Brain ──► planner ──► skills / agent ──► result
                              │                          (apps, files, browser, code, vision)
                              ▼
                  text-to-speech (Piper) ──► speakers
```

* **LLM:** NVIDIA NIM (primary) with OpenRouter fallback; the wake word, speech-to-text and text-to-speech all run
  locally.
* **UI:** PySide6 (Qt). The dashboard talks to the core through a thread-safe event bus.
* **Core:** asyncio event bus, voice engine, brain, autonomous agent, skills registry, plugin system.

---

## Requirements

* Windows 10 / 11
* Python 3.11 or later (tested with 3.11 and 3.13)
* A microphone and speakers
* Optional: an NVIDIA GPU (CUDA) makes speech recognition faster; everything also runs on CPU
* Optional: an NVIDIA NIM and/or OpenRouter API key for the language model

## Install

```powershell
git clone https://github.com/shivdhiksh/spidy-ai.git
cd spidy-ai

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Download the voice models

The speech and wake-word models are not stored in git (they are large). Download them once:

```powershell
python scripts/setup_models.py --tts --voice en_US-lessac-high   # Spidy's voice (female, American English)
python scripts/setup_models.py --stt                              # speech recognition
python scripts/setup_models.py --wake-word                        # openWakeWord feature models
```

The two small custom wake-word models ("Hey Spidy" and "Wake up Spidy", about 10 KB) are included in\nssets/models/wake_word/. To retrain them: python scripts/build_spidy_wake_models.py.\n\n### Add your API keys

```powershell
copy .env.example .env      # then edit .env
```

Put your `NVIDIA_API_KEY` and/or `OPENROUTER_API_KEY` in `.env`. The file is git-ignored; never commit it.

## Run

```powershell
python -m spidy.main                 # start Spidy with the dashboard
python -m spidy.main --background    # start hidden; the dashboard appears when you say "Hey Spidy"
python -m spidy.main --text-mode     # type instead of speak
python -m spidy.main --quit          # ask a running Spidy to close
```

### Start with Windows

```powershell
python -m spidy.main --install-autostart     # also: --status-autostart, --uninstall-autostart
```

### Using the window

| Button | What it does |
|---|---|
| **–** (minimize) | Sends Spidy to the taskbar; she keeps listening |
| **X** | Closes the window and stops speech and running tasks; she keeps listening for "Hey Spidy" |
| **Settings → Quit Spidy** (or tray → Quit) | Closes Spidy completely |

The menu on the left has a working screen for **Voice, Browser, Files, Apps, Automate, Code, Vision, Memory** and
**Settings**; every button either sends a request to Spidy (the answer appears in Chat) or acts directly.

---

## Configuration

Everything is in [`config/spidy_config.yaml`](config/spidy_config.yaml). The most useful settings:

| Setting | Meaning |
|---|---|
| `voice.wake_word.threshold` | Wake-word sensitivity |
| `voice.wake_word.verify_with_stt` | Whisper confirms the name before waking (recommended) |
| `voice.tts.voice` | Piper voice (default `en_US-lessac-high`) |
| `voice.wake_ack.phrases` | What she says when she wakes (default "Yes boss.") |
| `voice.stt.idle_unload_seconds` | Free the speech model after this much quiet (0 = keep loaded) |
| `voice.barge_in.interrupt_on_speech` | Experimental: talk over her to interrupt (off by default; unreliable on laptop speakers). "Spidy stop" always works |
| `ui.always_on_top`, `ui.weather_city` | Window behaviour and the weather shown in the header |

---

## Project structure

```
spidy/
├── core/          # app bootstrap, event bus, lifecycle, watchdog, single-instance guard
├── voice/         # session manager, barge-in, interruption, echo canceller, transcript guards
├── perception/    # microphone capture, wake word + verifier, STT (faster-whisper), TTS (Piper), context observers
├── brain/         # intent classification, planning, conversation, self-commands
├── agent/         # autonomous goal planner, executor, verifier, code agent
├── skills/        # desktop (apps, files, system), browser, and the tool registry
├── browser/       # browser automation backends
├── vision/        # screen understanding
├── memory/ knowledge/ learning/   # what Spidy remembers and learns
├── llm/           # NVIDIA NIM / OpenRouter clients and router
├── permissions/   # permission tiers and audit log
├── plugins/       # plugin discovery and sandboxing
├── ui/            # PySide6 dashboard, tray, global hotkey, notifications
│   └── dashboard/ # window, panels, the light-sphere face, working menu pages
└── desktop/       # autostart and desktop controller
config/            # spidy_config.yaml, system prompt
scripts/           # model download, wake-word training, live verification scripts
tests/             # unit, integration and acceptance tests
```

## Development

```powershell
pytest tests/unit -q                       # fast unit tests (offscreen Qt: set QT_QPA_PLATFORM=offscreen)
pytest tests/unit --cov=spidy              # with coverage
ruff check spidy tests                     # lint
```

Known issues: `tests/unit/test_browser_agent.py` has failing tests (an outdated fake backend), and one
integration acceptance test (`test_acceptance_03`) fails on a "save it to to" phrasing bug.

## Troubleshooting

* **"Hey Spidy" does not respond:** check the log in `%APPDATA%\Spidy\logs`. Wake attempts rejected by the
  Whisper check are logged as `[WAKE VERIFY] rejected ... heard='...'` with what the microphone heard.
* **High RAM:** the speech models load when the wake word starts to sound and unload after
  `voice.stt.idle_unload_seconds` of quiet.
* **Spidy is not running after you pressed Settings → Quit:** that fully closes her; start her again (or restart
  Windows if autostart is on).

## License

MIT © Shiva
