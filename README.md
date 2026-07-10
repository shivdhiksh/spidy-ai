# Spidy — AI Desktop Companion

> Your personal AI that lives inside Windows.

[![CI](https://github.com/yourusername/spidy/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/spidy/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Spidy?

Spidy is a production-quality, always-on AI Desktop Companion for Windows. Say **"Hey Spidy"** and a beautiful floating interface appears above every application — ready to help you with anything.

- 🎙️ Always-on wake word detection
- 🧠 Local AI reasoning (100% offline capable)
- 🗂️ File search, open, and management
- 🌐 Browser automation
- 💻 Coding assistance
- 👁️ Screen understanding & OCR
- 🔒 Permission-gated for safety

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- Windows 10/11

### Install

```powershell
# Clone the repository
git clone https://github.com/yourusername/spidy.git
cd spidy

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### Run

```powershell
python -m spidy.main
# or
spidy
```

---

## Development

```powershell
# Run tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=spidy --cov-report=term-missing

# Lint
ruff check spidy/ tests/

# Format
ruff format spidy/ tests/

# Type check
mypy spidy/ --ignore-missing-imports
```

---

## Project Structure

```
spidy/
├── core/          # Application bootstrap, event bus, lifecycle
├── config/        # YAML config + Pydantic validation
├── logging/       # Structured loguru logger
├── voice/         # Wake word, STT, TTS (Milestone 1)
├── conversation/  # Context, intent, response (Milestone 3)
├── reasoning/     # LLM client, planner, router (Milestone 3)
├── memory/        # Short/long-term memory (Milestone 6)
├── agents/        # Desktop, file, browser, screen agents
├── ui/            # PySide6 overlay interface (Milestone 2)
├── permissions/   # Permission tiers and audit log
└── plugins/       # Plugin discovery and sandboxing
```

---

## Milestone Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Foundation (config, logging, event bus) | ✅ In Progress |
| 1 | Voice Pipeline (wake word, STT, TTS) | 🔜 |
| 2 | Desktop Overlay UI | 🔜 |
| 3 | Reasoning & Conversation | 🔜 |
| 4 | Desktop & File Agents | 🔜 |
| 5 | Browser & Screen Agents | 🔜 |
| 6 | Memory Engine | 🔜 |
| 7 | Coding Agent | 🔜 |
| 8 | Vision & System Agents | 🔜 |
| 9 | Plugin System | 🔜 |
| 10 | Production Hardening | 🔜 |

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.

---

## License

MIT © Shiva
