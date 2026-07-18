"""
Pytest shared fixtures and configuration.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import yaml

# ─── Qt offscreen platform ────────────────────────────────────────────────────
# Must be set before ANY PySide6 / Qt import anywhere in the test process.
# Placing it here (conftest.py at the tests/ root) guarantees it is set
# before any test module is collected or imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ─── ChromaDB / PostHog telemetry ────────────────────────────────────────────
# ChromaDB initiates a PostHog telemetry network call on first client
# construction. In CI / automated tests this causes the test to hang until the
# connection times out (~30 s) and may interfere with event-loop teardown.
# Setting ANONYMIZED_TELEMETRY=false disables it before any import.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

# ─── Async test support ───────────────────────────────────────────────────────

# pytest-asyncio is configured via pyproject.toml (asyncio_mode = "auto")


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_config_file(tmp_path: Path) -> Path:
    """Return path to a minimal valid config file."""
    cfg = {"app": {"user_name": "TestUser"}}
    path = tmp_path / "spidy_config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def empty_config_file(tmp_path: Path) -> Path:
    """Return path to an empty config file (all defaults apply)."""
    path = tmp_path / "spidy_config.yaml"
    path.write_text("", encoding="utf-8")
    return path
