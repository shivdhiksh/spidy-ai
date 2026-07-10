"""
Pytest shared fixtures and configuration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

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
