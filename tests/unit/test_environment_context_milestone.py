"""
Tests for Environment Context Snapshot and Collector (Phase A)
"""

import pytest
from unittest.mock import MagicMock, patch
from spidy.perception.context.environment_context import (
    EnvironmentContext,
    EnvironmentContextCollector,
)


@pytest.mark.asyncio
async def test_environment_context_collection():
    collector = EnvironmentContextCollector()
    snap = await collector.collect(include_elements=False)
    assert isinstance(snap, EnvironmentContext)
    assert snap.timestamp is not None
    assert snap.platform in ("win32", "windows", "linux", "darwin")
    assert isinstance(snap.open_windows, list)
    assert isinstance(snap.running_apps, list)


@pytest.mark.asyncio
async def test_environment_context_formatting():
    collector = EnvironmentContextCollector()
    snap = await collector.collect(include_elements=False)
    summary = snap.format_summary()
    assert "Active Window:" in summary
    assert "Screen Size:" in summary


def test_environment_context_app_running():
    snap = EnvironmentContext(
        running_apps=["notepad.exe", "msedge.exe", "code.exe"],
        active_window={"title": "Untitled - Notepad", "process_name": "notepad.exe"},
    )
    assert snap.is_app_running("notepad") is True
    assert snap.is_app_running("edge") is True
    assert snap.is_app_running("calculator") is False
    assert snap.is_active_window("notepad") is True
    assert snap.is_active_window("calculator") is False
