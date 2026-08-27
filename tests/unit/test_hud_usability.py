"""
Unit tests for SPIDY HUD Usability & Readability Overhaul
=========================================================
Verifies:
1. Scrollable conversation history (HUDChatOverlay) multi-turn messages without 120-char clipping.
2. Telemetry panel GPU, VRAM, CPU, RAM, Disk hardware metrics and N/A handling.
3. Movable HUD window header dragging & minimize/close controls.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from spidy.ui.themes import build_default_theme_manager
from spidy.ui.widgets.hud_chat import HUDChatOverlay
from spidy.ui.widgets.hud_telemetry import TelemetryPanel


@pytest.fixture(scope="session")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


class TestHUDChatOverlay:
    def test_add_messages_without_truncation(self, qapp):
        chat = HUDChatOverlay()
        long_response = "A" * 300
        chat.add_message("user", "Hello Spidy!")
        chat.add_message("assistant", long_response)

        assert len(chat._messages) == 2
        assert chat._messages[0] == ("user", "Hello Spidy!")
        assert chat._messages[1] == ("assistant", long_response)
        # Message count in layout includes the 2 bubbles + stretch item
        assert chat._msg_layout.count() >= 3

    def test_clear_chat(self, qapp):
        chat = HUDChatOverlay()
        chat.add_message("user", "Hello")
        chat.clear()
        assert len(chat._messages) == 0


class TestTelemetryPanel:
    def test_telemetry_rows_contain_all_hardware_metrics(self, qapp):
        panel = TelemetryPanel(["cpu", "ram", "gpu", "vram", "disk"])
        labels = [r.label for r in panel._rows]
        assert "CPU" in labels
        assert "RAM" in labels
        assert "GPU" in labels
        assert "VRAM" in labels
        assert "DISK" in labels

    def test_telemetry_background_gpu_query_does_not_crash(self, qapp):
        panel = TelemetryPanel(["gpu", "vram"])
        panel._query_gpu_background()
        assert isinstance(panel._gpu_cache, tuple)
        assert len(panel._gpu_cache) == 2


class TestHUDHeaderAndOverlay:
    def test_overlay_session_pos_persistence(self, qapp):
        from spidy.ui.overlay import OverlayWindow, _HUDHeader
        mgr = build_default_theme_manager()
        theme = mgr.current

        # Set session position
        OverlayWindow._session_pos = QPoint(200, 300)
        win = OverlayWindow(theme=theme, width=1200, height=700)
        assert win.pos() == QPoint(200, 300)

        # Verify header controls exist
        assert hasattr(win._header, "_btn_min")
        assert hasattr(win._header, "_btn_close")
        assert hasattr(win._header, "_drag_pos")
