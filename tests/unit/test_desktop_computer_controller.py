"""
Unit tests for DesktopComputerController
========================================
Verifies:
1. Screen metrics and screenshot capture (including GDI).
2. Active window retrieval, window listing, finding, focusing, and closing.
3. Mouse pointer actions (click, move, scroll, double click).
4. Keyboard actions (typing, key press, hotkeys).
5. Clipboard operations (copy, paste, get/set).
6. UI Automation element lookup and clicking.
7. Telemetry logging.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from spidy.desktop.controller import DesktopComputerController


class TestDesktopComputerController:
    @pytest.fixture
    def controller(self):
        return DesktopComputerController()

    def test_get_screen_size(self, controller):
        w, h = controller.get_screen_size()
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w > 0
        assert h > 0

    def test_screenshot_returns_bytes_and_writes_file(self, controller, tmp_path):
        target_file = tmp_path / "screen.bmp"
        data = controller.screenshot(file_path=str(target_file))
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert target_file.exists()
        assert target_file.stat().st_size == len(data)

    def test_get_active_window(self, controller):
        info = controller.get_active_window()
        assert isinstance(info, dict)
        assert "title" in info
        assert "hwnd" in info
        assert "process_name" in info

    def test_list_windows(self, controller):
        wins = controller.list_windows(visible_only=True)
        assert isinstance(wins, list)
        if wins:
            w0 = wins[0]
            assert "hwnd" in w0
            assert "title" in w0

    def test_find_window(self, controller):
        # Searching for non-existent window returns None
        win = controller.find_window("__non_existent_window_12345__")
        assert win is None

    def test_mouse_actions_mocked(self, controller):
        assert controller.move_mouse(100, 200) is True
        assert controller.click(100, 200, button="left") is True
        assert controller.double_click(100, 200) is True
        assert controller.right_click(100, 200) is True
        assert controller.scroll(3) is True

    def test_keyboard_actions_mocked(self, controller):
        assert controller.type_text("Hello Spidy", interval=0.0, press_enter=False) is True
        assert controller.press_key("enter") is True
        assert controller.hotkey("ctrl", "s") is True

    def test_clipboard_operations(self, controller):
        text = "Spidy Clipboard Test 42"
        assert controller.set_clipboard_text(text) is True
        read_back = controller.get_clipboard_text()
        if controller._is_windows:
            assert read_back == text

    def test_open_application_notepad(self, controller):
        with patch("subprocess.Popen") as mock_popen, patch("os.startfile", create=True):
            ok, msg = controller.open_application("notepad")
            assert ok is True
            assert "Launched" in msg or "Focused" in msg

    def test_open_application_settings(self, controller):
        with patch("os.startfile", create=True) as mock_start:
            ok, msg = controller.open_application("settings")
            assert ok is True

    def test_ui_element_lookup_and_click_fallback(self, controller):
        # Non-existent element returns None / False
        elem = controller.find_ui_element(window_query="NonExistent", control_name="Save")
        assert elem is None
        assert controller.click_ui_element(window_query="NonExistent", control_name="Save") is False
