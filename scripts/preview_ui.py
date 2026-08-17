"""
preview_ui.py -- Spidy HUD visual state preview (M17.2)
=======================================================
Cycles through all 8 UI states without touching Brain/Voice/Agent.

Usage:
    py scripts/preview_ui.py

Sequence:
    1. SLEEPING (idle)
    2. WAKE BOOT  (wake_ready)
    3. LISTENING
    4. THINKING
    5. SPEAKING
    6. WORKING (with task progress)
    7. CONFIRMATION
    8. Back to SLEEPING

Each state shown for 2.5 seconds.
"""

from __future__ import annotations

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from spidy.ui.overlay import OverlayWindow
from spidy.ui.themes.dark import DARK_THEME


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Spidy HUD Preview")
    app.setQuitOnLastWindowClosed(True)

    win = OverlayWindow(DARK_THEME, animate=False)
    win.show()

    _STATES = [
        ("idle",        _noop,                  "SLEEPING"),
        ("wake_ready",  _noop,                  "WAKE BOOT"),
        ("listening",   _setup_listening,       "LISTENING"),
        ("thinking",    _noop,                  "THINKING"),
        ("speaking",    _setup_speaking,        "SPEAKING"),
        ("working",     _setup_working,         "WORKING (autonomous)"),
        ("working",     _setup_confirmation,    "CONFIRMATION"),
        ("idle",        _noop,                  "Back to SLEEPING"),
    ]

    idx = [0]

    def advance():
        if idx[0] >= len(_STATES):
            app.quit()
            return
        state, setup_fn, label = _STATES[idx[0]]
        print(f"[HUD Preview] -> {label}")
        win.set_state(state)
        setup_fn(win)
        idx[0] += 1
        QTimer.singleShot(2500, advance)

    QTimer.singleShot(600, advance)
    sys.exit(app.exec())


def _noop(win): pass


def _setup_listening(win: OverlayWindow) -> None:
    win.update_waveform([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.5, 0.3])
    win.add_message("user", "What is artificial intelligence?")


def _setup_speaking(win: OverlayWindow) -> None:
    win.update_waveform([0.6, 0.8, 0.9, 0.7, 0.8, 0.6, 0.5, 0.7])
    win.add_message(
        "assistant",
        "Artificial intelligence is the simulation of human intelligence in machines "
        "programmed to think and learn like humans.",
    )


def _setup_working(win: OverlayWindow) -> None:
    from spidy.ui.widgets.hud_task_console import TaskStep
    win.update_task_progress(
        "Upload my demo video to YouTube",
        [
            TaskStep("Find demo video",            "done"),
            TaskStep("Open browser",               "done"),
            TaskStep("Navigate to YouTube Studio", "done"),
            TaskStep("Upload video file",          "running"),
            TaskStep("Set title and description",  "pending"),
            TaskStep("Publish",                    "pending"),
        ],
    )


def _setup_confirmation(win: OverlayWindow) -> None:
    win.show_confirmation(
        task_id="preview-task-1",
        goal_id="preview-goal-1",
        description="Upload demo video to YouTube",
        prompt="This action will make the video publicly visible. Confirm to proceed.",
    )


if __name__ == "__main__":
    main()
