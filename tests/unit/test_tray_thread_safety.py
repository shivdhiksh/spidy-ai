"""
SystemTrayManager.set_overlay_visible is called from the asyncio thread (wake handlers)
and from the Qt thread (HUD restored). Rebuilding the tray menu off the Qt thread
corrupted the heap (0xc0000374) and killed Spidy right after "Hey Spidy".
"""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from spidy.ui.tray import SystemTrayManager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_menu_updates_only_run_on_the_qt_thread(qapp):
    tray = SystemTrayManager()
    seen: list[int] = []
    tray._update_menu = lambda: seen.append(threading.get_ident())  # type: ignore[method-assign]

    def worker(offset: int) -> None:
        for i in range(50):
            tray.set_overlay_visible(bool((i + offset) & 1))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    qapp.processEvents()

    assert seen, "expected at least one menu update"
    assert set(seen) == {threading.get_ident()}


def test_unchanged_visibility_does_not_rebuild_the_menu(qapp):
    tray = SystemTrayManager()
    calls: list[bool] = []
    tray._update_menu = lambda: calls.append(True)  # type: ignore[method-assign]

    tray.set_overlay_visible(True)   # already True by default
    qapp.processEvents()
    assert calls == []

    tray.set_overlay_visible(False)
    qapp.processEvents()
    assert calls == [True]
    assert tray._overlay_visible is False
