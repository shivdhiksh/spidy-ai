"""
The dashboard window must be a drop-in replacement for the old HUD and every control must
do what its label says. Offscreen Qt: no display needed (fonts render as boxes there, which
is irrelevant to behaviour).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spidy.ui.dashboard import DashboardWindow  # noqa: E402
from spidy.ui.dashboard.base import REF_W  # noqa: E402
from spidy.ui.dashboard.panels import DOCK, DOCK_X, NAV_ITEMS, QUICK, SUGGESTIONS  # noqa: E402
from spidy.ui.dashboard.stats import pretty_cpu, pretty_gpu  # noqa: E402

OVERLAY_API = [
    "set_state", "add_message", "update_waveform", "show_notification", "update_task_progress",
    "show_confirmation", "hide_confirmation", "apply_theme", "show_animated", "hide_animated",
    "minimize_hud", "restore_hud", "show_hud", "hide_hud", "toggle_hud", "_ensure_foreground",
    "record_valid_geometry", "ensure_valid_geometry",
]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    w = DashboardWindow(animate=False, weather=(False, 0.0, 0.0), city="Testville")
    w.stats.stop()
    yield w
    w.stats.stop()
    w.close()


def click(widget, ref_x: float, ref_y: float) -> None:
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(int(ref_x * widget.k), int(ref_y * widget.k)))


def test_is_a_drop_in_replacement_for_the_old_hud(win):
    for name in OVERLAY_API:
        assert callable(getattr(win, name)), name
    for sig in ("mic_button_clicked", "overlay_closed", "hud_minimized", "hud_restored", "command_submitted"):
        assert hasattr(win, sig), sig
    card = win._confirmation_card
    assert hasattr(card, "confirmed") and hasattr(card, "cancelled")


def test_navigation_switches_pages(win):
    win.show_page("Chat")
    assert not win.hero.isVisibleTo(win) and win.chat.isVisibleTo(win)
    win.show_page("Memory")
    assert win.pages["Memory"].isVisibleTo(win) and not win.chat.isVisibleTo(win)
    win.show_page("Home")
    assert win.hero.isVisibleTo(win) and not win.chat.isVisibleTo(win) and not any(p.isVisibleTo(win) for p in win.pages.values())


def test_clicking_a_nav_item_opens_its_page(win):
    idx = [n for n, _ in NAV_ITEMS].index("Chat")
    click(win.rail, 120, 150 + idx * 62 + 27)
    assert win.rail.current == "Chat" and win.chat.isVisibleTo(win)


@pytest.mark.parametrize("idx", range(6))
def test_dock_buttons_send_their_commands(win, idx):
    got: list[str] = []
    win.command_submitted.connect(got.append)
    cmd = DOCK[idx][2]
    click(win.dock, DOCK_X[idx] + 60, 36)
    if cmd.startswith("__page__:"):
        assert got == [] and win._page == cmd.split(":", 1)[1]
    else:
        assert got == [cmd]


def test_mic_button_is_the_microphone(win):
    hits: list[int] = []
    win.mic_button_clicked.connect(lambda: hits.append(1))
    click(win.voice, 420, 160)
    assert hits == [1]


def test_suggestions_run_a_command_or_open_a_page(win):
    got: list[str] = []
    win.command_submitted.connect(got.append)
    for i, (_icon, _label, cmd) in enumerate(SUGGESTIONS):
        click(win.sugg, 120, 40 + i * 48 + 23)
        if cmd.startswith("__page__:"):
            assert win._page == cmd.split(":", 1)[1]
            win.show_page("Home")
        else:
            assert got[-1] == cmd


def test_risky_quick_actions_need_confirmation(win):
    got: list[str] = []
    win.command_submitted.connect(got.append)
    click(win.quick, 35, 90)                       # Shutdown
    assert win.confirm.isVisibleTo(win) and got == []
    click(win.confirm, REF_W / 2 + 105, 587)       # Cancel
    assert got == [] and not win.confirm.isVisibleTo(win)
    click(win.quick, 35, 90)
    click(win.confirm, REF_W / 2 - 105, 587)       # Confirm
    assert got == [QUICK[0][2]]


def test_text_mode_types_commands(win):
    got: list[str] = []
    win.command_submitted.connect(got.append)
    click(win.voice, 560, 160)                     # Text mode pill
    assert win.voice.mode == "text" and win.input.isVisibleTo(win)
    win.input.setText("open notepad")
    QTest.keyClick(win.input, Qt.Key.Key_Return)
    assert got == ["open notepad"] and win.input.text() == ""
    click(win.voice, 280, 160)                     # back to Voice mode
    assert win.voice.mode == "voice" and not win.input.isVisibleTo(win)


def test_state_updates_voice_hero_and_status(win):
    win.set_state("listening")
    assert win.voice.state == "listening" and win.hero.state == "listening" and win.header.message == "Listening…"
    win.set_state("thinking")
    assert win.header.message == "Thinking…"
    win.set_state("idle")
    assert win.header.message.startswith("Ready")


def test_hero_eyes_are_brighter_when_listening_than_idle(win):
    win.set_state("idle")
    idle = win.hero.current_level()
    win.set_state("listening")
    assert win.hero.current_level() > idle + 0.25


def test_assistant_replies_show_in_status_chat_and_recent(win):
    win.add_message("user", "open vs code")
    win.add_message("assistant", "Opened vs code.")
    assert win.header.message == "Opened vs code."
    plain = win.chat.view.toPlainText()
    assert "Opened vs code." in plain and "open vs code" in plain
    assert [m for _r, m in win.recent.items] == ["open vs code", "Opened vs code."]


def test_task_progress_is_shown_and_cleared(win):
    win.update_task_progress("Open apps", [("Open VS Code", "done"), {"description": "Open WhatsApp", "status": "running"}])
    assert win.sugg.goal == "Open apps" and len(win.sugg.steps) == 2
    win.set_state("idle")
    assert win.sugg.goal == ""


def test_hide_and_restore_emit_the_signals_the_tray_listens_to(win):
    events: list[str] = []
    win.hud_minimized.connect(lambda: events.append("min"))
    win.hud_restored.connect(lambda: events.append("res"))
    win.restore_hud()
    assert win.isVisible()
    win.minimize_hud()
    assert not win.isVisible() and win._is_minimized_to_tray
    win.toggle_hud()
    assert win.isVisible()
    assert events == ["res", "min", "res"]


def test_has_a_taskbar_button_not_a_tool_window(win):
    flags = win.windowFlags()
    assert (flags & Qt.WindowType.WindowType_Mask) != Qt.WindowType.Tool   # Tool windows skip the taskbar
    assert flags & Qt.WindowType.WindowMinimizeButtonHint
    assert not win.windowIcon().isNull()


def test_minimize_button_goes_to_the_taskbar_and_wake_restores_it(win):
    events: list[str] = []
    win.hud_minimized.connect(lambda: events.append("min"))
    win.hud_restored.connect(lambda: events.append("res"))
    win.restore_hud()
    win.buttons.minimize.emit()
    assert win.isMinimized() and win.isVisible()      # still there, parked on the taskbar
    assert events[-1] == "min"

    win.restore_hud()                                  # a wake word
    assert not win.isMinimized() and win.isVisible() and win.windowOpacity() > 0.95
    assert not win._is_minimized_to_tray
    assert events[-1] == "res"


def test_close_button_closes_the_window_but_spidy_keeps_listening(win):
    closed, quit_asked = [], []
    win.overlay_closed.connect(lambda: closed.append("closed"))
    win.quit_requested.connect(lambda: quit_asked.append("quit"))
    win.restore_hud()
    win.buttons.close.emit()
    assert closed == ["closed"] and not win.isVisible()
    assert quit_asked == []                       # the program is still running
    win.restore_hud()                              # "Hey Spidy" brings the window back
    assert win.isVisible()


def test_idle_hide_still_tucks_the_hud_away_without_a_taskbar_button(win):
    win.restore_hud()
    win.buttons.minimize.emit()
    win.minimize_hud()                                 # e.g. the idle timeout
    assert not win.isVisible() and not win.isMinimized()


def test_live_stats_update_the_panels_and_build_sparkline_history(win):
    sample = {"cpu": 18.0, "gpu": 32.0, "ram": 54.0, "vram": 20.0, "online": False, "host": "box",
              "specs": {"cpu": ("Intel Core i5", "12th Gen")}, "weather": ("24°C", "Clear")}
    win._on_stats(sample)
    win._on_stats({**sample, "cpu": 25.0})
    assert win.stats_row.vals["CPU".lower()] == 25.0
    assert list(win.stats_row.hist["cpu"]) == [18.0, 25.0]
    assert win.stats_row.online is False and win.stats_row.host == "box"
    assert win.header.weather == ("24°C", "Clear")
    assert win.stats_row._spec("cpu") == "Intel Core i5 12th Gen"


def test_missing_gpu_shows_a_dash_not_a_crash(win):
    win._on_stats({"cpu": 5.0, "gpu": None, "ram": 40.0, "vram": None, "online": True, "host": "h", "specs": {}})
    assert win.stats_row.vals["gpu"] is None
    win.stats_row.grab()


@pytest.mark.parametrize("raw, expected", [
    ("12th Gen Intel(R) Core(TM) i5-12450H", ("Intel Core i5", "12th Gen")),
    ("AMD Ryzen 7 7435HS with Radeon Graphics", ("AMD Ryzen 7", "")),
])
def test_cpu_names_are_shortened(raw, expected):
    assert pretty_cpu(raw) == expected


def test_gpu_names_are_shortened():
    assert pretty_gpu("NVIDIA GeForce RTX 3050 Laptop GPU", 4096) == ("NVIDIA RTX 3050", "4GB VRAM")


def test_every_screen_paints_without_error(win):
    for page in ("Home", "Chat", "Voice"):
        win.show_page(page)
        win.grab()
    for state in ("idle", "listening", "thinking", "speaking", "working", "error"):
        win.show_page("Home")
        win.set_state(state)
        win.hero.grab()


# ── the menu screens are real ────────────────────────────────────────────────

def test_every_menu_item_has_a_working_screen(win):
    from spidy.ui.dashboard.panels import NAV_ITEMS
    for name, _icon in NAV_ITEMS:
        win.show_page(name)
        if name in ("Home", "Chat"):
            continue
        assert name in win.pages, name
        shown = [n for n, p in win.pages.items() if p.isVisibleTo(win)]
        assert shown == [name]


def test_page_buttons_send_requests_to_spidy_and_open_chat(win):
    from PySide6.QtWidgets import QPushButton
    sent: list[str] = []
    win.command_submitted.connect(sent.append)
    for name in ("Browser", "Automation", "Code Assistant", "Screen Vision", "Memory", "Voice"):
        win.show_page(name)
        before = len(sent)
        tile = next(b for b in win.pages[name].findChildren(QPushButton) if b.objectName() == "tile")
        tile.click()
        assert len(sent) == before + 1 and sent[-1].strip(), name
        assert win._page == "Chat" and win.rail.current == "Chat"


def test_entry_boxes_turn_text_into_the_right_request(win):
    from PySide6.QtWidgets import QLineEdit
    sent: list[str] = []
    win.command_submitted.connect(sent.append)

    def type_into(page, text):
        win.show_page(page)
        edit = win.pages[page].findChildren(QLineEdit)[0]
        edit.setText(text)
        edit.returnPressed.emit()

    type_into("Browser", "github.com")
    type_into("Browser", "best pizza near me")
    type_into("Files", "DBMS assignment")
    type_into("Code Assistant", "a number guessing game")
    type_into("Memory", "my favourite editor is VS Code")
    assert sent == ["open github.com", "search for best pizza near me", "find my file DBMS assignment",
                    "write a python program: a number guessing game", "remember that my favourite editor is VS Code"]


def test_files_and_apps_screens_list_real_things(win, monkeypatch, tmp_path):
    from spidy.ui.dashboard import pages
    doc = tmp_path / "Desktop" / "report.docx"
    doc.parent.mkdir()
    doc.write_text("x")
    monkeypatch.setattr(pages, "_user_folder", lambda name: tmp_path / name)
    names = [p.name for p in pages.recent_files()]
    assert names == ["report.docx"]

    opened: list[str] = []
    monkeypatch.setattr(pages, "open_path", lambda p: opened.append(str(p)) or True)
    page = win.pages["Files"]
    page._fill(pages.recent_files())
    page.list.itemActivated.emit(page.list.item(0))
    assert opened == [str(doc)]

    from types import SimpleNamespace
    apps = win.pages["Applications"]
    apps._fill([SimpleNamespace(name="Notepad"), SimpleNamespace(name="Calculator")])
    assert [apps.list.item(i).text() for i in range(apps.list.count())] == ["Calculator", "Notepad"]
    apps.filter.setText("note")
    assert [apps.list.item(i).text() for i in range(apps.list.count())] == ["Notepad"]


def test_settings_toggles_do_real_things(win, monkeypatch):
    import spidy.desktop.autostart as auto
    calls: list[str] = []
    monkeypatch.setattr(auto, "enable_autostart", lambda background=True, custom_cmd=None: calls.append("on") or True)
    monkeypatch.setattr(auto, "disable_autostart", lambda: calls.append("off") or True)
    monkeypatch.setattr(auto, "is_autostart_enabled", lambda: False)
    page = win.pages["Settings"]
    page.on_show()
    assert not page.autostart.isChecked()
    page.autostart.click()
    page.autostart.click()
    assert calls == ["on", "off"]

    win.pages["Settings"].ontop.click()                       # turn "above other windows" off
    assert not win.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    closed: list[str] = []
    win.quit_requested.connect(lambda: closed.append("quit"))
    page.quit_requested.emit()
    assert closed == ["quit"]


def test_voice_screen_follows_spidys_state(win):
    win.set_state("listening")
    assert "Listening" in win.pages["Voice"].status.text()
    win.add_message("user", "hello")
    assert win.pages["Memory"].history.count() == 1


def test_face_is_the_rotating_dot_sphere_inside_hud_rings(win):
    win.show()
    win.set_state("listening")
    first = win.hero.grab().toImage()
    win.hero._phase += 1.5                                  # the sphere turns
    second = win.hero.grab().toImage()
    assert first != second
    cyan = sum(1 for x in range(0, first.width(), 8) for y in range(0, first.height(), 8)
               if first.pixelColor(x, y).blue() > 110 and first.pixelColor(x, y).red() < 140)
    assert cyan > 40                                        # real light in the sphere and its rings
    win.set_state("error")                                  # errors turn the rings amber
    amber = win.hero.grab().toImage()
    assert any(amber.pixelColor(x, y).red() > 200 and amber.pixelColor(x, y).green() > 130 and amber.pixelColor(x, y).blue() < 110
               for x in range(0, amber.width(), 4) for y in range(0, amber.height(), 4))
