"""
Working screens behind the dashboard's menu (Voice, Browser, Files, Apps, Automate, Code, Vision, Memory,
Settings).

Every button does something real: it either sends a normal request through Spidy's own pipeline (the same
path as speaking or typing it, so the reply shows up in Chat) or acts locally (open a folder, launch an app,
turn autostart on or off).
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from spidy.logging.logger import get_logger
from spidy.ui.dashboard import art

log = get_logger(__name__)

PAGE_W, PAGE_H = 1190.0, 650.0
HOME = Path.home()
_DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md", ".csv", ".py", ".png", ".jpg",
            ".jpeg", ".mp4", ".mp3", ".zip"}


def _user_folder(name: str) -> Path:
    """A known folder, preferring the OneDrive-redirected one when it exists."""
    for base in (HOME / "OneDrive", HOME):
        p = base / name
        if p.exists():
            return p
    return HOME / name


def recent_files(limit: int = 40) -> list[Path]:
    """Newest documents in Desktop / Documents / Downloads (two levels deep, no hidden or system folders)."""
    found: list[tuple[float, Path]] = []
    for name in ("Desktop", "Documents", "Downloads"):
        root = _user_folder(name)
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith((".", "$", "node_modules", "__pycache__", "venv", ".venv"))]
            if len(Path(dirpath).parts) - base_depth >= 2:
                dirs[:] = []
            for f in files:
                p = Path(dirpath) / f
                if p.suffix.lower() in _DOC_EXT:
                    try:
                        found.append((p.stat().st_mtime, p))
                    except OSError:
                        pass
    found.sort(key=lambda t: t[0], reverse=True)
    seen, out = set(), []
    for _, p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= limit:
            break
    return out


def open_path(path: str | Path) -> bool:
    try:
        os.startfile(str(path))                      # noqa: S606 -- the user's own file or folder
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not open {p}: {e}", p=path, e=exc)
        return False


class _Worker(QObject):
    """Runs a function on a thread and hands the result back on the Qt thread."""
    done = Signal(object)

    def run(self, fn) -> None:
        def go() -> None:
            try:
                self.done.emit(fn())
            except Exception as exc:  # noqa: BLE001
                log.warning("Page worker failed: {e}", e=exc)
                self.done.emit(None)
        threading.Thread(target=go, daemon=True, name="dashboard-page").start()


class FunctionPage(QWidget):
    """Card with a title, a one-line description and a body of real controls."""

    submit = Signal(str)          # a request for Spidy (shown in Chat, handled by the brain)
    toast = Signal(str)           # short feedback line
    navigate = Signal(str)

    title = ""
    blurb = ""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.k = 1.0
        self.setObjectName("fpage")
        self._root = QVBoxLayout(self)
        self._head = QLabel(self.title)
        self._head.setObjectName("title")
        self._sub = QLabel(self.blurb)
        self._sub.setObjectName("sub")
        self._root.addWidget(self._head)
        self._root.addWidget(self._sub)
        self.body = QVBoxLayout()
        self._root.addLayout(self.body, 1)
        self.build()

    # -- to override
    def build(self) -> None:
        pass

    def on_show(self) -> None:
        """Called each time the page is opened (refresh dynamic content)."""

    # -- helpers
    def say(self, text: str) -> None:
        """Send a request to Spidy and jump to Chat to see the answer."""
        self.submit.emit(text)

    def section(self, label: str) -> QLabel:
        w = QLabel(label.upper())
        w.setObjectName("section")
        self.body.addWidget(w)
        return w

    def tiles(self, items: list[tuple[str, str, object]], cols: int = 4) -> QGridLayout:
        """A grid of buttons: (label, hint, action) where action is a request string or a callable."""
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, (label, hint, action) in enumerate(items):
            b = QPushButton(f"{label}\n{hint}" if hint else label)
            b.setObjectName("tile")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(hint)
            b.clicked.connect((lambda _=False, a=action: a() if callable(a) else self.say(str(a))))
            grid.addWidget(b, i // cols, i % cols)
        self.body.addLayout(grid)
        return grid

    def entry(self, placeholder: str, button: str, on_go) -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        go = QPushButton(button)
        go.setObjectName("go")
        go.setCursor(Qt.CursorShape.PointingHandCursor)

        def fire() -> None:
            text = edit.text().strip()
            if text:
                on_go(text)
                edit.clear()
        edit.returnPressed.connect(fire)
        go.clicked.connect(fire)
        row.addWidget(edit, 1)
        row.addWidget(go)
        self.body.addLayout(row)
        return edit

    def set_k(self, k: float) -> None:
        self.k = k
        px = lambda v: max(8, int(v * k))   # noqa: E731
        m = px(34)
        self._root.setContentsMargins(m, px(26), m, px(26))
        self._root.setSpacing(px(10))
        self.body.setSpacing(px(12))
        self.setStyleSheet(f"""
            QLabel {{ color: #eceef4; font-family: 'Segoe UI'; font-size: {px(15)}px; background: transparent; }}
            QLabel#title {{ font-size: {px(30)}px; font-weight: 300; letter-spacing: {px(2)}px; }}
            QLabel#sub {{ color: #8f95a6; font-size: {px(15)}px; margin-bottom: {px(6)}px; }}
            QLabel#section {{ color: #ff4a52; font-size: {px(11)}px; letter-spacing: {px(3)}px; margin-top: {px(8)}px; }}
            QLabel#note {{ color: #8f95a6; font-size: {px(13)}px; }}
            QPushButton#tile {{ background: rgba(255,255,255,0.04); color: #eceef4; border: 1px solid rgba(255,255,255,0.10);
                border-radius: {px(14)}px; padding: {px(14)}px; font-family: 'Segoe UI'; font-size: {px(15)}px; text-align: left; }}
            QPushButton#tile:hover {{ border-color: #ff4a52; background: rgba(226,36,44,0.12); }}
            QPushButton#tile:pressed {{ background: rgba(226,36,44,0.25); }}
            QPushButton#go {{ background: #e2242c; color: white; border: none; border-radius: {px(12)}px;
                padding: {px(10)}px {px(26)}px; font-size: {px(15)}px; font-family: 'Segoe UI'; }}
            QPushButton#go:hover {{ background: #ff4a52; }}
            QLineEdit {{ background: rgba(255,255,255,0.05); color: #eceef4; border: 1px solid rgba(255,255,255,0.14);
                border-radius: {px(12)}px; padding: {px(10)}px {px(14)}px; font-size: {px(15)}px; font-family: 'Segoe UI';
                selection-background-color: #e2242c; }}
            QLineEdit:focus {{ border-color: #ff4a52; }}
            QListWidget {{ background: rgba(255,255,255,0.03); color: #eceef4; border: 1px solid rgba(255,255,255,0.10);
                border-radius: {px(12)}px; font-size: {px(14)}px; font-family: 'Segoe UI'; outline: 0; }}
            QListWidget::item {{ padding: {px(8)}px {px(12)}px; }}
            QListWidget::item:hover {{ background: rgba(226,36,44,0.14); }}
            QListWidget::item:selected {{ background: rgba(226,36,44,0.30); color: white; }}
            QScrollBar:vertical {{ background: transparent; width: {px(8)}px; }}
            QScrollBar::handle:vertical {{ background: rgba(255,255,255,0.22); border-radius: 3px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(self.k, self.k)
        art.draw_card(p, QRectF(1, 1, PAGE_W - 2, PAGE_H - 2), 28)


def _note(text: str) -> QLabel:
    w = QLabel(text)
    w.setObjectName("note")
    w.setWordWrap(True)
    return w


# ── Voice ────────────────────────────────────────────────────────────────────

class VoicePage(FunctionPage):
    title = "Voice"
    blurb = "Talk to Spidy with your voice. Say “Hey Spidy” or “Wake up Spidy” any time."

    def build(self) -> None:
        self.status = QLabel("")
        self.body.addWidget(self.status)
        self.section("Try it")
        self.tiles([
            ("Test my voice", "Spidy answers out loud", "Say hello and tell me one fun fact."),
            ("What can you do?", "Spidy lists her abilities", "What can you do for me?"),
            ("Tell me the time", "spoken answer", "What time is it?"),
            ("Read my notes", "spoken answer", "Read my notes."),
        ], cols=4)
        self.section("Wake words")
        self.body.addWidget(_note("“Hey Spidy” and “Wake up Spidy”. Spidy confirms the name was really said before she wakes, "
                                  "so background voices and TV are ignored."))
        self.section("This computer")
        self.mic = QLabel("")
        self.body.addWidget(self.mic)
        self.body.addStretch(1)

    def set_status(self, state: str) -> None:
        text = {"idle": "Waiting for “Hey Spidy”.", "wake_ready": "Waiting for “Hey Spidy”.", "listening": "Listening to you now…",
                "thinking": "Thinking…", "speaking": "Speaking…", "working": "Working on a task…",
                "error": "Something went wrong."}.get(state, "")
        self.status.setText(f"Right now: {text}")

    def on_show(self) -> None:
        try:
            import sounddevice as sd
            d = sd.query_devices(kind="input")
            self.mic.setText(f"Microphone: {d['name']}")
        except Exception:  # noqa: BLE001
            self.mic.setText("Microphone: not detected")


# ── Browser ──────────────────────────────────────────────────────────────────

class BrowserPage(FunctionPage):
    title = "Browser"
    blurb = "Open any site or search the web. Spidy opens it in your browser."

    SITES = [("YouTube", "open youtube"), ("Gmail", "open gmail"), ("Google", "open google"), ("GitHub", "open github"),
             ("WhatsApp Web", "open whatsapp web"), ("Google Maps", "open google maps"), ("ChatGPT", "open chatgpt"),
             ("LinkedIn", "open linkedin")]

    def build(self) -> None:
        self.entry("Type a website or what to search for…", "Go", self._go)
        self.section("Quick sites")
        self.tiles([(n, "", c) for n, c in self.SITES], cols=4)
        self.section("Ask Spidy")
        self.tiles([("Play music on YouTube", "", "Play some relaxing music on YouTube"),
                    ("Latest news", "", "Open today's top news"),
                    ("Weather", "", "What's the weather today?"),
                    ("Open a new tab", "", "Open a new browser tab")], cols=4)
        self.body.addStretch(1)

    def _go(self, text: str) -> None:
        looks_like_site = " " not in text and "." in text
        self.say(f"open {text}" if looks_like_site else f"search for {text}")


# ── Files ────────────────────────────────────────────────────────────────────

class FilesPage(FunctionPage):
    title = "Files"
    blurb = "Find a file by name or open your folders. Double-click a recent file to open it."

    def build(self) -> None:
        self.entry("Find a file, e.g. “DBMS assignment”…", "Find", lambda t: self.say(f"find my file {t}"))
        self.section("Folders")
        folders = [(n, "", (lambda n=n: self._folder(n))) for n in ("Desktop", "Documents", "Downloads", "Pictures")]
        self.tiles(folders, cols=4)
        self.section("Recently changed")
        self.list = QListWidget()
        self.list.itemActivated.connect(self._open_item)
        self.list.itemDoubleClicked.connect(self._open_item)
        self.body.addWidget(self.list, 1)
        self._worker = _Worker()
        self._worker.done.connect(self._fill)

    def _folder(self, name: str) -> None:
        if not open_path(_user_folder(name)):
            self.toast.emit(f"Could not open {name}")

    def on_show(self) -> None:
        self._worker.run(recent_files)

    def _fill(self, files) -> None:
        self.list.clear()
        for p in files or []:
            it = QListWidgetItem(f"{p.name}    —    {p.parent.name}")
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            self.list.addItem(it)
        if not files:
            self.list.addItem("No recent files found.")

    def _open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and not open_path(path):
            self.toast.emit("Could not open that file")


# ── Applications ─────────────────────────────────────────────────────────────

class AppsPage(FunctionPage):
    title = "Applications"
    blurb = "Everything installed on this computer. Double-click to launch."

    def build(self) -> None:
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter apps…")
        self.filter.textChanged.connect(self._apply_filter)
        self.body.addWidget(self.filter)
        self.list = QListWidget()
        self.list.itemActivated.connect(self._launch)
        self.list.itemDoubleClicked.connect(self._launch)
        self.body.addWidget(self.list, 1)
        self._apps: list = []
        self._worker = _Worker()
        self._worker.done.connect(self._fill)
        self._loading = False

    def on_show(self) -> None:
        if self._apps or self._loading:
            return
        self._loading = True
        self.list.addItem("Loading your apps…")

        def load():
            from spidy.skills.desktop.app_catalog import load_start_apps
            return load_start_apps()
        self._worker.run(load)

    def _fill(self, apps) -> None:
        self._loading = False
        self._apps = sorted(apps or [], key=lambda a: a.name.lower())
        self._apply_filter()

    def _apply_filter(self) -> None:
        q = self.filter.text().strip().lower()
        self.list.clear()
        for a in self._apps:
            if q in a.name.lower():
                it = QListWidgetItem(a.name)
                it.setData(Qt.ItemDataRole.UserRole, a)
                self.list.addItem(it)
        if self.list.count() == 0:
            self.list.addItem("No apps found." if self._apps else "Could not read the app list.")

    def _launch(self, item: QListWidgetItem) -> None:
        app = item.data(Qt.ItemDataRole.UserRole)
        if app is None:
            return
        try:
            from spidy.skills.desktop.app_catalog import launch_start_app
            launch_start_app(app)
            self.toast.emit(f"Opening {app.name}…")
        except Exception as exc:  # noqa: BLE001
            log.warning("Launch failed: {e}", e=exc)
            self.toast.emit(f"Could not open {app.name}")


# ── Automation ───────────────────────────────────────────────────────────────

class AutomationPage(FunctionPage):
    title = "Automation"
    blurb = "One click to set up a whole routine, or describe any task and Spidy will plan and do it."

    def build(self) -> None:
        self.entry("Describe a task, e.g. “organise my Downloads by file type”…", "Run", self.say)
        self.section("Routines")
        self.tiles([
            ("Work mode", "VS Code, Gmail, Calendar", "Open VS Code and open Gmail and open Google Calendar"),
            ("Study mode", "Notepad + lo-fi music", "Open notepad and play lofi study music on YouTube"),
            ("Chill mode", "Relaxing music", "Play relaxing music on YouTube"),
            ("News", "Today's top stories", "Open today's top news"),
            ("Clean desktop", "Tidy Desktop files", "Organise the files on my Desktop into folders by type"),
            ("Focus 25 min", "Timer + quiet", "Set a timer for 25 minutes"),
            ("Daily briefing", "Weather, time, news", "Give me a quick briefing: the time, the weather and the top news"),
            ("Battery & system", "Health check", "Show system info"),
        ], cols=4)
        self.body.addStretch(1)


# ── Code assistant ───────────────────────────────────────────────────────────

class CodePage(FunctionPage):
    title = "Code Assistant"
    blurb = "Ask Spidy to write, explain or fix code, or jump into your tools."

    def build(self) -> None:
        self.entry("What should Spidy build? e.g. “a Python program that renames photos by date”…", "Build",
                   lambda t: self.say(f"write a python program: {t}"))
        self.section("Start something")
        self.tiles([
            ("New Python script", "hello-world to start", "Create a python program that prints hello world"),
            ("Explain code", "paste or point to a file", "Explain the code in my clipboard"),
            ("Fix an error", "Spidy reads the error", "Help me fix the error I just got"),
            ("Write a function", "describe it, Spidy writes it", "Write a python function to sort a list of names"),
        ], cols=4)
        self.section("Tools")
        self.tiles([
            ("VS Code", "open the editor", "open vs code"),
            ("Terminal", "open a terminal", "open terminal"),
            ("Project folder", "open Spidy's folder", lambda: open_path(Path(__file__).resolve().parents[3])),
            ("GitHub", "open in browser", "open github"),
        ], cols=4)
        self.body.addStretch(1)


# ── Screen vision ────────────────────────────────────────────────────────────

class VisionPage(FunctionPage):
    title = "Screen Vision"
    blurb = "Spidy can look at your screen and tell you what is on it."

    def build(self) -> None:
        self.section("Ask about your screen")
        self.tiles([
            ("What's on my screen?", "Spidy describes it", "What is on my screen right now?"),
            ("Read the text", "reads visible text", "Read the text on my screen"),
            ("Summarise this page", "short summary", "Summarise what I am looking at"),
            ("Which app is open?", "active window", "Which app am I using right now?"),
        ], cols=4)
        self.section("Capture")
        self.tiles([
            ("Take a screenshot", "saved to Pictures", "take a screenshot"),
            ("Open screenshots", "folder", lambda: open_path(_user_folder("Pictures"))),
        ], cols=4)
        self.body.addWidget(_note("Screen understanding uses Spidy's vision model; it needs an internet connection."))
        self.body.addStretch(1)


# ── Memory ───────────────────────────────────────────────────────────────────

class MemoryPage(FunctionPage):
    title = "Memory"
    blurb = "Teach Spidy things about you and your work, and ask what she remembers."

    def build(self) -> None:
        self.entry("Teach Spidy something, e.g. “my favourite editor is VS Code”…", "Remember",
                   lambda t: self.say(f"remember that {t}"))
        self.section("Ask")
        self.tiles([
            ("What do you know about me?", "", "What do you remember about me?"),
            ("What did we talk about?", "", "What did we talk about earlier?"),
            ("Read my notes", "", "Read my notes."),
            ("Add a note", "", "Take a note"),
        ], cols=4)
        self.section("This session")
        self.history = QListWidget()
        self.body.addWidget(self.history, 1)

    def add_line(self, role: str, text: str) -> None:
        who = "You" if role.lower() in ("user", "you") else "Spidy"
        self.history.addItem(f"{who}:  {text}")
        if self.history.count() > 200:
            self.history.takeItem(0)
        self.history.scrollToBottom()


# ── Settings ─────────────────────────────────────────────────────────────────

class SettingsPage(FunctionPage):
    title = "Settings"
    blurb = "How Spidy starts, behaves and where her files live."

    always_on_top_changed = Signal(bool)
    quit_requested = Signal()

    def build(self) -> None:
        self.section("Behaviour")
        row = QGridLayout()
        row.setSpacing(12)
        self.autostart = self._toggle("Start Spidy when Windows starts", self._set_autostart)
        self.ontop = self._toggle("Keep Spidy above other windows", self.always_on_top_changed.emit)
        row.addWidget(self.autostart, 0, 0)
        row.addWidget(self.ontop, 0, 1)
        self.body.addLayout(row)
        self.section("Files")
        root = Path(__file__).resolve().parents[3]
        self.tiles([
            ("Open settings file", "spidy_config.yaml", lambda: open_path(root / "config" / "spidy_config.yaml")),
            ("Open logs folder", "see what Spidy did", lambda: open_path(Path(os.environ.get("APPDATA", str(HOME))) / "Spidy" / "logs")),
            ("Open Spidy folder", "project files", lambda: open_path(root)),
            ("Show system info", "RAM, CPU, battery", "show system info"),
        ], cols=4)
        self.section("Spidy")
        self.tiles([("Quit Spidy", "stop listening and close", self.quit_requested.emit)], cols=4)
        self.body.addWidget(_note("Changes to the settings file apply the next time Spidy starts."))
        self.body.addStretch(1)

    def _toggle(self, label: str, on_change) -> QPushButton:
        b = QPushButton()
        b.setObjectName("tile")
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b._label = label                                         # type: ignore[attr-defined]
        b.toggled.connect(lambda on, b=b: (self._paint_toggle(b), on_change(on)))
        self._paint_toggle(b)
        return b

    @staticmethod
    def _paint_toggle(b: QPushButton) -> None:
        b.setText(f"{b._label}\n{'ON' if b.isChecked() else 'OFF'}")        # type: ignore[attr-defined]

    def _set_autostart(self, on: bool) -> None:
        try:
            from spidy.desktop.autostart import disable_autostart, enable_autostart
            ok = enable_autostart(background=True) if on else disable_autostart()
            self.toast.emit(("Autostart on" if on else "Autostart off") if ok else "Could not change autostart")
        except Exception as exc:  # noqa: BLE001
            log.warning("Autostart change failed: {e}", e=exc)
            self.toast.emit("Could not change autostart")

    def set_always_on_top(self, on: bool) -> None:
        self.ontop.blockSignals(True)
        self.ontop.setChecked(on)
        self._paint_toggle(self.ontop)
        self.ontop.blockSignals(False)

    def on_show(self) -> None:
        try:
            from spidy.desktop.autostart import is_autostart_enabled
            on = bool(is_autostart_enabled())
        except Exception:  # noqa: BLE001
            on = False
        self.autostart.blockSignals(True)
        self.autostart.setChecked(on)
        self._paint_toggle(self.autostart)
        self.autostart.blockSignals(False)


PAGES: dict[str, type[FunctionPage]] = {
    "Voice": VoicePage, "Browser": BrowserPage, "Files": FilesPage, "Applications": AppsPage,
    "Automation": AutomationPage, "Code Assistant": CodePage, "Screen Vision": VisionPage,
    "Memory": MemoryPage, "Settings": SettingsPage,
}
