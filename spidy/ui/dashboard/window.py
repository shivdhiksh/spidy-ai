"""
DashboardWindow: the Spidy home screen (minimal near-black design with a 3D spider face).

It exposes the same interface as the old ``OverlayWindow`` (set_state, add_message,
update_waveform, show_confirmation, restore_hud, ...), so SpidyApp, the tray, the hotkey and
the brain bridge keep working unchanged. Typed/clicked commands leave through
``command_submitted`` and are routed to the brain like spoken ones.
"""

from __future__ import annotations

import html
import platform

from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QPropertyAnimation, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QTextBrowser, QWidget

from spidy.logging.logger import get_logger
from spidy.ui.dashboard import art, palette as pal
from spidy.ui.dashboard.base import REF_H, REF_W, RefWidget, text
from spidy.ui.dashboard.icons import draw_icon
from spidy.ui.dashboard.panels import (
    CENTER, LIGHT, MEDIUM, NAV_ITEMS, CommandInput, DockRow, HeaderBar, HeroHead, NavRail, QuickRow,
    RecentCard, StatsRow, SuggestionList, VoiceControls, WindowButtons, caps,
)
from spidy.ui.dashboard.pages import PAGES, MemoryPage, SettingsPage, VoicePage
from spidy.ui.dashboard.stats import StatsWorker
from spidy.ui.widgets.hud_confirmation import HUDConfirmation

log = get_logger(__name__)

# Reference-pixel position (x, y) of each panel; sizes come from the widgets themselves.
LAYOUT = {
    "rail": (40, 60), "header": (320, 56), "hero": (300, 150), "voice": (300, 560), "dock": (300, 764),
    "sugg": (1180, 170), "quick": (1180, 520), "recent": (1180, 690), "stats": (300, 840), "buttons": (1432, 34),
}
INPUT_RECT = (430.0, 566.0, 580.0, 44.0)
PAGE_RECT = (300.0, 170.0, 1190.0, 650.0)

PAGE_BLURBS = {
    "Voice": "Microphone, wake-word sensitivity and voice settings.",
    "Browser": "Open tabs, history and browser automation.",
    "Files": "Search and manage files and folders.",
    "Applications": "Installed and running applications.",
    "Automation": "Build and run automations and routines.",
    "Code Assistant": "Write, run and fix code in a project.",
    "Screen Vision": "See and understand what is on your screen.",
    "Memory": "What Spidy remembers about you and your work.",
    "Settings": "Appearance, voice, privacy and behaviour.",
}

STATUS_DEFAULT = {
    "idle": "Ready for your command…", "wake_ready": "Ready for your command…",
    "listening": "Listening…", "thinking": "Thinking…", "speaking": "Speaking…",
    "working": "Working on it…", "error": "Something went wrong",
}


class ChatPage(QWidget):
    """Scrollable conversation history."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.k = 1.0
        self.view = QTextBrowser(self)
        self.view.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.view.setOpenExternalLinks(False)
        self._blocks: list[str] = []

    def set_k(self, k: float) -> None:
        self.k = k
        px = max(11, int(18 * k))
        self.view.setStyleSheet(
            f"QTextBrowser {{ background: transparent; color: #eceef4; font-family: 'Segoe UI'; font-size: {px}px;"
            f" padding: 8px; selection-background-color: #e2242c; }}"
            f"QScrollBar:vertical {{ background: transparent; width: {max(6, int(8 * k))}px; }}"
            f"QScrollBar::handle:vertical {{ background: rgba(255,255,255,0.22); border-radius: 3px; min-height: 30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}")
        self.view.setGeometry(int(30 * k), int(24 * k), int((PAGE_RECT[2] - 60) * k), int((PAGE_RECT[3] - 48) * k))

    def add(self, role: str, msg: str) -> None:
        user = role.lower() in ("user", "you")
        who, color = ("YOU", "#9aa0b0") if user else ("SPIDY", "#ff4a52")
        self._blocks.append(
            f"<p style='margin:12px 0 2px 0'><span style='color:{color};font-weight:500;letter-spacing:3px;font-size:11px'>{who}</span></p>"
            f"<p style='margin:0 0 10px 0;color:#eceef4'>{html.escape(msg).replace(chr(10), '<br>')}</p>")
        self.view.setHtml("".join(self._blocks[-200:]))
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(self.k, self.k)
        art.draw_card(p, QRectF(1, 1, PAGE_RECT[2] - 2, PAGE_RECT[3] - 2), 28)


class ConfirmOverlay(RefWidget):
    """'Are you sure?' card for risky quick actions (shutdown, restart...)."""

    confirmed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(REF_W, REF_H, parent)
        self.title, self.body, self.command = "", "", ""
        self._hover = ""
        self.hide()

    def ask(self, title: str, body: str, command: str) -> None:
        self.title, self.body, self.command = title, body, command
        self.show()
        self.raise_()
        self.update()

    def _buttons(self) -> dict[str, QRectF]:
        return {"ok": QRectF(REF_W / 2 - 200, 560, 190, 54), "no": QRectF(REF_W / 2 + 10, 560, 190, 54)}

    def mouseMoveEvent(self, e) -> None:
        pos = self.ref_pos(e)
        hover = next((k for k, r in self._buttons().items() if r.contains(pos)), "")
        if hover != self._hover:
            self._hover = hover
            self.update()

    def mousePressEvent(self, e) -> None:
        pos = self.ref_pos(e)
        for key, r in self._buttons().items():
            if r.contains(pos):
                self.hide()
                (self.confirmed.emit(self.command) if key == "ok" else self.cancelled.emit())
                return

    def paintEvent(self, _e) -> None:
        p = self.begin()
        p.fillRect(QRectF(0, 0, REF_W, REF_H), QColor(0, 0, 0, 190))
        card = QRectF(REF_W / 2 - 290, 390, 580, 250)
        p.setBrush(QBrush(QColor(12, 12, 16, 250)))
        p.setPen(QPen(pal.with_alpha(pal.RED_BRIGHT, 180), 1.3))
        p.drawRoundedRect(card, 26, 26)
        text(p, QRectF(card.left(), 416, card.width(), 50), self.title, pal.font(30, LIGHT), pal.TEXT, CENTER)
        text(p, QRectF(card.left() + 36, 470, card.width() - 72, 70), self.body, pal.font(16), pal.MUTED, CENTER)
        for key, r in self._buttons().items():
            ok = key == "ok"
            p.setBrush(QBrush(pal.with_alpha(pal.RED, 150 if ok else 0)))
            p.setPen(QPen(pal.RED_BRIGHT if ok else pal.HAIR_STRONG, 1.8 if self._hover == key else 1.2))
            p.drawRoundedRect(r, 27, 27)
            caps(p, r, "Confirm" if ok else "Cancel", 13, pal.TEXT, 3.0, CENTER)


class DashboardWindow(QWidget):
    _session_pos: QPointF | None = None

    mic_button_clicked = Signal()
    overlay_closed = Signal()
    quit_requested = Signal()           # really quit Spidy (Settings > Quit); the X button only closes the window
    hud_minimized = Signal()
    hud_restored = Signal()
    command_submitted = Signal(str)

    def __init__(self, theme=None, width: int = 0, height: int = 0, edge: str = "center",
                 always_on_top: bool = True, animate: bool = True,
                 weather: tuple[bool, float, float] = (False, 0.0, 0.0), city: str = "") -> None:
        super().__init__(None)
        self._animate = animate
        self._is_minimized_to_tray = False
        self._drag: QPointF | None = None
        self._page = "Home"
        self._state = "idle"
        self._toast = ""
        self._bg: QPixmap | None = None
        self._anim: QPropertyAnimation | None = None

        # A normal top-level window (not Qt.Tool) so Spidy has a taskbar button while it is open and
        # can be minimized to it; the minimize hint keeps the taskbar click-to-minimize working.
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowMinimizeButtonHint
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setWindowTitle("Spidy")
        self.setWindowIcon(self._make_icon())

        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        w = min(1500, int(avail.width() * 0.94)) if avail else 1440
        h = int(w * REF_H / REF_W)
        if avail and h > avail.height() * 0.95:
            h = int(avail.height() * 0.95)
            w = int(h * REF_W / REF_H)
        self._w = max(960, w)
        self._h = int(self._w * REF_H / REF_W)
        self.setFixedSize(self._w, self._h)
        self.k = self._w / REF_W

        self.hero = HeroHead(self)
        self.header = HeaderBar(city or "", self)
        self.status = self.header                      # status line lives in the header
        self.rail = NavRail(self)
        self.nav = self.rail
        self.voice = VoiceControls(self)
        self.dock = DockRow(self)
        self.sugg = SuggestionList(self)
        self.quick = QuickRow(self)
        self.recent = RecentCard(self)
        self.stats_row = StatsRow(self)
        self.buttons = WindowButtons(self)
        self.input = CommandInput(self)
        self.chat = ChatPage(self)
        self.pages = {name: cls(self) for name, cls in PAGES.items()}     # every menu screen is real
        self.chat.hide()
        for page in self.pages.values():
            page.hide()
            page.submit.connect(self._page_submit)
            page.toast.connect(lambda t: self.show_notification("", t))
        self.confirm = ConfirmOverlay(self)
        self._confirmation_card = HUDConfirmation(self)
        self._home = [self.hero, self.voice, self.dock, self.sugg, self.quick, self.recent]
        self._layout()

        self.rail.navigate.connect(self.show_page)
        settings = self.pages["Settings"]
        settings.always_on_top_changed.connect(self._set_always_on_top)
        settings.quit_requested.connect(self.quit_requested)
        settings.set_always_on_top(always_on_top)
        self.voice.mic_clicked.connect(self.mic_button_clicked)
        self.voice.mode_changed.connect(self._on_mode)
        self.dock.command.connect(self._on_dock)
        self.sugg.command.connect(self._on_dock)
        self.input.submitted.connect(self.command_submitted)
        self.quick.command.connect(self._on_quick)
        self.confirm.confirmed.connect(self.command_submitted)
        self.buttons.minimize.connect(self.minimize_to_taskbar)
        self.buttons.close.connect(self._on_close)

        self.stats = StatsWorker(weather=weather)
        self.stats.updated.connect(self._on_stats)
        self.stats.start()

        self._place_initial()
        self.setWindowOpacity(0.0)

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._clear_toast)

    # ── layout ───────────────────────────────────────────────────────────
    def _layout(self) -> None:
        k = self.k
        items = {"rail": self.rail, "header": self.header, "hero": self.hero, "voice": self.voice,
                 "dock": self.dock, "sugg": self.sugg, "quick": self.quick, "recent": self.recent,
                 "stats": self.stats_row, "buttons": self.buttons}
        for key, widget in items.items():
            x, y = LAYOUT[key]
            widget.set_k(k)
            widget.setGeometry(round(x * k), round(y * k), round(widget.ref_w * k) + 1, round(widget.ref_h * k) + 1)
        ix, iy, iw, ih = INPUT_RECT
        self.input.setGeometry(round(ix * k), round(iy * k), round(iw * k), round(ih * k))
        self.input.apply_scale(k)
        x, y, w, h = PAGE_RECT
        for page in (self.chat, *self.pages.values()):
            page.setGeometry(round(x * k), round(y * k), round(w * k), round(h * k))
        self.chat.set_k(k)
        for page in self.pages.values():
            page.set_k(k)
        self.confirm.setGeometry(0, 0, self._w, self._h)
        self.confirm.set_k(k)
        self._confirmation_card.setGeometry(round(self._w * 0.3), round(self._h * 0.3), round(self._w * 0.4), round(self._h * 0.4))

    def _place_initial(self) -> None:
        if DashboardWindow._session_pos is not None:
            self.move(DashboardWindow._session_pos.toPoint())
            return
        screen = QGuiApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.move(g.x() + (g.width() - self._w) // 2, g.y() + (g.height() - self._h) // 2)

    def paintEvent(self, _e) -> None:
        dpr = self.devicePixelRatioF()
        if self._bg is None or self._bg.devicePixelRatio() != dpr:
            self._bg = art.make_background(self._w, self._h, dpr, self.k)
        p = QPainter(self)
        p.drawPixmap(0, 0, self._bg)

    # ── interaction ──────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition() - QPointF(self.frameGeometry().topLeft())

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move((e.globalPosition() - self._drag).toPoint())

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.minimize_hud()
        else:
            super().keyPressEvent(e)

    def _on_mode(self, mode: str) -> None:
        on = mode == "text"
        self.input.setVisible(on)
        if on:
            self.input.setFocus()
        self.voice.update()

    def _on_dock(self, cmd: str) -> None:
        if cmd.startswith("__page__:"):
            name = cmd.split(":", 1)[1]
            self.show_page(name)
            self.rail.current = name
            self.rail.update()
        else:
            self.command_submitted.emit(cmd)

    def _on_quick(self, cmd: str, label: str) -> None:
        self.confirm.ask(f"{label}?", f"Spidy will {label.lower()} this computer. Save your work first.", cmd)

    def _on_close(self) -> None:
        """X closes the window: Spidy stops talking and drops running tasks but keeps listening for "Hey Spidy"."""
        self.record_valid_geometry()
        self.hide()
        self.overlay_closed.emit()

    @staticmethod
    def _make_icon() -> QIcon:
        """The spider face as a taskbar / window icon."""
        pm = QPixmap(64, 64)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        art.draw_head(p, 32, 30, 0.085, level=0.9)
        p.end()
        return QIcon(pm)

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == QEvent.Type.WindowStateChange and self._is_minimized_to_tray \
                and not self.isMinimized() and self.isVisible():
            # Restored from the taskbar button (not by a wake word).
            self._is_minimized_to_tray = False
            self.setWindowOpacity(1.0)
            self.hud_restored.emit()

    def _page_submit(self, text: str) -> None:
        """A button on a menu screen: send it to Spidy and show the conversation."""
        self.command_submitted.emit(text)
        self.show_page("Chat")
        self.rail.current = "Chat"
        self.rail.update()

    def _set_always_on_top(self, on: bool) -> None:
        pos = self.pos()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.show()
        self.move(pos)

    def show_page(self, name: str) -> None:
        self._page = name
        home = name == "Home"
        for w in self._home:
            w.setVisible(home)
        self.input.setVisible(home and self.voice.mode == "text")
        self.chat.setVisible(name == "Chat")
        for key, page in self.pages.items():
            page.setVisible(key == name)
        if name in self.pages:
            self.pages[name].on_show()
        self.confirm.raise_()
        self._confirmation_card.raise_()

    # ── live data ────────────────────────────────────────────────────────
    def _on_stats(self, d: dict) -> None:
        self.stats_row.set_data(d)
        self.header.set_weather(d.get("weather"))

    # ── OverlayWindow-compatible API ─────────────────────────────────────
    def set_state(self, state: str) -> None:
        self._state = state
        self.voice.set_state(state)
        self.hero.set_state(state)
        self.pages["Voice"].set_status(state)
        if state in ("idle", "wake_ready", "error"):
            self.sugg.clear_task()
        self._refresh_status()

    def add_message(self, role: str, msg: str) -> None:
        self.chat.add(role, msg)
        self.recent.add(role, msg)
        self.pages["Memory"].add_line(role, msg)
        if role.lower() not in ("user", "you"):
            self._toast = msg.replace("\n", " ")
            self._toast_timer.start(9000)
            self._refresh_status()

    def update_waveform(self, amplitudes) -> None:
        try:
            level = max(amplitudes) if amplitudes else 0.0
        except TypeError:
            level = float(amplitudes or 0.0)
        self.voice.set_level(float(level))
        self.hero.set_voice_level(float(level))

    def show_notification(self, title: str, msg: str, level: str = "info") -> None:
        self._toast = f"{title}: {msg}" if title else msg
        self._toast_timer.start(6000)
        self._refresh_status()

    def update_task_progress(self, goal: str, steps) -> None:
        out: list[tuple[str, str]] = []
        for s in steps or []:
            if isinstance(s, dict):
                out.append((str(s.get("description", "")), str(s.get("status", "pending"))))
            elif hasattr(s, "description"):
                out.append((str(s.description), str(getattr(s, "status", "pending"))))
            elif isinstance(s, (list, tuple)) and len(s) == 2:
                out.append((str(s[0]), str(s[1])))
        self.sugg.set_task(goal, out)

    def show_confirmation(self, task_id: str, goal_id: str, description: str, prompt: str = "") -> None:
        card = self._confirmation_card
        card.setGeometry(round(self._w * 0.28), round(self._h * 0.28), round(self._w * 0.44), round(self._h * 0.44))
        card.show_confirmation(task_id, goal_id, description, prompt)
        card.raise_()

    def hide_confirmation(self) -> None:
        self._confirmation_card.hide_confirmation()

    def apply_theme(self, theme) -> None:  # the dashboard has one fixed look
        pass

    def _clear_toast(self) -> None:
        self._toast = ""
        self._refresh_status()

    def _refresh_status(self) -> None:
        msg = (self._toast[:90] + ("…" if len(self._toast) > 90 else "")) if self._toast \
            else STATUS_DEFAULT.get(self._state, STATUS_DEFAULT["idle"])
        self.header.set_message(msg)

    # ── show / hide ──────────────────────────────────────────────────────
    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.hero.set_animating(True)
        self.voice.set_animating(True)
        self.stats_row.set_animating(True)
        self.stats.resume()

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self.hero.set_animating(False)
        self.voice.set_animating(False)
        self.stats_row.set_animating(False)
        self.stats.pause()

    def record_valid_geometry(self) -> None:
        DashboardWindow._session_pos = QPointF(self.pos())

    def ensure_valid_geometry(self) -> None:
        screen = QGuiApplication.screenAt(self.frameGeometry().center()) or QGuiApplication.primaryScreen()
        if screen and not screen.availableGeometry().intersects(self.frameGeometry()):
            g = screen.availableGeometry()
            self.move(g.x() + (g.width() - self._w) // 2, g.y() + (g.height() - self._h) // 2)

    def _fade(self, to: float, then=None) -> None:
        if self._anim is not None:
            self._anim.stop()
        if not self._animate:
            self.setWindowOpacity(to)
            if then:
                then()
            return
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(170)
        self._anim.setStartValue(self.windowOpacity())
        self._anim.setEndValue(to)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        if then:
            self._anim.finished.connect(then)
        self._anim.start()

    def show_animated(self) -> None:
        if self.isMinimized():                      # waiting on the taskbar: bring it back up
            self.showNormal()
            self.setWindowOpacity(1.0)
        if self.isVisible() and self.windowOpacity() >= 0.95:
            self.raise_()
            self.activateWindow()
            return
        if DashboardWindow._session_pos is not None:
            self.move(DashboardWindow._session_pos.toPoint())
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade(1.0)

    def hide_animated(self) -> None:
        self.record_valid_geometry()
        self._fade(0.0, self.hide)

    def minimize_hud(self) -> None:
        """Tuck the HUD away entirely (idle timeout, Esc, tray): no taskbar button until the next wake."""
        self._is_minimized_to_tray = True
        if self.isMinimized():
            self.showNormal()
        self.hide_animated()
        self.hud_minimized.emit()

    def minimize_to_taskbar(self) -> None:
        """The minimize button: Spidy stays on the taskbar and keeps listening; a wake word restores it."""
        self._is_minimized_to_tray = True
        self.record_valid_geometry()
        self.showMinimized()
        self.hud_minimized.emit()

    def restore_hud(self) -> None:
        self._is_minimized_to_tray = False
        self.ensure_valid_geometry()
        self.show_animated()
        self._ensure_foreground()
        self.hud_restored.emit()

    def show_hud(self) -> None:
        self.restore_hud()

    def hide_hud(self) -> None:
        self.minimize_hud()

    def toggle_hud(self) -> None:
        if self.isVisible() and self.windowOpacity() > 0.05 and not self._is_minimized_to_tray \
                and not self.isMinimized():
            self.minimize_hud()
        else:
            self.restore_hud()

    def _ensure_foreground(self) -> None:
        """Bring the window to the front (best effort: Windows may refuse to steal focus)."""
        self.raise_()
        self.activateWindow()
        if platform.system() == "Windows":
            try:
                import ctypes
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 5)
                user32.SetForegroundWindow(hwnd)
                user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
            except Exception as exc:  # noqa: BLE001
                log.debug("Foreground activation failed: {exc}", exc=exc)
