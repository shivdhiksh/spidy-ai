"""
hud_telemetry.py -- Compact HUD telemetry modules
==================================================
Lightweight system metrics panel for the Spidy HUD.

Uses psutil (already a Spidy dependency) with 1-second polling.
Results are cached -- no per-frame system calls.
Displays as mini QPainter progress bars.
Falls back to "N/A" for any unavailable metric.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from spidy.ui.themes.base import Theme

# How often to update metrics (ms)
_POLL_INTERVAL_MS = 1200  # ~1 fps


def _try_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _try_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


class _Metric:
    """Stores a single metric name + current value (0-100 or N/A)."""
    __slots__ = ("label", "value", "unit")

    def __init__(self, label: str, value: float | None = None, unit: str = "%"):
        self.label = label
        self.value = value  # None == N/A
        self.unit  = unit


class TelemetryPanel(QWidget):
    """
    Compact vertical metrics panel.

    Layout (each row):
        LABEL  [===---]  value%

    Parameters
    ----------
    metrics : list[str]
        Which metrics to show. Options: "cpu", "ram", "gpu", "vram",
        "net_up", "net_down", "disk"
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._requested = metrics or ["cpu", "ram", "gpu", "vram", "disk"]
        self._rows: list[_Metric] = self._build_rows()

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._c_primary   = QColor("#00E5FF")
        self._c_secondary = QColor("#00ACC1")
        self._c_dim       = QColor("#004D5E")
        self._c_text      = QColor("#B2EBF2")
        self._c_bg        = QColor("#0A1929")

        self._psutil = _try_psutil()
        self._nvml   = _try_nvml()
        self._net_prev: tuple[int, int] | None = None
        self._net_time: float = 0.0
        self._net_up:   float = 0.0
        self._net_down: float = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()  # populate immediately

    # ---- Public API ---------------------------------------------------------

    def apply_theme(self, theme: "Theme") -> None:
        c = theme.colors
        self._c_primary   = QColor(c.hud_primary)
        self._c_secondary = QColor(c.hud_secondary)
        self._c_dim       = QColor(c.hud_dim)
        self._c_text      = QColor(c.hud_text)
        self._c_bg        = QColor(c.hud_grid)
        self.update()

    # ---- Painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        row_h = self.height() / max(1, len(self._rows))
        # Proportions: label | bar | value
        # label=28%, bar=40%, gap=8px total, value gets the remainder.
        # This ensures "100%" fits even at the minimum panel width.
        label_w = w * 0.28
        bar_x   = label_w + 4
        bar_w   = w * 0.40
        val_x   = bar_x + bar_w + 4

        for i, m in enumerate(self._rows):
            y_top = i * row_h
            y_mid = y_top + row_h * 0.5

            # Label
            label_col = QColor(self._c_text)
            label_col.setAlphaF(0.70)
            painter.setPen(QPen(label_col))
            font = QFont("Consolas", max(7, min(12, int(row_h * 0.32))))
            painter.setFont(font)
            painter.drawText(
                QRectF(0, y_top, label_w, row_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                m.label,
            )

            # Bar background
            bar_rect = QRectF(bar_x, y_mid - row_h * 0.15, bar_w, row_h * 0.30)
            bg_col = QColor(self._c_dim)
            bg_col.setAlphaF(0.25)
            painter.setBrush(QBrush(bg_col))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_rect, 2, 2)

            if m.value is not None:
                frac = max(0.0, min(1.0, m.value / 100.0))
                fill_col = self._bar_color(frac)
                fill_rect = QRectF(bar_x, bar_rect.y(), bar_w * frac, bar_rect.height())
                painter.setBrush(QBrush(fill_col))
                painter.drawRoundedRect(fill_rect, 2, 2)

                # Value text
                val_txt = f"{m.value:.0f}{m.unit}"
            else:
                val_txt = "N/A"

            painter.setPen(QPen(label_col))
            painter.drawText(
                QRectF(val_x, y_top, w - val_x, row_h),
                Qt.AlignmentFlag.AlignVCenter,
                val_txt,
            )

        painter.end()

    # ---- Internal -----------------------------------------------------------

    def _bar_color(self, frac: float) -> QColor:
        """Green→cyan→amber→red gradient based on load."""
        if frac < 0.60:
            return QColor(self._c_primary)
        elif frac < 0.80:
            return QColor(self._c_secondary)
        elif frac < 0.90:
            return QColor("#E8A020")
        return QColor("#FF2444")

    def _build_rows(self) -> list[_Metric]:
        label_map = {
            "cpu":       "CPU",
            "ram":       "RAM",
            "gpu":       "GPU",
            "vram":      "VRAM",
            "disk":      "DISK",
            "net_up":    "NET↑",
            "net_down":  "NET↓",
        }
        rows = []
        for m in self._requested:
            rows.append(_Metric(label_map.get(m, m.upper())))
        return rows

    def _poll(self) -> None:
        ps = self._psutil
        nv = self._nvml
        now = time.monotonic()

        for m in self._rows:
            key = m.label

            try:
                if key == "CPU":
                    m.value = ps.cpu_percent(interval=None) if ps else None
                elif key == "RAM":
                    m.value = ps.virtual_memory().percent if ps else None
                elif key == "GPU":
                    if nv:
                        h = nv.nvmlDeviceGetHandleByIndex(0)
                        util = nv.nvmlDeviceGetUtilizationRates(h)
                        m.value = float(util.gpu)
                    else:
                        m.value = None
                elif key == "VRAM":
                    if nv:
                        h = nv.nvmlDeviceGetHandleByIndex(0)
                        mem = nv.nvmlDeviceGetMemoryInfo(h)
                        m.value = (mem.used / mem.total) * 100
                    else:
                        m.value = None
                elif key == "DISK":
                    m.value = ps.disk_usage("/").percent if ps else None
                elif key in ("NET↑", "NET↓"):
                    self._update_net(ps, now)
                    m.unit = "KB"
                    if key == "NET↑":
                        m.value = self._net_up
                    else:
                        m.value = self._net_down
                        m.value = min(100.0, m.value) if m.value is not None else None
            except Exception:
                m.value = None

        self._net_time = now
        self.update()

    def _update_net(self, ps, now: float) -> None:
        if not ps:
            self._net_up = self._net_down = None
            return
        try:
            io = ps.net_io_counters()
            if self._net_prev is not None:
                dt = max(0.001, now - self._net_time)
                self._net_up   = (io.bytes_sent - self._net_prev[0]) / dt / 1024
                self._net_down = (io.bytes_recv - self._net_prev[1]) / dt / 1024
            self._net_prev = (io.bytes_sent, io.bytes_recv)
        except Exception:
            self._net_up = self._net_down = None

