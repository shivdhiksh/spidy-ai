"""
Live data for the dashboard: CPU/RAM/GPU load, hardware specs, network state and weather.

A single background thread samples everything and emits one ``updated`` signal (queued onto
the Qt thread). GPU numbers come from ``nvidia-smi`` (no extra dependency); every value is
optional and the UI shows a dash when a source is unavailable.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
from typing import Any

import psutil
from PySide6.QtCore import QObject, Signal

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_WEATHER_TEXT = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle", 61: "Rain", 63: "Rain", 65: "Heavy rain",
    71: "Snow", 73: "Snow", 75: "Heavy snow", 80: "Showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             creationflags=_NO_WINDOW, encoding="utf-8", errors="replace")
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def read_gpu() -> dict[str, Any] | None:
    """GPU load/VRAM/name from nvidia-smi, or None if there is no NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    line = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                 "--format=csv,noheader,nounits"]).splitlines()
    if not line:
        return None
    parts = [x.strip() for x in line[0].split(",")]
    try:
        return {"util": float(parts[0]), "vram_used": float(parts[1]),
                "vram_total": float(parts[2]), "name": parts[3]}
    except (IndexError, ValueError):
        return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return ""


def pretty_cpu(raw: str) -> tuple[str, str]:
    """'12th Gen Intel(R) Core(TM) i5-12450H' -> ('Intel Core i5', '12th Gen')."""
    if not raw:
        return "CPU", ""
    gen = re.search(r"(\d+)(?:st|nd|rd|th) Gen", raw)
    core = re.search(r"Core\(TM\)\s*(i\d|Ultra\s*\d)", raw)
    ryzen = re.search(r"(Ryzen\s+\d)", raw)
    if core:
        return f"Intel Core {core.group(1)}", f"{gen.group(1)}th Gen" if gen else ""
    if ryzen:
        return f"AMD {ryzen.group(1)}", ""
    return re.sub(r"\((?:R|TM)\)|CPU|@.*", "", raw).strip()[:22], ""


def pretty_gpu(raw: str, vram_mb: float) -> tuple[str, str]:
    name = re.sub(r"GeForce |Laptop GPU|Graphics| GPU", "", raw).strip()
    return name or "GPU", f"{vram_mb / 1024:.0f}GB VRAM" if vram_mb else ""


def pretty_disk() -> str:
    try:
        total_gb = shutil.disk_usage("C:\\" if sys.platform == "win32" else "/").total / 1e9
    except OSError:
        return "—"
    for size in (128, 256, 512, 1024, 2048, 4096):
        if total_gb <= size * 1.02:
            return f"{size // 1024} TB" if size >= 1024 else f"{size} GB"
    return f"{total_gb / 1000:.0f} TB"


def network_state() -> tuple[bool, str]:
    name = socket.gethostname()
    try:
        stats = psutil.net_if_stats()
        up = any(s.isup and not n.lower().startswith(("lo", "loopback")) for n, s in stats.items())
    except Exception:  # noqa: BLE001
        up = False
    return up, name


def fetch_weather(lat: float, lon: float) -> tuple[str, str] | None:
    """(temperature text, condition) from Open-Meteo, or None when offline."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&current=temperature_2m,weather_code")
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            cur = json.loads(resp.read().decode("utf-8"))["current"]
        return f"{round(cur['temperature_2m'])}°C", _WEATHER_TEXT.get(int(cur["weather_code"]), "—")
    except Exception:  # noqa: BLE001
        return None


class StatsWorker(QObject):
    """Samples system stats every few seconds and weather every 30 minutes."""

    updated = Signal(dict)

    def __init__(self, weather: tuple[bool, float, float] = (False, 0.0, 0.0), interval: float = 2.5) -> None:
        super().__init__()
        self._weather_on, self._lat, self._lon = weather
        self._interval = interval
        self._stop = threading.Event()
        self._active = threading.Event()
        self._active.set()
        self._thread: threading.Thread | None = None
        self._specs: dict[str, Any] = {}
        self._weather: tuple[str, str] | None = None
        self._weather_at = -1e9

    def pause(self) -> None:
        """No sampling while the window is hidden (saves a nvidia-smi spawn every few seconds)."""
        self._active.clear()

    def resume(self) -> None:
        self._active.set()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="DashboardStats", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _load_specs(self, gpu: dict[str, Any] | None) -> None:
        cpu1, cpu2 = pretty_cpu(cpu_name())
        gpu1, gpu2 = pretty_gpu(gpu["name"], gpu["vram_total"]) if gpu else ("No GPU", "")
        self._specs = {
            "cpu": (cpu1, cpu2), "gpu": (gpu1, gpu2),
            "ram": (f"{-(-psutil.virtual_memory().total // 2**30 // 4) * 4:.0f} GB", "RAM"),  # 15.4 usable -> 16 GB
            "disk": (pretty_disk(), "SSD"),
        }

    def _loop(self) -> None:
        import time
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            if not self._active.is_set():
                self._active.wait(1.0)
                continue
            gpu = read_gpu()
            if not self._specs:
                self._load_specs(gpu)
            up, host = network_state()
            now = time.monotonic()
            if self._weather_on and now - self._weather_at > 1800:
                self._weather_at = now
                self._weather = fetch_weather(self._lat, self._lon) or self._weather
            vm = psutil.virtual_memory()
            self.updated.emit({
                "cpu": psutil.cpu_percent(interval=None),
                "ram": vm.percent,
                "gpu": gpu["util"] if gpu else None,
                "vram": (100.0 * gpu["vram_used"] / gpu["vram_total"]) if gpu and gpu["vram_total"] else None,
                "specs": self._specs, "online": up, "host": host, "weather": self._weather,
            })
            self._stop.wait(self._interval)
