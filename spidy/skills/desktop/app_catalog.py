"""
App catalog — find any installed Windows application by its spoken name.

``AppSkill`` only knew a fixed list of executables, so anything else (WhatsApp, Store
apps, apps not on PATH) could not be launched. This module adds:

- ``normalize_app_name``   fixes common speech-recognition mistakes ("VS Core" → "vs code")
- ``find_start_app``       fuzzy lookup in the Start menu (``Get-StartApps``), which lists
                           classic apps *and* Microsoft Store apps with their AppUserModelID
- ``launch_start_app``     starts an entry via ``shell:AppsFolder\\<AppID>``
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Whisper regularly mishears short app names.
_SPOKEN_FIXES: dict[str, str] = {
    "vs core": "vs code", "vs coat": "vs code", "vs cord": "vs code", "vs cold": "vs code",
    "vs codes": "vs code", "vs coke": "vs code", "bs code": "vs code", "vis code": "vs code",
    "v s code": "vs code", "vscode": "vs code", "vs": "vs code", "visual studio coat": "vs code",
    "whats app": "whatsapp", "what's app": "whatsapp", "whatsapp web": "whatsapp",
    "watsapp": "whatsapp", "what sap": "whatsapp", "whats up": "whatsapp",
    "google chrome": "chrome", "microsoft edge": "edge", "file explorer": "explorer",
}

_STRIP_TRAILING = re.compile(r"\s+(?:app|application|program|for me|please|now|again|too)$")
_STRIP_LEADING = re.compile(r"^(?:the|my|a|an)\s+")


def normalize_app_name(name: str) -> str:
    """Lower-case, strip filler words/punctuation and fix common mishearings."""
    text = re.sub(r"[^\w\s'+.-]", " ", name.lower())
    text = re.sub(r"\s+", " ", text).strip(" .-'")
    if text in _SPOKEN_FIXES:          # "whats app" must be fixed before " app" is stripped
        return _SPOKEN_FIXES[text]
    for _ in range(3):  # peel stacked fillers: "the whatsapp app please"
        new = _STRIP_LEADING.sub("", _STRIP_TRAILING.sub("", text)).strip(" .-'")
        if new == text:
            break
        text = new
    return _SPOKEN_FIXES.get(text, text)


@dataclass(frozen=True)
class StartApp:
    name: str      # display name, e.g. "WhatsApp"
    app_id: str    # AppUserModelID or path, e.g. "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"


_CACHE_TTL_SECONDS = 600.0
_cache: list[StartApp] = []
_cache_time = 0.0
_cache_lock = threading.Lock()


def load_start_apps(force: bool = False) -> list[StartApp]:
    """Return the Start menu apps (cached for 10 minutes). Empty list off-Windows."""
    global _cache, _cache_time
    if sys.platform != "win32":
        return []
    with _cache_lock:
        if _cache and not force and time.monotonic() - _cache_time < _CACHE_TTL_SECONDS:
            return _cache
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-StartApps | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=25, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            data = json.loads(proc.stdout) if proc.stdout.strip() else []
            if isinstance(data, dict):
                data = [data]
            apps = [StartApp(str(d["Name"]), str(d["AppID"])) for d in data if d.get("Name") and d.get("AppID")]
        except Exception as exc:  # noqa: BLE001
            log.warning("AppCatalog: Get-StartApps failed: {exc}", exc=exc)
            return _cache  # keep any stale list rather than nothing
        _cache, _cache_time = apps, time.monotonic()
        log.info("AppCatalog: loaded {n} Start menu apps.", n=len(apps))
        return _cache


def find_start_app(spoken: str, apps: list[StartApp] | None = None) -> StartApp | None:
    """Best Start menu match for a spoken name, or None."""
    wanted = normalize_app_name(spoken)
    if not wanted:
        return None
    apps = load_start_apps() if apps is None else apps
    lowered = [(a.name.lower(), a) for a in apps]

    for key, app in lowered:                       # exact
        if key == wanted:
            return app
    words = wanted.split()
    # "vs code" should find "Visual Studio Code": accept the well-known expansion
    expansions = {"vs code": ("visual studio code",)}.get(wanted, ())
    for expanded in expansions:
        for key, app in lowered:
            if key == expanded:
                return app
    starts = [(len(k), a) for k, a in lowered if k.startswith(wanted)]
    if starts:                                     # "whatsapp" → "WhatsApp Desktop"
        return min(starts, key=lambda t: t[0])[1]
    contains = [(len(k), a) for k, a in lowered if all(w in k for w in words)]
    if contains:
        return min(contains, key=lambda t: t[0])[1]
    close = difflib.get_close_matches(wanted, [k for k, _ in lowered], n=1, cutoff=0.78)
    if close:
        return next(a for k, a in lowered if k == close[0])
    return None


def launch_start_app(app: StartApp) -> subprocess.Popen:
    """Start a Start menu entry (works for Store/UWP and classic apps)."""
    return subprocess.Popen(
        ["explorer.exe", f"shell:AppsFolder\\{app.app_id}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
