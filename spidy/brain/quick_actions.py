"""
Quick actions: everyday requests handled directly and reliably, before the planner or the language model.

The menu check showed the slow path failing on simple things (``open google`` returned raw JSON, searches
hit a captcha in the hidden browser, the model said it had no weather or screen access). These are answered
here with plain code: open a site in the user's own browser, Google search, weather and a briefing from
Open-Meteo and Google News, timers, folders and file search, clipboard code help, screenshots and screen
questions, and facts the user asks Spidy to remember.

``try_handle`` returns a ``QuickResult``: a finished ``reply``, or a ``rewrite`` (a better prompt for the
language model, e.g. with the clipboard contents attached), or ``None`` when the request is not ours.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from spidy.logging.logger import get_logger

log = get_logger(__name__)

HOME = Path.home()
DATA_DIR = Path(os.environ.get("APPDATA", str(HOME))) / "Spidy"
FACTS_FILE = DATA_DIR / "remembered_facts.json"
# NVIDIA NIM vision models, best first (checked against the live model list, 2026-11)
_VISION_MODELS = ("nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

SITES: dict[str, tuple[str, str]] = {
    "google": ("Google", "https://www.google.com"),
    "youtube": ("YouTube", "https://www.youtube.com"),
    "gmail": ("Gmail", "https://mail.google.com"),
    "mail": ("Gmail", "https://mail.google.com"),
    "github": ("GitHub", "https://github.com"),
    "whatsapp": ("WhatsApp Web", "https://web.whatsapp.com"),
    "whatsapp web": ("WhatsApp Web", "https://web.whatsapp.com"),
    "google maps": ("Google Maps", "https://maps.google.com"),
    "maps": ("Google Maps", "https://maps.google.com"),
    "chatgpt": ("ChatGPT", "https://chatgpt.com"),
    "chat gpt": ("ChatGPT", "https://chatgpt.com"),
    "linkedin": ("LinkedIn", "https://www.linkedin.com"),
    "news": ("Google News", "https://news.google.com"),
    "google news": ("Google News", "https://news.google.com"),
    "top news": ("Google News", "https://news.google.com"),
    "today's top news": ("Google News", "https://news.google.com"),
    "todays top news": ("Google News", "https://news.google.com"),
    "today's news": ("Google News", "https://news.google.com"),
    "google calendar": ("Google Calendar", "https://calendar.google.com"),
    "calendar": ("Google Calendar", "https://calendar.google.com"),
    "google drive": ("Google Drive", "https://drive.google.com"),
    "drive": ("Google Drive", "https://drive.google.com"),
    "netflix": ("Netflix", "https://www.netflix.com"),
    "amazon": ("Amazon", "https://www.amazon.in"),
    "flipkart": ("Flipkart", "https://www.flipkart.com"),
    "instagram": ("Instagram", "https://www.instagram.com"),
    "twitter": ("X", "https://x.com"),
    "stack overflow": ("Stack Overflow", "https://stackoverflow.com"),
    "wikipedia": ("Wikipedia", "https://www.wikipedia.org"),
}
FOLDERS = ("desktop", "documents", "downloads", "pictures", "music", "videos")
_WEATHER_TEXT = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy",
                 51: "drizzly", 53: "drizzly", 55: "drizzly", 61: "rainy", 63: "rainy", 65: "pouring",
                 71: "snowy", 73: "snowy", 75: "snowy", 80: "showery", 81: "showery", 82: "stormy",
                 95: "stormy", 96: "stormy", 99: "stormy"}


@dataclass
class QuickResult:
    reply: str | None = None        # finished answer
    rewrite: str | None = None      # better prompt to send to the language model instead
    code: tuple[str, str] | None = None   # (language, description): write a program to a real file


# ── helpers ───────────────────────────────────────────────────────────────────

_LEAD = re.compile(r"^(?:(?:hey|hi|ok|okay)\s+)?(?:spidy|spidey)?[\s,!.]*(?:(?:can|could|would)\s+you\s+|please\s+|kindly\s+)*",
                   re.IGNORECASE)


def normalize(text: str) -> str:
    t = _LEAD.sub("", (text or "").strip())
    t = re.sub(r"\s+(?:please|for me|now|right now)$", "", t.strip(), flags=re.IGNORECASE)
    return t.strip().rstrip(".!?").strip()


# ── code requests ─────────────────────────────────────────────────────────────

# language name -> (canonical language, file extension)
CODE_LANGUAGES: dict[str, tuple[str, str]] = {
    "html": ("html", ".html"), "css": ("html", ".html"), "website": ("html", ".html"), "web page": ("html", ".html"),
    "webpage": ("html", ".html"), "landing page": ("html", ".html"), "web app": ("html", ".html"),
    "portfolio": ("html", ".html"), "site": ("html", ".html"),
    "python": ("python", ".py"), "py": ("python", ".py"),
    "javascript": ("javascript", ".js"), "js": ("javascript", ".js"), "node": ("javascript", ".js"),
    "nodejs": ("javascript", ".js"), "typescript": ("typescript", ".ts"), "ts": ("typescript", ".ts"),
    "react": ("react", ".jsx"), "java": ("java", ".java"), "kotlin": ("kotlin", ".kt"),
    "c++": ("c++", ".cpp"), "cpp": ("c++", ".cpp"), "c plus plus": ("c++", ".cpp"), "c#": ("c#", ".cs"),
    "c sharp": ("c#", ".cs"), "csharp": ("c#", ".cs"), "c": ("c", ".c"), "go": ("go", ".go"), "golang": ("go", ".go"),
    "rust": ("rust", ".rs"), "php": ("php", ".php"), "ruby": ("ruby", ".rb"), "swift": ("swift", ".swift"),
    "sql": ("sql", ".sql"), "bash": ("bash", ".sh"), "shell": ("bash", ".sh"), "powershell": ("powershell", ".ps1"),
    "batch": ("batch", ".bat"), "dart": ("dart", ".dart"), "r": ("r", ".r"), "lua": ("lua", ".lua"),
    "json": ("json", ".json"), "yaml": ("yaml", ".yaml"), "markdown": ("markdown", ".md"),
}
_CODE_VERB = r"(?:create|make|write|build|generate|code|develop|design|program|craft|prepare|give\s+me|i\s+(?:want|need))"
_CODE_NOUN = (r"(?:website|web\s*site|web\s*page|webpage|landing\s+page|portfolio|web\s+app|site|program|programme|"
              r"script|app|application|game|tool|code|function|class|module|api|bot|calculator|query|"
              r"(?:html|css|python|java\w*|c\+\+|c#|\w+)\s+file|page)")
_LANG_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in CODE_LANGUAGES), key=len, reverse=True))
                      + r")(?![\w+#])", re.IGNORECASE)


def parse_code_request(text: str) -> tuple[str, str] | None:
    """
    ``(language, request)`` if ``text`` asks Spidy to write code, else None.

    Handles "create a website using HTML", "make a calculator website", "write a python script that
    renames files", "build a snake game in JavaScript", "open VS Code, create an HTML file, write
    calculator code". "Create a file called notes" (no language) stays a file operation.
    """
    t = normalize(text)
    low = re.sub(r"\b(?:vs|visual\s+studio)\s+code\b", "vscode", t.lower())
    # the thing to build must follow the verb closely: "make me a simple calculator website",
    # not "make the chrome app full screen" (verb ... noun ... more words is still fine)
    if not re.search(rf"\b{_CODE_VERB}\b(?:\W+(?!(?:the|my|this|that)\b)\w+){{0,5}}?\W+{_CODE_NOUN}\b", low):
        return None
    if re.search(rf"\b{_CODE_VERB}\W+(?:the|my|this|that)\s+\w*\s*(?:app|application|site|page|game)\b", low) \
            and not re.search(r"\b(?:code|program|script)\b", low):
        return None
    if re.search(r"\b(?:folder|directory|note|reminder|alarm|playlist|account|password|shortcut|backup)\b", low) \
            and not re.search(r"\b(?:code|program|script|website|app)\b", low):
        return None
    lang_m = None
    for m in _LANG_RE.finditer(low):
        key = m.group(1).lower()
        if key in ("c", "r", "go", "site", "page") and not re.search(
                rf"\b(?:in|using|with|language)\s+{re.escape(key)}\b|\b{re.escape(key)}\s+(?:program|code|file|language|script)\b",
                low):
            if key == "site" and re.search(r"\bweb\s*site|\bsite\b", low):
                lang_m = lang_m or m
            continue
        lang_m = lang_m or m
    if lang_m is not None:
        language = CODE_LANGUAGES[lang_m.group(1).lower()][0]
    elif re.search(r"\b(?:website|web\s*site|web\s*page|webpage|landing\s+page|portfolio|web\s+app)\b", low):
        language = "html"
    else:
        noun = re.search(r"\b(?:program|programme|script|app|application|game|tool|code|function|class|bot|calculator)\b", low)
        if not noun:            # "create a file called x" / "make a page" with nothing code-like
            return None
        language = "python"
    return language, t


def user_folder(name: str) -> Path:
    for base in (HOME / "OneDrive", HOME):
        p = base / name.capitalize()
        if p.exists():
            return p
    return HOME / name.capitalize()


def ssl_context():
    """certifi's CA bundle: this PC's Windows store has an expired root that breaks some sites (Open-Meteo)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _http_json(url: str, timeout: float = 6.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Spidy/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _settings():
    try:
        from spidy.config.manager import ConfigManager
        return ConfigManager().load()
    except Exception:  # noqa: BLE001
        return None


def load_facts() -> list[str]:
    try:
        return list(json.loads(FACTS_FILE.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return []


def save_facts(facts: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FACTS_FILE.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")


def facts_context() -> str:
    """Remembered facts for the language model's system prompt ('' when there are none)."""
    facts = load_facts()
    if not facts:
        return ""
    return "\n\nThings the user asked you to remember about them:\n" + "\n".join(f"- {f}" for f in facts[-40:])


def _you(text: str) -> str:
    """'my editor is VS Code' -> 'your editor is VS Code' (for repeating a fact back)."""
    swaps = {"my": "your", "i": "you", "i'm": "you're", "am": "are", "me": "you", "mine": "yours", "myself": "yourself"}
    return re.sub(r"\b(my|i|i'm|am|me|mine|myself)\b", lambda m: swaps[m.group(1).lower()], text, flags=re.IGNORECASE)


def read_clipboard() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"], capture_output=True,
                             text=True, timeout=6, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def take_screenshot() -> Path | None:
    """Full-screen PNG in Pictures/Spidy (built-in .NET, no extra packages)."""
    folder = user_folder("pictures") / "Spidy"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"screenshot-{_dt.datetime.now():%Y%m%d-%H%M%S}.png"
    script = ("Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
              "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen;"
              "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
              "$g=[System.Drawing.Graphics]::FromImage($bmp);"
              "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size);"
              "$bmp.Save($env:SPIDY_SHOT,[System.Drawing.Imaging.ImageFormat]::Png)")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script], timeout=15,
                       env={**os.environ, "SPIDY_SHOT": str(path)},
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
        return path if path.exists() else None
    except Exception as exc:  # noqa: BLE001
        log.warning("Screenshot failed: {e}", e=exc)
        return None


def active_app() -> str:
    """Title of the top-most visible window that is not Spidy itself."""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        own = os.getpid()
        hwnd = user32.GetTopWindow(None)
        while hwnd:
            if user32.IsWindowVisible(hwnd):
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                n = user32.GetWindowTextLengthW(hwnd)
                if pid.value != own and n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    title = buf.value.strip()
                    if title and title not in ("Program Manager", "Spidy") and "Spidy" != title:
                        return title
            hwnd = user32.GetWindow(hwnd, 2)          # GW_HWNDNEXT
    except Exception:  # noqa: BLE001
        pass
    return ""


def ask_vision(image: Path, question: str) -> str | None:
    """Ask an NVIDIA NIM vision model about an image. None when no key / model is reachable."""
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        return None
    data = base64.b64encode(image.read_bytes()).decode()
    body_base = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]}],
        "max_tokens": 400, "temperature": 0.2}
    for model in _VISION_MODELS:
        req = urllib.request.Request("https://integrate.api.nvidia.com/v1/chat/completions",
                                     data=json.dumps({**body_base, "model": model}).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out = json.loads(resp.read().decode())
            return out["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            log.info("Vision model {m} unavailable: {e}", m=model, e=exc)
    return None


def find_files(query: str, limit: int = 5, seconds: float = 6.0) -> list[Path]:
    """Files whose name contains every word of ``query`` (Desktop, Documents, Downloads, Pictures, OneDrive)."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in ("the", "my", "a", "file", "files")]
    if not words:
        return []
    roots = [user_folder(n) for n in ("desktop", "documents", "downloads", "pictures")] + [HOME / "OneDrive"]
    deadline = time.monotonic() + seconds
    hits: list[tuple[float, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith((".", "$", "__pycache__", "node_modules", "venv", ".venv"))]
            for f in files:
                low = f.lower()
                if all(w in low for w in words):
                    p = Path(dirpath) / f
                    if p not in seen:
                        seen.add(p)
                        try:
                            hits.append((p.stat().st_mtime, p))
                        except OSError:
                            pass
            if time.monotonic() > deadline:
                break
        if time.monotonic() > deadline:
            break
    hits.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in hits[:limit]]


# ── the handler ───────────────────────────────────────────────────────────────

class QuickActions:
    """Deterministic handlers for common requests. ``schedule`` runs a coroutine later (timers)."""

    def __init__(self, announce: Callable[[str], Any] | None = None) -> None:
        self._announce = announce          # async callable(text): speak / show a message later
        self._timers: list[asyncio.TimerHandle] = []

    async def try_handle(self, text: str) -> QuickResult | None:
        t = normalize(text)
        low = t.lower()
        if not low:
            return None
        for handler in (self._create_code, self._site, self._new_tab, self._folder, self._find_file, self._search, self._weather,
                        self._briefing, self._timer, self._clipboard_code, self._screen, self._remember):
            try:
                res = await handler(t, low)
            except Exception as exc:  # noqa: BLE001
                log.warning("Quick action {h} failed: {e}", h=handler.__name__, e=exc)
                res = None
            if res is not None:
                log.info("[QUICK] {h} handled '{t}'", h=handler.__name__, t=t[:80])
                return res
        return None

    # -- write a program to a file
    async def _create_code(self, t: str, low: str) -> QuickResult | None:
        req = parse_code_request(t)
        return QuickResult(code=req) if req else None

    # -- websites
    async def _site(self, t: str, low: str) -> QuickResult | None:
        m = re.fullmatch(r"(?:open|launch|go to|show me|show|visit)\s+(?:the\s+|a\s+)?(.+?)"
                         r"(?:\s+(?:website|site|page|web))?(?:\s+in\s+(?:the\s+|my\s+)?(?:browser|chrome))?", low)
        if not m:
            return None
        name = m.group(1).strip()
        if name not in SITES:
            return None
        title, url = SITES[name]
        await asyncio.to_thread(webbrowser.open_new_tab, url)
        return QuickResult(reply=f"Opened {title}.")

    async def _new_tab(self, t: str, low: str) -> QuickResult | None:
        if re.fullmatch(r"(?:open\s+)?(?:a\s+)?new\s+(?:browser\s+)?tab", low):
            await asyncio.to_thread(webbrowser.open_new_tab, "https://www.google.com")
            return QuickResult(reply="Opened a new tab.")
        return None

    async def _search(self, t: str, low: str) -> QuickResult | None:
        if "youtube" in low:
            return None
        m = re.fullmatch(r"(?:search(?:\s+the\s+web|\s+google|\s+online)?(?:\s+for)?|google|look\s+up)\s+(.+)", t,
                         flags=re.IGNORECASE)
        if not m:
            return None
        q = m.group(1).strip()
        if re.match(r"(?:my\s+)?files?\b", q, re.IGNORECASE):
            return None
        await asyncio.to_thread(webbrowser.open_new_tab, "https://www.google.com/search?q=" + urllib.parse.quote_plus(q))
        return QuickResult(reply=f"Here are the results for {q}.")

    # -- folders and files
    async def _folder(self, t: str, low: str) -> QuickResult | None:
        if re.fullmatch(r"open\s+(?:the\s+|my\s+)?(?:file\s+explorer|files|my\s+files|explorer|this\s+pc|my\s+computer)", low):
            subprocess.Popen(["explorer.exe"])
            return QuickResult(reply="Opened File Explorer.")
        m = re.fullmatch(r"open\s+(?:the\s+|my\s+)?(" + "|".join(FOLDERS) + r")(?:\s+folder)?", low)
        if m:
            os.startfile(str(user_folder(m.group(1))))       # noqa: S606
            return QuickResult(reply=f"Opened your {m.group(1).capitalize()} folder.")
        return None

    async def _find_file(self, t: str, low: str) -> QuickResult | None:
        m = re.fullmatch(r"(?:find|locate|search\s+for|where(?:'s|\s+is))\s+(?:my\s+|the\s+|a\s+)?"
                         r"(?:files?|documents?)(?:\s+on\s+my\s+(?:computer|pc|laptop))?(?:\s+(?:called|named))?\s*(.*)", low)
        if not m:
            m = re.fullmatch(r"(?:find|locate|where(?:'s|\s+is))\s+my\s+(.+?)(?:\s+file)?", low)
            if not m:
                return None
        query = m.group(1).strip()
        if not query:
            return QuickResult(reply="Sure! What's the file called?")
        hits = await asyncio.to_thread(find_files, query)
        if not hits:
            return QuickResult(reply=f"Sorry, I couldn't find a file called {query} in your Desktop, Documents, "
                                     "Downloads or Pictures.")
        best = hits[0]
        subprocess.Popen(["explorer.exe", f"/select,{best}"])
        more = f" I found {len(hits) - 1} more with a similar name." if len(hits) > 1 else ""
        return QuickResult(reply=f"Found it! {best.name} is in {best.parent.name}. I've shown it in File Explorer.{more}")

    # -- weather, briefing
    def _weather_now(self) -> str | None:
        s = _settings()
        ui = getattr(s, "ui", None)
        lat, lon = getattr(ui, "weather_latitude", 17.385), getattr(ui, "weather_longitude", 78.4867)
        city = (getattr(ui, "weather_city", "") or "Hyderabad").split(",")[0]
        d = _http_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                       "&current=temperature_2m,weather_code&daily=temperature_2m_max,temperature_2m_min,"
                       "precipitation_probability_max&timezone=auto&forecast_days=1")
        cur, day = d["current"], d["daily"]
        cond = _WEATHER_TEXT.get(int(cur["weather_code"]), "")
        rain = day.get("precipitation_probability_max", [None])[0]
        text = (f"It's {round(cur['temperature_2m'])} degrees and {cond} in {city} right now, with a high of "
                f"{round(day['temperature_2m_max'][0])} and a low of {round(day['temperature_2m_min'][0])}.")
        if rain is not None and rain >= 40:
            text += f" There's a {rain} percent chance of rain, so keep an umbrella handy."
        return text

    async def _weather(self, t: str, low: str) -> QuickResult | None:
        if not re.search(r"\bweather\b|\b(?:how hot|how cold|temperature outside|is it going to rain|will it rain)\b", low):
            return None
        if "briefing" in low:
            return None
        try:
            return QuickResult(reply=await asyncio.to_thread(self._weather_now))
        except Exception as exc:  # noqa: BLE001
            log.info("Weather lookup failed: {e}", e=exc)
            return QuickResult(reply="Sorry, I couldn't reach the weather service right now.")

    @staticmethod
    def _headlines(n: int = 3) -> list[str]:
        req = urllib.request.Request("https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
                                     headers={"User-Agent": "Spidy/1.0"})
        with urllib.request.urlopen(req, timeout=6, context=ssl_context()) as resp:
            root = ET.fromstring(resp.read())
        titles = [i.findtext("title") or "" for i in root.iter("item")]
        return [re.sub(r"\s+-\s+[^-]+$", "", x).strip() for x in titles if x][:n]

    async def _briefing(self, t: str, low: str) -> QuickResult | None:
        if not re.search(r"\b(briefing|brief me|morning update|daily update)\b", low):
            return None
        parts = [f"Here's your briefing. It's {_dt.datetime.now():%I:%M %p}".replace(" 0", " ") + "."]
        try:
            parts.append(await asyncio.to_thread(self._weather_now))
        except Exception:  # noqa: BLE001
            parts.append("I couldn't get the weather.")
        try:
            heads = await asyncio.to_thread(self._headlines)
            if heads:
                parts.append("Top news: " + ". ".join(heads) + ".")
        except Exception:  # noqa: BLE001
            parts.append("I couldn't get the news.")
        return QuickResult(reply=" ".join(parts))

    # -- timer
    async def _timer(self, t: str, low: str) -> QuickResult | None:
        m = (re.search(r"\btimer\b.*?\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", low)
             or re.search(r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+timer\b", low))
        if not m:
            return None
        n, unit = float(m.group(1)), m.group(2)
        secs = n * (3600 if unit.startswith(("hour", "hr")) else 60 if unit.startswith("min") else 1)
        label = f"{m.group(1)} {'hour' if unit.startswith(('hour', 'hr')) else 'minute' if unit.startswith('min') else 'second'}"
        label += "" if n == 1 else "s"
        loop = asyncio.get_running_loop()

        def ring() -> None:
            if self._announce is not None:
                asyncio.ensure_future(self._announce(f"Time's up! Your {label} timer is done."))
        self._timers.append(loop.call_later(secs, ring))
        return QuickResult(reply=f"Timer set for {label}. I'll tell you when it's done!")

    # -- code help from the clipboard
    async def _clipboard_code(self, t: str, low: str) -> QuickResult | None:
        if "\n" in t:                      # already carries the code (our own rewrite)
            return None
        if not (re.search(r"\b(clipboard|copied|i copied|i just got|this code|this error)\b", low)
                and re.search(r"\b(explain|fix|debug|review|help|what does|solve)\b", low)):
            return None
        clip = await asyncio.to_thread(read_clipboard)
        if not clip:
            return QuickResult(reply="Copy the code or the error first, then ask me again and I'll take a look.")
        task = "Explain what this code does, simply" if "explain" in low or "what does" in low else \
            "Find the problem and give the fixed version, with a one-line reason"
        return QuickResult(rewrite=f"{task}:\n\n{clip[:6000]}")

    # -- screen
    async def _screen(self, t: str, low: str) -> QuickResult | None:
        if re.fullmatch(r"(?:take|capture|grab)\s+(?:a\s+)?(?:screen\s*shot|screenshot|screen\s+capture)(?:\s+of\s+my\s+screen)?", low):
            path = await asyncio.to_thread(take_screenshot)
            if path is None:
                return QuickResult(reply="Sorry, I couldn't take the screenshot.")
            return QuickResult(reply=f"Done! Screenshot saved in your Pictures, in the Spidy folder.")
        if re.search(r"\bwhich\s+(?:app|application|window|program)\b.*\b(?:using|open|on)\b|\bwhat\s+app\b", low):
            title = await asyncio.to_thread(active_app)
            return QuickResult(reply=f"You're using {title}." if title else "I couldn't tell which app is in front.")
        if not re.search(r"\b(on my screen|my screen|what i am looking at|what i'm looking at|this page|"
                         r"read the text|screen right now)\b", low):
            return None
        path = await asyncio.to_thread(take_screenshot)
        if path is None:
            return QuickResult(reply="Sorry, I couldn't capture your screen.")
        if "read" in low:
            q = "Read out the main text visible on this screen, briefly, in plain sentences."
        elif "summar" in low:
            q = "Summarise what the user is looking at on this screen in three short sentences."
        else:
            q = "Describe what is on this computer screen in two or three short sentences."
        answer = await asyncio.to_thread(ask_vision, path, q)
        if answer is None:
            title = await asyncio.to_thread(active_app)
            where = f" You have {title} in front." if title else ""
            return QuickResult(reply="Sorry, my vision model isn't reachable right now, so I can't look at the "
                                     f"screen.{where} I saved a screenshot in Pictures, in the Spidy folder.")
        return QuickResult(reply=answer)

    # -- memory
    async def _remember(self, t: str, low: str) -> QuickResult | None:
        m = re.fullmatch(r"remember\s+(?:that\s+)?(.+)", t, flags=re.IGNORECASE)
        if m:
            fact = m.group(1).strip()
            facts = [f for f in load_facts() if f.lower() != fact.lower()] + [fact]
            await asyncio.to_thread(save_facts, facts)
            return QuickResult(reply=f"Got it! I'll remember that {_you(fact)}.")
        if re.search(r"\bwhat\s+do\s+you\s+(?:remember|know)\s+about\s+me\b|\bwhat\s+have\s+i\s+told\s+you\b", low):
            facts = load_facts()
            if not facts:
                return QuickResult(reply="You haven't asked me to remember anything yet. Just say, remember that, "
                                         "and tell me something.")
            return QuickResult(reply="Here's what I remember: " + ". ".join(_you(f) for f in facts[-10:]) + ".")
        m = re.fullmatch(r"forget\s+(?:that\s+)?(.+)", t, flags=re.IGNORECASE)
        if m:
            target = m.group(1).strip().lower()
            facts = load_facts()
            keep = [f for f in facts if target not in f.lower()]
            if len(keep) == len(facts):
                return None
            await asyncio.to_thread(save_facts, keep)
            return QuickResult(reply="Okay, I've forgotten that.")
        return None
