"""
IntentClassifier — Utterance → Intent Mapping
===============================================
Classifies a user utterance into a structured Intent.

Strategy: Heuristic / rule-based (fast, 100% testable, no LLM call)
------------------------------------------------------------------------
The classifier runs in two stages:

  Stage 1 — Pre-processor checks (URL detection, pure math expressions)
      These are high-confidence patterns that run before any rules.

  Stage 2 — Keyword rule matching
      Ordered rules; higher priority rules are declared first.
      The first rule whose confidence > current best wins.

The class is designed as a **strategy**: a future LLM-based classifier
can replace this implementation by subclassing ``BaseIntentClassifier``
and overriding ``classify()``.

Recognised action categories
-----------------------------
  url          : open_url (bare domain / http URL detected)
  file_*       : open_file, search_files, read_file, delete_file
  app_*        : launch_app, close_app, bring_app_to_foreground, detect_running_apps
  web_*        : search_web, search_google, search_youtube, search_bing,
                 search_duckduckgo, search_wikipedia, search_maps, open_url
  system_*     : set_volume, set_brightness, sleep_system, lock_workstation,
                 shutdown_system, restart_system, empty_recycle_bin, get_system_info
  note_*       : take_note, read_notes, find_note
  timer_*      : set_timer, set_alarm, set_reminder
  calculate    : safe math/expression evaluation
  take_screenshot : capture the screen
  greet/farewell/introduce : social interaction
  chat         : general conversation (fallback)

Entity extraction
-----------------
Simple regex / split extraction. Entities are attached to Intent as
named Entity objects. Future versions will use NER models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from spidy.brain.types import Entity, Intent
from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Confidence levels for pattern matches
_HIGH_CONFIDENCE = 0.9
_MED_CONFIDENCE = 0.75
_LOW_CONFIDENCE = 0.5

# ─── URL Detection ────────────────────────────────────────────────────────────

# Well-known TLDs for bare-domain detection (e.g. "github.com" → open_url)
_KNOWN_TLDS = frozenset({
    "com", "org", "net", "io", "dev", "ai", "co", "gov", "edu",
    "uk", "ca", "au", "de", "fr", "jp", "in", "us", "eu",
})

# Common "navigate to" prefixes to strip before URL extraction
_URL_PREFIXES = re.compile(
    r"^(?:open|go\s+to|navigate\s+to|visit|browse\s+to|launch|take\s+me\s+to)\s+",
    re.IGNORECASE,
)

# Full URL pattern (http/https/www)
_FULL_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)

# Bare domain pattern: something.tld or something.tld/path
_BARE_DOMAIN_RE = re.compile(
    r"\b([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)"
    r"\.([a-zA-Z]{2,10})"
    r"(?:/[^\s]*)?\b",
)


def _normalise_url(raw: str) -> str:
    """Ensure a URL has a scheme; add https:// if missing."""
    raw = raw.strip().rstrip(".")
    if raw.startswith(("http://", "https://")):
        return raw
    return "https://" + raw


def _detect_url(text: str) -> str | None:
    """
    Detect a URL in the utterance.

    Returns the normalised URL string, or None if no URL is found.
    Handles:
    - Full URLs: https://github.com, www.google.com
    - Bare domains: github.com, youtube.com/watch?v=...
    - Prefixed: "open github.com", "go to stackoverflow.com"
    """
    # Strip action prefix
    stripped = _URL_PREFIXES.sub("", text.strip())

    # Check for full URL first
    m = _FULL_URL_RE.search(stripped)
    if m:
        return _normalise_url(m.group(0))

    # Check for bare domain
    m = _BARE_DOMAIN_RE.search(stripped)
    if m:
        tld = m.group(2).lower()
        if tld in _KNOWN_TLDS:
            return _normalise_url(m.group(0))

    return None


# ─── Math Expression Detection ────────────────────────────────────────────────

_MATH_OPERATOR_RE = re.compile(r"[\d\s]*[\+\-\*\/\%\^]+[\d\s]+")
_MATH_WORDS_RE = re.compile(
    r"\b(plus|minus|times|divided\s+by|multiplied\s+by|percent\s+of|"
    r"square\s+root|power\s+of|mod\s+|modulo\s+)\b",
    re.IGNORECASE,
)

# Detects natural-language math expressions like "what is 5 plus 3"
# or "compute 100 divided by 4". Used in Stage 1c of classify().
_WORD_MATH_TRIGGER_RE = re.compile(
    r'(?:\d|(?:calculate|compute|work\s+out|solve|what\s+is|how\s+much\s+is))'  # digit or math keyword
    r'.+?'  # any chars (non-greedy)
    r'\b(?:plus|minus|times|divided\s+by|multiplied\s+by|percent\s+of|'
    r'square\s+root|mod(?:ulo)?|over|power\s+of)\b',  # word operator present
    re.IGNORECASE,
)

def _extract_math_expression(text: str) -> list[Entity]:
    """Extract a math expression from the utterance."""
    # Remove common prefixes
    cleaned = re.sub(
        r"^(calculate|compute|what\s+is|eval|evaluate|solve|work\s+out|"
        r"what\'s|whats)\s+",
        "", text.lower().strip()
    )
    cleaned = re.sub(r"\s*=\s*\??$", "", cleaned).strip()
    if cleaned:
        return [Entity(name="expression", value=cleaned)]
    return []


# ─── Pattern Rule ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Rule:
    """A single classification rule."""
    action: str
    patterns: tuple[str, ...]    # Keyword phrases; any match → intent
    confidence: float = _HIGH_CONFIDENCE
    entity_extractor: Callable[[str], list[Entity]] | None = None


# ─── Entity Extractors ────────────────────────────────────────────────────────

def _extract_query(text: str) -> list[Entity]:
    """Extract the search query from the utterance."""
    cleaned = re.sub(
        r"^(search\s+web\s+for|search\s+the\s+web\s+for|search\s+for|"
        r"look\s+for|look\s+up|find\s+out\s+about|find|google|search|"
        r"bing|query)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_google_query(text: str) -> list[Entity]:
    """Extract the search query for a Google search."""
    cleaned = re.sub(
        r"^(search\s+google\s+for|google\s+search\s+for|search\s+on\s+google\s+for|"
        r"google\s+for|google|search\s+google)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_youtube_query(text: str) -> list[Entity]:
    """Extract the search query from a YouTube search utterance.

    Examples
    --------
    "search youtube for python" → "python"
    "youtube python tutorial"   → "python tutorial"
    "youtube cats"              → "cats"
    """
    cleaned = re.sub(
        r"^(search\s+youtube\s+for|search\s+on\s+youtube\s+for"
        r"|search\s+youtube|youtube\s+search\s+for|youtube\s+for"
        r"|youtube\s+search|youtube)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_bing_query(text: str) -> list[Entity]:
    """Extract the search query for a Bing search."""
    cleaned = re.sub(
        r"^(search\s+bing\s+for|bing\s+search\s+for|search\s+on\s+bing\s+for|"
        r"bing\s+for|bing)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_ddg_query(text: str) -> list[Entity]:
    """Extract the search query for a DuckDuckGo search."""
    cleaned = re.sub(
        r"^(search\s+duckduckgo\s+for|duckduckgo\s+for|duckduckgo|"
        r"duck\s+duck\s+go|search\s+ddg\s+for|ddg)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_wikipedia_query(text: str) -> list[Entity]:
    """Extract the search query for a Wikipedia search."""
    cleaned = re.sub(
        r"^(search\s+wikipedia\s+for|wikipedia\s+for|look\s+up\s+on\s+wikipedia|"
        r"wikipedia|wiki\s+search|search\s+wiki\s+for|wiki)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_maps_query(text: str) -> list[Entity]:
    """Extract a location/directions query."""
    cleaned = re.sub(
        r"^(directions\s+to|get\s+directions\s+to|navigate\s+to|maps\s+to|"
        r"find\s+on\s+maps|google\s+maps|open\s+maps\s+for|show\s+maps\s+for|"
        r"show\s+map\s+of|map\s+of)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_app_name(text: str) -> list[Entity]:
    """Extract an app name from the utterance.

    Stops at conjunctions like " and " so compound commands such as
    "open edge and search for python" correctly extract only "edge".
    """
    match = re.search(
        r"(?:open|launch|start|run|close|quit|exit|switch\s+to|focus)\s+"
        r"(.+?)(?:\s+and\b|\s+(?:app|application))?$",
        text.lower().strip()
    )
    if match:
        name = match.group(1).strip()
        name = re.sub(r"^the\s+", "", name).strip()
        if name:
            return [Entity(name="name", value=name)]
    return []


def _extract_app_and_query(text: str) -> list[Entity]:
    """Extract app name and search query from compound commands.

    Handles utterances like:
      "open edge and search for python"   → app_name="edge", query="python"
      "open chrome and search for cats"   → app_name="chrome", query="cats"
      "launch firefox and google python"  → app_name="firefox", query="python"
    """
    match = re.search(
        r"(?:open|launch|start)\s+(.+?)\s+and\s+"
        r"(?:search|look\s+up|google|find)(?:\s+for)?\s+(.+)$",
        text.lower().strip(),
    )
    if match:
        app_raw = re.sub(r"^the\s+", "", match.group(1).strip()).strip()
        query_raw = match.group(2).strip()
        entities: list[Entity] = []
        if app_raw:
            entities.append(Entity(name="app_name", value=app_raw))
        if query_raw:
            entities.append(Entity(name="query", value=query_raw))
        return entities
    return []


def _extract_filename(text: str) -> list[Entity]:
    """Extract a filename / file query from the utterance."""
    match = re.search(
        r"(?:open|read|find|search\s+for|show\s+me|look\s+for|where\s+is|"
        r"locate|find\s+my)\s+(.+?)(?:\s+file)?$",
        text.lower().strip()
    )
    if match:
        return [Entity(name="filename", value=match.group(1).strip())]
    return []


def _extract_note_content(text: str) -> list[Entity]:
    """Extract note content from 'take a note: ...' utterances."""
    match = re.search(r"(?:note|remember|write\s+down)[:\s]+(.+)$", text.strip(), re.I)
    if match:
        return [Entity(name="content", value=match.group(1).strip())]
    return []


def _extract_timer_duration(text: str) -> list[Entity]:
    """Extract timer duration from utterance."""
    match = re.search(r"(\d+)\s*(second|minute|hour|sec|min|hr)s?", text.lower())
    if match:
        return [
            Entity(name="amount", value=match.group(1)),
            Entity(name="unit", value=match.group(2)),
        ]
    return []


def _extract_folder_name(text: str) -> list[Entity]:
    """Extract folder name from utterances like 'create a folder named AI Projects'.

    Returns Entity(name="folder_name", value="New Folder") as a safe default
    when the user says "create a new folder" without specifying a name, so
    FileSkill always receives a usable folder name.
    """
    # Named folder: "named X", "called X", "named 'X'"
    named = re.search(
        r"(?:named?|called|with\s+name|with\s+the\s+name)\s+['\"]?([\\w\\s\\-\\.]+?)['\"]?(?:\s+on|\s+in|\s+at|$)",
        text.strip(),
        re.IGNORECASE,
    )
    if named:
        return [Entity(name="folder_name", value=named.group(1).strip())]

    # "create a folder X" where X is last word(s)
    fallback = re.search(
        r"(?:create|make|mkdir|new)\s+(?:a\s+)?(?:folder|directory|dir)\s+['\"]?([\w\s\-\.]+?)['\"]?$",
        text.strip(),
        re.IGNORECASE,
    )
    if fallback:
        name = fallback.group(1).strip()
        if name and name.lower() not in ("on the desktop", "on desktop", "here"):
            return [Entity(name="folder_name", value=name)]

    # Default: "create a new folder" with no name → use "New Folder"
    create_re = re.search(
        r"(?:create|make|new|mkdir)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory|dir)\b",
        text.strip(),
        re.IGNORECASE,
    )
    if create_re:
        return [Entity(name="folder_name", value="New Folder")]

    return []


# Spoken folder-name → real filesystem path mapping used by open_folder
_SPOKEN_FOLDER_PATHS: dict[str, str] = {
    "downloads": "downloads",
    "download": "downloads",
    "documents": "documents",
    "document": "documents",
    "desktop": "desktop",
    "pictures": "pictures",
    "photos": "pictures",
    "images": "pictures",
    "music": "music",
    "songs": "music",
    "videos": "videos",
    "movies": "videos",
    "home": "home",
}


def _extract_folder_path(text: str) -> list[Entity]:
    """
    Extract the folder path entity from open_folder utterances.

    Maps spoken shorthand names ("downloads", "desktop") to canonical path
    labels that FileSkill._resolve_parent_path() understands.
    Returns an empty list if no known folder name is found (FileSkill
    will default to the Desktop in that case).
    """
    text_lower = text.lower()
    for spoken, canonical in _SPOKEN_FOLDER_PATHS.items():
        if spoken in text_lower:
            return [Entity(name="path", value=canonical)]
    return []



def _extract_project_name(text: str) -> list[Entity]:
    """Extract project name from utterances like 'open my Portfolio project'."""
    patterns = [
        # "called/named X"
        r"(?:named?|called)\s+['\"]?([\w\s\-\.]+?)['\"]?(?:\s+project|\s+in|\s+with|$)",
        # "open ... Portfolio project"
        r"(?:open|launch|start|load)\s+(?:my\s+)?([\w\s\-\.]+?)\s+(?:project|folder|workspace|repo)",
        # "open Portfolio in vs code"
        r"(?:open|launch)\s+(?:my\s+)?([\w\s\-\.]+?)\s+(?:in|with)\s+(?:vs\s*code|vscode|visual\s+studio)",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name and len(name) >= 2:
                return [Entity(name="project_name", value=name)]
    return []


def _extract_file_type(text: str) -> list[Entity]:
    """Extract file type from 'find all PDF files' type utterances."""
    ext_map = {
        "pdf": "pdf", "pdfs": "pdf",
        "doc": "doc", "docx": "docx", "word": "docx",
        "xls": "xls", "xlsx": "xlsx", "excel": "xlsx",
        "ppt": "ppt", "pptx": "pptx", "powerpoint": "pptx",
        "txt": "txt", "text": "txt",
        "jpg": "jpg", "jpeg": "jpg", "png": "png", "image": "png",
        "mp3": "mp3", "audio": "mp3",
        "mp4": "mp4", "video": "mp4",
        "zip": "zip",
        "py": "py", "python": "py",
        "js": "js", "javascript": "js",
    }
    text_l = text.lower()
    for keyword, ext in ext_map.items():
        if keyword in text_l:
            return [Entity(name="extensions", value=ext)]
    return []


def _extract_volume_params(text: str) -> list[Entity]:
    """Extract volume level or direction from utterance."""
    entities: list[Entity] = []

    # Numeric level: "set volume to 75", "volume 50"
    level_match = re.search(r"\b(\d{1,3})\s*(?:percent|%)?\b", text.lower())
    if level_match:
        level = int(level_match.group(1))
        if 0 <= level <= 100:
            entities.append(Entity(name="level", value=str(level)))
            return entities

    # Direction words
    text_lower = text.lower()
    if any(w in text_lower for w in ("mute", "silence", "quiet")):
        entities.append(Entity(name="direction", value="mute"))
    elif "unmute" in text_lower or "un-mute" in text_lower:
        entities.append(Entity(name="direction", value="unmute"))
    elif any(w in text_lower for w in ("up", "higher", "louder", "increase", "raise", "turn up", "max", "maximum")):
        entities.append(Entity(name="direction", value="up"))
    elif any(w in text_lower for w in ("down", "lower", "quieter", "decrease", "reduce", "turn down", "min", "minimum")):
        entities.append(Entity(name="direction", value="down"))

    return entities


def _extract_brightness_level(text: str) -> list[Entity]:
    """Extract brightness level from utterance."""
    match = re.search(r"\b(\d{1,3})\s*(?:percent|%)?\b", text.lower())
    if match:
        level = int(match.group(1))
        if 0 <= level <= 100:
            return [Entity(name="level", value=str(level))]
    # Direction-based defaults
    text_lower = text.lower()
    if any(w in text_lower for w in ("max", "maximum", "full", "brightest")):
        return [Entity(name="level", value="100")]
    if any(w in text_lower for w in ("min", "minimum", "dim", "dimmest", "off")):
        return [Entity(name="level", value="10")]
    if any(w in text_lower for w in ("up", "higher", "increase", "brighter", "raise")):
        return [Entity(name="level", value="80")]
    if any(w in text_lower for w in ("down", "lower", "decrease", "darker", "reduce")):
        return [Entity(name="level", value="40")]
    return []


# ─── Rule Definitions ─────────────────────────────────────────────────────────
#
# Priority ordering (first matching rule with highest confidence wins):
#   1. Greetings / identity (HIGH) — must beat generic search
#   2. Compound browser+search (HIGH) — before launch_app
#   3. YouTube search (HIGH) — before generic search
#   4. Bing / DuckDuckGo / Wikipedia / Maps (HIGH) — specific engines
#   5. Google / generic web search (MED)
#   6. App operations (HIGH for specific, MED for generic)
#   7. System control (HIGH)
#   8. Files, Notes, Timers (HIGH)
#   9. Calculator / Screenshot (HIGH)
#   10. Help / shutdown (HIGH)
#
_RULES: list[_Rule] = [
    # ── Greetings / Identity ── HIGH priority so they beat generic search ──
    _Rule(
        action="greet",
        patterns=(
            "hey spidy", "hi spidy", "hello spidy",
            "hi there", "hey there",
            "good morning", "good afternoon", "good evening", "good night",
            "howdy", "greetings",
            "hi", "hey", "hello",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="farewell",
        patterns=(
            "goodbye", "bye bye", "see you", "see ya",
            "farewell", "talk to you later", "good night spidy",
            "bye spidy", "goodbye spidy", "bye",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="introduce",
        patterns=(
            "who are you", "what are you", "introduce yourself",
            "tell me about yourself", "what can you do for me",
            "are you an ai", "are you a bot",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Screenshot ─────────────────────────────────────────────────────────
    _Rule(
        action="take_screenshot",
        patterns=(
            "take a screenshot", "take screenshot", "screenshot",
            "capture screen", "capture the screen", "grab screenshot",
            "screen capture", "print screen", "snip screen",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Calculator ─────────────────────────────────────────────────────────
    _Rule(
        action="calculate",
        patterns=(
            "calculate", "compute", "computation", "evaluate",
            "what is the result of", "what is the answer to",
            "solve", "work out", "how much is", "how many is",
        ),
        entity_extractor=_extract_math_expression,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── File / Folder operations ────────────────────────────────────────────
    _Rule(
        action="open_file",
        patterns=("open file", "open the file", "read file", "read the file"),
        entity_extractor=_extract_filename,
    ),
    _Rule(
        action="search_files",
        patterns=(
            # Generic
            "find file", "search file", "find files", "look for file",
            "search for file", "where is the file", "where is my",
            "find my resume", "find my document", "find my pdf",
            "locate file", "find the document",
            # PDF-specific (M13.1)
            "find all pdf", "find pdf", "search for pdf",
            "find all pdf files", "find pdf files",
            "search downloads for pdf", "find pdfs",
            # DOCX / images
            "find all docx", "find word files", "find all word",
            "find all images", "find all photos", "find all pictures",
            "find all mp3", "find all mp4", "find all videos",
            # General type search
            "find all files", "search for files",
        ),
        entity_extractor=lambda t: _extract_file_type(t) or _extract_filename(t),
        confidence=_HIGH_CONFIDENCE,
    ),
    # Create folder (M13.1)
    _Rule(
        action="create_folder",
        patterns=(
            "create a folder", "create folder", "make a folder", "make folder",
            "new folder", "mkdir", "make a directory", "make directory",
            "create a directory", "create directory", "create new folder",
            "create a new folder", "make a new folder", "add a folder",
            "create a folder on the desktop", "make a folder on the desktop",
            "create folder on desktop",
        ),
        entity_extractor=_extract_folder_name,
        confidence=_HIGH_CONFIDENCE,
    ),
    # Delete folder (M13.1)
    _Rule(
        action="delete_folder",
        patterns=(
            "delete folder", "remove folder", "delete directory",
            "remove directory", "delete the folder", "remove the folder",
        ),
        entity_extractor=_extract_folder_name,
        confidence=_HIGH_CONFIDENCE,
    ),
    # Open / reveal folder (M13.1)
    _Rule(
        action="open_folder",
        patterns=(
            "open downloads", "open the downloads folder", "open downloads folder",
            "open documents", "open documents folder", "open the documents folder",
            "open desktop folder", "show me the desktop folder",
            "open pictures folder", "open music folder", "open videos folder",
            "open my documents", "open my downloads", "open my desktop",
            "show folder", "browse folder", "open folder in explorer",
        ),
        entity_extractor=_extract_folder_path,
        confidence=_HIGH_CONFIDENCE,
    ),
    # List recent files (M13.1)
    _Rule(
        action="list_recent_files",
        patterns=(
            "recent files", "recently opened", "recently used files",
            "show recent files", "list recent files", "what did i open recently",
            "show recently used",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Compound browser+search ── BEFORE launch_app ───────────────────────
    _Rule(
        action="open_browser_and_search",
        patterns=(
            "open edge and search", "open chrome and search",
            "open firefox and search", "open browser and search",
            "launch edge and search", "launch chrome and search",
            "launch firefox and search", "open edge and google",
            "open chrome and google", "open edge and look up",
            "open chrome and look up",
        ),
        entity_extractor=_extract_app_and_query,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── YouTube ── before generic search ───────────────────────────────────
    _Rule(
        action="search_youtube",
        patterns=(
            "search youtube for", "search youtube", "youtube search for",
            "search on youtube for", "search on youtube",
            "youtube for", "find on youtube", "look up on youtube",
            "youtube",   # bare "youtube X" pattern — entity extractor handles rest
        ),
        entity_extractor=_extract_youtube_query,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Bing ───────────────────────────────────────────────────────────────
    _Rule(
        action="search_bing",
        patterns=(
            "search bing for", "bing search for", "search on bing for",
            "bing for", "search bing", "bing",
        ),
        entity_extractor=_extract_bing_query,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── DuckDuckGo ─────────────────────────────────────────────────────────
    _Rule(
        action="search_duckduckgo",
        patterns=(
            "search duckduckgo for", "duckduckgo for", "duckduckgo",
            "duck duck go", "search ddg for", "ddg",
            "search privately for",
        ),
        entity_extractor=_extract_ddg_query,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Wikipedia ──────────────────────────────────────────────────────────
    _Rule(
        action="search_wikipedia",
        patterns=(
            "search wikipedia for", "wikipedia for", "wikipedia",
            "wiki search", "search wiki for", "wiki",
            "look up on wikipedia", "look it up on wikipedia",
        ),
        entity_extractor=_extract_wikipedia_query,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Maps / Directions ──────────────────────────────────────────────────
    _Rule(
        action="search_maps",
        patterns=(
            "directions to", "get directions to", "navigate to",
            "show me how to get to", "how do i get to",
            "maps to", "find on maps", "google maps",
            "open maps for", "show maps for", "show map of", "map of",
        ),
        entity_extractor=_extract_maps_query,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Google / generic web search ────────────────────────────────────────
    _Rule(
        action="search_web",
        patterns=(
            "search the web", "search web for", "search for",
            "google search for", "search google for", "google",
            "look up", "find out about",
            "what is", "who is", "tell me about",
            "what are", "how do", "how does", "how to",
            "when did", "where is", "why does", "why is",
        ),
        entity_extractor=_extract_query,
        confidence=_MED_CONFIDENCE,
    ),

    # ── URL navigation ─────────────────────────────────────────────────────
    _Rule(
        action="open_url",
        patterns=(
            "go to website", "open website", "open url",
            "go to http", "visit", "browse to",
            "open https", "go to https",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── App operations ─────────────────────────────────────────────────────
    _Rule(
        action="launch_app",
        patterns=(
            # Generic verb + "app"
            "open app", "start app", "launch app", "run app",
            "open application", "launch application", "start application",
            # Specific apps — expanded for M13.1
            "open notepad", "launch notepad", "start notepad", "run notepad",
            "open calculator", "launch calculator", "start calculator",
            "open chrome", "launch chrome", "start chrome", "run chrome",
            "open google chrome", "launch google chrome",
            "open firefox", "launch firefox", "start firefox",
            "open edge", "launch edge", "open microsoft edge",
            "open spotify", "launch spotify", "start spotify",
            "open code", "open vscode", "open vs code",
            "launch vs code", "launch vscode",
            "open visual studio code", "launch visual studio code",
            "start vs code", "start vscode",
            "open word", "open excel", "open powerpoint",
            "open terminal", "open cmd", "open command prompt",
            "open powershell", "start terminal", "launch terminal",
            "open explorer", "open file explorer", "open task manager",
            "open discord", "launch discord",
            "open teams", "launch teams",
            "open zoom", "launch zoom",
            "open slack", "launch slack",
            "open paint", "launch paint",
            "open snipping tool",
            # macOS / cross-platform aliases (M13.2)
            "open finder", "launch finder", "start finder",  # Finder → File Explorer
            "open file manager", "launch file manager",       # Generic alias
            "open file browser", "launch file browser",       # Generic alias
            "open files",                                     # GNOME Files / generic
            "open iterm", "open iterm2",                      # macOS Terminal aliases
            "open bash", "open zsh", "open shell", "open console",
            # Bare "launch" verb
            "launch",
        ),
        entity_extractor=_extract_app_name,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="close_app",
        patterns=(
            "close app", "quit app", "exit app", "close application",
            "kill app", "close window", "close notepad", "close chrome",
            "close firefox", "close spotify", "close edge",
            "quit chrome", "quit firefox", "kill chrome",
        ),
        entity_extractor=_extract_app_name,
    ),

    # ── Show Desktop (M13.2) ── HIGH priority, BEFORE bring_app_to_foreground ─────
    # Must appear before bring_app_to_foreground so "go to desktop" hits this
    # rule instead of the generic "go to" pattern below.
    _Rule(
        action="show_desktop",
        patterns=(
            "go to the desktop", "go to desktop",
            "show the desktop", "show desktop",
            "view the desktop", "view desktop",
            "minimize all", "minimize all windows", "minimize everything",
            "hide all windows", "show my desktop",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Desktop context menu (M13.2) ───────────────────────────────────────
    # No skill implements this yet; the "no skill" path gives an honest answer.
    _Rule(
        action="desktop_context_menu",
        patterns=(
            "right click on the desktop", "right click the desktop",
            "right-click on the desktop", "right-click the desktop",
            "right click desktop", "desktop right click",
            "context menu on the desktop", "desktop context menu",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    _Rule(
        action="bring_app_to_foreground",
        patterns=(
            # NOTE: "go to" intentionally removed here (M13.2 fix).
            # It was too broad and caused "go to desktop" to misfire.
            "switch to", "bring up", "focus on",
            "bring to front", "show window", "focus window",
            "switch back to", "go back to",
        ),
        entity_extractor=_extract_app_name,
        confidence=_MED_CONFIDENCE,
    ),

    # ── System control ─────────────────────────────────────────────────────
    _Rule(
        action="set_volume",
        patterns=(
            "set volume", "volume up", "volume down", "mute",
            "increase volume", "decrease volume", "turn up volume",
            "turn down volume", "unmute", "volume to",
            "turn volume up", "turn volume down", "raise volume",
            "lower volume", "max volume", "minimum volume",
        ),
        entity_extractor=_extract_volume_params,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="set_brightness",
        patterns=(
            "set brightness", "brightness to", "increase brightness",
            "decrease brightness", "dim the screen", "dim screen",
            "brighten screen", "screen brightness", "turn up brightness",
            "turn down brightness", "lower brightness", "raise brightness",
            "make screen brighter", "make screen darker",
        ),
        entity_extractor=_extract_brightness_level,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="lock_workstation",
        patterns=(
            "lock screen", "lock computer", "lock my pc", "lock the screen",
            "lock workstation", "lock my computer", "lock session",
            "secure the computer", "lock windows", "lock my laptop",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="sleep_system",
        patterns=(
            "put to sleep", "sleep mode", "hibernate",
            "put computer to sleep", "suspend",
            "put my computer to sleep", "put laptop to sleep",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="shutdown_system",
        patterns=(
            "shutdown", "shut down", "shut down the computer",
            "shut down my computer", "power off", "turn off computer",
            "turn off my computer", "turn off the pc", "power down",
            "shut down laptop", "shutdown computer", "shutdown pc",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="restart_system",
        patterns=(
            "restart", "reboot", "restart the computer",
            "reboot the computer", "restart my computer",
            "reboot system", "restart laptop", "reboot laptop",
            "restart windows", "restart pc",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="empty_recycle_bin",
        patterns=(
            "empty recycle bin", "clear recycle bin",
            "empty trash", "delete recycle bin contents",
            "clean recycle bin",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="get_system_info",
        patterns=(
            "system info", "system status", "computer status",
            "cpu usage", "ram usage", "memory usage", "battery level",
            "battery status", "how is my computer doing",
            "check system", "what's my cpu",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Notes ──────────────────────────────────────────────────────────────
    _Rule(
        action="take_note",
        patterns=(
            "take a note", "note that", "remember this",
            "write down", "make a note", "note:",
            "add a note", "save a note",
        ),
        entity_extractor=_extract_note_content,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="read_notes",
        patterns=(
            "read my notes", "show notes", "show my notes",
            "what are my notes", "list notes", "read notes",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Timer / Alarm ──────────────────────────────────────────────────────
    _Rule(
        action="set_timer",
        patterns=(
            "set a timer", "timer for", "start timer",
            "set timer for", "countdown", "start a timer",
            "remind me in", "set countdown",
        ),
        entity_extractor=_extract_timer_duration,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="set_alarm",
        patterns=(
            "set an alarm", "alarm at", "wake me at", "alarm for",
            "remind me at", "set alarm", "wake me up at",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Clipboard ──────────────────────────────────────────────────────────
    _Rule(
        action="get_clipboard",
        patterns=(
            "what's in the clipboard", "show clipboard", "read clipboard",
            "paste contents", "clipboard contents", "what did i copy",
            "show me the clipboard",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Help ───────────────────────────────────────────────────────────────
    _Rule(
        action="help",
        patterns=(
            "help", "what can you do", "what are your capabilities",
            "show me what you can do", "list capabilities",
            "list commands", "show commands", "what commands",
            "how do i use you", "user guide",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Webpage / Knowledge operations (M13.1) ─────────────────────────────

    _Rule(
        action="read_webpage",
        patterns=(
            "read this webpage", "read this page", "read the current page",
            "read the webpage", "read current webpage",
            "summarize this webpage", "summarize this page",
            "summarize the current page", "summarize the current webpage",
            "summarize current page", "summarize webpage",
            "what does this page say", "what is on this page",
            "summarize current browser page", "read the current browser page",
            "extract text from this page", "extract webpage content",
            "get the text from this page", "read the open webpage",
            "summarize what's on this page",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Project / Workspace operations (M13.1) ─────────────────────────────
    _Rule(
        action="open_project",
        patterns=(
            "open my project", "open my portfolio", "open portfolio project",
            "open portfolio in vs code", "open my portfolio in vs code",
            "open project in vs code", "open project in vscode",
            "open my project in vs code", "open my project folder",
            "load my project", "load project",
            "open the project", "show my project",
        ),
        entity_extractor=_extract_project_name,
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Goal cancellation (M13.1) ───────────────────────────────────────────
    _Rule(
        action="cancel_goal",
        patterns=(
            "cancel", "cancel that", "cancel the current goal",
            "cancel goal", "stop that", "stop what you're doing",
            "never mind", "nevermind", "abort", "abort that",
            "forget it", "stop the task", "stop the goal",
            "cancel the task", "halt", "stop everything",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Terminal / Shell operations (M13.1) ────────────────────────────────
    _Rule(
        action="open_terminal",
        patterns=(
            "open a terminal", "open terminal", "open a command prompt",
            "open command prompt", "open cmd", "open powershell",
            "start a terminal", "launch terminal", "get me a terminal",
            "open a shell", "open shell",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Running apps detection (M13.1) ─────────────────────────────────────
    _Rule(
        action="detect_running_apps",
        patterns=(
            "what is running", "what's running", "list open apps",
            "show running apps", "what apps are open", "running apps",
            "list running processes", "what processes are running",
            "show open applications", "what is currently open",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),

    # ── Project creation (M13.1) ────────────────────────────────────────────
    # Catches "create a X project" / "new X project" patterns before they
    # fall through to chat. TaskDecomposer handles the actual decomposition.
    _Rule(
        action="create_project",
        patterns=(
            # Generic
            "create a project", "create project", "new project",
            "start a project", "start new project",
            # Typed
            "create a react project", "create react project", "new react project",
            "create a python project", "create python project", "new python project",
            "create a flask project", "create flask project", "new flask project",
            "create a node project", "create node project",
            "create a django project", "create django project",
            "create a vue project", "create vue project",
            "create an angular project", "create angular project",
            "create a fastapi project", "create fastapi project",
            "create a typescript project", "create typescript project",
            "build a react app", "build a python app", "build a web app",
            "make a react app", "make a python app",
        ),
        entity_extractor=_extract_project_name,
        confidence=_HIGH_CONFIDENCE,
    ),
]


# ─── Compound Intent Detection ────────────────────────────────────────────────
#
# Detects utterances that contain multiple sequential instructions, e.g.:
#   "Create a folder named AI Projects, open it in VS Code, create main.py"
#   "open edge and search for python, then open notepad"
#
# Strategy: split on a set of coordination patterns, classify each part,
# and return action="compound" with sub-intents encoded as JSON entities.

# Patterns that signal task chaining between distinct instructions
_COMPOUND_SPLIT_RE = re.compile(
    r"""(?x)          # verbose mode
    \s*               # optional leading space
    (?:               # non-capturing group
        ,\s*then\b    # ", then"
      | ,\s*after\s+that\b  # ", after that"
      | ,\s*next\b    # ", next"
      | ,\s*finally\b # ", finally"
      | ,\s*also\b    # ", also"
      | ;\s*          # semicolon
      | \bthen\s+     # "then " (standalone)
      | \bafter\s+that\s+  # "after that "
      | \band\s+then\b     # "and then"
    )
    """,
    re.IGNORECASE,
)

# A compound utterance must have at least this many parts to trigger multi-step
_MIN_COMPOUND_PARTS = 2
# Upper bound on parts to prevent runaway splits
_MAX_COMPOUND_PARTS = 8
# Minimum part length (chars) to be considered a real instruction
_MIN_PART_LENGTH = 4


def _split_compound_utterance(text: str) -> list[str] | None:
    """
    Attempt to split a compound utterance into ordered sub-instructions.

    Returns None if the utterance doesn't look compound (single intent).
    Returns a list of 2+ non-empty part strings if compound.
    """
    parts = _COMPOUND_SPLIT_RE.split(text.strip())
    # Filter out empty / trivial parts
    cleaned = [
        p.strip()
        for p in parts
        if p and p.strip() and len(p.strip()) >= _MIN_PART_LENGTH
    ]
    if len(cleaned) >= _MIN_COMPOUND_PARTS:
        return cleaned[:_MAX_COMPOUND_PARTS]
    return None


# ─── Classifier ───────────────────────────────────────────────────────────────


class IntentClassifier:
    """
    Heuristic intent classifier for Spidy v1.0.

    Classifies utterances against a set of keyword rules.
    Falls back to ``action="chat"`` when no rule matches.

    Stage 1 — Pre-processors: URL detection, pure math expressions
    Stage 2 — Keyword rule matching

    Future improvement: Replace or augment with an LLM-based classifier
    by overriding the ``classify()`` method in a subclass.

    Parameters
    ----------
    min_confidence:
        Minimum confidence required to act on an intent.
        Below this threshold, the DecisionEngine uses ``CLARIFY``.
    """

    def __init__(self, min_confidence: float = 0.6) -> None:
        self._min_confidence = min_confidence
        self._rules = _RULES

    async def classify(self, utterance: str) -> Intent:
        """
        Classify a raw utterance into an Intent.

        Parameters
        ----------
        utterance:
            The user's spoken or typed input.

        Returns
        -------
        Intent
            Always returns an Intent — never raises.
            action="chat" is returned when no rule matches.
        """
        if not utterance or not utterance.strip():
            return Intent(
                action="chat",
                confidence=_LOW_CONFIDENCE,
                raw_utterance=utterance,
                source="heuristic",
            )

        text_lower = utterance.lower().strip()

        # ── Stage 1d: Compound intent detection ──────────────────────────
        #
        # Check for multi-step utterances BEFORE all other classification
        # so that "create a folder, open VS Code, create main.py" is
        # treated as 3 sequential steps rather than misclassified.
        #
        # We skip compound detection when:
        # - The utterance starts with a greeting / farewell / intro
        # - The utterance is a math expression (handled by 1b/1c)
        _skip_compound = (
            text_lower.startswith(("hi ", "hey ", "hello", "bye", "who are", "what are", "what is", "how do", "how does"))
            or bool(_MATH_OPERATOR_RE.search(text_lower))
            or bool(_WORD_MATH_TRIGGER_RE.search(text_lower))
        )
        if not _skip_compound:
            sub_parts = _split_compound_utterance(utterance)
            if sub_parts and len(sub_parts) >= _MIN_COMPOUND_PARTS:
                # Classify each sub-part independently and encode as entities
                sub_intents_data: list[dict] = []
                for part in sub_parts:
                    sub = await self._classify_single(part)
                    sub_intents_data.append({
                        "utterance": part,
                        "action": sub.action,
                        "entities": [
                            {"name": e.name, "value": e.value}
                            for e in sub.entities
                        ],
                        "confidence": sub.confidence,
                    })
                log.debug(
                    "Compound intent detected: {n} parts",
                    n=len(sub_intents_data),
                )
                return Intent(
                    action="compound",
                    entities=tuple(
                        Entity(
                            name=f"step_{i}",
                            value=json.dumps(data),
                        )
                        for i, data in enumerate(sub_intents_data)
                    ),
                    confidence=_HIGH_CONFIDENCE,
                    raw_utterance=utterance,
                    source="heuristic_compound",
                )

        # ── Stage 1: Pre-processors ──────────────────────────────────────

        # 1a. URL detection (highest priority: "open github.com" is unambiguous)
        detected_url = _detect_url(utterance)
        if detected_url:
            log.debug(
                "URL detected in utterance → open_url: {url}",
                url=detected_url,
            )
            return Intent(
                action="open_url",
                entities=(Entity(name="url", value=detected_url),),
                confidence=_HIGH_CONFIDENCE,
                raw_utterance=utterance,
                source="heuristic_url",
            )

        # 1b. Pure math expression detection (symbol operators):
        #     "15 * 32", "100 / 4 + 7" — no keyword needed
        if _MATH_OPERATOR_RE.fullmatch(text_lower.strip()) or (
            _MATH_OPERATOR_RE.search(text_lower) and not any(
                kw in text_lower for kw in ("open", "launch", "search", "find", "go")
            )
        ):
            entities = _extract_math_expression(utterance)
            if entities:
                log.debug("Math expression detected → calculate")
                return Intent(
                    action="calculate",
                    entities=tuple(entities),
                    confidence=_HIGH_CONFIDENCE,
                    raw_utterance=utterance,
                    source="heuristic_math",
                )

        # 1c. Natural-language math: "what is 5 plus 3", "compute 100 divided by 4"
        #     Matches utterances with at least one digit AND a word math operator.
        #     Must run BEFORE Stage 2 keyword rules so search_web doesn't steal them.
        if _WORD_MATH_TRIGGER_RE.search(text_lower):
            entities = _extract_math_expression(utterance)
            log.debug("Word-operator math detected → calculate")
            return Intent(
                action="calculate",
                entities=tuple(entities),
                confidence=_HIGH_CONFIDENCE,
                raw_utterance=utterance,
                source="heuristic_math",
            )

        return await self._classify_single(utterance)

    async def _classify_single(self, utterance: str) -> Intent:
        """
        Classify a single (non-compound) utterance through URL/math/rule pipeline.

        This is the original classify() logic, refactored so compound intent
        detection can re-use it for each sub-part.
        """
        text_lower = utterance.lower().strip()

        # Stage 1a: URL
        detected_url = _detect_url(utterance)
        if detected_url:
            return Intent(
                action="open_url",
                entities=(Entity(name="url", value=detected_url),),
                confidence=_HIGH_CONFIDENCE,
                raw_utterance=utterance,
                source="heuristic_url",
            )

        # Stage 1b: Pure math
        if _MATH_OPERATOR_RE.fullmatch(text_lower.strip()) or (
            _MATH_OPERATOR_RE.search(text_lower) and not any(
                kw in text_lower for kw in ("open", "launch", "search", "find", "go")
            )
        ):
            entities = _extract_math_expression(utterance)
            if entities:
                return Intent(
                    action="calculate",
                    entities=tuple(entities),
                    confidence=_HIGH_CONFIDENCE,
                    raw_utterance=utterance,
                    source="heuristic_math",
                )

        # Stage 1c: Word-math
        if _WORD_MATH_TRIGGER_RE.search(text_lower):
            entities = _extract_math_expression(utterance)
            return Intent(
                action="calculate",
                entities=tuple(entities),
                confidence=_HIGH_CONFIDENCE,
                raw_utterance=utterance,
                source="heuristic_math",
            )

        # ── Stage 2: Keyword rule matching ───────────────────────────────

        best_action = "chat"
        best_confidence = _LOW_CONFIDENCE
        best_entities: list[Entity] = []

        for rule in self._rules:
            for pattern in rule.patterns:
                # Use word-boundary matching to avoid false positives:
                # - Short patterns (≤4 chars): "hi", "bye" can match inside longer words
                # - Single-word patterns (no spaces, ≤8 chars): "compute" in "computer"
                is_single_word = " " not in pattern
                if len(pattern) <= 4 or (is_single_word and len(pattern) <= 8):
                    matched = bool(re.search(r'\b' + re.escape(pattern) + r'\b', text_lower))
                else:
                    matched = pattern in text_lower

                if matched:
                    if rule.confidence > best_confidence:
                        best_confidence = rule.confidence
                        best_action = rule.action
                        best_entities = (
                            rule.entity_extractor(utterance)
                            if rule.entity_extractor
                            else []
                        )
                    break  # Found a match for this rule, stop checking patterns

        if best_action != "chat":
            log.debug(
                "Intent classified: {action} ({conf:.0%}) ← '{utt}'",
                action=best_action,
                conf=best_confidence,
                utt=utterance[:60],
            )
        else:
            log.debug(
                "No rule matched — defaulting to chat: '{utt}'",
                utt=utterance[:60],
            )

        return Intent(
            action=best_action,
            entities=tuple(best_entities),
            confidence=best_confidence,
            raw_utterance=utterance,
            source="heuristic",
        )

    @staticmethod
    def extract_compound_steps(intent: "Intent") -> list[dict]:
        """
        Parse the sub-intent data from a compound Intent's entities.

        Parameters
        ----------
        intent:
            A compound Intent (action=="compound") produced by classify().

        Returns
        -------
        list[dict]
            Ordered list of sub-intent dicts:
            ``{"utterance": str, "action": str, "entities": list, "confidence": float}``
        """
        steps: list[dict] = []
        for entity in intent.entities:
            if entity.name.startswith("step_"):
                try:
                    data = json.loads(entity.value)
                    steps.append(data)
                except (json.JSONDecodeError, ValueError):
                    pass
        return steps
