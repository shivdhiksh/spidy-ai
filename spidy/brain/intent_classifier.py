"""
IntentClassifier — Utterance → Intent Mapping
===============================================
Classifies a user utterance into a structured Intent.

Milestone 3 implementation: Heuristic / rule-based
----------------------------------------------------
This classifier uses keyword patterns to classify intents.
It is fast (no LLM call needed) and 100% testable.

The class is designed as a **strategy**: a future LLM-based classifier
can replace this implementation by subclassing ``BaseIntentClassifier``
and overriding ``classify()``. The Brain never depends on the specific
implementation, only on the return type ``Intent``.

Recognised action categories
-----------------------------
  file_*       : open_file, search_files, read_file, delete_file
  app_*        : launch_app, close_app, bring_app_to_foreground, detect_running_apps
  web_*        : search_web, open_url, navigate
  system_*     : set_volume, set_brightness, sleep, lock
  note_*       : take_note, read_notes, find_note
  timer_*      : set_timer, set_alarm, set_reminder
  chat         : general conversation (fallback)

Entity extraction
-----------------
Simple regex / split extraction. Future versions will use NER models.
Entities are attached to the Intent as named Entity objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from spidy.brain.types import Entity, Intent
from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Confidence level for pattern matches
_HIGH_CONFIDENCE = 0.9
_MED_CONFIDENCE = 0.75
_LOW_CONFIDENCE = 0.5


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
    # Remove leading action words and return the rest as query.
    # Longer alternatives must come FIRST so the regex engine does not
    # greedily consume "search" before it can match "search for".
    cleaned = re.sub(
        r"^(search web for|search the web for|search for|look for|look up|"
        r"find out about|find|google|search)\s+",
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
    """
    cleaned = re.sub(
        r"^(search\s+youtube\s+for|search\s+on\s+youtube\s+for"
        r"|search\s+youtube|youtube\s+search\s+for|youtube\s+for"
        r"|youtube)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_app_name(text: str) -> list[Entity]:
    """Extract an app name from the utterance."""
    match = re.search(
        r"(?:open|launch|start|run|close|quit|exit|switch to|focus)\s+(.+?)(?:\s+(?:app|application))?$",
        text.lower().strip()
    )
    if match:
        name = match.group(1).strip()
        # Strip leading article "the" (e.g. "open the notepad" → "notepad")
        name = re.sub(r"^the\s+", "", name).strip()
        if name:
            return [Entity(name="name", value=name)]
    return []


def _extract_filename(text: str) -> list[Entity]:
    """Extract a filename / file query from the utterance."""
    match = re.search(
        r"(?:open|read|find|search for|show me|look for)\s+(.+?)(?:\s+file)?$",
        text.lower().strip()
    )
    if match:
        return [Entity(name="filename", value=match.group(1).strip())]
    return []


def _extract_note_content(text: str) -> list[Entity]:
    """Extract note content from 'take a note: ...' utterances."""
    match = re.search(r"(?:note|remember|write down)[:\s]+(.+)$", text.strip(), re.I)
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


# ─── Rule Definitions ─────────────────────────────────────────────────────────

_RULES: list[_Rule] = [
    # Greetings / identity  — HIGH priority so they beat generic search_web rules
    _Rule(
        action="greet",
        patterns=(
            "hey spidy", "hi spidy", "hello spidy",
            "hi there", "hey there",
            "good morning", "good afternoon", "good evening", "good night",
            "howdy", "greetings",
            # bare single-word greetings — matched via exact contain
            "hi", "hey", "hello",
        ),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="farewell",
        patterns=(
            "goodbye", "bye", "bye bye", "see you", "see ya",
            "farewell", "talk to you later", "good night spidy",
            "bye spidy", "goodbye spidy",
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
    # File operations
    _Rule(
        action="open_file",
        patterns=("open file", "open the file", "read file", "read the file"),
        entity_extractor=_extract_filename,
    ),
    _Rule(
        action="search_files",
        patterns=("find file", "search file", "find files", "look for file",
                  "search for file", "where is the file"),
        entity_extractor=_extract_filename,
    ),
    # App operations — action names MUST match AppSkill.capabilities() exactly
    _Rule(
        action="launch_app",
        patterns=(
            # Generic verb + "app"
            "open app", "start app", "launch app", "run app",
            "open application", "launch application", "start application",
            # Specific apps — bare-name variants
            "open notepad", "launch notepad", "start notepad", "run notepad",
            "open calculator", "launch calculator", "start calculator",
            "open chrome", "launch chrome", "start chrome",
            "open firefox", "launch firefox", "start firefox",
            "open edge", "launch edge",
            "open spotify", "launch spotify", "start spotify",
            "open code", "open vscode", "open visual studio",
            "launch vscode", "launch code",
            "open word", "open excel", "open powerpoint",
            "open terminal", "open cmd", "open powershell",
            "open explorer", "open task manager",
            # Bare "launch" verb with anything following
            "launch",
        ),
        entity_extractor=_extract_app_name,
    ),
    _Rule(
        action="close_app",
        patterns=("close app", "quit app", "exit app", "close application",
                  "kill app", "close window", "close notepad", "close chrome",
                  "close firefox", "close spotify"),
        entity_extractor=_extract_app_name,
    ),
    _Rule(
        action="bring_app_to_foreground",
        patterns=("switch to", "go to", "bring up", "focus on",
                  "bring to front", "show window", "focus window"),
        entity_extractor=_extract_app_name,
        confidence=_MED_CONFIDENCE,
    ),
    _Rule(
        action="search_youtube",
        patterns=(
            "search youtube for", "search youtube", "youtube search for",
            "search on youtube for", "search on youtube",
            "youtube for", "find on youtube", "look up on youtube",
        ),
        entity_extractor=_extract_youtube_query,
        confidence=_HIGH_CONFIDENCE,
    ),
    # Generic web search — must come AFTER search_youtube so "search youtube"
    # doesn't accidentally match "search for" pattern here
    _Rule(
        action="search_web",
        patterns=("search the web", "search web for", "search for",
                  "google", "look up", "find out about",
                  "what is", "who is", "tell me about"),
        entity_extractor=_extract_query,
        confidence=_MED_CONFIDENCE,
    ),
    _Rule(
        action="open_url",
        patterns=("go to website", "open website", "open url",
                  "navigate to", "go to http", "visit"),
        confidence=_HIGH_CONFIDENCE,
    ),
    # System operations
    _Rule(
        action="set_volume",
        patterns=("set volume", "volume up", "volume down", "mute",
                  "increase volume", "decrease volume", "turn up volume",
                  "turn down volume"),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="lock_screen",
        patterns=("lock screen", "lock computer", "lock my pc",
                  "lock the screen"),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="sleep_system",
        patterns=("sleep", "put to sleep", "hibernate"),
        confidence=_HIGH_CONFIDENCE,
    ),
    # Notes
    _Rule(
        action="take_note",
        patterns=("take a note", "note that", "remember this",
                  "write down", "make a note", "note:"),
        entity_extractor=_extract_note_content,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="read_notes",
        patterns=("read my notes", "show notes", "show my notes",
                  "what are my notes", "list notes"),
        confidence=_HIGH_CONFIDENCE,
    ),
    # Timer / Alarm
    _Rule(
        action="set_timer",
        patterns=("set a timer", "timer for", "start timer",
                  "set timer for", "countdown"),
        entity_extractor=_extract_timer_duration,
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="set_alarm",
        patterns=("set an alarm", "alarm at", "wake me at", "alarm for",
                  "remind me at"),
        confidence=_HIGH_CONFIDENCE,
    ),
    # Shutdown / help
    _Rule(
        action="shutdown",
        patterns=("stop listening", "go to sleep", "goodbye", "bye spidy",
                  "shut down", "exit spidy", "stop spidy"),
        confidence=_HIGH_CONFIDENCE,
    ),
    _Rule(
        action="help",
        patterns=("help", "what can you do", "what are your capabilities",
                  "show me what you can do", "list capabilities"),
        confidence=_HIGH_CONFIDENCE,
    ),
]


# ─── Classifier ───────────────────────────────────────────────────────────────


class IntentClassifier:
    """
    Heuristic intent classifier for Milestone 3.

    Classifies utterances against a set of keyword rules.
    Falls back to ``action="chat"`` when no rule matches.

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

        best_action = "chat"
        best_confidence = _LOW_CONFIDENCE
        best_entities: list[Entity] = []

        for rule in self._rules:
            for pattern in rule.patterns:
                if pattern in text_lower:
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
