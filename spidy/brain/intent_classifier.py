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
  app_*        : open_app, close_app, switch_app
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
    # Remove leading action words and return the rest as query
    cleaned = re.sub(
        r"^(search|find|look up|google|look for|search for|search web for)\s+",
        "", text.lower().strip()
    )
    if cleaned:
        return [Entity(name="query", value=cleaned)]
    return []


def _extract_app_name(text: str) -> list[Entity]:
    """Extract an app name from the utterance."""
    match = re.search(
        r"(?:open|launch|start|close|quit|exit|switch to|focus)\s+(.+?)(?:\s+(?:app|application))?$",
        text.lower().strip()
    )
    if match:
        return [Entity(name="app_name", value=match.group(1).strip())]
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
    # App operations
    _Rule(
        action="open_app",
        patterns=("open app", "launch", "start app", "open application",
                  "open chrome", "open firefox", "open code",
                  "open notepad", "open calculator", "open spotify",
                  "open vscode", "open visual studio"),
        entity_extractor=_extract_app_name,
    ),
    _Rule(
        action="close_app",
        patterns=("close app", "quit app", "exit app", "close application",
                  "kill app", "close window"),
        entity_extractor=_extract_app_name,
    ),
    _Rule(
        action="switch_app",
        patterns=("switch to", "go to", "bring up", "focus on"),
        entity_extractor=_extract_app_name,
        confidence=_MED_CONFIDENCE,
    ),
    # Web operations
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
