"""
ContextResolver — Anaphora & Reference Resolution
===================================================
Resolves vague references in user utterances against the recent
conversation history before the IntentClassifier processes them.

This enables conversational continuity like:
  User: "Open VS Code"
  User: "Create a Python project"
  User: "Open it"          ← "it" resolved to "VS Code"

  User: "Find my resume"
  User: "Open that file"   ← "that file" resolved to resume path

Strategy
--------
The resolver uses a two-pass approach:

  Pass 1 — Entity extraction
    Scan the last N turns for concrete entities:
      - App names     (resolved from launch_app / close_app intents)
      - File paths    (resolved from open_file / search_files results)
      - URLs          (resolved from open_url / search_web intents)
      - Search queries (resolved from search_* intents)
      - Folder/paths  (resolved from any file/folder operations)

  Pass 2 — Reference matching
    Scan the current utterance for reference patterns:
      - Pronouns: "it", "that", "this", "them", "those"
      - Descriptors: "the file", "the app", "the browser", "the result",
                     "that folder", "the previous one", "my project"
      - Recency: "the latest", "the last one", "that thing"

  When a reference matches, substitute it with the resolved entity value.

Design
------
- Pure Python, no LLM dependency — fast and always available
- Heuristic only: falls back gracefully if no match found
- Returns the rewritten utterance (unchanged if nothing to resolve)
- Also returns a list of ResolvedRef objects for logging/debugging

Thread safety
-------------
ContextResolver is stateless — all state lives in the turns list passed
to resolve(). Safe to call from multiple coroutines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.types import ConversationTurn

log = get_logger(__name__)

# ─── Reference Patterns ───────────────────────────────────────────────────────

# Pronoun-style references that can substitute a single entity
_PRONOUN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bit\b", re.I), "pronoun"),
    (re.compile(r"\bthat\b", re.I), "pronoun"),
    (re.compile(r"\bthis\b", re.I), "pronoun"),
    (re.compile(r"\bthose\b", re.I), "pronoun"),
    (re.compile(r"\bthem\b", re.I), "pronoun"),
]

# Descriptor-style references like "the file", "the app", "that folder"
_DESCRIPTOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bthe\s+(?:file|document|pdf|doc)\b", re.I), "file"),
    (re.compile(r"\bthat\s+(?:file|document|pdf|doc)\b", re.I), "file"),
    (re.compile(r"\bthe\s+(?:app|application|program|window)\b", re.I), "app"),
    (re.compile(r"\bthat\s+(?:app|application|program|window)\b", re.I), "app"),
    (re.compile(r"\bthe\s+(?:browser|chrome|edge|firefox)\b", re.I), "app"),
    (re.compile(r"\bthe\s+(?:folder|directory|project|workspace)\b", re.I), "path"),
    (re.compile(r"\bthat\s+(?:folder|directory|project|workspace)\b", re.I), "path"),
    (re.compile(r"\bthe\s+(?:result|results|output|answer)\b", re.I), "query"),
    (re.compile(r"\bthe\s+(?:previous|last|latest)\s+(?:file|document|search|result)\b", re.I), "any"),
    (re.compile(r"\bmy\s+(?:project|workspace|folder)\b", re.I), "path"),
    (re.compile(r"\bmy\s+(?:resume|cv|document|file)\b", re.I), "file"),
    (re.compile(r"\bmy\s+(?:downloads|desktop)\b", re.I), "path"),
]

# Intent-to-entity-type mapping — what entity does each intent produce?
_INTENT_ENTITY_TYPES: dict[str, str] = {
    "launch_app": "app",
    "close_app": "app",
    "bring_app_to_foreground": "app",
    "open_browser_and_search": "app",
    "open_file": "file",
    "search_files": "file",
    "read_file": "file",
    "open_url": "url",
    "search_web": "query",
    "search_google": "query",
    "search_youtube": "query",
    "search_bing": "query",
    "search_duckduckgo": "query",
    "search_wikipedia": "query",
}

# Special patterns that indicate the user is referencing something, not naming it
_OPEN_WITH_REFERENCE = re.compile(
    r"^(?:open|launch|start|show|go to|use)\s+(?:it|that|this)\s*$",
    re.I,
)
_CLOSE_WITH_REFERENCE = re.compile(
    r"^(?:close|quit|exit|kill)\s+(?:it|that|this)\s*$",
    re.I,
)


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class ResolvedRef:
    """Records a single anaphora resolution for logging/debug."""
    original: str       # What was in the utterance ("it", "the file")
    resolved: str       # What it was replaced with ("VS Code", "/path/to/file")
    entity_type: str    # "app" | "file" | "url" | "query" | "path"


@dataclass
class ResolutionResult:
    """Output of a ContextResolver.resolve() call."""
    utterance: str              # Final utterance (may be rewritten)
    resolved_refs: list[ResolvedRef]   # What was resolved (empty if no changes)

    @property
    def was_resolved(self) -> bool:
        """True if the utterance was modified."""
        return bool(self.resolved_refs)


# ─── Entity Extraction ────────────────────────────────────────────────────────


def _extract_entities_from_turns(
    turns: list["ConversationTurn"],
    max_turns: int = 10,
) -> dict[str, list[str]]:
    """
    Scan recent turns for concrete entities grouped by type.

    Returns a dict:
        {
          "app":   ["VS Code", "Chrome"],
          "file":  ["/path/to/resume.pdf"],
          "url":   ["https://github.com"],
          "query": ["python tutorials"],
          "path":  ["AI Projects"],
        }
    Most recent entities appear first.
    """
    from spidy.brain.types import TurnRole

    entities: dict[str, list[str]] = {
        "app": [],
        "file": [],
        "url": [],
        "query": [],
        "path": [],
    }

    # Walk turns newest-first, limited to max_turns
    recent = list(turns)[-max_turns:]
    for turn in reversed(recent):
        if turn.role != TurnRole.USER or turn.intent is None:
            continue
        intent = turn.intent
        entity_type = _INTENT_ENTITY_TYPES.get(intent.action)
        if entity_type is None:
            continue
        for entity in intent.entities:
            value = entity.value.strip()
            if value and value not in entities.get(entity_type, []):
                entities.setdefault(entity_type, []).append(value)

    return entities


def _best_entity(entities: dict[str, list[str]], preferred_type: str) -> str | None:
    """Return the most recent entity of a preferred type, or any if not found."""
    candidates = entities.get(preferred_type, [])
    if candidates:
        return candidates[0]
    # Fallback: try all types in priority order
    for t in ("app", "file", "path", "url", "query"):
        if t != preferred_type and entities.get(t):
            return entities[t][0]
    return None


# ─── Resolver ─────────────────────────────────────────────────────────────────


class ContextResolver:
    """
    Resolves anaphoric references in utterances using conversation history.

    Parameters
    ----------
    max_history_turns:
        How many recent turns to scan for entity extraction.
        Higher = better recall, slower for very long conversations.
    """

    def __init__(self, max_history_turns: int = 10) -> None:
        self._max_turns = max_history_turns

    def resolve(
        self,
        utterance: str,
        turns: list["ConversationTurn"],
    ) -> ResolutionResult:
        """
        Resolve references in the utterance using conversation history.

        Parameters
        ----------
        utterance:
            The raw user utterance (after any STT processing).
        turns:
            The current conversation window (oldest first).

        Returns
        -------
        ResolutionResult
            Contains the (possibly rewritten) utterance and what was resolved.
        """
        if not turns or not utterance.strip():
            return ResolutionResult(utterance=utterance, resolved_refs=[])

        entities = _extract_entities_from_turns(turns, self._max_turns)

        # Short-circuit: if no entities have been mentioned yet, nothing to resolve
        has_entities = any(v for v in entities.values())
        if not has_entities:
            return ResolutionResult(utterance=utterance, resolved_refs=[])

        resolved_refs: list[ResolvedRef] = []
        rewritten = utterance

        # ── Pattern 1: Whole-utterance shortcuts ("open it", "close that") ────
        if _OPEN_WITH_REFERENCE.match(utterance.strip()):
            app = _best_entity(entities, "app")
            if app:
                verb = utterance.strip().split()[0]
                rewritten = f"{verb} {app}"
                resolved_refs.append(ResolvedRef(
                    original=utterance.strip(),
                    resolved=rewritten,
                    entity_type="app",
                ))
                log.debug(
                    "ContextResolver: '{orig}' → '{resolved}'",
                    orig=utterance.strip(),
                    resolved=rewritten,
                )
                return ResolutionResult(utterance=rewritten, resolved_refs=resolved_refs)

        if _CLOSE_WITH_REFERENCE.match(utterance.strip()):
            app = _best_entity(entities, "app")
            if app:
                verb = utterance.strip().split()[0]
                rewritten = f"{verb} {app}"
                resolved_refs.append(ResolvedRef(
                    original=utterance.strip(),
                    resolved=rewritten,
                    entity_type="app",
                ))
                return ResolutionResult(utterance=rewritten, resolved_refs=resolved_refs)

        # ── Pattern 2: Descriptor substitution ("open the file", "use that app") ──
        for pattern, preferred_type in _DESCRIPTOR_PATTERNS:
            match = pattern.search(rewritten)
            if match:
                entity = _best_entity(entities, preferred_type)
                if entity:
                    rewritten = pattern.sub(entity, rewritten)
                    resolved_refs.append(ResolvedRef(
                        original=match.group(0),
                        resolved=entity,
                        entity_type=preferred_type,
                    ))

        # ── Pattern 3: Pronoun substitution ("search for it", "install that") ──
        # Only applies if no descriptor already resolved it
        if not resolved_refs:
            for pattern, ref_type in _PRONOUN_PATTERNS:
                # Don't substitute pronouns that are part of common phrases like
                # "what is that", "tell me about it" (conversational LLM territory)
                match = pattern.search(rewritten)
                if not match:
                    continue
                # Check that the pronoun is preceded by an action verb
                before = rewritten[: match.start()].strip().lower()
                action_verbs = (
                    "open", "close", "launch", "run", "install", "use",
                    "show", "find", "search for", "delete", "read", "edit",
                    "create in", "open in", "move to",
                )
                if any(before.endswith(v) for v in action_verbs):
                    entity = _best_entity(entities, "any")
                    if entity:
                        rewritten = pattern.sub(entity, rewritten, count=1)
                        resolved_refs.append(ResolvedRef(
                            original=match.group(0),
                            resolved=entity,
                            entity_type="any",
                        ))
                        break  # One pronoun resolution per call

        if resolved_refs:
            log.debug(
                "ContextResolver: resolved {n} ref(s) | '{orig}' → '{resolved}'",
                n=len(resolved_refs),
                orig=utterance[:60],
                resolved=rewritten[:60],
            )

        return ResolutionResult(utterance=rewritten, resolved_refs=resolved_refs)
