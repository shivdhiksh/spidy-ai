"""
WakeWordStripper — Transcript Wake-Prefix Normalizer
======================================================
Removes wake-word prefixes from STT transcripts *before* intent
classification.

Problem
-------
The STT engine captures the full audio including the wake phrase.
When the user says "Hey Jarvis, open Edge", the transcript is:
    "Hey Jarvis. Open Edge."

Without stripping, the Brain receives the wake phrase and may respond
as if it were the command (e.g. greeting instead of opening Edge).

Design rules
------------
1. Only strip when the wake phrase appears at the VERY START of the
   transcript (prefix match).
2. Punctuation at the boundary (, . ! ?) is consumed.
3. Leading/trailing whitespace in the remainder is stripped.
4. The match is case-insensitive.
5. If the transcript is ONLY the wake phrase (no command follows),
   the result is an empty string — the caller treats this as a
   no-command utterance.
6. "jarvis" or any individual word is NEVER stripped mid-sentence.
   Only known full wake patterns are removed from the prefix.

Supported patterns (configurable)
----------------------------------
    hey jarvis
    hey spidy
    hey spidy!
    ok jarvis
    okay jarvis
    wake up spidy

Usage
-----
    stripper = WakeWordStripper()
    clean = stripper.strip_wake_prefix("Hey Jarvis, open Edge.")
    # → "open Edge."

    clean = stripper.strip_wake_prefix("Hey Jarvis.")
    # → ""  (no command)

    clean = stripper.strip_wake_prefix("what is Python")
    # → "what is Python"  (no wake phrase — unchanged)
"""

from __future__ import annotations

import re

# Default wake patterns in order of longest-first so the greedy match
# hits "hey jarvis" before "jarvis" if both were listed.
_DEFAULT_PATTERNS: list[str] = [
    "hey jarvis",
    "okay jarvis",
    "ok jarvis",
    "wake up spidy",   # future custom model phrase (strip only; no model yet)
    "hey spidy",
    "okay spidy",
    "ok spidy",
]

# Punctuation characters that may follow the wake phrase before the command.
_BOUNDARY_PUNCT = r"[,.\s!?]*"


def _build_pattern(wake_phrases: list[str]) -> re.Pattern[str]:
    r"""
    Compile a single regex that matches any wake phrase at the start.

    Each phrase is escaped and joined with | (alternation).
    The boundary group consumes trailing punctuation and whitespace.

    Example compiled pattern (simplified)::

        ^(hey jarvis|ok jarvis)[,.\s!?]*\s*
    """
    escaped = [re.escape(p) for p in wake_phrases]
    alternation = "|".join(escaped)
    return re.compile(
        rf"^(?:{alternation}){_BOUNDARY_PUNCT}\s*",
        re.IGNORECASE,
    )


class WakeWordStripper:
    """
    Strips wake-word prefixes from STT transcripts.

    Parameters
    ----------
    wake_patterns:
        List of wake phrases to recognise (case-insensitive).
        Defaults to the standard Spidy/Jarvis phrases.
    """

    def __init__(
        self,
        wake_patterns: list[str] | None = None,
    ) -> None:
        self._patterns = wake_patterns or _DEFAULT_PATTERNS
        self._re = _build_pattern(self._patterns)

    def strip_wake_prefix(self, text: str) -> str:
        """
        Remove the wake-word prefix from *text* if present.

        Parameters
        ----------
        text:
            Raw STT transcript.

        Returns
        -------
        str
            The transcript with the wake prefix removed and surrounding
            whitespace stripped.  Empty string if the transcript contained
            only the wake phrase.

        Examples
        --------
        >>> s = WakeWordStripper()
        >>> s.strip_wake_prefix("Hey Jarvis, open Edge.")
        'open Edge.'
        >>> s.strip_wake_prefix("Hey Jarvis. What is Python?")
        'What is Python?'
        >>> s.strip_wake_prefix("Hey Jarvis.")
        ''
        >>> s.strip_wake_prefix("what time is it")
        'what time is it'
        >>> s.strip_wake_prefix("I asked Jarvis about the weather")
        'I asked Jarvis about the weather'
        """
        stripped = self._re.sub("", text).strip()
        return stripped

    def has_wake_prefix(self, text: str) -> bool:
        """
        Return True if *text* starts with a recognised wake phrase.

        Parameters
        ----------
        text:
            Raw STT transcript.
        """
        return bool(self._re.match(text))

    def is_wake_only(self, text: str) -> bool:
        """
        Return True when *text* consists entirely of wake phrase(s) — with
        no real command remaining.

        This catches multi-repetition transcripts like::

            "Hey Jarvis. Hey Jarvis."   → wake-only  → True
            "Hey Jarvis, Hey Jarvis."   → wake-only  → True
            "Okay Jarvis. Okay Jarvis." → wake-only  → True
            "Hey Jarvis"                → wake-only  → True

        But preserves legitimate commands::

            "Hey Jarvis, open Edge."            → False
            "Hey Jarvis, what is Python?"       → False
            "Who is Jarvis?"                    → False  (no wake prefix)

        Algorithm
        ---------
        Repeatedly strip the wake prefix until no more prefix remains.
        If the final remainder is empty (or blank punctuation only), the
        entire text was composed of wake phrases.

        Parameters
        ----------
        text:
            Raw STT transcript (not yet stripped).

        Returns
        -------
        bool
            True if the transcript is wake-phrase-only.
        """
        current = text.strip()
        if not current:
            return False  # empty string — handled by the empty-text guard elsewhere

        # Strip wake prefixes iteratively until no more match at the head.
        # Safety cap: max 10 iterations (a real sentence won't have 10 wake
        # phrases, so this prevents any pathological infinite-loop risk).
        for _ in range(10):
            if not self.has_wake_prefix(current):
                break
            current = self._re.sub("", current).strip()
            # After stripping, remove any leading punctuation-only debris
            # (e.g. a period left behind between phrases like "Jarvis. Hey…")
            current = current.lstrip(".,!? \t")

        # If nothing meaningful remains → wake-only
        # "Meaningful" = at least one word character that is NOT part of
        # another wake phrase starter.
        remaining = current.strip(".,!? \t\n")
        return len(remaining) == 0

    @property
    def patterns(self) -> list[str]:
        """The configured wake patterns."""
        return list(self._patterns)
