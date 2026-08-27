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
# hits "wake up spidy" / "hey jarvis" before shorter prefixes.
_DEFAULT_PATTERNS: list[str] = [
    "wake up spidy",
    "wake up spidey",
    "hey spidy",
    "hey spidey",
    "okay spidy",
    "okay spidey",
    "ok spidy",
    "ok spidey",
    "hey jarvis",
    "okay jarvis",
    "ok jarvis",
    "spidy",
    "spidey",
]

# Boundary pattern: word boundary, trailing punctuation, and optional filler
# conjunction (e.g. "and" in "Wake up Spidy and search...")
_BOUNDARY_PUNCT = r"\b[,.\s!?]*(?:and(?:\s+|$))?\s*"


def _build_pattern(wake_phrases: list[str], ack_phrases: list[str] | None = None) -> re.Pattern[str]:
    r"""
    Compile a single regex that matches any wake phrase (or optional ack preamble) at the start.
    """
    all_phrases = list(wake_phrases)
    if ack_phrases:
        for ack in ack_phrases:
            clean_ack = ack.strip().rstrip(".,!?")
            if clean_ack and len(clean_ack) >= 2:
                all_phrases.append(clean_ack)

    sorted_phrases = sorted(set(all_phrases), key=len, reverse=True)
    escaped = [re.escape(p) for p in sorted_phrases]
    alternation = "|".join(escaped)
    return re.compile(
        rf"^(?:{alternation}){_BOUNDARY_PUNCT}",
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
    ack_phrases:
        Optional list of wake acknowledgement phrases (e.g. from config)
        to strip if caught at the very start of a transcript.
    """

    def __init__(
        self,
        wake_patterns: list[str] | None = None,
        ack_phrases: list[str] | None = None,
    ) -> None:
        self._patterns = wake_patterns or _DEFAULT_PATTERNS
        self._ack_phrases = ack_phrases or []
        self._re = _build_pattern(self._patterns, self._ack_phrases)

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
            whitespace stripped. Empty string if the transcript contained
            only the wake phrase.

        Examples
        --------
        >>> s = WakeWordStripper()
        >>> s.strip_wake_prefix("Hey Spidy, open Edge.")
        'open Edge.'
        >>> s.strip_wake_prefix("Wake up Spidy and search YouTube for Python tutorials.")
        'search YouTube for Python tutorials.'
        >>> s.strip_wake_prefix("Hey Spidy.")
        ''
        >>> s.strip_wake_prefix("what time is it")
        'what time is it'
        >>> s.strip_wake_prefix("Who is Spidy?")
        'Who is Spidy?'
        """
        current = text.strip()
        for _ in range(10):
            if not self.has_wake_prefix(current):
                break
            prev = current
            current = self._re.sub("", current).strip()
            current = current.lstrip(".,!? \t")
            if current == prev:
                break
        return current

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

        This catches single and multi-repetition transcripts like::

            "Hey Spidy"                 → wake-only  → True
            "Wake up Spidy"             → wake-only  → True
            "Hey Jarvis. Hey Jarvis."   → wake-only  → True
            "Wake up Spidy, open Edge"  → False

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

        stripped = self.strip_wake_prefix(current)
        remaining = stripped.strip(".,!? \t\n")
        return len(remaining) == 0 or remaining.lower() in ("and",)

    @property
    def patterns(self) -> list[str]:
        """The configured wake patterns."""
        return list(self._patterns)

