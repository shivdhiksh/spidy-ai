"""
Commands addressed to Spidy itself (as opposed to the computer).

"Close yourself" used to fall through to the chat model, which answered "I can't close
myself". These are recognised deterministically so they actually work.
"""

from __future__ import annotations

import re

# Whisper spells the name many ways ("Spidy", "Spidey", "smarty"...), so the verb +
# "yourself" form is accepted on its own; the name form needs a spelling of Spidy.
# "stop" is deliberately NOT a quit word: "Spidy stop" is the barge-in phrase that only
# interrupts speech.
_NAME = r"(?:spid(?:y|ey|ie|i)|spiddy)"
_QUIT_RE = re.compile(
    r"\b(?:"
    rf"(?:close|quit|exit|terminate|kill|turn\s+off|power\s+off|shut\s*(?:down|off))"
    rf"\s+(?:your\s*self|yourselves|{_NAME})"
    r"|shut\s+(?:your\s*self)\s+(?:down|off)"
    r"|turn\s+(?:your\s*self)\s+off"
    rf"|{_NAME}\W+(?:please\W+)?(?:shut\s*(?:down|off)|quit|exit)"
    r")\b",
    re.IGNORECASE,
)


def is_quit_request(text: str) -> bool:
    """True if the user is asking Spidy to close/exit itself."""
    return bool(text and _QUIT_RE.search(text))


QUIT_FAREWELL = "Okay, closing now. Goodbye!"
