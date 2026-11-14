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
# Like a phone assistant, Spidy is never "closed" by a casual request: "close yourself" only dismisses
# her (window hidden, current task stopped) and she keeps listening for "Hey Spidy". Exiting the program
# takes an unmistakable request: shut down / turn off / power off / terminate, or "quit ... completely".
_FULL = r"(?:completely|fully|entirely|for\s+good|totally)"
_QUIT_RE = re.compile(
    r"\b(?:"
    rf"(?:terminate|kill|turn\s+off|power\s+off|shut\s*(?:down|off))\s+(?:your\s*self|yourselves|{_NAME})"
    r"|shut\s+(?:your\s*self)\s+(?:down|off)"
    r"|turn\s+(?:your\s*self)\s+off"
    rf"|{_NAME}\W+(?:please\W+)?(?:shut\s*(?:down|off)|turn\s+off|power\s+off)"
    rf"|(?:close|quit|exit|stop)\s+(?:your\s*self|{_NAME}|the\s+app|the\s+program|spidy\s+app)\s+{_FULL}"
    rf"|(?:quit|exit|close)\s+{_FULL}"
    r")\b",
    re.IGNORECASE,
)
_DISMISS_RE = re.compile(
    r"\b(?:"
    rf"(?:close|quit|exit|hide|minimi[sz]e|dismiss)\s+(?:your\s*self|yourselves|{_NAME})"
    rf"|{_NAME}\W+(?:please\W+)?(?:quit|exit|close|go\s+away|go\s+to\s+sleep|hide|dismiss)"
    rf"|(?:go\s+away|go\s+to\s+sleep|that'?s\s+all(?:\s+for\s+now)?|(?:good)?\s*bye\s+{_NAME})"
    r")\b",
    re.IGNORECASE,
)


def is_quit_request(text: str) -> bool:
    """True if the user wants to exit Spidy completely (shut down / turn off / quit completely)."""
    return bool(text and _QUIT_RE.search(text))


def is_dismiss_request(text: str) -> bool:
    """True if the user wants Spidy out of the way for now ("close yourself"); she keeps listening."""
    return bool(text and not is_quit_request(text) and _DISMISS_RE.search(text))


DISMISS_REPLY = "Okay! I'll be right here. Just say Hey Spidy when you need me."
QUIT_FAREWELL = "Okay, closing now. Goodbye!"


# "Who developed you?" must always get the same answer. The language model would otherwise credit the
# company whose API powers it (NVIDIA, OpenAI...), so this is answered here and never reaches the model.
_ASK = r"(?:developed|created|made|built|designed|programmed|coded|trained|invented|owns?|founded|authored|wrote)"
_YOU = rf"(?:you|u|{_NAME}|this\s+(?:assistant|app|program|software))"
_CREATOR_RE = re.compile(
    r"\b(?:"
    rf"(?:who|whom|which\s+(?:company|person|team)|what\s+(?:company|person|team))\s+(?:\w+\s+){{0,3}}{_ASK}\s+{_YOU}"
    rf"|who(?:'s|\s+is|\s+was|\s+are)\s+(?:your|{_NAME}(?:'s)?)\s+(?:developer|developers|creator|creators|maker|makers|author|owner|boss|inventor|programmer|founder|parent)"
    rf"|who(?:'s|\s+is|\s+was)\s+behind\s+{_YOU}"
    rf"|(?:are|were)\s+you\s+{_ASK}\s+by\b"
    rf"|(?:tell\s+me\s+)?(?:about\s+)?(?:your|{_NAME}(?:'s)?)\s+(?:developer|creator|maker)"
    r")",
    re.IGNORECASE,
)


def is_creator_question(text: str) -> bool:
    """True if the user asks who developed / created / owns Spidy."""
    return bool(text and _CREATOR_RE.search(text))


CREATOR_REPLY = (
    "I'm SPIDY, an AI desktop assistant developed by my boss Shiva Dhikshith. My architecture, features, "
    "and capabilities are built by my developer. I use AI models and services from providers like NVIDIA "
    "to power some of my intelligence, but Shiva Dhikshith Sir is my creator and developer."
)
