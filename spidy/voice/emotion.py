"""
Emotional delivery for Spidy's voice.

Piper voices have no "emotion" switch, but how a sentence sounds can still be shaped per sentence:
speed, how much the pitch and timing vary (expressiveness), loudness and the pause that follows. Good
news comes out brighter and quicker, questions lift, apologies are slower and softer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceStyle:
    name: str
    speed: float = 1.0          # >1 faster
    expressiveness: float = 0.667  # Piper noise_scale: pitch/energy variation
    rhythm: float = 0.8         # Piper noise_w_scale: timing variation
    volume: float = 1.0
    pause_after: float = 0.08   # seconds of silence after the sentence


CALM = VoiceStyle("calm")
EXCITED = VoiceStyle("excited", speed=1.14, expressiveness=0.85, rhythm=0.75, volume=1.12, pause_after=0.12)
CURIOUS = VoiceStyle("curious", speed=1.0, expressiveness=0.76, rhythm=0.85, volume=1.0, pause_after=0.16)
SORRY = VoiceStyle("sorry", speed=0.9, expressiveness=0.55, rhythm=0.7, volume=0.88, pause_after=0.22)
WARM = VoiceStyle("warm", speed=0.98, expressiveness=0.72, rhythm=0.85, volume=1.0, pause_after=0.12)

_SORRY_WORDS = re.compile(
    r"\b(sorry|unfortunately|sadly|apolog\w*|couldn'?t|can'?t|cannot|unable|failed|failure|error|problem|"
    r"didn'?t work|not working|not able|oops)\b", re.IGNORECASE)
_HAPPY_WORDS = re.compile(
    r"\b(great|awesome|amazing|perfect|excellent|fantastic|wonderful|congrat\w*|yay|wow|love|nice|brilliant|"
    r"done|all set|ready|success\w*|good news)\b", re.IGNORECASE)
_WARM_WORDS = re.compile(
    r"\b(hello|hi|hey|welcome|thank\w*|good (?:morning|afternoon|evening|night)|boss|happy to|glad|"
    r"of course|sure)\b", re.IGNORECASE)


def style_for(sentence: str) -> VoiceStyle:
    """Pick how one sentence should sound."""
    s = sentence.strip()
    if not s:
        return CALM
    if _SORRY_WORDS.search(s):
        return SORRY
    if s.endswith("!") or _HAPPY_WORDS.search(s):
        return EXCITED
    if s.endswith("?"):
        return CURIOUS
    if _WARM_WORDS.search(s):
        return WARM
    return CALM
