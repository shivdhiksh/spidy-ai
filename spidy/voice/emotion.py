"""
Emotional delivery for Spidy's voice.

Piper voices have no "emotion" switch, so each sentence is shaped the way people actually change their
voice when they feel something:

* **pitch** -- excited and happy speech sits higher and *rises*; sad speech sits lower and *falls*.
  Piper cannot change pitch, so the sentence is synthesised slightly slower and then resampled
  (see :func:`apply_pitch`), which raises/lowers the pitch while keeping the timing.
* **speed**, **loudness** and how much the voice varies (Piper's noise scales),
* the **pause** after the sentence (sad and thoughtful sentences linger).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoiceStyle:
    name: str
    speed: float = 1.0             # >1 faster
    expressiveness: float = 0.667  # Piper noise_scale: pitch/energy variation
    rhythm: float = 0.8            # Piper noise_w_scale: timing variation
    volume: float = 1.0
    pause_after: float = 0.08      # seconds of silence after the sentence
    pitch: float = 1.0             # pitch at the start of the sentence (1.1 = 10% higher)
    pitch_end: float | None = None  # pitch at the end (None = same as start): a rising/falling contour


CALM = VoiceStyle("calm", pitch=1.0, pitch_end=0.98)
EXCITED = VoiceStyle("excited", speed=1.1, expressiveness=0.9, rhythm=0.75, volume=1.15, pause_after=0.14,
                     pitch=1.1, pitch_end=1.15)
HAPPY = VoiceStyle("happy", speed=1.05, expressiveness=0.8, rhythm=0.8, volume=1.08, pause_after=0.12,
                   pitch=1.06, pitch_end=1.08)
SURPRISED = VoiceStyle("surprised", speed=1.04, expressiveness=0.95, rhythm=0.8, volume=1.12, pause_after=0.18,
                       pitch=1.14, pitch_end=1.06)
CURIOUS = VoiceStyle("curious", speed=1.0, expressiveness=0.78, rhythm=0.85, volume=1.0, pause_after=0.16,
                     pitch=1.0, pitch_end=1.1)
SORRY = VoiceStyle("sorry", speed=0.86, expressiveness=0.5, rhythm=0.7, volume=0.84, pause_after=0.3,
                   pitch=0.95, pitch_end=0.89)
CARING = VoiceStyle("caring", speed=0.92, expressiveness=0.6, rhythm=0.8, volume=0.92, pause_after=0.22,
                    pitch=0.98, pitch_end=0.95)
WARM = VoiceStyle("warm", speed=0.98, expressiveness=0.74, rhythm=0.85, volume=1.02, pause_after=0.12,
                  pitch=1.03, pitch_end=1.02)

_SORRY_WORDS = re.compile(
    r"\b(sorry|unfortunately|sadly|apolog\w*|couldn'?t|can'?t|cannot|unable|failed|failure|error|problem|"
    r"didn'?t work|not working|not able|oops|oh no|aww|that'?s sad|bad news|i'?m afraid)\b", re.IGNORECASE)
_CARING_WORDS = re.compile(
    r"\b(don'?t worry|it'?s okay|it'?s alright|i understand|take care|i'?m here|you'?ve got this|"
    r"feel better|rest|be careful|stay safe|hang in there)\b", re.IGNORECASE)
_SURPRISE_WORDS = re.compile(r"^(?:oh|whoa|wow|really|no way|what)\b|\b(whoa|no way|incredible|unbelievable)\b",
                             re.IGNORECASE)
_EXCITED_WORDS = re.compile(
    r"\b(awesome|amazing|fantastic|incredible|congrat\w*|yay|woohoo|hooray|let'?s go|brilliant|wow)\b",
    re.IGNORECASE)
_HAPPY_WORDS = re.compile(
    r"\b(great|perfect|excellent|wonderful|love|nice|done|all set|ready|success\w*|good news|glad|happy|"
    r"sure thing|of course|here you go)\b", re.IGNORECASE)
_WARM_WORDS = re.compile(
    r"\b(hello|hi|hey|welcome|thank\w*|good (?:morning|afternoon|evening|night)|boss|happy to|"
    r"sure)\b", re.IGNORECASE)


def style_for(sentence: str) -> VoiceStyle:
    """Pick how one sentence should sound."""
    s = sentence.strip()
    if not s:
        return CALM
    if _SORRY_WORDS.search(s):
        return SORRY
    if _CARING_WORDS.search(s):
        return CARING
    if _SURPRISE_WORDS.search(s) and (s.endswith("!") or s.endswith("?")):
        return SURPRISED
    if _EXCITED_WORDS.search(s) or (s.endswith("!") and _HAPPY_WORDS.search(s)):
        return EXCITED
    if s.endswith("!") or _HAPPY_WORDS.search(s):
        return HAPPY
    if s.endswith("?"):
        return CURIOUS
    if _WARM_WORDS.search(s):
        return WARM
    return CALM


def synth_speed(style: VoiceStyle | None) -> float:
    """Speed to synthesise at so that, after the pitch shift, the sentence has the style's speed."""
    if style is None:
        return 1.0
    mean_pitch = (style.pitch + (style.pitch_end if style.pitch_end is not None else style.pitch)) / 2
    return style.speed / mean_pitch


def apply_pitch(samples: np.ndarray, style: VoiceStyle | None) -> np.ndarray:
    """
    Shift the pitch of ``samples`` along the style's contour by resampling.

    Reading the waveform faster raises the pitch and shortens it; the sentence was synthesised
    correspondingly slower (:func:`synth_speed`), so the timing comes out right.
    """
    if style is None or samples.size < 2:
        return samples
    p0 = style.pitch
    p1 = style.pitch_end if style.pitch_end is not None else p0
    if abs(p0 - 1.0) < 1e-3 and abs(p1 - 1.0) < 1e-3:
        return samples
    n_out = max(2, int(len(samples) / ((p0 + p1) / 2)))
    rate = np.linspace(p0, p1, n_out)
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    pos = np.clip(pos, 0, len(samples) - 1)
    out = np.interp(pos, np.arange(len(samples)), samples.astype(np.float32))
    return out.astype(np.float32)
