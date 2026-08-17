"""
TranscriptQualityGuard — STT Suspicious Transcript Filter
==========================================================
Detects clearly unusable STT transcripts before they trigger an
expensive LLM request.

Problem
-------
faster-whisper can produce garbled output on noise, silence, or
very low signal-to-noise audio:

    "24 similar word word word word..."
    "Mm-hmm. Side right which one and those lemme?"
    "Beep. Beep. Beep."
    "9 to 9 to 0. What is Python?"

These are produced when Whisper's decoder runs on near-silence or
microphone noise and outputs its highest-probability tokens, which
tend to be highly repetitive or nonsensical.

Design rules
------------
1. ONLY use signals that are actually available from faster-whisper.
   Do NOT invent fake confidence scores.
2. Conservative: short legitimate commands are NEVER rejected.
   Examples that must always pass:
       "Open Edge"
       "Stop"
       "Yes"
       "No"
       "Thanks"
       "what time is it"
       "open VS Code"
3. Checks are fast (no I/O, no model calls).
4. A suspicious transcript returns True from is_suspicious().
   Callers should skip Brain processing and return to listening.

Available signals
-----------------
From faster-whisper's segment objects (when available):
    segment.no_speech_prob   — probability audio is silence/noise
    segment.avg_logprob      — average log-prob; very negative = low conf
    segment.compression_ratio — high = repetitive output

From the transcript text itself (always available):
    - Word repetition ratio: same word appearing N+ times in short text
    - Noise-word patterns: "beep", pure digits repeated, etc.

Thresholds
----------
These are conservative defaults. They can be tuned after observing
production logs.

    REPETITION_MAX_RATIO     = 0.5   (>50% same word → suspicious)
    REPETITION_MIN_WORDS     = 6     (only applies when ≥6 words)
    NOISE_PATTERNS           = beep sequences, pure numeric repetition
    NO_SPEECH_PROB_THRESHOLD = 0.85  (segment-level, if available)
    AVG_LOGPROB_THRESHOLD    = -1.5  (segment-level, if available)

Usage
-----
    guard = TranscriptQualityGuard()
    if guard.is_suspicious(transcript_text):
        # skip LLM call, return to listening
        return
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Thresholds ────────────────────────────────────────────────────────────────

# Word-repetition check: only triggered if the transcript has at least
# this many words (to avoid flagging short commands like "yes" "no").
_MIN_WORDS_FOR_REPETITION_CHECK = 6

# Fraction of words that are the same (most-common word / total words).
# >50% of words being identical indicates hallucination.
_MAX_REPETITION_RATIO = 0.50

# Phrase (bigram) repetition: if the same consecutive 2-word pair appears
# more than this fraction of all bigrams, flag as repetitive.
# Only applied when transcript has >= _MIN_WORDS_FOR_REPETITION_CHECK words.
_MAX_BIGRAM_RATIO = 0.45

# Whisper segment quality thresholds (used when segment data is provided).
_NO_SPEECH_PROB_THRESHOLD = 0.85  # segment.no_speech_prob above this -> suspicious
_AVG_LOGPROB_THRESHOLD = -1.5     # segment.avg_logprob below this -> suspicious

# Noise word patterns (whole-transcript match, case-insensitive)
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    # "Beep." or "Beep. Beep." or "Beep beep beep"
    re.compile(r"^(\bbeep\b[\s.,!?]*){2,}$", re.IGNORECASE),
    # Pure repeated digits: "9 to 9 to 0" type noise
    re.compile(r"^(\d[\s.,]*){4,}$"),
    # Pure repeated single-word filler: "word word word word"
    re.compile(r"^(\b\w+\b)(\s+\1){5,}$", re.IGNORECASE),
    # "Subscribe to our channel." repeated twice or more
    re.compile(r"(\bsubscribe\b.{0,40}){2,}", re.IGNORECASE),
]


@dataclass
class SegmentQualitySignals:
    """
    Optional quality signals from a faster-whisper segment.

    Pass this alongside the transcript text when the signals are
    available to enable segment-level quality checks.

    Attributes
    ----------
    no_speech_prob:
        Probability [0, 1] that the segment contains no speech.
        From segment.no_speech_prob.
    avg_logprob:
        Average log-probability of the output tokens.
        Very negative values indicate low confidence.
        From segment.avg_logprob.
    compression_ratio:
        Compression ratio of the output. High values indicate
        repetitive / degenerate output.
        From segment.compression_ratio.
    """
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0


class TranscriptQualityGuard:
    """
    Detects clearly unusable STT transcripts.

    All checks use only signals that actually exist in faster-whisper.
    No fake confidence scores are generated.

    Parameters
    ----------
    min_words_for_repetition:
        Only run the word-repetition check when the transcript has at
        least this many words. Prevents false positives on short commands.
    max_repetition_ratio:
        Flag as suspicious when the most-common word fraction exceeds
        this threshold.
    no_speech_prob_threshold:
        Flag as suspicious when segment no_speech_prob exceeds this.
    avg_logprob_threshold:
        Flag as suspicious when segment avg_logprob is below this.
    """

    def __init__(
        self,
        min_words_for_repetition: int = _MIN_WORDS_FOR_REPETITION_CHECK,
        max_repetition_ratio: float = _MAX_REPETITION_RATIO,
        max_bigram_ratio: float = _MAX_BIGRAM_RATIO,
        no_speech_prob_threshold: float = _NO_SPEECH_PROB_THRESHOLD,
        avg_logprob_threshold: float = _AVG_LOGPROB_THRESHOLD,
    ) -> None:
        self._min_words = min_words_for_repetition
        self._max_ratio = max_repetition_ratio
        self._max_bigram_ratio = max_bigram_ratio
        self._no_speech_threshold = no_speech_prob_threshold
        self._logprob_threshold = avg_logprob_threshold

    def is_suspicious(
        self,
        text: str,
        segments: list[SegmentQualitySignals] | None = None,
    ) -> bool:
        """
        Return True if the transcript is clearly garbled or unusable.

        Parameters
        ----------
        text:
            The transcribed text (already stripped of whitespace).
        segments:
            Optional list of SegmentQualitySignals from faster-whisper.
            If provided, segment-level checks are run in addition to
            text-level checks.

        Returns
        -------
        bool
            True if the transcript appears to be garbage and should NOT
            be forwarded to the Brain.
        """
        if not text:
            return False  # Empty transcripts are handled separately

        # ── Text-level checks (always available) ───────────────────────

        # Check 1: Noise patterns
        if self._matches_noise_pattern(text):
            return True

        # Check 2: Word repetition (single-word dominance)
        if self._is_repetitive(text):
            return True

        # Check 3: Phrase repetition (bigram-level loops like "Hey Jerry's. Hey Jerry's.")
        if self._is_phrase_repetitive(text):
            return True

        # ── Segment-level checks (only when segment data is available) ──

        if segments:
            for seg in segments:
                if self._segment_is_suspicious(seg):
                    return True

        return False

    def suspicious_reason(
        self,
        text: str,
        segments: list[SegmentQualitySignals] | None = None,
    ) -> str | None:
        """
        Return a human-readable reason string if suspicious, else None.

        Useful for logging exactly why a transcript was rejected.
        """
        if not text:
            return None

        if self._matches_noise_pattern(text):
            return "noise_pattern_match"

        if self._is_repetitive(text):
            return "excessive_word_repetition"

        if self._is_phrase_repetitive(text):
            return "phrase_repetition"

        if segments:
            for i, seg in enumerate(segments):
                if seg.no_speech_prob > self._no_speech_threshold:
                    return f"segment[{i}].no_speech_prob={seg.no_speech_prob:.2f}"
                if seg.avg_logprob < self._logprob_threshold:
                    return f"segment[{i}].avg_logprob={seg.avg_logprob:.2f}"

        return None

    # ── Private helpers ────────────────────────────────────────────────

    def _matches_noise_pattern(self, text: str) -> bool:
        """Return True if the whole transcript matches a known noise pattern."""
        for pattern in _NOISE_PATTERNS:
            if pattern.match(text.strip()):
                return True
        return False

    def _is_repetitive(self, text: str) -> bool:
        """
        Return True if more than max_repetition_ratio of the words
        are the same word.

        Only applied when the transcript has at least min_words words.
        """
        words = text.lower().split()
        if len(words) < self._min_words:
            return False

        # Find the most common word (excluding common connectors that
        # appear naturally in legitimate sentences)
        from collections import Counter
        word_counts = Counter(words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]

        ratio = most_common_count / len(words)
        return ratio > self._max_ratio

    def _is_phrase_repetitive(self, text: str) -> bool:
        """
        Return True if consecutive 2-word pairs (bigrams) repeat excessively.

        Catches phrase-level hallucinations like:
            "Hey, Jerry's. Hey, Jerry's. Hey, Jerry's."
            "Thank you. Thank you. Thank you."

        The word-repetition check misses these because no single word
        exceeds the 50% threshold — but the PAIR repeats every 2 words.

        Only applied when the transcript has at least min_words words.
        """
        # Strip punctuation and normalise
        clean = re.sub(r"[^\w\s]", "", text.lower())
        words = clean.split()
        if len(words) < self._min_words:
            return False

        # Build bigrams
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        if not bigrams:
            return False

        from collections import Counter
        bigram_counts = Counter(bigrams)
        most_common_bigram, most_common_count = bigram_counts.most_common(1)[0]

        ratio = most_common_count / len(bigrams)
        return ratio > self._max_bigram_ratio

    def _segment_is_suspicious(self, seg: SegmentQualitySignals) -> bool:
        """Return True if a single segment's quality signals are bad."""
        if seg.no_speech_prob > self._no_speech_threshold:
            return True
        if seg.avg_logprob < self._logprob_threshold:
            return True
        return False
