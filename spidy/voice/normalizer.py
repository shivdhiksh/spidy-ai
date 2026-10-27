"""
TranscriptNormalizer — ASR Recovery & Speech Normalization Layer
================================================================
Cleans speech-recognition transcripts before Intent Classification and
Task Decomposition.

Key Responsibilities:
---------------------
1. Strip wake-word prefixes ("Hey Spidy", "Wake up Spidy", "Hey Jarvis")
   and ack phrases ("Yes Shiva", "Yes", "Hmm?").
2. Remove noise preambles ("In history, clear the number", "That's Shiva").
3. Repair ASR punctuation artifacts:
   - "Open VS Code and create a Python. File name. Edge." -> "Open VS Code and create a Python file named Edge.py"
   - "File name. Edge.py" -> "file named Edge.py"
   - "File name Edge" -> "file named Edge.py"
4. Normalize spaced extensions:
   - "edge . py" -> "edge.py"
   - "python . py" -> "python.py"
   - "notes . txt" -> "notes.txt"
   - "report . pdf" -> "report.pdf"
5. De-duplicate stuttered/repeated words ("open open notepad" -> "open notepad").
6. Normalize speech arithmetic ("25 times 18", "25 multiplied by 18" -> "25 * 18").
7. Expose structured metadata:
   - raw_transcript
   - normalized_transcript
   - normalization_reason
   - confidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Noise preambles to strip when speech recognition captures preamble noise
_NOISE_PREAMBLES: list[re.Pattern[str]] = [
    re.compile(r"^(?:in history[,.\s]+clear the (?:number|screen)[,.\s]+)", re.IGNORECASE),
    re.compile(r"^(?:that\'s\s+[\w]+[,.\s]+)", re.IGNORECASE),
    re.compile(r"^(?:clear (?:the )?(?:number|screen|history)[,.\s]+)", re.IGNORECASE),
]

# Wake & Ack phrases
_WAKE_ACK_PATTERNS: list[str] = [
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
    "yes shiva",
    "yes",
    "hmm",
]

_BOUNDARY_PUNCT = r"\b[,.\s!?]*(?:and(?:\s+|$))?\s*"


@dataclass
class NormalizationResult:
    """Structured result of transcript cleanup."""
    raw_transcript: str
    normalized_transcript: str
    normalization_reason: str = "clean"
    confidence: float = 1.0

    @property
    def original_transcript(self) -> str:
        return self.raw_transcript


class TranscriptNormalizer:
    """
    Normalizes raw STT transcripts and repairs fragmented ASR text.
    """

    def __init__(self, ack_phrases: list[str] | None = None) -> None:
        all_patterns = list(_WAKE_ACK_PATTERNS)
        if ack_phrases:
            for ack in ack_phrases:
                c = ack.strip().rstrip(".,!?")
                if c and len(c) >= 2:
                    all_patterns.append(c)

        sorted_patterns = sorted(set(all_patterns), key=len, reverse=True)
        escaped = [re.escape(p) for p in sorted_patterns]
        self._wake_ack_re = re.compile(
            rf"^(?:{'|'.join(escaped)}){_BOUNDARY_PUNCT}",
            re.IGNORECASE,
        )

    def normalize(self, raw_transcript: str) -> NormalizationResult:
        """
        Clean, normalize, and repair raw speech transcript.
        """
        raw = (raw_transcript or "").strip()
        if not raw:
            return NormalizationResult(raw_transcript="", normalized_transcript="", confidence=1.0)

        reasons: list[str] = []
        text = raw

        # 1. Strip noise preambles
        for p in _NOISE_PREAMBLES:
            m = p.match(text)
            if m:
                text = text[m.end():].strip()
                reasons.append("stripped_noise_preamble")

        # 2. Strip wake word / ack phrases repeatedly from start
        while True:
            m = self._wake_ack_re.match(text)
            if m:
                text = text[m.end():].strip()
                reasons.append("stripped_wake_ack")
            else:
                break

        if not text:
            return NormalizationResult(
                raw_transcript=raw,
                normalized_transcript="",
                normalization_reason="wake_only",
                confidence=1.0,
            )

        # 3. Repair fragmented noun phrases & ASR punctuation
        # E.g. "create a Python. File name. Edge." -> "create a Python file named Edge.py"
        # "create a python. file name edge" -> "create a python file named edge.py"
        m_file_frag = re.search(
            r'\b(?:create|make|write)\s+(?:a\s+)?(python|react|flask|rust|c\+\+|javascript|typescript|text)\b[.\s,]+file\s+name[.\s,]+([\w\-]+)(?:\.([\w]+))?',
            text,
            re.IGNORECASE,
        )
        if m_file_frag:
            lang = m_file_frag.group(1).strip()
            fname = m_file_frag.group(2).strip()
            ext = m_file_frag.group(3)
            if not ext:
                ext_map = {"python": "py", "javascript": "js", "typescript": "ts", "text": "txt", "rust": "rs"}
                ext = ext_map.get(lang.lower(), "py")
            replacement = f"create a {lang} file named {fname}.{ext}"
            text = text[:m_file_frag.start()] + replacement + text[m_file_frag.end():]
            reasons.append("merged_fragmented_file_phrase")

        # E.g. "file name. edge.py" -> "file named edge.py"
        text_re = re.sub(
            r'\bfile\s+name[.\s,]+([\w\-]+(?:\.[\w]+)?)',
            r'file named \1',
            text,
            flags=re.IGNORECASE,
        )
        if text_re != text:
            text = text_re
            reasons.append("normalized_file_named")

        # 4. Fix spaced extensions: "edge . py" -> "edge.py", "notes . txt" -> "notes.txt"
        text_ext = re.sub(
            r'\b([\w\-]+)\s*\.\s*(py|txt|json|md|csv|pdf|docx|html|js|ts|rs|cpp|h)\b',
            r'\1.\2',
            text,
            flags=re.IGNORECASE,
        )
        if text_ext != text:
            text = text_ext
            reasons.append("fixed_extension_spacing")

        # 5. Normalize arithmetic expressions: "25 times 18" -> "25 * 18"
        text_math = re.sub(
            r'\b(\d+)\s*(?:times|multiplied by|x)\s*(\d+)\b',
            r'\1 * \2',
            text,
            flags=re.IGNORECASE,
        )
        if text_math != text:
            text = text_math
            reasons.append("normalized_arithmetic")

        # 6. Remove stutter / duplicate consecutive words: "open open notepad" -> "open notepad"
        text_dedup = re.sub(
            r'\b([a-zA-Z]{2,})\s+\1\b',
            r'\1',
            text,
            flags=re.IGNORECASE,
        )
        if text_dedup != text:
            text = text_dedup
            reasons.append("deduplicated_repeated_words")

        # 7. Merge sentence splits inside commands:
        # "Search YouTube. For Python tutorials" -> "Search YouTube for Python tutorials"
        text_merge = re.sub(
            r'\b(search\s+[\w]+)\.\s+for\b',
            r'\1 for',
            text,
            flags=re.IGNORECASE,
        )
        if text_merge != text:
            text = text_merge
            reasons.append("merged_search_sentence_break")

        # Clean trailing punctuation
        clean_text = text.strip()
        confidence = 0.98 if reasons else 1.0
        reason_str = ", ".join(reasons) if reasons else "clean"

        log.info(
            "[VOICE] raw='{raw}' | normalized='{norm}' | cleanup_confidence={conf:.2f} reason='{reason}'",
            raw=raw[:60],
            norm=clean_text[:60],
            conf=confidence,
            reason=reason_str,
        )

        return NormalizationResult(
            raw_transcript=raw,
            normalized_transcript=clean_text,
            normalization_reason=reason_str,
            confidence=confidence,
        )
