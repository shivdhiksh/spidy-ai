"""
TTSTextFilter — Pre-TTS Response Text Cleaner
==============================================
Converts LLM markdown responses into clean, natural-sounding prose
suitable for speech synthesis, WITHOUT modifying the original text
shown in the UI/chat.

Problem this solves
-------------------
LLM responses often contain:
  - Markdown formatting  (**bold**, *italic*, `code`)
  - Headers              (# Section, ## Subsection)
  - Bullet/ordered lists (-, *, 1., 2.)
  - Code fences          (```python ... ```)
  - URLs                 (https://...)
  - Excessive whitespace / newlines

A TTS engine reads these literally, causing:
  - "asterisk asterisk bold asterisk asterisk"
  - "hash hash header"
  - Code blocks spoken word-by-word over 30+ seconds

This filter strips or replaces those elements so Spidy speaks naturally.

Scope
-----
- Applied ONLY to the text passed to TTS
- The original response_text is NEVER modified
- Purely deterministic string transformation — no LLM calls
- Configurable via voice.tts_filter in SpidyConfig

Design principles
-----------------
1. Safe over clever: prefer explicit rules over complex regex
2. Fail open: if a rule fails, return the partially-cleaned text
3. Measurable: return a FilterResult with char counts and block stats
4. No state: TTSTextFilter is a stateless transformer

Usage
-----
    filt = TTSTextFilter(code_block_replacement="I've written the code. You can see it in the chat.")
    result = filt.filter("Here's the code:\\n```python\\nprint('hi')\\n```\\nEnjoy!")
    # result.tts_text  → "Here's the code. I've written the code. You can see it in the chat. Enjoy!"
    # result.code_blocks_removed  → 1
    # result.original_chars       → 54
    # result.filtered_chars       → 72
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from spidy.logging.logger import get_logger

log = get_logger(__name__)

# ── Compiled patterns (module-level for performance) ──────────────────────────

# Fenced code blocks: ```lang\n...\n``` or ~~~lang\n...\n~~~
# Non-greedy to avoid merging multiple blocks.
_CODE_FENCE = re.compile(
    r'(`{3,}|~{3,})[^\n]*\n.*?\1',
    re.DOTALL,
)

# Inline code: `something`
_INLINE_CODE = re.compile(r'`[^`\n]+`')

# Markdown headers: # H1 / ## H2 / ### H3 etc.  (up to 6 levels)
_HEADER = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)

# Bold: **text** or __text__
_BOLD = re.compile(r'\*{2}(.+?)\*{2}|_{2}(.+?)_{2}', re.DOTALL)

# Italic: *text* or _text_ (single markers, not already consumed by bold)
_ITALIC = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)')

# Strikethrough: ~~text~~
_STRIKETHROUGH = re.compile(r'~~(.+?)~~', re.DOTALL)

# Unordered list markers at start of line: -, *, +
_UNORDERED_LIST = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)

# Ordered list markers: 1. 2. 3.
_ORDERED_LIST = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)

# Markdown links: [text](url) → keep text only
_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')

# Bare URLs: http(s)://...
_URL = re.compile(r'https?://\S+')

# Horizontal rules: --- or *** or ___
_HORIZONTAL_RULE = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)

# Blockquote markers: > text
_BLOCKQUOTE = re.compile(r'^\s*>\s?', re.MULTILINE)

# HTML tags
_HTML_TAG = re.compile(r'<[^>]+>')

# Multiple consecutive blank lines → single newline
_MULTI_BLANK = re.compile(r'\n{3,}')

# Multiple spaces
_MULTI_SPACE = re.compile(r' {2,}')


@dataclass
class FilterResult:
    """
    Result of a TTSTextFilter.filter() call.

    Attributes
    ----------
    tts_text:
        The cleaned text ready for TTS synthesis.
    original_chars:
        Character count of the input text.
    filtered_chars:
        Character count of tts_text.
    code_blocks_removed:
        Number of fenced code blocks replaced.
    """
    tts_text: str
    original_chars: int
    filtered_chars: int
    code_blocks_removed: int = 0


class TTSTextFilter:
    """
    Converts markdown LLM responses to clean TTS-ready prose.

    Parameters
    ----------
    code_block_replacement:
        Text to substitute for each fenced code block.
        Default: "I've written the code. You can see it in the chat."
    url_replacement:
        Text to substitute for bare URLs.
        Default: "the link"
    enabled:
        Master switch. When False, filter() is a pass-through.
    """

    def __init__(
        self,
        code_block_replacement: str = "I've written the code. You can see it in the chat.",
        url_replacement: str = "the link",
        enabled: bool = True,
    ) -> None:
        self._code_replacement = code_block_replacement.strip()
        self._url_replacement = url_replacement.strip()
        self._enabled = enabled

    # ── Public API ────────────────────────────────────────────────────────

    def filter(self, text: str) -> FilterResult:
        """
        Apply all filters to produce TTS-ready text.

        The input text is NEVER modified in place — a new string is returned.
        Fails open: any exception returns the best partial result available.

        Parameters
        ----------
        text:
            Raw LLM response (may contain markdown, code blocks, etc.)

        Returns
        -------
        FilterResult
            Cleaned text + statistics.
        """
        if not self._enabled:
            return FilterResult(
                tts_text=text,
                original_chars=len(text),
                filtered_chars=len(text),
                code_blocks_removed=0,
            )

        original_chars = len(text)

        try:
            cleaned, code_blocks_removed = self._apply_all(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("TTSTextFilter: unexpected error (fail-open): {exc}", exc=exc)
            cleaned = text
            code_blocks_removed = 0

        return FilterResult(
            tts_text=cleaned,
            original_chars=original_chars,
            filtered_chars=len(cleaned),
            code_blocks_removed=code_blocks_removed,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _apply_all(self, text: str) -> tuple[str, int]:
        """Apply all rules in order. Returns (cleaned_text, code_blocks_removed)."""
        t = text

        # 1. Count and replace fenced code blocks first (before other rules
        #    consume their content)
        code_blocks_removed = len(_CODE_FENCE.findall(t))
        if code_blocks_removed > 0:
            t = _CODE_FENCE.sub(self._code_replacement, t)

        # 2. Remove HTML tags
        t = _HTML_TAG.sub('', t)

        # 3. Strip inline code backticks (keep content)
        t = _INLINE_CODE.sub(lambda m: m.group(0)[1:-1], t)

        # 4. Strip markdown headers → keep header text
        t = _HEADER.sub(lambda m: m.group(1), t)

        # 5. Strip bold/italic → keep inner text
        t = _BOLD.sub(lambda m: m.group(1) or m.group(2) or '', t)
        t = _ITALIC.sub(lambda m: m.group(1) or m.group(2) or '', t)

        # 6. Strip strikethrough
        t = _STRIKETHROUGH.sub(lambda m: m.group(1), t)

        # 7. Blockquotes → keep content
        t = _BLOCKQUOTE.sub('', t)

        # 8. Horizontal rules → remove
        t = _HORIZONTAL_RULE.sub('', t)

        # 9. List markers → remove (keep content)
        t = _UNORDERED_LIST.sub('', t)
        t = _ORDERED_LIST.sub('', t)

        # 10. Links → keep link text, discard URL
        t = _LINK.sub(lambda m: m.group(1), t)

        # 11. Bare URLs → replacement word
        t = _URL.sub(self._url_replacement, t)

        # 12. Collapse whitespace
        t = _MULTI_BLANK.sub('\n\n', t)
        t = t.replace('\n', ' ')       # newlines → spaces for TTS flow
        t = _MULTI_SPACE.sub(' ', t)
        t = t.strip()

        return t, code_blocks_removed
