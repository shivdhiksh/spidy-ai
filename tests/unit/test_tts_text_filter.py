"""
Tests — TTSTextFilter (Stage 1 TTS Latency Fix)
================================================
Comprehensive tests for the markdown-to-speech text filter.

Groups:
  A. Code block removal (the highest-impact fix)
  B. Markdown formatting stripping
  C. List marker removal
  D. URL normalization
  E. Pass-through / edge cases
  F. FilterResult statistics
  G. Disabled filter pass-through
  H. ContinuousVoiceController integration (filter applied to TTS, UI untouched)
  I. StreamingTTSWrapper timing instrumentation (logs present)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.voice.tts_text_filter import FilterResult, TTSTextFilter


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def filt() -> TTSTextFilter:
    """Default-configured filter (enabled=True)."""
    return TTSTextFilter()


@pytest.fixture
def disabled_filt() -> TTSTextFilter:
    return TTSTextFilter(enabled=False)


@pytest.fixture
def custom_filt() -> TTSTextFilter:
    return TTSTextFilter(
        code_block_replacement="Code is ready.",
        url_replacement="a website",
    )


# ── Group A: Code block removal ───────────────────────────────────────────────


class TestCodeBlockRemoval:

    def test_python_code_fence_replaced(self, filt: TTSTextFilter) -> None:
        """Triple-backtick Python code block → replacement sentence."""
        text = "Here's some code:\n```python\ndef hello():\n    print('hi')\n```\nEnjoy!"
        result = filt.filter(text)
        assert "def hello" not in result.tts_text
        assert "print" not in result.tts_text
        assert filt._code_replacement in result.tts_text
        assert result.code_blocks_removed == 1

    def test_plain_code_fence_replaced(self, filt: TTSTextFilter) -> None:
        """Triple-backtick without language tag."""
        text = "```\nsome code here\n```"
        result = filt.filter(text)
        assert "some code here" not in result.tts_text
        assert result.code_blocks_removed == 1

    def test_multiple_code_blocks_all_replaced(self, filt: TTSTextFilter) -> None:
        """Two separate code blocks both replaced."""
        text = (
            "Block 1:\n```python\nfoo()\n```\n"
            "Block 2:\n```javascript\nbar();\n```"
        )
        result = filt.filter(text)
        assert "foo" not in result.tts_text
        assert "bar" not in result.tts_text
        assert result.code_blocks_removed == 2

    def test_tilde_fence_replaced(self, filt: TTSTextFilter) -> None:
        """Triple-tilde code fence also handled."""
        text = "~~~\nsome code\n~~~"
        result = filt.filter(text)
        assert "some code" not in result.tts_text
        assert result.code_blocks_removed == 1

    def test_custom_code_replacement_text(self, custom_filt: TTSTextFilter) -> None:
        """Custom replacement text is used."""
        text = "```python\nprint('hello')\n```"
        result = custom_filt.filter(text)
        assert "Code is ready." in result.tts_text
        assert result.code_blocks_removed == 1

    def test_empty_code_replacement_drops_block(self) -> None:
        """Empty replacement → code block removed with no spoken text."""
        filt = TTSTextFilter(code_block_replacement="")
        text = "Before.\n```python\ncode\n```\nAfter."
        result = filt.filter(text)
        assert "code" not in result.tts_text
        assert result.code_blocks_removed == 1
        assert "Before" in result.tts_text
        assert "After" in result.tts_text

    def test_inline_code_backticks_stripped(self, filt: TTSTextFilter) -> None:
        """Inline `code` → content without backticks."""
        result = filt.filter("Use the `print` function.")
        assert "`" not in result.tts_text
        assert "print" in result.tts_text

    def test_code_block_count_zero_for_plain_text(self, filt: TTSTextFilter) -> None:
        result = filt.filter("This is plain text with no code.")
        assert result.code_blocks_removed == 0


# ── Group B: Markdown formatting ─────────────────────────────────────────────


class TestMarkdownFormatting:

    def test_bold_asterisk_stripped(self, filt: TTSTextFilter) -> None:
        result = filt.filter("This is **important** text.")
        assert "**" not in result.tts_text
        assert "important" in result.tts_text

    def test_bold_underscore_stripped(self, filt: TTSTextFilter) -> None:
        result = filt.filter("This is __important__ text.")
        assert "__" not in result.tts_text
        assert "important" in result.tts_text

    def test_italic_asterisk_stripped(self, filt: TTSTextFilter) -> None:
        result = filt.filter("This is *emphasized* text.")
        assert "*" not in result.tts_text
        assert "emphasized" in result.tts_text

    def test_h1_header_keeps_text(self, filt: TTSTextFilter) -> None:
        result = filt.filter("# Introduction\nThis is the body.")
        assert "#" not in result.tts_text
        assert "Introduction" in result.tts_text

    def test_h2_header_keeps_text(self, filt: TTSTextFilter) -> None:
        result = filt.filter("## Section Two\nSome content.")
        assert "##" not in result.tts_text
        assert "Section Two" in result.tts_text

    def test_strikethrough_stripped(self, filt: TTSTextFilter) -> None:
        result = filt.filter("This is ~~deleted~~ text.")
        assert "~~" not in result.tts_text
        assert "deleted" in result.tts_text

    def test_horizontal_rule_removed(self, filt: TTSTextFilter) -> None:
        result = filt.filter("Before\n---\nAfter")
        assert "---" not in result.tts_text
        assert "Before" in result.tts_text
        assert "After" in result.tts_text

    def test_blockquote_marker_removed(self, filt: TTSTextFilter) -> None:
        result = filt.filter("> This is a quote.")
        assert ">" not in result.tts_text
        assert "This is a quote" in result.tts_text

    def test_html_tags_removed(self, filt: TTSTextFilter) -> None:
        result = filt.filter("Some <b>bold</b> text.")
        assert "<b>" not in result.tts_text
        assert "</b>" not in result.tts_text
        assert "bold" in result.tts_text


# ── Group C: List markers ─────────────────────────────────────────────────────


class TestListMarkers:

    def test_unordered_dash_list(self, filt: TTSTextFilter) -> None:
        text = "Items:\n- First\n- Second\n- Third"
        result = filt.filter(text)
        assert "- First" not in result.tts_text
        assert "First" in result.tts_text
        assert "Second" in result.tts_text

    def test_unordered_asterisk_list(self, filt: TTSTextFilter) -> None:
        text = "* Alpha\n* Beta"
        result = filt.filter(text)
        assert "* Alpha" not in result.tts_text
        assert "Alpha" in result.tts_text

    def test_ordered_list_markers_removed(self, filt: TTSTextFilter) -> None:
        text = "1. First item\n2. Second item\n3. Third item"
        result = filt.filter(text)
        assert "1." not in result.tts_text
        assert "2." not in result.tts_text
        assert "First item" in result.tts_text
        assert "Second item" in result.tts_text


# ── Group D: URL normalization ────────────────────────────────────────────────


class TestURLNormalization:

    def test_bare_https_url_replaced(self, filt: TTSTextFilter) -> None:
        result = filt.filter("Visit https://www.example.com for more.")
        assert "https://" not in result.tts_text
        assert "the link" in result.tts_text

    def test_bare_http_url_replaced(self, filt: TTSTextFilter) -> None:
        result = filt.filter("Go to http://example.com now.")
        assert "http://" not in result.tts_text
        assert "the link" in result.tts_text

    def test_markdown_link_text_kept(self, filt: TTSTextFilter) -> None:
        """[text](url) → text only."""
        result = filt.filter("See [the docs](https://docs.example.com) for details.")
        assert "https://" not in result.tts_text
        assert "the docs" in result.tts_text

    def test_custom_url_replacement(self, custom_filt: TTSTextFilter) -> None:
        result = custom_filt.filter("Visit https://example.com please.")
        assert "a website" in result.tts_text


# ── Group E: Pass-through / edge cases ───────────────────────────────────────


class TestEdgeCases:

    def test_empty_string_returns_empty(self, filt: TTSTextFilter) -> None:
        result = filt.filter("")
        assert result.tts_text == ""
        assert result.original_chars == 0
        assert result.filtered_chars == 0

    def test_plain_text_unchanged_content(self, filt: TTSTextFilter) -> None:
        text = "Python is a programming language."
        result = filt.filter(text)
        assert "Python is a programming language" in result.tts_text

    def test_newlines_converted_to_spaces(self, filt: TTSTextFilter) -> None:
        result = filt.filter("Line one.\nLine two.\nLine three.")
        assert "\n" not in result.tts_text
        assert "Line one" in result.tts_text

    def test_multiple_spaces_collapsed(self, filt: TTSTextFilter) -> None:
        result = filt.filter("Hello   world   test.")
        assert "  " not in result.tts_text
        assert "Hello world test" in result.tts_text


# ── Group F: FilterResult statistics ─────────────────────────────────────────


class TestFilterResultStatistics:

    def test_original_chars_matches_input(self, filt: TTSTextFilter) -> None:
        text = "Hello world."
        result = filt.filter(text)
        assert result.original_chars == len(text)

    def test_filtered_chars_matches_output(self, filt: TTSTextFilter) -> None:
        text = "**Bold** text."
        result = filt.filter(text)
        assert result.filtered_chars == len(result.tts_text)

    def test_code_blocks_reduced_chars(self, filt: TTSTextFilter) -> None:
        """Code block replacement reduces char count for large code."""
        code = "\n".join(f"    line_{i} = {i}" for i in range(20))
        text = f"Here:\n```python\n{code}\n```\nDone."
        result = filt.filter(text)
        # The filtered text must be shorter than the original (code replaced with short phrase)
        assert result.filtered_chars < result.original_chars
        assert result.code_blocks_removed == 1

    def test_filter_result_is_dataclass(self, filt: TTSTextFilter) -> None:
        result = filt.filter("test")
        assert isinstance(result, FilterResult)
        assert hasattr(result, "tts_text")
        assert hasattr(result, "original_chars")
        assert hasattr(result, "filtered_chars")
        assert hasattr(result, "code_blocks_removed")


# ── Group G: Disabled filter ──────────────────────────────────────────────────


class TestDisabledFilter:

    def test_disabled_filter_passes_through_verbatim(self, disabled_filt: TTSTextFilter) -> None:
        text = "```python\nprint('hello')\n```"
        result = disabled_filt.filter(text)
        assert result.tts_text == text

    def test_disabled_filter_no_code_blocks_removed(self, disabled_filt: TTSTextFilter) -> None:
        result = disabled_filt.filter("```python\ncode\n```")
        assert result.code_blocks_removed == 0

    def test_disabled_filter_chars_equal(self, disabled_filt: TTSTextFilter) -> None:
        text = "**Bold** and *italic* and `code`."
        result = disabled_filt.filter(text)
        assert result.original_chars == result.filtered_chars
        assert result.tts_text == text

    def test_enabled_property(self) -> None:
        assert TTSTextFilter(enabled=True).enabled is True
        assert TTSTextFilter(enabled=False).enabled is False


# ── Group H: CVC integration — UI gets original, TTS gets filtered ────────────


class TestCVCIntegration:
    """
    Verify that in ContinuousVoiceController._on_brain_response:
    - The TTS receives the FILTERED text
    - The UI event (BrainResponseReadyEvent.response_text) is NOT filtered
    """

    def _make_cvc(self, filter_enabled: bool = True):
        """Build a minimal CVC with mocked dependencies."""
        from spidy.voice.continuous import ContinuousVoiceController
        from spidy.voice.session import VoiceSessionManager
        from spidy.voice.streaming_tts import StreamingTTSWrapper
        from spidy.voice.interruption import InterruptionHandler
        from spidy.voice.tts_text_filter import TTSTextFilter

        bus = MagicMock()
        bus.subscribe = MagicMock()
        bus.unsubscribe = MagicMock()
        bus.publish = AsyncMock()

        session = MagicMock(spec=VoiceSessionManager)
        session.is_active = True
        session.is_paused = False
        session.session_id = "test-sid"
        session.mark_speaking = AsyncMock()
        session.mark_response_complete = AsyncMock()

        tts = MagicMock(spec=StreamingTTSWrapper)
        tts.is_speaking = False
        tts.speak = AsyncMock()

        interruption = MagicMock(spec=InterruptionHandler)
        brain = MagicMock()

        engine = MagicMock()
        engine._capture_engine.set_mode = MagicMock()
        engine._wake_model.reset_buffer = MagicMock()
        engine.signal_relisten = MagicMock()
        engine._on_brain_response = MagicMock()

        tts_filter = TTSTextFilter(
            code_block_replacement="I've written the code. You can see it in the chat.",
            enabled=filter_enabled,
        )

        cvc = ContinuousVoiceController(
            voice_engine=engine,
            brain=brain,
            bus=bus,
            session_manager=session,
            streaming_tts=tts,
            interruption_handler=interruption,
            tts_filter=tts_filter,
            continuous_mode=False,
            measure_latency=False,
        )
        return cvc, tts

    @pytest.mark.asyncio
    async def test_code_response_filtered_for_tts(self) -> None:
        """Code block in response → TTS receives short phrase, not code."""
        cvc, tts = self._make_cvc(filter_enabled=True)

        event = MagicMock()
        event.response_text = (
            "Sure! Here's the code:\n"
            "```python\n"
            "def area(w, h):\n"
            "    return w * h\n"
            "```\n"
            "This calculates the area."
        )

        await cvc._on_brain_response(event)

        # TTS must NOT have received the raw code
        speak_call_text = tts.speak.call_args[0][0]
        assert "def area" not in speak_call_text
        assert "return w * h" not in speak_call_text
        # TTS must have the replacement phrase
        assert "I've written the code" in speak_call_text

    @pytest.mark.asyncio
    async def test_plain_response_passes_through(self) -> None:
        """Plain prose response → TTS receives it (with minor whitespace normalization)."""
        cvc, tts = self._make_cvc(filter_enabled=True)

        event = MagicMock()
        event.response_text = "Python is a programming language."

        await cvc._on_brain_response(event)

        speak_call_text = tts.speak.call_args[0][0]
        assert "Python" in speak_call_text
        assert "programming language" in speak_call_text

    @pytest.mark.asyncio
    async def test_original_response_text_untouched(self) -> None:
        """The event.response_text attribute is never modified by the filter."""
        cvc, tts = self._make_cvc(filter_enabled=True)

        code_response = (
            "Here:\n```python\nprint('hi')\n```\nDone."
        )
        event = MagicMock()
        event.response_text = code_response

        await cvc._on_brain_response(event)

        # The original event object must be unmodified
        assert event.response_text == code_response
        assert "```python" in event.response_text

    @pytest.mark.asyncio
    async def test_filter_disabled_passes_raw_text(self) -> None:
        """When filter is disabled, raw LLM text (including markdown) goes to TTS."""
        cvc, tts = self._make_cvc(filter_enabled=False)

        event = MagicMock()
        event.response_text = "```python\nprint('hi')\n```"

        await cvc._on_brain_response(event)

        speak_call_text = tts.speak.call_args[0][0]
        # With filter disabled, code fence and content pass through verbatim
        assert "```python" in speak_call_text


# ── Group I: StreamingTTS timing instrumentation ──────────────────────────────


class TestStreamingTTSTimingLogs:
    """Verify timing log entries are emitted (via log.info mock)."""

    @pytest.mark.asyncio
    async def test_response_timing_log_emitted(self) -> None:
        """[TTS] response: chars=... sentences=... log is emitted."""
        from spidy.voice.streaming_tts import StreamingTTSWrapper

        mock_engine = MagicMock()
        mock_engine.speak = AsyncMock()
        mock_engine.is_speaking = False

        wrapper = StreamingTTSWrapper(engine=mock_engine)

        with patch("spidy.voice.streaming_tts.log") as mock_log:
            await wrapper.speak("Hello world. How are you?")

        calls = [str(c) for c in mock_log.info.call_args_list]
        assert any("[TTS] response" in c for c in calls), (
            f"Expected '[TTS] response' log line, got: {calls}"
        )

    @pytest.mark.asyncio
    async def test_per_sentence_timing_log_emitted(self) -> None:
        """[TTS] sentence i/n: chars=... elapsed=... log is emitted per sentence."""
        from spidy.voice.streaming_tts import StreamingTTSWrapper

        mock_engine = MagicMock()
        mock_engine.speak = AsyncMock()
        mock_engine.is_speaking = False

        wrapper = StreamingTTSWrapper(engine=mock_engine)

        with patch("spidy.voice.streaming_tts.log") as mock_log:
            await wrapper.speak("First sentence. Second sentence.")

        calls = [str(c) for c in mock_log.info.call_args_list]
        sentence_logs = [c for c in calls if "[TTS] sentence" in c]
        assert len(sentence_logs) >= 1, (
            f"Expected per-sentence [TTS] logs, got: {calls}"
        )

    @pytest.mark.asyncio
    async def test_total_timing_log_emitted(self) -> None:
        """[TTS] total: sentences=... elapsed=... log is emitted at end."""
        from spidy.voice.streaming_tts import StreamingTTSWrapper

        mock_engine = MagicMock()
        mock_engine.speak = AsyncMock()
        mock_engine.is_speaking = False

        wrapper = StreamingTTSWrapper(engine=mock_engine)

        with patch("spidy.voice.streaming_tts.log") as mock_log:
            await wrapper.speak("Hello. World.")

        calls = [str(c) for c in mock_log.info.call_args_list]
        assert any("[TTS] total" in c for c in calls), (
            f"Expected '[TTS] total' log line, got: {calls}"
        )
