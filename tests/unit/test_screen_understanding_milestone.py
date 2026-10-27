"""
Tests for Screen Understanding Engine (Phase B)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from spidy.vision.screen_understanding import (
    ScreenUnderstandingEngine,
    ScreenUnderstandingResult,
)


@pytest.mark.asyncio
async def test_screen_understanding_engine_defaults():
    engine = ScreenUnderstandingEngine()
    res = await engine.understand_screen(need_screenshot=False, need_ocr=False)
    assert isinstance(res, ScreenUnderstandingResult)
    assert res.timestamp > 0
    summary = res.summary()
    assert "Active Window:" in summary


@pytest.mark.asyncio
async def test_screen_understanding_error_detection():
    engine = ScreenUnderstandingEngine()
    with patch.object(
        engine,
        "understand_screen",
        new=AsyncMock(
            return_value=ScreenUnderstandingResult(
                active_window="Command Prompt",
                ocr_text="Traceback (most recent call last):\n  File 'app.py', line 5\nSyntaxError: invalid syntax",
                detected_errors=["SyntaxError: invalid syntax"],
                has_error=True,
            )
        ),
    ):
        has_err, errors = await engine.detect_screen_errors()
        assert has_err is True
        assert len(errors) > 0
        assert "SyntaxError" in errors[0]


@pytest.mark.asyncio
async def test_screen_understanding_answer_question():
    engine = ScreenUnderstandingEngine()
    with patch.object(
        engine,
        "understand_screen",
        new=AsyncMock(
            return_value=ScreenUnderstandingResult(
                active_window="Calculator",
                ocr_text="Display 42",
                ui_elements=[{"name": "42", "type": "Text"}],
            )
        ),
    ):
        ans = await engine.answer_screen_question("what is on the screen?")
        assert "Calculator" in ans or "42" in ans
