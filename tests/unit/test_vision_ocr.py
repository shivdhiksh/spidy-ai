"""
Tests for spidy.vision.ocr — OCREngine

All tests mock easyocr so they run without installing it (~500 MB).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spidy.vision.ocr import OCREngine
from spidy.vision.types import OCRBlock, OCRResult


# ─── Availability ─────────────────────────────────────────────────────────────


class TestOCREngineAvailability:
    def test_is_not_available_when_easyocr_absent(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", True):
                engine = OCREngine()
                assert engine.is_available is False

    def test_is_not_available_when_numpy_absent(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", True):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", False):
                engine = OCREngine()
                assert engine.is_available is False

    def test_is_available_when_both_present(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", True):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", True):
                engine = OCREngine()
                assert engine.is_available is True

    def test_constructor_defaults(self):
        engine = OCREngine()
        assert engine._language == "en"
        assert engine._confidence_threshold == 0.5
        assert engine._gpu is False

    def test_constructor_custom(self):
        engine = OCREngine(language="fr", confidence_threshold=0.7, gpu=True)
        assert engine._language == "fr"
        assert engine._confidence_threshold == 0.7
        assert engine._gpu is True


# ─── read_image() when unavailable ────────────────────────────────────────────


class TestReadImageUnavailable:
    def test_returns_empty_ocr_result_when_unavailable(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            engine = OCREngine()
            result = engine.read_image(b"\x89PNG")
            assert isinstance(result, OCRResult)
            assert result.available is False
            assert result.text == ""
            assert result.confidence == 0.0
            assert result.block_count == 0

    def test_logs_warning_once(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            engine = OCREngine()
            # Call twice — should only warn once
            engine.read_image(b"")
            engine.read_image(b"")
            assert engine._unavailable_warned is True

    def test_does_not_raise(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            engine = OCREngine()
            # Must not raise any exception
            result = engine.read_image(b"garbage data")
            assert isinstance(result, OCRResult)


# ─── read_file() when unavailable ─────────────────────────────────────────────


class TestReadFileUnavailable:
    def test_returns_empty_when_unavailable(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            engine = OCREngine()
            result = engine.read_file("/path/to/image.png")
            assert result.available is False
            assert result.text == ""


# ─── extract_text() convenience wrapper ───────────────────────────────────────


class TestExtractText:
    def test_returns_string_when_unavailable(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            engine = OCREngine()
            text = engine.extract_text(b"")
            assert isinstance(text, str)
            assert text == ""

    def test_returns_text_from_read_image(self):
        # Mock is_available + reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Hello World", 0.95),
        ]

        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", True):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", True):
                engine = OCREngine(confidence_threshold=0.0)
                engine._reader = mock_reader

                # Patch PIL to avoid actual image decoding
                with patch("spidy.vision.ocr.OCREngine.read_image") as mock_read:
                    mock_read.return_value = OCRResult(
                        text="Hello World",
                        confidence=0.95,
                        blocks=(OCRBlock(text="Hello World", confidence=0.95),),
                    )
                    text = engine.extract_text(b"\x89PNG")

        assert text == "Hello World"


# ─── _parse_results() ─────────────────────────────────────────────────────────


class TestParseResults:
    def _make_engine(self) -> OCREngine:
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", True):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", True):
                return OCREngine(confidence_threshold=0.5)

    def test_empty_raw(self):
        engine = self._make_engine()
        result = engine._parse_results([])
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.block_count == 0
        assert result.available is True

    def test_single_block(self):
        engine = self._make_engine()
        raw = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Hello", 0.9),
        ]
        result = engine._parse_results(raw)
        assert result.text == "Hello"
        assert result.confidence == pytest.approx(0.9)
        assert result.block_count == 1
        assert result.blocks[0].text == "Hello"
        assert result.blocks[0].confidence == pytest.approx(0.9)

    def test_multiple_blocks(self):
        engine = self._make_engine()
        raw = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Hello", 0.9),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "World", 0.8),
        ]
        result = engine._parse_results(raw)
        assert "Hello" in result.text
        assert "World" in result.text
        assert result.block_count == 2
        # Mean confidence
        assert result.confidence == pytest.approx(0.85)

    def test_filters_low_confidence_blocks(self):
        engine = self._make_engine()  # threshold = 0.5
        raw = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "HighConf", 0.9),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "LowConf", 0.3),  # below threshold
        ]
        result = engine._parse_results(raw)
        assert "HighConf" in result.text
        assert "LowConf" not in result.text
        assert result.block_count == 1

    def test_bbox_conversion(self):
        engine = self._make_engine()
        raw = [
            ([[10, 20], [110, 20], [110, 50], [10, 50]], "text", 0.9),
        ]
        result = engine._parse_results(raw)
        bbox = result.blocks[0].bbox
        assert bbox[0] == 10   # x
        assert bbox[1] == 20   # y
        assert bbox[2] == 100  # width = 110 - 10
        assert bbox[3] == 30   # height = 50 - 20

    def test_malformed_item_skipped(self):
        engine = self._make_engine()
        raw = [
            ("malformed",),  # too few items
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "OK", 0.9),
        ]
        result = engine._parse_results(raw)
        assert result.text == "OK"
        assert result.block_count == 1

    def test_language_preserved(self):
        engine = OCREngine(language="de", confidence_threshold=0.0)
        result = engine._parse_results([])
        assert result.language == "de"


# ─── _get_reader() lazy init ──────────────────────────────────────────────────


class TestGetReader:
    def test_returns_none_when_unavailable(self):
        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", False):
            engine = OCREngine()
            assert engine._get_reader() is None

    def test_lazy_init_creates_reader(self):
        mock_reader = MagicMock()

        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", True):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", True):
                with patch("spidy.vision.ocr._easyocr") as mock_easyocr:
                    mock_easyocr.Reader.return_value = mock_reader
                    engine = OCREngine()
                    assert engine._reader is None  # not yet created
                    reader = engine._get_reader()
                    assert reader is mock_reader
                    assert engine._reader is mock_reader  # cached

    def test_reader_created_only_once(self):
        mock_reader = MagicMock()

        with patch("spidy.vision.ocr._EASYOCR_AVAILABLE", True):
            with patch("spidy.vision.ocr._NUMPY_AVAILABLE", True):
                with patch("spidy.vision.ocr._easyocr") as mock_easyocr:
                    mock_easyocr.Reader.return_value = mock_reader
                    engine = OCREngine()
                    engine._get_reader()
                    engine._get_reader()
                    # Reader constructor only called once
                    assert mock_easyocr.Reader.call_count == 1
