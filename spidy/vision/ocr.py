"""
OCR Engine — On-Screen Text Extraction
=======================================
Reads text from screenshots or image files using ``easyocr``.

Supported operations
--------------------
- ``read_image(image_data)`` — OCR on raw PNG/JPEG bytes
- ``read_file(path)``        — OCR on a file path
- ``extract_text(image_data)`` — convenience wrapper returning plain str

Graceful Degradation
--------------------
``easyocr`` is a heavy optional dependency (~500 MB with models).
When absent:
  - ``is_available`` returns ``False``
  - All read methods return an ``OCRResult(text="", confidence=0.0, available=False)``
  - A warning is logged once at first use
  - No exceptions are raised to the caller

The reader is lazy-initialized on first use so import time is fast.

Usage (when deps available)
---------------------------
    engine = OCREngine(language="en", confidence_threshold=0.5)
    if engine.is_available:
        result = engine.read_image(png_bytes)
        print(result.text)
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.vision.types import OCRBlock, OCRResult

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ── Lazy optional imports ─────────────────────────────────────────────────────

try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    _easyocr = None  # type: ignore[assignment]

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    _np = None  # type: ignore[assignment]


class OCREngine:
    """
    Text extraction engine backed by ``easyocr``.

    Parameters
    ----------
    language:
        ISO 639-1 language code (e.g. ``"en"``). EasyOCR supports 80+ langs.
    confidence_threshold:
        Minimum confidence to include a block in results (0.0–1.0).
    gpu:
        Whether to use GPU acceleration. Defaults to False (CPU-safe).
    """

    def __init__(
        self,
        language: str = "en",
        confidence_threshold: float = 0.5,
        gpu: bool = False,
    ) -> None:
        self._language = language
        self._confidence_threshold = confidence_threshold
        self._gpu = gpu
        self._reader: object | None = None   # easyocr.Reader, lazy
        self._unavailable_warned = False

    # ── Availability ──────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when ``easyocr`` and ``numpy`` are installed."""
        return _EASYOCR_AVAILABLE and _NUMPY_AVAILABLE

    # ── Lazy Initialisation ───────────────────────────────────────────────────

    def _get_reader(self) -> object | None:
        """
        Lazy-load the easyocr.Reader on first use.

        The reader downloads models on first call (~500 MB), so we defer
        this until the first actual OCR request.
        """
        if not self.is_available:
            return None

        if self._reader is None:
            log.info(
                "Initialising EasyOCR reader (language={lang}) — "
                "first use may download models.",
                lang=self._language,
            )
            try:
                self._reader = _easyocr.Reader(  # type: ignore[union-attr]
                    [self._language],
                    gpu=self._gpu,
                    verbose=False,
                )
                log.info("EasyOCR reader ready.")
            except Exception as exc:
                log.warning("EasyOCR reader failed to initialise: {exc}", exc=exc)
                self._reader = None

        return self._reader

    # ── Read operations ───────────────────────────────────────────────────────

    def read_image(self, image_data: bytes) -> OCRResult:
        """
        Perform OCR on raw image bytes (PNG, JPEG, BMP, etc.).

        Parameters
        ----------
        image_data:
            Raw image bytes. Typically PNG output from ScreenshotEngine.

        Returns
        -------
        OCRResult
            Extracted text with confidence scores and block positions.
            Returns an empty result (``available=False``) when deps absent.
        """
        if not self.is_available:
            return self._unavailable_result()

        reader = self._get_reader()
        if reader is None:
            return self._unavailable_result()

        try:
            # Decode bytes → numpy array via PIL-compatible approach
            import PIL.Image  # type: ignore[import]
            img = PIL.Image.open(io.BytesIO(image_data))
            img_array = _np.array(img)  # type: ignore[union-attr]

            raw_results = reader.readtext(img_array)  # type: ignore[union-attr]
            return self._parse_results(raw_results)

        except ImportError:
            # Pillow not available — try numpy direct decode
            try:
                img_array = _np.frombuffer(image_data, dtype=_np.uint8)  # type: ignore[union-attr]
                raw_results = reader.readtext(img_array)  # type: ignore[union-attr]
                return self._parse_results(raw_results)
            except Exception as exc:
                log.warning("OCR read_image failed: {exc}", exc=exc)
                return OCRResult(text="", confidence=0.0, blocks=(), language=self._language, available=True)

        except Exception as exc:
            log.warning("OCR read_image failed: {exc}", exc=exc)
            return OCRResult(text="", confidence=0.0, blocks=(), language=self._language, available=True)

    def read_file(self, path: str) -> OCRResult:
        """
        Perform OCR on an image file on disk.

        Parameters
        ----------
        path:
            Absolute or relative path to the image file.

        Returns
        -------
        OCRResult
            Extracted text with confidence and block positions.
        """
        if not self.is_available:
            return self._unavailable_result()

        reader = self._get_reader()
        if reader is None:
            return self._unavailable_result()

        try:
            raw_results = reader.readtext(path)  # type: ignore[union-attr]
            return self._parse_results(raw_results)
        except Exception as exc:
            log.warning("OCR read_file failed (path={path}): {exc}", path=path, exc=exc)
            return OCRResult(text="", confidence=0.0, blocks=(), language=self._language, available=True)

    def extract_text(self, image_data: bytes) -> str:
        """
        Convenience wrapper that returns only the extracted text string.

        Parameters
        ----------
        image_data:
            Raw image bytes.

        Returns
        -------
        str
            Extracted text, or empty string when unavailable.
        """
        return self.read_image(image_data).text

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_results(self, raw: list) -> OCRResult:
        """
        Convert easyocr raw output to our typed ``OCRResult``.

        easyocr returns: ``[([[bbox_points], text, confidence]), ...]``
        where bbox_points is 4 corner points [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
        """
        blocks: list[OCRBlock] = []
        total_confidence = 0.0

        for item in raw:
            if len(item) < 3:
                continue
            bbox_points, text, confidence = item[0], item[1], item[2]

            if confidence < self._confidence_threshold:
                continue

            # Convert 4-corner points → (x, y, w, h)
            try:
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x = int(min(xs))
                y = int(min(ys))
                w = int(max(xs) - min(xs))
                h = int(max(ys) - min(ys))
                bbox = (x, y, w, h)
            except (IndexError, TypeError, ValueError):
                bbox = (0, 0, 0, 0)

            blocks.append(OCRBlock(text=str(text), confidence=float(confidence), bbox=bbox))
            total_confidence += float(confidence)

        mean_confidence = total_confidence / len(blocks) if blocks else 0.0
        full_text = "\n".join(b.text for b in blocks)

        return OCRResult(
            text=full_text,
            confidence=mean_confidence,
            blocks=tuple(blocks),
            language=self._language,
            available=True,
        )

    def _unavailable_result(self) -> OCRResult:
        """Return an empty OCRResult signalling deps are absent."""
        if not self._unavailable_warned:
            log.warning(
                "easyocr is not installed — OCR unavailable. "
                "Install with: pip install easyocr"
            )
            self._unavailable_warned = True

        return OCRResult(
            text="",
            confidence=0.0,
            blocks=(),
            language=self._language,
            available=False,
        )
