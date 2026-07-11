"""
Tests for spidy.vision.types
"""
import pytest
from spidy.vision.types import (
    MonitorInfo,
    OCRBlock,
    OCRResult,
    ScreenAnalysis,
    ScreenshotResult,
    UIRegion,
    VisionDependencyError,
)


# ─── VisionDependencyError ────────────────────────────────────────────────────


class TestVisionDependencyError:
    def test_is_runtime_error(self):
        err = VisionDependencyError("deps missing")
        assert isinstance(err, RuntimeError)

    def test_message(self):
        err = VisionDependencyError("mss not installed")
        assert "mss" in str(err)

    def test_can_raise_and_catch(self):
        with pytest.raises(VisionDependencyError):
            raise VisionDependencyError("test")


# ─── MonitorInfo ──────────────────────────────────────────────────────────────


class TestMonitorInfo:
    def test_defaults(self):
        m = MonitorInfo()
        assert m.index == 0
        assert m.width == 0
        assert m.height == 0
        assert m.x == 0
        assert m.y == 0
        assert m.is_primary is True

    def test_with_values(self):
        m = MonitorInfo(index=1, width=1920, height=1080, x=1920, y=0, is_primary=False)
        assert m.index == 1
        assert m.width == 1920
        assert m.height == 1080
        assert m.x == 1920
        assert m.is_primary is False

    def test_frozen(self):
        m = MonitorInfo(index=0, width=1920, height=1080)
        with pytest.raises((AttributeError, TypeError)):
            m.width = 2560  # type: ignore[misc]

    def test_str_primary(self):
        m = MonitorInfo(index=0, width=1920, height=1080, x=0, y=0, is_primary=True)
        s = str(m)
        assert "primary" in s
        assert "1920" in s

    def test_str_secondary(self):
        m = MonitorInfo(index=1, width=2560, height=1440, is_primary=False)
        s = str(m)
        assert "primary" not in s
        assert "2560" in s


# ─── ScreenshotResult ─────────────────────────────────────────────────────────


class TestScreenshotResult:
    def test_defaults(self):
        r = ScreenshotResult()
        assert r.image_data == b""
        assert r.width == 0
        assert r.height == 0
        assert r.monitor_index == 0
        assert r.region is None
        assert r.timestamp == 0.0
        assert r.source == "fullscreen"

    def test_size_bytes_empty(self):
        r = ScreenshotResult()
        assert r.size_bytes == 0

    def test_size_bytes_with_data(self):
        data = b"PNG\x89" * 100
        r = ScreenshotResult(image_data=data, width=100, height=100)
        assert r.size_bytes == 400

    def test_is_empty_true(self):
        r = ScreenshotResult()
        assert r.is_empty is True

    def test_is_empty_false(self):
        r = ScreenshotResult(image_data=b"\x89PNG\r\n")
        assert r.is_empty is False

    def test_source_variants(self):
        for source in ("fullscreen", "window", "region"):
            r = ScreenshotResult(source=source)
            assert r.source == source

    def test_region_stored(self):
        r = ScreenshotResult(region=(10, 20, 100, 200))
        assert r.region == (10, 20, 100, 200)

    def test_frozen(self):
        r = ScreenshotResult()
        with pytest.raises((AttributeError, TypeError)):
            r.width = 1920  # type: ignore[misc]

    def test_str(self):
        r = ScreenshotResult(
            image_data=b"\x89PNG",
            width=1920, height=1080,
            source="fullscreen",
        )
        s = str(r)
        assert "fullscreen" in s
        assert "1920" in s


# ─── OCRBlock ─────────────────────────────────────────────────────────────────


class TestOCRBlock:
    def test_defaults(self):
        b = OCRBlock()
        assert b.text == ""
        assert b.confidence == 0.0
        assert b.bbox == (0, 0, 0, 0)

    def test_with_values(self):
        b = OCRBlock(text="Hello World", confidence=0.95, bbox=(10, 20, 100, 30))
        assert b.text == "Hello World"
        assert b.confidence == 0.95
        assert b.bbox == (10, 20, 100, 30)

    def test_frozen(self):
        b = OCRBlock(text="test")
        with pytest.raises((AttributeError, TypeError)):
            b.text = "other"  # type: ignore[misc]

    def test_str(self):
        b = OCRBlock(text="Hello", confidence=0.9)
        s = str(b)
        assert "Hello" in s
        assert "0.90" in s


# ─── OCRResult ────────────────────────────────────────────────────────────────


class TestOCRResult:
    def test_defaults(self):
        r = OCRResult()
        assert r.text == ""
        assert r.confidence == 0.0
        assert r.blocks == ()
        assert r.language == "en"
        assert r.available is True

    def test_block_count(self):
        blocks = (OCRBlock(text="a"), OCRBlock(text="b"))
        r = OCRResult(blocks=blocks)
        assert r.block_count == 2

    def test_block_count_empty(self):
        r = OCRResult()
        assert r.block_count == 0

    def test_is_empty_true(self):
        r = OCRResult(text="")
        assert r.is_empty is True

    def test_is_empty_false(self):
        r = OCRResult(text="Hello")
        assert r.is_empty is False

    def test_is_empty_whitespace(self):
        r = OCRResult(text="   \n  ")
        assert r.is_empty is True

    def test_unavailable(self):
        r = OCRResult(available=False)
        assert r.available is False

    def test_str(self):
        r = OCRResult(text="Hello", confidence=0.85, blocks=())
        s = str(r)
        assert "0.85" in s

    def test_frozen(self):
        r = OCRResult()
        with pytest.raises((AttributeError, TypeError)):
            r.text = "modified"  # type: ignore[misc]


# ─── UIRegion ─────────────────────────────────────────────────────────────────


class TestUIRegion:
    def test_defaults(self):
        r = UIRegion()
        assert r.name == "unknown"
        assert r.bbox == (0, 0, 0, 0)
        assert r.confidence == 0.0

    def test_with_values(self):
        r = UIRegion(name="button", bbox=(50, 100, 200, 40), confidence=0.8)
        assert r.name == "button"
        assert r.bbox == (50, 100, 200, 40)
        assert r.confidence == 0.8

    def test_frozen(self):
        r = UIRegion()
        with pytest.raises((AttributeError, TypeError)):
            r.name = "changed"  # type: ignore[misc]

    def test_str(self):
        r = UIRegion(name="window", bbox=(0, 0, 100, 100), confidence=0.75)
        s = str(r)
        assert "window" in s
        assert "0.75" in s


# ─── ScreenAnalysis ───────────────────────────────────────────────────────────


class TestScreenAnalysis:
    def test_defaults(self):
        a = ScreenAnalysis()
        assert a.active_app == ""
        assert a.active_window == ""
        assert a.visible_windows == ()
        assert a.ui_regions == ()
        assert a.timestamp == 0.0

    def test_window_count(self):
        a = ScreenAnalysis(visible_windows=("Code.exe", "Chrome"))
        assert a.window_count == 2

    def test_region_count(self):
        r1 = UIRegion(name="r1")
        r2 = UIRegion(name="r2")
        a = ScreenAnalysis(ui_regions=(r1, r2))
        assert a.region_count == 2

    def test_to_description_full(self):
        a = ScreenAnalysis(
            active_app="Code.exe",
            active_window="main.py — VS Code",
            visible_windows=("Code.exe", "Chrome"),
        )
        desc = a.to_description()
        assert "Code.exe" in desc
        assert "VS Code" in desc

    def test_to_description_empty(self):
        a = ScreenAnalysis()
        desc = a.to_description()
        assert "unavailable" in desc.lower()

    def test_frozen(self):
        a = ScreenAnalysis()
        with pytest.raises((AttributeError, TypeError)):
            a.active_app = "changed"  # type: ignore[misc]

    def test_str(self):
        a = ScreenAnalysis(active_app="chrome.exe", visible_windows=("A", "B"))
        s = str(a)
        assert "chrome.exe" in s
