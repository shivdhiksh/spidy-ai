"""
Tests for Milestone 7 — Browser Types
=======================================
Unit tests for all browser type dataclasses:
- PageInfo
- TabInfo
- DownloadResult
- HistoryEntry
- SearchResult
"""

from __future__ import annotations

import pytest

from spidy.browser.types import (
    DownloadResult,
    HistoryEntry,
    PageInfo,
    SearchResult,
    TabInfo,
)


# ──────────────────────────────────────────────────────────────────────────────
# PageInfo
# ──────────────────────────────────────────────────────────────────────────────


class TestPageInfo:
    def test_defaults(self):
        p = PageInfo()
        assert p.title == ""
        assert p.url == ""
        assert p.tab_id == 0
        assert p.load_time_ms == 0

    def test_construction_with_values(self):
        p = PageInfo(title="Google", url="https://google.com", tab_id=2, load_time_ms=350)
        assert p.title == "Google"
        assert p.url == "https://google.com"
        assert p.tab_id == 2
        assert p.load_time_ms == 350

    def test_frozen(self):
        p = PageInfo(title="Test")
        with pytest.raises(Exception):  # FrozenInstanceError
            p.title = "Modified"  # type: ignore[misc]

    def test_to_dict_keys(self):
        p = PageInfo(title="A", url="https://a.com", tab_id=1, load_time_ms=100)
        d = p.to_dict()
        assert d == {"title": "A", "url": "https://a.com", "tab_id": 1, "load_time_ms": 100}

    def test_equality(self):
        p1 = PageInfo(title="X", url="https://x.com")
        p2 = PageInfo(title="X", url="https://x.com")
        assert p1 == p2


# ──────────────────────────────────────────────────────────────────────────────
# TabInfo
# ──────────────────────────────────────────────────────────────────────────────


class TestTabInfo:
    def test_defaults(self):
        t = TabInfo()
        assert t.tab_id == 0
        assert t.title == ""
        assert t.url == ""
        assert t.is_active is False

    def test_active_tab(self):
        t = TabInfo(tab_id=3, title="YouTube", url="https://youtube.com", is_active=True)
        assert t.is_active is True

    def test_to_dict(self):
        t = TabInfo(tab_id=1, title="GitHub", url="https://github.com", is_active=False)
        d = t.to_dict()
        assert d["tab_id"] == 1
        assert d["title"] == "GitHub"
        assert d["url"] == "https://github.com"
        assert d["is_active"] is False

    def test_frozen(self):
        t = TabInfo()
        with pytest.raises(Exception):
            t.tab_id = 99  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# DownloadResult
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadResult:
    def test_defaults(self):
        d = DownloadResult()
        assert d.filename == ""
        assert d.path == ""
        assert d.size_bytes == 0
        assert d.success is False
        assert d.error == ""

    def test_successful_download(self):
        d = DownloadResult(
            filename="report.pdf",
            path="/home/user/Downloads/report.pdf",
            size_bytes=204800,
            success=True,
        )
        assert d.success is True
        assert d.error == ""

    def test_failed_download(self):
        d = DownloadResult(success=False, error="Timeout after 60s")
        assert d.success is False
        assert "Timeout" in d.error

    def test_to_dict(self):
        d = DownloadResult(filename="file.zip", path="/tmp/file.zip",
                           size_bytes=1024, success=True, error="")
        result = d.to_dict()
        assert result["filename"] == "file.zip"
        assert result["size_bytes"] == 1024
        assert result["success"] is True


# ──────────────────────────────────────────────────────────────────────────────
# HistoryEntry
# ──────────────────────────────────────────────────────────────────────────────


class TestHistoryEntry:
    def test_defaults(self):
        h = HistoryEntry()
        assert h.title == ""
        assert h.url == ""
        assert h.visit_time == ""

    def test_with_values(self):
        h = HistoryEntry(title="Stack Overflow", url="https://stackoverflow.com",
                         visit_time="2026-07-11T10:00:00")
        assert h.title == "Stack Overflow"
        assert "stackoverflow" in h.url

    def test_to_dict(self):
        h = HistoryEntry(title="T", url="https://t.com", visit_time="2026-01-01T00:00:00")
        d = h.to_dict()
        assert set(d.keys()) == {"title", "url", "visit_time"}

    def test_frozen(self):
        h = HistoryEntry(title="A")
        with pytest.raises(Exception):
            h.title = "B"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# SearchResult
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchResult:
    def test_defaults(self):
        s = SearchResult()
        assert s.title == ""
        assert s.url == ""
        assert s.snippet == ""

    def test_to_dict(self):
        s = SearchResult(title="Python", url="https://python.org", snippet="Official site")
        d = s.to_dict()
        assert d == {"title": "Python", "url": "https://python.org", "snippet": "Official site"}

    def test_frozen(self):
        s = SearchResult(title="X")
        with pytest.raises(Exception):
            s.title = "Y"  # type: ignore[misc]
