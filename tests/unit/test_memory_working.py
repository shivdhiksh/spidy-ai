"""
Tests for spidy.memory.working — WorkingMemory
"""

from __future__ import annotations

import pytest

from spidy.memory.types import MemoryEntry, MemoryType
from spidy.memory.working import WorkingMemory


class TestWorkingMemory:
    """WorkingMemory — in-process rolling session buffer."""

    # ── Construction ──────────────────────────────────────────────────────

    def test_default_max_messages(self):
        wm = WorkingMemory()
        assert wm._max_messages == 20

    def test_custom_max_messages(self):
        wm = WorkingMemory(max_messages=5)
        assert wm._max_messages == 5

    def test_max_messages_minimum_is_one(self):
        wm = WorkingMemory(max_messages=0)
        assert wm._max_messages == 1

    # ── add() ─────────────────────────────────────────────────────────────

    def test_add_returns_memory_entry(self):
        wm = WorkingMemory()
        entry = wm.add("hello", session_id="s1")
        assert isinstance(entry, MemoryEntry)
        assert entry.content == "hello"
        assert entry.session_id == "s1"

    def test_add_memory_type_is_working(self):
        wm = WorkingMemory()
        entry = wm.add("test", session_id="s1")
        assert entry.memory_type == MemoryType.WORKING

    def test_add_with_tags(self):
        wm = WorkingMemory()
        entry = wm.add("tagged", session_id="s1", tags=["work", "urgent"])
        assert "work" in entry.tags
        assert "urgent" in entry.tags

    def test_add_with_metadata(self):
        wm = WorkingMemory()
        entry = wm.add("meta", session_id="s1", metadata={"key": "val"})
        assert entry.metadata == {"key": "val"}

    def test_add_default_session_id(self):
        wm = WorkingMemory()
        entry = wm.add("no session", session_id="")
        assert entry.session_id == "default"

    # ── Capacity / eviction ───────────────────────────────────────────────

    def test_rolling_eviction(self):
        wm = WorkingMemory(max_messages=3)
        for i in range(5):
            wm.add(f"msg{i}", session_id="s1")
        entries = wm.get_session("s1")
        assert len(entries) == 3
        # Most recent 3 should remain
        contents = [e.content for e in entries]
        assert "msg2" in contents
        assert "msg3" in contents
        assert "msg4" in contents
        # Oldest evicted
        assert "msg0" not in contents
        assert "msg1" not in contents

    # ── get_session() ─────────────────────────────────────────────────────

    def test_get_session_empty(self):
        wm = WorkingMemory()
        result = wm.get_session("nonexistent")
        assert result == []

    def test_get_session_returns_all_in_order(self):
        wm = WorkingMemory()
        wm.add("first", session_id="s1")
        wm.add("second", session_id="s1")
        wm.add("third", session_id="s1")
        entries = wm.get_session("s1")
        assert len(entries) == 3
        assert entries[0].content == "first"
        assert entries[2].content == "third"

    # ── Session isolation ─────────────────────────────────────────────────

    def test_sessions_are_isolated(self):
        wm = WorkingMemory()
        wm.add("session-a message", session_id="a")
        wm.add("session-b message", session_id="b")
        a_entries = wm.get_session("a")
        b_entries = wm.get_session("b")
        assert len(a_entries) == 1
        assert len(b_entries) == 1
        assert a_entries[0].content == "session-a message"
        assert b_entries[0].content == "session-b message"

    # ── get_context_window() ──────────────────────────────────────────────

    def test_context_window_limit(self):
        wm = WorkingMemory()
        for i in range(10):
            wm.add(f"m{i}", session_id="s1")
        recent = wm.get_context_window("s1", limit=3)
        assert len(recent) == 3
        # Should be the last 3
        assert recent[-1].content == "m9"

    def test_context_window_no_limit(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.add(f"m{i}", session_id="s1")
        all_entries = wm.get_context_window("s1", limit=None)
        assert len(all_entries) == 5

    # ── find() ────────────────────────────────────────────────────────────

    def test_find_keyword_match(self):
        wm = WorkingMemory()
        wm.add("I love Python programming", session_id="s1")
        wm.add("The weather is nice today", session_id="s1")
        results = wm.find("python", session_id="s1")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_find_case_insensitive(self):
        wm = WorkingMemory()
        wm.add("I use PYTHON daily", session_id="s1")
        results = wm.find("python", session_id="s1")
        assert len(results) == 1

    def test_find_no_match(self):
        wm = WorkingMemory()
        wm.add("something", session_id="s1")
        results = wm.find("nomatch", session_id="s1")
        assert results == []

    def test_find_respects_limit(self):
        wm = WorkingMemory(max_messages=20)
        for i in range(10):
            wm.add(f"python message {i}", session_id="s1")
        results = wm.find("python", session_id="s1", limit=3)
        assert len(results) == 3

    def test_find_across_all_sessions(self):
        wm = WorkingMemory()
        wm.add("python in session a", session_id="a")
        wm.add("python in session b", session_id="b")
        # Empty session_id = search all
        results = wm.find("python", session_id="")
        assert len(results) == 2

    # ── clear() ───────────────────────────────────────────────────────────

    def test_clear_session(self):
        wm = WorkingMemory()
        wm.add("msg1", session_id="s1")
        wm.add("msg2", session_id="s1")
        wm.add("other", session_id="s2")
        count = wm.clear("s1")
        assert count == 2
        assert wm.get_session("s1") == []
        # s2 should be untouched
        assert len(wm.get_session("s2")) == 1

    def test_clear_all(self):
        wm = WorkingMemory()
        wm.add("a", session_id="s1")
        wm.add("b", session_id="s2")
        count = wm.clear("")
        assert count == 2
        assert wm.session_count() == 0

    def test_clear_nonexistent_session(self):
        wm = WorkingMemory()
        count = wm.clear("nope")
        assert count == 0

    # ── entry_count / session_count ───────────────────────────────────────

    def test_entry_count(self):
        wm = WorkingMemory()
        wm.add("a", session_id="s1")
        wm.add("b", session_id="s1")
        assert wm.entry_count("s1") == 2

    def test_entry_count_total(self):
        wm = WorkingMemory()
        wm.add("a", session_id="s1")
        wm.add("b", session_id="s2")
        assert wm.entry_count() == 2

    def test_session_count(self):
        wm = WorkingMemory()
        wm.add("a", session_id="s1")
        wm.add("b", session_id="s2")
        assert wm.session_count() == 2
