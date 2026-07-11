"""
Tests for Milestone 4 — Built-in Skills
========================================
Unit tests for all 6 built-in skills:
- HelpSkill
- TimeSkill
- GreetSkill
- NoteSkill
- TimerSkill
- SystemSkill
- register_builtin_skills() loader
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from spidy.skills.base import SkillCapability, SkillContext, SkillResult


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ctx(action: str = "", params: dict | None = None, **kw) -> SkillContext:
    return SkillContext(action=action, params=params or {}, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# 1. HelpSkill
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpSkill:
    def _make(self, registry=None):
        from spidy.skills.builtin.help_skill import HelpSkill
        return HelpSkill(registry=registry)

    def test_capabilities_declared(self):
        skill = self._make()
        caps = skill.capabilities()
        assert any(c.action == "show_help" for c in caps)

    def test_permission_tier_t0(self):
        skill = self._make()
        for cap in skill.capabilities():
            assert cap.permission_tier == "T0"

    @pytest.mark.asyncio
    async def test_show_help_no_registry(self):
        skill = self._make()
        result = await skill.execute("show_help", _ctx("show_help"))
        assert result.success
        assert "Spidy" in result.message or "companion" in result.message.lower()

    @pytest.mark.asyncio
    async def test_show_help_with_registry(self):
        from spidy.skills.registry import SkillRegistry
        from spidy.skills.builtin.time_skill import TimeSkill
        registry = SkillRegistry()
        registry.register(TimeSkill())
        skill = self._make(registry=registry)
        result = await skill.execute("show_help", _ctx("show_help"))
        assert result.success

    @pytest.mark.asyncio
    async def test_unknown_action_fails(self):
        skill = self._make()
        result = await skill.execute("unknown", _ctx("unknown"))
        assert not result.success

    def test_skill_name(self):
        from spidy.skills.builtin.help_skill import HelpSkill
        assert HelpSkill.name == "help_skill"


# ──────────────────────────────────────────────────────────────────────────────
# 2. TimeSkill
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeSkill:
    def _make(self):
        from spidy.skills.builtin.time_skill import TimeSkill
        return TimeSkill()

    def test_capabilities_declared(self):
        skill = self._make()
        caps = skill.capabilities()
        actions = {c.action for c in caps}
        assert {"get_time", "get_date", "get_day", "get_datetime"} <= actions

    def test_all_tiers_t0(self):
        skill = self._make()
        assert all(c.permission_tier == "T0" for c in skill.capabilities())

    @pytest.mark.asyncio
    async def test_get_time_success(self):
        skill = self._make()
        result = await skill.execute("get_time", _ctx("get_time"))
        assert result.success
        assert "time" in result.message.lower() or ":" in result.message

    @pytest.mark.asyncio
    async def test_get_date_success(self):
        skill = self._make()
        result = await skill.execute("get_date", _ctx("get_date"))
        assert result.success
        assert result.data is not None
        assert "date" in result.data

    @pytest.mark.asyncio
    async def test_get_day_success(self):
        skill = self._make()
        result = await skill.execute("get_day", _ctx("get_day"))
        assert result.success
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(d in result.message for d in days)

    @pytest.mark.asyncio
    async def test_get_datetime_success(self):
        skill = self._make()
        result = await skill.execute("get_datetime", _ctx("get_datetime"))
        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_unknown_action_fails(self):
        skill = self._make()
        result = await skill.execute("fly_to_moon", _ctx("fly_to_moon"))
        assert not result.success


# ──────────────────────────────────────────────────────────────────────────────
# 3. GreetSkill
# ──────────────────────────────────────────────────────────────────────────────

class TestGreetSkill:
    def _make(self):
        from spidy.skills.builtin.greet_skill import GreetSkill
        return GreetSkill()

    def test_capabilities_declared(self):
        skill = self._make()
        actions = {c.action for c in skill.capabilities()}
        assert {"greet", "farewell", "introduce"} <= actions

    @pytest.mark.asyncio
    async def test_greet_includes_user_name(self):
        skill = self._make()
        ctx = _ctx("greet", user_name="Alice")
        result = await skill.execute("greet", ctx)
        assert result.success
        assert "Alice" in result.message

    @pytest.mark.asyncio
    async def test_farewell_includes_user_name(self):
        skill = self._make()
        ctx = _ctx("farewell", user_name="Bob")
        result = await skill.execute("farewell", ctx)
        assert result.success
        assert "Bob" in result.message

    @pytest.mark.asyncio
    async def test_introduce_mentions_spidy(self):
        skill = self._make()
        result = await skill.execute("introduce", _ctx("introduce"))
        assert result.success
        assert "Spidy" in result.message

    @pytest.mark.asyncio
    async def test_unknown_action_fails(self):
        skill = self._make()
        result = await skill.execute("fly", _ctx("fly"))
        assert not result.success


# ──────────────────────────────────────────────────────────────────────────────
# 4. NoteSkill
# ──────────────────────────────────────────────────────────────────────────────

class TestNoteSkill:
    def _make(self, tmpdir):
        from spidy.skills.builtin.note_skill import NoteSkill
        notes_file = str(Path(tmpdir) / "notes.jsonl")
        return NoteSkill(notes_file=notes_file)

    @pytest.mark.asyncio
    async def test_take_note_success(self, tmp_path):
        skill = self._make(tmp_path)
        ctx = _ctx("take_note", params={"content": "Buy milk"})
        result = await skill.execute("take_note", ctx)
        assert result.success
        assert "Buy milk" in result.message

    @pytest.mark.asyncio
    async def test_take_note_empty_fails(self, tmp_path):
        skill = self._make(tmp_path)
        ctx = _ctx("take_note", params={"content": ""})
        result = await skill.execute("take_note", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_read_notes_empty(self, tmp_path):
        skill = self._make(tmp_path)
        result = await skill.execute("read_notes", _ctx("read_notes"))
        assert result.success
        assert "no notes" in result.message.lower()

    @pytest.mark.asyncio
    async def test_read_notes_after_write(self, tmp_path):
        skill = self._make(tmp_path)
        await skill.execute("take_note", _ctx("take_note", params={"content": "Test note"}))
        result = await skill.execute("read_notes", _ctx("read_notes"))
        assert result.success
        assert "Test note" in result.message

    @pytest.mark.asyncio
    async def test_find_note_match(self, tmp_path):
        skill = self._make(tmp_path)
        await skill.execute("take_note", _ctx("take_note", params={"content": "Buy apples from store"}))
        result = await skill.execute("find_note", _ctx("find_note", params={"query": "apples"}))
        assert result.success
        assert "apples" in result.message.lower()

    @pytest.mark.asyncio
    async def test_find_note_no_match(self, tmp_path):
        skill = self._make(tmp_path)
        result = await skill.execute("find_note", _ctx("find_note", params={"query": "xyz_not_there"}))
        assert result.success
        assert "No notes found" in result.message

    @pytest.mark.asyncio
    async def test_clear_notes(self, tmp_path):
        skill = self._make(tmp_path)
        await skill.execute("take_note", _ctx("take_note", params={"content": "to be cleared"}))
        result = await skill.execute("clear_notes", _ctx("clear_notes"))
        assert result.success
        # After clear, reading should show empty
        read_result = await skill.execute("read_notes", _ctx("read_notes"))
        assert "no notes" in read_result.message.lower()

    @pytest.mark.asyncio
    async def test_note_permission_tiers(self, tmp_path):
        skill = self._make(tmp_path)
        tier_map = {c.action: c.permission_tier for c in skill.capabilities()}
        assert tier_map["take_note"] == "T1"
        assert tier_map["read_notes"] == "T0"
        assert tier_map["find_note"] == "T0"
        assert tier_map["clear_notes"] == "T2"


# ──────────────────────────────────────────────────────────────────────────────
# 5. TimerSkill
# ──────────────────────────────────────────────────────────────────────────────

class TestTimerSkill:
    def _make(self):
        from spidy.skills.builtin.timer_skill import TimerSkill
        return TimerSkill()

    @pytest.mark.asyncio
    async def test_set_timer_success(self):
        skill = self._make()
        ctx = _ctx("set_timer", params={"duration": "5 seconds", "name": "test"})
        result = await skill.execute("set_timer", ctx)
        assert result.success
        assert "Timer" in result.message or "5" in result.message
        # Cancel the background task
        for t in skill._timers.values():
            if t.task:
                t.task.cancel()

    @pytest.mark.asyncio
    async def test_set_timer_invalid_duration(self):
        skill = self._make()
        ctx = _ctx("set_timer", params={"duration": "blah blah", "name": "test"})
        result = await skill.execute("set_timer", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_timers_empty(self):
        skill = self._make()
        result = await skill.execute("list_timers", _ctx("list_timers"))
        assert result.success
        assert "No active" in result.message

    @pytest.mark.asyncio
    async def test_cancel_timer(self):
        skill = self._make()
        ctx = _ctx("set_timer", params={"duration": "60 seconds", "name": "mytimer"})
        await skill.execute("set_timer", ctx)
        result = await skill.execute("cancel_timer", _ctx("cancel_timer", params={"name": "mytimer"}))
        assert result.success

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_timer(self):
        skill = self._make()
        result = await skill.execute("cancel_timer", _ctx("cancel_timer", params={"name": "ghost"}))
        assert not result.success

    def test_parse_duration_seconds(self):
        from spidy.skills.builtin.timer_skill import TimerSkill
        assert TimerSkill._parse_duration("30 seconds") == 30.0
        assert TimerSkill._parse_duration("5 minutes") == 300.0
        assert TimerSkill._parse_duration("1 hour") == 3600.0
        assert TimerSkill._parse_duration("2 min 30 sec") == 150.0

    def test_parse_duration_bare_number(self):
        from spidy.skills.builtin.timer_skill import TimerSkill
        assert TimerSkill._parse_duration("45") == 45.0

    def test_parse_duration_invalid(self):
        from spidy.skills.builtin.timer_skill import TimerSkill
        assert TimerSkill._parse_duration("") is None
        assert TimerSkill._parse_duration("next tuesday") is None

    def test_format_duration(self):
        from spidy.skills.builtin.timer_skill import TimerSkill
        assert "30 second" in TimerSkill._format_duration(30)
        assert "5 minute" in TimerSkill._format_duration(300)
        assert "1 hour" in TimerSkill._format_duration(3600)

    def test_all_tiers_t0(self):
        skill = self._make()
        assert all(c.permission_tier == "T0" for c in skill.capabilities())


# ──────────────────────────────────────────────────────────────────────────────
# 6. SystemSkill
# ──────────────────────────────────────────────────────────────────────────────

class TestSystemSkill:
    def _make(self):
        from spidy.skills.builtin.system_skill import SystemSkill
        return SystemSkill()

    @pytest.mark.asyncio
    async def test_get_system_info_success(self):
        skill = self._make()
        result = await skill.execute("get_system_info", _ctx("get_system_info"))
        assert result.success
        assert result.data is not None
        assert "platform" in result.data

    @pytest.mark.asyncio
    async def test_set_volume_stub_success(self):
        skill = self._make()
        ctx = _ctx("set_volume", params={"level": 50})
        result = await skill.execute("set_volume", ctx)
        # Stub in M4 — returns success with explanation
        assert result.success
        assert "Milestone 7" in result.message or "not yet" in result.message.lower()

    @pytest.mark.asyncio
    async def test_lock_screen_stub_success(self):
        skill = self._make()
        result = await skill.execute("lock_screen", _ctx("lock_screen"))
        assert result.success
        assert "not yet" in result.message.lower() or "Milestone 7" in result.message

    def test_get_system_info_tier_t0(self):
        skill = self._make()
        tier_map = {c.action: c.permission_tier for c in skill.capabilities()}
        assert tier_map["get_system_info"] == "T0"

    def test_set_volume_tier_t1(self):
        skill = self._make()
        tier_map = {c.action: c.permission_tier for c in skill.capabilities()}
        assert tier_map["set_volume"] == "T1"


# ──────────────────────────────────────────────────────────────────────────────
# 7. register_builtin_skills()
# ──────────────────────────────────────────────────────────────────────────────

class TestRegisterBuiltinSkills:
    def test_all_skills_registered_by_default(self):
        from spidy.skills.builtin import register_builtin_skills
        from spidy.skills.registry import SkillRegistry
        registry = SkillRegistry()
        registered = register_builtin_skills(registry)
        assert len(registered) == 6
        assert "help_skill" in registered
        assert "time_skill" in registered
        assert "greet_skill" in registered
        assert "note_skill" in registered
        assert "timer_skill" in registered
        assert "system_skill" in registered

    def test_disabled_skill_not_registered(self):
        from spidy.config.manager import SkillsConfig
        from spidy.skills.builtin import register_builtin_skills
        from spidy.skills.registry import SkillRegistry
        config = SkillsConfig(help_skill_enabled=False, time_skill_enabled=False)
        registry = SkillRegistry()
        registered = register_builtin_skills(registry, config=config)
        assert "help_skill" not in registered
        assert "time_skill" not in registered
        assert "greet_skill" in registered  # still enabled

    def test_skills_findable_by_action(self):
        from spidy.skills.builtin import register_builtin_skills
        from spidy.skills.registry import SkillRegistry
        registry = SkillRegistry()
        register_builtin_skills(registry)
        # All key actions should be discoverable
        for action in ["show_help", "get_time", "greet", "take_note", "set_timer", "get_system_info"]:
            assert registry.find_skill_for_action(action) is not None, f"Action '{action}' not found"

    def test_skills_config_notes_file(self, tmp_path):
        from spidy.config.manager import SkillsConfig
        from spidy.skills.builtin import register_builtin_skills
        from spidy.skills.registry import SkillRegistry
        notes_path = str(tmp_path / "my_notes.jsonl")
        config = SkillsConfig(notes_file=notes_path)
        registry = SkillRegistry()
        registered = register_builtin_skills(registry, config=config)
        assert "note_skill" in registered
