"""
Unit tests for SkillRegistry
=============================
Tests registration, duplicate detection, action lookup,
unregistration, thread safety, and capacity reporting.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from spidy.skills.base import (
    BaseSkill, ParamSchema, SkillCapability, SkillContext, SkillResult,
)
from spidy.skills.registry import SkillRegistry


# ─── Fake skills ──────────────────────────────────────────────────────────────


class FileSkill(BaseSkill):
    name = "file_skill"
    version = "1.0.0"

    def capabilities(self):
        return [
            SkillCapability("read_file", "Read a file", "T0"),
            SkillCapability("write_file", "Write a file", "T1"),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        return SkillResult.ok(f"done: {action}")


class BrowserSkill(BaseSkill):
    name = "browser_skill"
    version = "1.0.0"

    def capabilities(self):
        return [
            SkillCapability("open_url", "Open a URL", "T0"),
            SkillCapability("click_element", "Click a page element", "T0"),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        return SkillResult.ok(f"done: {action}")


class ConflictingSkill(BaseSkill):
    """A skill that tries to claim 'read_file' — already owned by FileSkill."""
    name = "conflict_skill"
    version = "1.0.0"

    def capabilities(self):
        return [
            SkillCapability("read_file", "Also reads files?", "T0"),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        return SkillResult.ok("conflict")


# ─── SkillRegistry Tests ──────────────────────────────────────────────────────


class TestSkillRegistryBasic:
    def test_empty_registry(self):
        r = SkillRegistry()
        assert len(r) == 0
        assert r.all_skills() == []
        assert r.all_capabilities() == []

    def test_register_single_skill(self):
        r = SkillRegistry()
        r.register(FileSkill())
        assert len(r) == 1
        assert "file_skill" in r

    def test_register_multiple_skills(self):
        r = SkillRegistry()
        r.register(FileSkill())
        r.register(BrowserSkill())
        assert len(r) == 2
        assert "file_skill" in r
        assert "browser_skill" in r

    def test_all_capabilities_aggregates_all(self):
        r = SkillRegistry()
        r.register(FileSkill())
        r.register(BrowserSkill())
        caps = r.all_capabilities()
        actions = {c.action for c in caps}
        assert "read_file" in actions
        assert "write_file" in actions
        assert "open_url" in actions
        assert "click_element" in actions
        assert len(caps) == 4

    def test_repr(self):
        r = SkillRegistry()
        r.register(FileSkill())
        assert "file_skill" in repr(r)


class TestSkillRegistryLookup:
    def test_find_skill_for_known_action(self):
        r = SkillRegistry()
        r.register(FileSkill())
        skill = r.find_skill_for_action("read_file")
        assert skill is not None
        assert skill.name == "file_skill"

    def test_find_skill_for_unknown_action_returns_none(self):
        r = SkillRegistry()
        r.register(FileSkill())
        assert r.find_skill_for_action("fly_to_moon") is None

    def test_find_skill_by_name(self):
        r = SkillRegistry()
        r.register(BrowserSkill())
        skill = r.find_skill_by_name("browser_skill")
        assert skill is not None
        assert skill.name == "browser_skill"

    def test_find_skill_by_name_missing_returns_none(self):
        r = SkillRegistry()
        assert r.find_skill_by_name("not_there") is None

    def test_all_skills_returns_instances(self):
        r = SkillRegistry()
        fs = FileSkill()
        bs = BrowserSkill()
        r.register(fs)
        r.register(bs)
        skills = r.all_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"file_skill", "browser_skill"}


class TestSkillRegistryDuplicates:
    def test_duplicate_registration_raises(self):
        r = SkillRegistry()
        r.register(FileSkill())
        with pytest.raises(ValueError, match="already registered"):
            r.register(FileSkill())

    def test_overwrite_replaces_skill(self):
        r = SkillRegistry()
        original = FileSkill()
        replacement = FileSkill()
        r.register(original)
        r.register(replacement, overwrite=True)
        assert len(r) == 1
        assert r.find_skill_by_name("file_skill") is replacement

    def test_action_conflict_logs_warning(self):
        """Registering a skill that shadows an existing action should not raise,
        just log a warning (we test that overwrite=True allows it)."""
        r = SkillRegistry()
        r.register(FileSkill())
        # ConflictingSkill claims 'read_file' — should warn but succeed with overwrite
        r.register(ConflictingSkill(), overwrite=True)
        # Now conflict_skill owns read_file
        skill = r.find_skill_for_action("read_file")
        assert skill is not None
        assert skill.name == "conflict_skill"


class TestSkillRegistryUnregister:
    def test_unregister_removes_skill(self):
        r = SkillRegistry()
        r.register(FileSkill())
        r.unregister("file_skill")
        assert "file_skill" not in r
        assert len(r) == 0

    def test_unregister_removes_actions(self):
        r = SkillRegistry()
        r.register(FileSkill())
        r.unregister("file_skill")
        assert r.find_skill_for_action("read_file") is None
        assert r.find_skill_for_action("write_file") is None

    def test_unregister_nonexistent_is_noop(self):
        r = SkillRegistry()
        r.unregister("not_there")  # must not raise

    def test_unregister_leaves_other_skills_intact(self):
        r = SkillRegistry()
        r.register(FileSkill())
        r.register(BrowserSkill())
        r.unregister("file_skill")
        assert len(r) == 1
        assert r.find_skill_for_action("open_url") is not None
        assert r.find_skill_for_action("read_file") is None


class TestSkillRegistryThreadSafety:
    """Concurrent registration/unregistration must not corrupt state."""

    def test_concurrent_register_does_not_corrupt(self):
        r = SkillRegistry()
        errors: list[Exception] = []

        def register_and_unregister(suffix: int) -> None:
            try:
                class DynamicSkill(BaseSkill):
                    name = f"dynamic_skill_{suffix}"
                    version = "1.0.0"

                    def capabilities(self):
                        return [SkillCapability(f"action_{suffix}", "desc", "T0")]

                    async def execute(self, action, context):
                        return SkillResult.ok("ok")

                s = DynamicSkill()
                r.register(s)
                r.find_skill_for_action(f"action_{suffix}")
                r.unregister(f"dynamic_skill_{suffix}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_and_unregister, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
