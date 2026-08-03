"""
tests/unit/test_planner_aliases.py — Planner Action Alias Tests
================================================================
Verifies that the _ACTION_ALIASES table correctly remaps intent action names
to canonical skill action names during plan construction.
"""

from __future__ import annotations

import pytest

from spidy.brain.planner import Planner, _ACTION_ALIASES
from spidy.brain.types import Decision, DecisionMode, Intent, Entity


def make_intent(action: str, entities: tuple[Entity, ...] = ()) -> Intent:
    return Intent(
        action=action,
        entities=entities,
        confidence=0.9,
        raw_utterance=f"test utterance for {action}",
        source="heuristic",
    )


def make_decision(action: str, mode: DecisionMode = DecisionMode.SKILL) -> Decision:
    return Decision(
        mode=mode,
        intent=make_intent(action),
        skill_name=action,
        rationale="test",
    )


@pytest.fixture
def planner():
    return Planner()


class TestActionAliasTable:
    """Validate the _ACTION_ALIASES table contents."""

    def test_lock_screen_alias(self):
        assert _ACTION_ALIASES["lock_screen"] == "lock_workstation"

    def test_shutdown_alias(self):
        assert _ACTION_ALIASES["shutdown"] == "shutdown_system"

    def test_restart_alias(self):
        assert _ACTION_ALIASES["restart"] == "restart_system"

    def test_search_web_alias(self):
        assert _ACTION_ALIASES["search_web"] == "search_google"

    def test_math_alias(self):
        assert _ACTION_ALIASES["math"] == "calculate"

    def test_no_alias_for_known_actions(self):
        """Actions that don't need remapping should not be in the alias table."""
        assert "launch_app" not in _ACTION_ALIASES
        assert "search_google" not in _ACTION_ALIASES
        assert "search_youtube" not in _ACTION_ALIASES
        assert "calculate" not in _ACTION_ALIASES
        assert "take_screenshot" not in _ACTION_ALIASES


class TestPlannerAliasApplication:
    """Verify that Planner._skill_step() applies aliases correctly."""

    @pytest.mark.asyncio
    async def test_lock_screen_remapped(self, planner):
        decision = make_decision("lock_screen")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.step_type == "skill"
        assert plan.first.action == "lock_workstation"

    @pytest.mark.asyncio
    async def test_shutdown_remapped(self, planner):
        decision = make_decision("shutdown")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "shutdown_system"

    @pytest.mark.asyncio
    async def test_restart_remapped(self, planner):
        decision = make_decision("restart")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "restart_system"

    @pytest.mark.asyncio
    async def test_search_web_remapped(self, planner):
        decision = make_decision("search_web")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "search_google"

    @pytest.mark.asyncio
    async def test_math_remapped(self, planner):
        decision = make_decision("math")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "calculate"

    @pytest.mark.asyncio
    async def test_no_alias_passthrough(self, planner):
        """Actions without aliases should pass through unchanged."""
        decision = make_decision("launch_app")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "launch_app"

    @pytest.mark.asyncio
    async def test_search_youtube_no_alias(self, planner):
        """search_youtube is a direct action — no alias needed."""
        decision = make_decision("search_youtube")
        plan = await planner.plan(decision, session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "search_youtube"

    @pytest.mark.asyncio
    async def test_entities_preserved_through_alias(self, planner):
        """Entities should be preserved even when action name is aliased."""
        intent = make_intent(
            "lock_screen",
            entities=(Entity(name="reason", value="going_out"),)
        )
        decision = Decision(
            mode=DecisionMode.SKILL,
            intent=intent,
            skill_name="lock_screen",
            rationale="test",
        )
        plan = await planner.plan(decision, session_id="test")
        assert plan.first.action == "lock_workstation"
        assert plan.first.params.get("reason") == "going_out"
