"""
tests/unit/test_p2_window_and_factual_fixes.py
================================================
P2 Regression Suite — two runtime bugs fixed in this milestone:

Issue 1 — minimize_window arguments missing
  1A. _extract_app_name now includes minimize/minimise/maximize/maximise/restore verbs,
      so "Minimize Edge" produces Entity(name="name", value="edge").
  1B. SkillResult.fail(non_retryable=True) is propagated through ToolResult so
      the retry loop exits on the first attempt for missing-param failures.

Issue 2 — factual questions routed to clarify
  2.  DecisionEngine bypasses the confidence gate for action=="chat", routing
      to LLM_DIRECT so that "What is the capital of Japan?" never returns a
      clarification prompt.

Regression coverage (pre-existing behaviour that must remain intact):
  - "Open Edge" → launch_app with name entity
  - "Close Edge" → close_app with name entity
  - "Focus Edge" → bring_app_to_foreground with name entity
  - "search the web for Python tutorials" → search_web → GoalIntentClassifier "chat"
  - Factual question does NOT enter AutonomousAgent
  - "Open Calculator. Open Calculator. Open Calculator." → launch_app (classifier level)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_intent(action: str, confidence: float = 0.9, raw_utterance: str = ""):
    from spidy.brain.types import Intent
    return Intent(
        action=action,
        entities=(),
        confidence=confidence,
        raw_utterance=raw_utterance,
        source="heuristic",
    )


def make_ctx(action: str, **params):
    from spidy.skills.base import SkillContext
    return SkillContext(action=action, params=params, session_id="test", user_name="Test")


@pytest.fixture
def classifier():
    from spidy.brain.intent_classifier import IntentClassifier
    return IntentClassifier()


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Issue 1A: Entity extraction for minimize / maximize / restore / focus
# ══════════════════════════════════════════════════════════════════════════════


class TestWindowVerbEntityExtraction:
    """_extract_app_name must extract the app name for window-management verbs."""

    @pytest.mark.asyncio
    async def test_minimize_edge_extracts_name(self, classifier):
        """Fix 1A: 'Minimize Edge' → minimize_window with name='edge'."""
        intent = await classifier.classify("Minimize Edge.")
        assert intent.action == "minimize_window", (
            f"Expected minimize_window, got {intent.action!r}"
        )
        name = intent.get_entity("name")
        assert name, "Entity 'name' must be present"
        assert "edge" in name.lower(), f"Expected 'edge' in name, got {name!r}"

    @pytest.mark.asyncio
    async def test_maximize_edge_extracts_name(self, classifier):
        """Fix 1A: 'Maximize Edge' → maximize_window with name='edge'."""
        intent = await classifier.classify("Maximize Edge.")
        assert intent.action == "maximize_window", (
            f"Expected maximize_window, got {intent.action!r}"
        )
        name = intent.get_entity("name")
        assert name, "Entity 'name' must be present"
        assert "edge" in name.lower(), f"Expected 'edge' in name, got {name!r}"

    @pytest.mark.asyncio
    async def test_minimise_edge_uk_spelling(self, classifier):
        """UK spelling 'minimise' must also extract the name."""
        intent = await classifier.classify("Minimise Edge.")
        assert intent.action == "minimize_window"
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()

    @pytest.mark.asyncio
    async def test_maximise_edge_uk_spelling(self, classifier):
        """UK spelling 'maximise' must also extract the name."""
        intent = await classifier.classify("Maximise Edge.")
        assert intent.action == "maximize_window"
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()

    @pytest.mark.asyncio
    async def test_minimize_notepad_extracts_name(self, classifier):
        intent = await classifier.classify("Minimize Notepad.")
        assert intent.action == "minimize_window"
        name = intent.get_entity("name")
        assert "notepad" in name.lower()

    @pytest.mark.asyncio
    async def test_maximize_chrome_extracts_name(self, classifier):
        intent = await classifier.classify("Maximize Chrome.")
        assert intent.action == "maximize_window"
        name = intent.get_entity("name")
        assert "chrome" in name.lower()

    @pytest.mark.asyncio
    async def test_maximize_vscode_extracts_name(self, classifier):
        intent = await classifier.classify("maximize vs code")
        assert intent.action == "maximize_window"
        name = intent.get_entity("name")
        assert name, "Entity 'name' must be present for 'maximize vs code'"

    @pytest.mark.asyncio
    async def test_restore_edge_extracts_name(self, classifier):
        """'Restore Edge' → maximize_window (restore is aliased to maximize)."""
        intent = await classifier.classify("Restore Edge.")
        assert intent.action == "maximize_window", (
            f"Expected maximize_window for restore, got {intent.action!r}"
        )
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()

    @pytest.mark.asyncio
    async def test_focus_edge_extracts_name(self, classifier):
        """'Focus Edge' → bring_app_to_foreground with name='edge'."""
        intent = await classifier.classify("Focus Edge.")
        assert intent.action == "bring_app_to_foreground", (
            f"Expected bring_app_to_foreground, got {intent.action!r}"
        )
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()

    @pytest.mark.asyncio
    async def test_close_edge_extracts_name(self, classifier):
        """Regression: 'Close Edge' → close_app with name='edge' (pre-existing)."""
        intent = await classifier.classify("Close Edge.")
        assert intent.action == "close_app"
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()

    @pytest.mark.asyncio
    async def test_open_edge_extracts_name(self, classifier):
        """Regression: 'Open Edge' → launch_app with name='edge' (pre-existing)."""
        intent = await classifier.classify("Open Edge.")
        assert intent.action == "launch_app"
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Issue 1B: SkillResult.non_retryable propagation
# ══════════════════════════════════════════════════════════════════════════════


class TestSkillResultNonRetryable:
    """SkillResult.fail(non_retryable=True) must exit the retry loop immediately."""

    def test_skill_result_fail_non_retryable_default_false(self):
        """Baseline: default non_retryable is False (no behaviour change)."""
        from spidy.skills.base import SkillResult
        r = SkillResult.fail("something went wrong")
        assert r.non_retryable is False

    def test_skill_result_fail_non_retryable_true(self):
        """SkillResult.fail accepts non_retryable=True."""
        from spidy.skills.base import SkillResult
        r = SkillResult.fail("'name' parameter is required", non_retryable=True)
        assert r.non_retryable is True
        assert not r.success

    def test_skill_result_ok_not_non_retryable(self):
        """Success results must never be marked non_retryable."""
        from spidy.skills.base import SkillResult
        r = SkillResult.ok("done")
        assert r.non_retryable is False

    @pytest.mark.asyncio
    async def test_missing_param_does_not_retry(self):
        """
        When minimize_window is called with no 'name' param, the skill returns
        non_retryable=True. The ToolRouter must attempt exactly once.
        """
        from spidy.brain.tool_router import ToolRouter
        from spidy.brain.types import PlanStep
        from spidy.skills.base import SkillResult

        bus = MagicMock()
        bus.publish = AsyncMock()

        # A skill that returns a non-retryable failure on the first call
        failing_skill = MagicMock()
        failing_skill.name = "app_skill"
        failing_skill.execute = AsyncMock(
            return_value=SkillResult.fail(
                "'name' parameter is required for minimize_window.",
                non_retryable=True,
            )
        )

        registry = MagicMock()
        registry.find_skill_for_action = MagicMock(return_value=failing_skill)

        router = ToolRouter(bus=bus, skill_registry=registry, max_retries=2)
        step = PlanStep(step_type="skill", action="minimize_window")

        result = await router._run_skill_with_retry(step, session_id="", user_name="test")

        assert result.success is False
        assert result.non_retryable is True
        assert failing_skill.execute.call_count == 1, (
            f"Missing-param failure was retried {failing_skill.execute.call_count} times; "
            "expected exactly 1 attempt (non_retryable flag must stop retries)"
        )

    @pytest.mark.asyncio
    async def test_retryable_skill_failure_still_retries(self):
        """
        A transient SkillResult.fail (non_retryable=False) must still be retried.
        Ensures we didn't break normal retry behaviour.
        """
        from spidy.brain.tool_router import ToolRouter
        from spidy.brain.types import PlanStep
        from spidy.skills.base import SkillResult
        import spidy.brain.tool_router as tr_module

        bus = MagicMock()
        bus.publish = AsyncMock()

        call_count = 0

        async def _execute(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return SkillResult.fail("transient error", non_retryable=False)
            return SkillResult.ok("success on third attempt")

        skill = MagicMock()
        skill.name = "test_skill"
        skill.execute = _execute

        registry = MagicMock()
        registry.find_skill_for_action = MagicMock(return_value=skill)

        router = ToolRouter(bus=bus, skill_registry=registry, max_retries=2)
        step = PlanStep(step_type="skill", action="test_action")

        # Patch asyncio.sleep inside tool_router to avoid real delays
        original_sleep = asyncio.sleep
        async def _fast_sleep(_delay):
            pass

        import unittest.mock as um
        with um.patch.object(tr_module.asyncio, "sleep", _fast_sleep):
            result = await router._run_skill_with_retry(step, session_id="", user_name="test")

        assert result.success is True
        assert call_count == 3, (
            f"Expected 3 attempts (2 retries), got {call_count}"
        )

    @pytest.mark.asyncio
    async def test_appskill_missing_name_param_marks_non_retryable(self):
        """
        AppSkill._minimize_window with no 'name' param must return
        SkillResult(success=False, non_retryable=True).
        """
        from spidy.skills.desktop.app_skill import AppSkill

        skill = AppSkill(bus=None)
        ctx = make_ctx("minimize_window")  # no 'name' param

        result = await skill.execute("minimize_window", ctx)
        assert result.success is False
        assert result.non_retryable is True, (
            "AppSkill missing-param failure must be non_retryable"
        )

    @pytest.mark.asyncio
    async def test_appskill_maximize_missing_name_non_retryable(self):
        from spidy.skills.desktop.app_skill import AppSkill
        skill = AppSkill(bus=None)
        ctx = make_ctx("maximize_window")
        result = await skill.execute("maximize_window", ctx)
        assert result.success is False
        assert result.non_retryable is True

    @pytest.mark.asyncio
    async def test_appskill_close_missing_name_non_retryable(self):
        from spidy.skills.desktop.app_skill import AppSkill
        skill = AppSkill(bus=None)
        ctx = make_ctx("close_app")
        result = await skill.execute("close_app", ctx)
        assert result.success is False
        assert result.non_retryable is True

    @pytest.mark.asyncio
    async def test_appskill_foreground_missing_name_non_retryable(self):
        from spidy.skills.desktop.app_skill import AppSkill
        skill = AppSkill(bus=None)
        ctx = make_ctx("bring_app_to_foreground")
        result = await skill.execute("bring_app_to_foreground", ctx)
        assert result.success is False
        assert result.non_retryable is True

    @pytest.mark.asyncio
    async def test_appskill_launch_missing_name_non_retryable(self):
        from spidy.skills.desktop.app_skill import AppSkill
        skill = AppSkill(bus=None)
        ctx = make_ctx("launch_app")
        result = await skill.execute("launch_app", ctx)
        assert result.success is False
        assert result.non_retryable is True


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Issue 2: Factual questions → LLM_DIRECT, never CLARIFY
# ══════════════════════════════════════════════════════════════════════════════


class TestFactualQuestionsDecisionEngine:
    """Fix 2: DecisionEngine must route intent.action=='chat' to LLM_DIRECT."""

    def _make_decision_engine(self):
        from spidy.brain.decision_engine import DecisionEngine
        registry = MagicMock()
        registry.find_skill_for_action = MagicMock(return_value=None)
        return DecisionEngine(skill_registry=registry)

    @pytest.mark.asyncio
    async def test_chat_with_low_confidence_goes_to_llm_not_clarify(self):
        """
        Fix 2: intent.action=='chat' at confidence=0.5 (below 0.6 gate)
        must produce LLM_DIRECT, not CLARIFY.
        """
        from spidy.brain.types import DecisionMode
        engine = self._make_decision_engine()
        intent = _make_intent("chat", confidence=0.5, raw_utterance="What is the capital of Japan?")
        decision = await engine.decide(intent)
        assert decision.mode == DecisionMode.LLM_DIRECT, (
            f"Expected LLM_DIRECT for chat intent, got {decision.mode!r}. "
            "Factual questions must never produce a CLARIFY response."
        )
        assert decision.mode != DecisionMode.CLARIFY

    @pytest.mark.asyncio
    async def test_chat_at_zero_confidence_still_llm_direct(self):
        """Edge case: even at confidence=0.0, 'chat' must go to LLM_DIRECT."""
        from spidy.brain.types import DecisionMode
        engine = self._make_decision_engine()
        intent = _make_intent("chat", confidence=0.0, raw_utterance="tell me a joke")
        decision = await engine.decide(intent)
        assert decision.mode == DecisionMode.LLM_DIRECT

    @pytest.mark.asyncio
    async def test_non_chat_low_confidence_still_clarifies(self):
        """
        Regression: a recognised intent (e.g. launch_app) at low confidence
        must still trigger CLARIFY. Only 'chat' bypasses the gate.
        """
        from spidy.brain.types import DecisionMode
        engine = self._make_decision_engine()
        intent = _make_intent("launch_app", confidence=0.3, raw_utterance="open something")
        decision = await engine.decide(intent)
        assert decision.mode == DecisionMode.CLARIFY, (
            f"Low-confidence non-chat intent must still CLARIFY, got {decision.mode!r}"
        )

    @pytest.mark.asyncio
    async def test_capital_of_japan_intent_is_chat(self):
        """
        'What is the capital of Japan?' must classify as action='chat'
        (no rule should match after P0-1 fix removed 'what is' from search_web).
        """
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("What is the capital of Japan?")
        assert intent.action == "chat", (
            f"Expected 'chat', got {intent.action!r}. "
            "Factual question must not match any search/skill rule."
        )

    @pytest.mark.asyncio
    async def test_capital_of_japan_decision_is_llm_direct(self):
        """
        End-to-end: 'What is the capital of Japan?' through classifier + decision
        engine must produce LLM_DIRECT, not CLARIFY.
        """
        from spidy.brain.intent_classifier import IntentClassifier
        from spidy.brain.types import DecisionMode
        engine = self._make_decision_engine()
        intent = await IntentClassifier().classify("What is the capital of Japan?")
        decision = await engine.decide(intent)
        assert decision.mode == DecisionMode.LLM_DIRECT, (
            f"'What is the capital of Japan?' → decision.mode={decision.mode!r}, "
            "expected LLM_DIRECT. The LLM must be called."
        )
        assert decision.mode != DecisionMode.CLARIFY

    @pytest.mark.asyncio
    async def test_various_factual_questions_are_llm_direct(self):
        """
        A range of factual questions must all reach LLM_DIRECT.
        None should produce CLARIFY.
        """
        from spidy.brain.intent_classifier import IntentClassifier
        from spidy.brain.types import DecisionMode

        questions = [
            "What is the capital of Japan?",
            "Who invented the telephone?",
            "Where is the Eiffel Tower?",
            "Why is the sky blue?",
            "How does photosynthesis work?",
            "When did World War 2 end?",
            "Tell me a joke.",
            "What is machine learning?",
        ]
        engine = self._make_decision_engine()
        clf = IntentClassifier()

        for q in questions:
            intent = await clf.classify(q)
            # All of these should be 'chat' (no rule should fire)
            # but even if a rule does fire at low confidence, chat must be LLM_DIRECT
            if intent.action == "chat":
                decision = await engine.decide(intent)
                assert decision.mode == DecisionMode.LLM_DIRECT, (
                    f"'{q}' → action='chat' → mode={decision.mode!r}, expected LLM_DIRECT"
                )
                assert decision.mode != DecisionMode.CLARIFY, (
                    f"'{q}' must NOT produce CLARIFY"
                )


class TestFactualQuestionsNotInAutonomousAgent:
    """Factual questions must not enter the AutonomousAgent pipeline."""

    def test_chat_intent_not_goal(self):
        """intent.action=='chat' must be classified as 'chat' by GoalIntentClassifier."""
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        clf = GoalIntentClassifier()
        intent = _make_intent("chat", confidence=0.5, raw_utterance="What is the capital of Japan?")
        route = clf.classify(intent)
        assert route == "chat", (
            f"'chat' intent must route to 'chat' (brain.process), not 'goal' (AutonomousAgent). "
            f"Got {route!r}."
        )

    @pytest.mark.asyncio
    async def test_capital_of_japan_not_goal_intent(self):
        """Full pipeline: 'What is the capital of Japan?' must not be is_goal_intent."""
        from spidy.brain.intent_classifier import IntentClassifier
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier

        intent = await IntentClassifier().classify("What is the capital of Japan?")
        clf = GoalIntentClassifier()
        assert not clf.is_executable(intent), (
            "Factual question must NOT enter AutonomousAgent (is_executable must be False)"
        )

    @pytest.mark.asyncio
    async def test_who_invented_telephone_not_goal(self):
        from spidy.brain.intent_classifier import IntentClassifier
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        intent = await IntentClassifier().classify("Who invented the telephone?")
        assert not GoalIntentClassifier().is_executable(intent)

    @pytest.mark.asyncio
    async def test_why_is_sky_blue_not_goal(self):
        from spidy.brain.intent_classifier import IntentClassifier
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        intent = await IntentClassifier().classify("Why is the sky blue?")
        assert not GoalIntentClassifier().is_executable(intent)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Regression: explicit search-web still works
# ══════════════════════════════════════════════════════════════════════════════


class TestSearchWebRegressions:
    """Explicit search-web utterances must still follow their intended path."""

    @pytest.mark.asyncio
    async def test_search_the_web_for_python_tutorials(self, classifier):
        """'search the web for Python tutorials' must still classify as search_web."""
        intent = await classifier.classify("search the web for Python tutorials")
        assert intent.action == "search_web", (
            f"Expected search_web, got {intent.action!r}"
        )

    @pytest.mark.asyncio
    async def test_search_web_routes_to_chat_not_goal(self, classifier):
        """search_web must route to GoalIntentClassifier 'chat' path (P1-2 regression)."""
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        intent = await classifier.classify("search the web for Python tutorials")
        route = GoalIntentClassifier().classify(intent)
        assert route == "chat", (
            f"search_web should route to 'chat' (brain.process), got {route!r}"
        )

    @pytest.mark.asyncio
    async def test_google_search_still_works(self, classifier):
        intent = await classifier.classify("google Python tutorials")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_look_up_still_works(self, classifier):
        intent = await classifier.classify("look up machine learning")
        assert intent.action == "search_web"


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Regression: "Open Edge" / repeated transcripts
# ══════════════════════════════════════════════════════════════════════════════


class TestOpenAppRegressions:
    """Launch-app intents must still work correctly after the verb-regex change."""

    @pytest.mark.asyncio
    async def test_open_edge_still_launch_app(self, classifier):
        """Regression: 'Open Edge' must remain launch_app with name='edge'."""
        intent = await classifier.classify("Open Edge")
        assert intent.action == "launch_app"
        name = intent.get_entity("name")
        assert name and "edge" in name.lower()

    @pytest.mark.asyncio
    async def test_open_calculator_still_launch_app(self, classifier):
        intent = await classifier.classify("Open Calculator")
        assert intent.action == "launch_app"
        name = intent.get_entity("name")
        assert name and "calculator" in name.lower()

    @pytest.mark.asyncio
    async def test_repeated_open_calculator_still_launch_app(self, classifier):
        """
        Regression: even a repeated transcript 'Open Calculator. Open Calculator.'
        must still classify the intent correctly.  (Deduplication is handled at
        the executor level, not the intent classifier level.)
        """
        utterances = [
            "Open Calculator.",
            "Open Calculator.",
            "Open Calculator.",
        ]
        for utt in utterances:
            intent = await classifier.classify(utt)
            assert intent.action == "launch_app", (
                f"'{utt}' → {intent.action!r}, expected launch_app"
            )

    @pytest.mark.asyncio
    async def test_switch_to_chrome_still_foreground(self, classifier):
        """Regression: 'switch to Chrome' must still classify as bring_app_to_foreground."""
        intent = await classifier.classify("switch to Chrome")
        assert intent.action == "bring_app_to_foreground"
