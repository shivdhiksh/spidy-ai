"""
tests/unit/test_p0_p1_latency_fixes.py
=======================================
Regression suite for the P0 and P1 latency / correctness fixes.

Covers:
  P0-1  -- Factual-question utterances ("what is X", "who is Y") must NOT route
           to search_web.  They should fall to chat (LLM-direct path).
  P0-2  -- BrowserSkill no longer calls the non-existent agent.navigate();
           it uses agent.open_url() for Bing, DuckDuckGo, Wikipedia, Maps.
  P1-1  -- AttributeError / TypeError / ImportError / NotImplementedError are
           detected as non-retryable; the retry loop exits immediately.
  P1-2  -- search_web is in CHAT_ACTIONS, not GOAL_ACTIONS.
  P1-3  -- GoalIntentClassifier factual-question fast-path routes to "chat"
           even when the classifier produces a search action.
  P1-4  -- SemanticMemory warmup does not initialise twice (idempotency check).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(action: str, confidence: float = 0.9, raw_utterance: str = ""):
    from spidy.brain.types import Intent
    return Intent(
        action=action,
        entities=(),
        confidence=confidence,
        raw_utterance=raw_utterance,
        source="heuristic",
    )


# ---------------------------------------------------------------------------
# P0-1: Factual-question routing (IntentClassifier)
# ---------------------------------------------------------------------------

class TestP01FactualQuestionRouting:
    """'what is', 'who is', etc. must NOT produce search_web."""

    @pytest.fixture
    def classifier(self):
        from spidy.brain.intent_classifier import IntentClassifier
        return IntentClassifier()

    @pytest.mark.asyncio
    async def test_what_is_capital_of_japan(self, classifier):
        intent = await classifier.classify("What is the capital of Japan?")
        assert intent.action != "search_web", (
            f"'What is the capital of Japan?' must NOT route to search_web "
            f"after P0-1 fix (got {intent.action!r})"
        )

    @pytest.mark.asyncio
    async def test_who_is_einstein(self, classifier):
        intent = await classifier.classify("Who is Albert Einstein?")
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_where_is_eiffel_tower(self, classifier):
        intent = await classifier.classify("where is the Eiffel Tower")
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_how_does_photosynthesis_work(self, classifier):
        intent = await classifier.classify("how does photosynthesis work")
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_why_is_sky_blue(self, classifier):
        intent = await classifier.classify("why is the sky blue")
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_when_did_ww2_end(self, classifier):
        intent = await classifier.classify("when did World War 2 end")
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_search_for_explicit_still_search_web(self, classifier):
        intent = await classifier.classify("search for Python tutorials")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_google_search_still_search_web(self, classifier):
        intent = await classifier.classify("google python tutorials")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_look_up_still_search_web(self, classifier):
        intent = await classifier.classify("look up machine learning")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_wikipedia_search_preserved(self, classifier):
        intent = await classifier.classify("search Wikipedia for Japan")
        assert intent.action == "search_wikipedia"

    @pytest.mark.asyncio
    async def test_calculate_math_not_search_web(self, classifier):
        intent = await classifier.classify("what is 100 divided by 4")
        assert intent.action == "calculate"
        assert intent.action != "search_web"


# ---------------------------------------------------------------------------
# P0-2: BrowserSkill navigate() -> open_url() fix
# ---------------------------------------------------------------------------

class TestP02BrowserSkillNavigateFix:
    """BrowserSkill must call agent.open_url(), not agent.navigate()."""

    def test_no_navigate_attribute_used_in_source(self):
        """Source-level check: agent.navigate must not appear in browser_skill.py."""
        import re
        source = open("spidy/skills/browser/browser_skill.py", encoding="utf-8").read()
        matches = re.findall(r'\bagent\.navigate\b', source)
        assert not matches, (
            f"Found {len(matches)} occurrence(s) of 'agent.navigate' in "
            f"browser_skill.py after P0-2 fix"
        )

    @pytest.mark.asyncio
    async def test_wikipedia_calls_open_url(self):
        from spidy.skills.browser.browser_skill import BrowserSkill
        from spidy.skills.base import SkillContext
        from spidy.browser.types import PageInfo

        pi = PageInfo(title="Result", url="https://en.wikipedia.org/wiki/Japan", tab_id=0)
        agent = MagicMock()
        agent.open_url = AsyncMock(return_value=pi)

        skill = BrowserSkill(bus=None)
        skill._agent = agent
        ctx = SkillContext(action="search_wikipedia", params={"query": "Japan"}, session_id="t")
        with patch.object(skill, '_get_or_create_agent', return_value=agent):
            with patch.object(skill, '_emit_event_obj', new=AsyncMock()):
                await skill._search_wikipedia(ctx)

        agent.open_url.assert_called_once()
        url_arg = agent.open_url.call_args.args[0]
        assert "wikipedia.org" in url_arg

    @pytest.mark.asyncio
    async def test_bing_calls_open_url(self):
        from spidy.skills.browser.browser_skill import BrowserSkill
        from spidy.skills.base import SkillContext
        from spidy.browser.types import PageInfo

        pi = PageInfo(title="Result", url="https://www.bing.com/search?q=test", tab_id=0)
        agent = MagicMock()
        agent.open_url = AsyncMock(return_value=pi)

        skill = BrowserSkill(bus=None)
        ctx = SkillContext(action="search_bing", params={"query": "test query"}, session_id="t")
        with patch.object(skill, '_get_or_create_agent', return_value=agent):
            with patch.object(skill, '_emit_event_obj', new=AsyncMock()):
                await skill._search_bing(ctx)

        agent.open_url.assert_called_once()
        url_arg = agent.open_url.call_args.args[0]
        assert "bing.com" in url_arg

    @pytest.mark.asyncio
    async def test_duckduckgo_calls_open_url(self):
        from spidy.skills.browser.browser_skill import BrowserSkill
        from spidy.skills.base import SkillContext
        from spidy.browser.types import PageInfo

        pi = PageInfo(title="Result", url="https://duckduckgo.com/?q=test", tab_id=0)
        agent = MagicMock()
        agent.open_url = AsyncMock(return_value=pi)

        skill = BrowserSkill(bus=None)
        ctx = SkillContext(action="search_duckduckgo", params={"query": "test query"}, session_id="t")
        with patch.object(skill, '_get_or_create_agent', return_value=agent):
            with patch.object(skill, '_emit_event_obj', new=AsyncMock()):
                await skill._search_duckduckgo(ctx)

        agent.open_url.assert_called_once()
        url_arg = agent.open_url.call_args.args[0]
        assert "duckduckgo.com" in url_arg

    @pytest.mark.asyncio
    async def test_maps_calls_open_url(self):
        from spidy.skills.browser.browser_skill import BrowserSkill
        from spidy.skills.base import SkillContext
        from spidy.browser.types import PageInfo

        pi = PageInfo(title="Result", url="https://www.google.com/maps/search/Paris", tab_id=0)
        agent = MagicMock()
        agent.open_url = AsyncMock(return_value=pi)

        skill = BrowserSkill(bus=None)
        ctx = SkillContext(action="search_maps", params={"query": "Paris"}, session_id="t")
        with patch.object(skill, '_get_or_create_agent', return_value=agent):
            with patch.object(skill, '_emit_event_obj', new=AsyncMock()):
                await skill._search_maps(ctx)

        agent.open_url.assert_called_once()
        url_arg = agent.open_url.call_args.args[0]
        assert "maps" in url_arg


# ---------------------------------------------------------------------------
# P1-1: Non-retryable errors
# ---------------------------------------------------------------------------

class TestP11NonRetryableErrors:
    """AttributeError / TypeError / ImportError / NotImplementedError must not be retried."""

    def _make_router(self, max_retries=2):
        from spidy.brain.tool_router import ToolRouter
        bus = MagicMock()
        bus.publish = AsyncMock()
        registry = MagicMock()
        return ToolRouter(bus=bus, skill_registry=registry, max_retries=max_retries)

    def _make_step(self, action="test_action"):
        from spidy.brain.types import PlanStep
        return PlanStep(step_type="skill", action=action)

    @pytest.mark.asyncio
    async def test_attribute_error_not_retried(self):
        router = self._make_router(max_retries=2)
        failing_skill = MagicMock()
        failing_skill.name = "bad_skill"
        failing_skill.execute = AsyncMock(
            side_effect=AttributeError("no attribute navigate")
        )
        router._registry.find_skill_for_action = MagicMock(return_value=failing_skill)
        step = self._make_step()
        result = await router._run_skill_with_retry(step, session_id="", user_name="test")
        assert result.success is False
        assert result.non_retryable is True
        assert failing_skill.execute.call_count == 1, (
            f"AttributeError was retried {failing_skill.execute.call_count} times; expected 1"
        )

    @pytest.mark.asyncio
    async def test_type_error_not_retried(self):
        router = self._make_router(max_retries=2)
        failing_skill = MagicMock()
        failing_skill.name = "bad_skill"
        failing_skill.execute = AsyncMock(side_effect=TypeError("bad arg"))
        router._registry.find_skill_for_action = MagicMock(return_value=failing_skill)
        result = await router._run_skill_with_retry(
            self._make_step(), session_id="", user_name="test"
        )
        assert result.non_retryable is True
        assert failing_skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_not_implemented_error_not_retried(self):
        router = self._make_router(max_retries=2)
        failing_skill = MagicMock()
        failing_skill.name = "bad_skill"
        failing_skill.execute = AsyncMock(side_effect=NotImplementedError("stub"))
        router._registry.find_skill_for_action = MagicMock(return_value=failing_skill)
        result = await router._run_skill_with_retry(
            self._make_step(), session_id="", user_name="test"
        )
        assert result.non_retryable is True
        assert failing_skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_import_error_not_retried(self):
        router = self._make_router(max_retries=2)
        failing_skill = MagicMock()
        failing_skill.name = "bad_skill"
        failing_skill.execute = AsyncMock(side_effect=ImportError("no module"))
        router._registry.find_skill_for_action = MagicMock(return_value=failing_skill)
        result = await router._run_skill_with_retry(
            self._make_step(), session_id="", user_name="test"
        )
        assert result.non_retryable is True
        assert failing_skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_transient_error_is_retried(self):
        """Generic RuntimeError must still be retried."""
        router = self._make_router(max_retries=2)
        call_count = 0

        async def _execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient failure")
            from spidy.skills.base import SkillResult
            return SkillResult(success=True, message="done")

        failing_skill = MagicMock()
        failing_skill.name = "transient_skill"
        failing_skill.execute = _execute
        router._registry.find_skill_for_action = MagicMock(return_value=failing_skill)
        with patch("spidy.brain.tool_router.asyncio.sleep", new=AsyncMock()):
            result = await router._run_skill_with_retry(
                self._make_step(), session_id="", user_name="test"
            )
        assert result.success is True
        assert call_count == 3

    def test_tool_result_non_retryable_defaults_false(self):
        from spidy.brain.types import ToolResult
        r = ToolResult.fail(message="err", action="x", error="oops")
        assert r.non_retryable is False


# ---------------------------------------------------------------------------
# P1-2: search_web classification
# ---------------------------------------------------------------------------

class TestP12SearchWebClassification:
    def test_search_web_not_in_goal_actions(self):
        from spidy.brain.goal_intent_classifier import GOAL_ACTIONS
        assert "search_web" not in GOAL_ACTIONS

    def test_search_web_in_chat_actions(self):
        from spidy.brain.goal_intent_classifier import CHAT_ACTIONS
        assert "search_web" in CHAT_ACTIONS

    def test_explicit_browser_actions_still_goal(self):
        from spidy.brain.goal_intent_classifier import GOAL_ACTIONS
        for a in ("search_google", "search_youtube", "search_bing",
                  "search_duckduckgo", "search_wikipedia", "search_maps"):
            assert a in GOAL_ACTIONS, f"'{a}' must remain in GOAL_ACTIONS"

    def test_search_web_routes_to_chat(self):
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        clf = GoalIntentClassifier()
        intent = _make_intent("search_web", confidence=0.9, raw_utterance="search for x")
        assert clf.classify(intent) == "chat"

    def test_search_google_still_goal(self):
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        clf = GoalIntentClassifier()
        intent = _make_intent("search_google", confidence=0.9, raw_utterance="search google for x")
        assert clf.classify(intent) == "goal"


# ---------------------------------------------------------------------------
# P1-3: Factual-question fast-path (GoalIntentClassifier)
# ---------------------------------------------------------------------------

class TestP13FactualQuestionFastPath:
    def setup_method(self):
        from spidy.brain.goal_intent_classifier import GoalIntentClassifier
        self.clf = GoalIntentClassifier()

    def test_what_is_capital_routes_to_chat(self):
        intent = _make_intent("chat", 0.5, "what is the capital of Japan?")
        assert self.clf.classify(intent) == "chat"

    def test_who_is_einstein_routes_to_chat(self):
        intent = _make_intent("chat", 0.5, "who is Albert Einstein")
        assert self.clf.classify(intent) == "chat"

    def test_where_is_paris_with_search_web_routes_to_chat(self):
        intent = _make_intent("search_web", 0.75, "where is Paris located")
        assert self.clf.classify(intent) == "chat"

    def test_how_does_with_search_web_routes_to_chat(self):
        intent = _make_intent("search_web", 0.75, "how does photosynthesis work")
        assert self.clf.classify(intent) == "chat"

    def test_why_is_sky_blue_routes_to_chat(self):
        intent = _make_intent("search_web", 0.75, "why is the sky blue")
        assert self.clf.classify(intent) == "chat"

    def test_high_conf_non_search_goal_not_overridden(self):
        """set_volume at high confidence must NOT be fast-pathed."""
        intent = _make_intent("set_volume", 0.9, "what is the volume set to")
        assert self.clf.classify(intent) == "goal"

    def test_search_wikipedia_with_factual_prefix_fast_pathed(self):
        # "what is" IS in _FACTUAL_QUESTION_PREFIXES; search_wikipedia is in _SEARCH_ACTIONS
        # so the fast-path applies and returns "chat"
        intent = _make_intent("search_wikipedia", 0.85, "what is in the Japan wikipedia article")
        assert self.clf.classify(intent) == "chat"

    def test_low_confidence_always_chat(self):
        intent = _make_intent("launch_app", 0.3, "open notepad")
        assert self.clf.classify(intent) == "chat"


# ---------------------------------------------------------------------------
# P1-4: SemanticMemory warmup idempotency
# ---------------------------------------------------------------------------

class TestP14SemanticMemoryWarmup:
    @pytest.mark.asyncio
    async def test_initialize_twice_warmup_only_once(self):
        """Calling initialize() twice schedules the warmup task at most once."""
        from spidy.memory.manager import MemoryManager

        mm = MemoryManager()
        mm._semantic = MagicMock()
        mm._semantic._ensure_initialized = MagicMock()
        mm._episodic = None

        tasks = []

        def _fake_create_task(coro):
            tasks.append(coro)
            coro.close()

        with patch("asyncio.create_task", side_effect=_fake_create_task):
            await mm.initialize()
            await mm.initialize()

        assert len(tasks) == 1, f"Expected 1 warmup task, got {len(tasks)}"

    @pytest.mark.asyncio
    async def test_warmup_exception_does_not_crash(self):
        """A failing warmup must be logged and swallowed, not propagated."""
        from spidy.memory.manager import MemoryManager

        mm = MemoryManager()
        mm._episodic = None
        semantic_mock = MagicMock()
        semantic_mock._ensure_initialized = MagicMock(
            side_effect=RuntimeError("model load failed")
        )
        mm._semantic = semantic_mock

        # Let initialize() run; the warmup task runs asynchronously
        await mm.initialize()
        # Give the event loop a chance to run the background task
        await asyncio.sleep(0.1)
        # If we got here without an unhandled exception, the test passes

    @pytest.mark.asyncio
    async def test_no_warmup_when_semantic_none(self):
        """When semantic memory is disabled, no warmup task is created."""
        from spidy.memory.manager import MemoryManager

        mm = MemoryManager()
        mm._semantic = None
        mm._episodic = None

        tasks = []

        def _fake_create_task(coro):
            tasks.append(coro)
            coro.close()

        with patch("asyncio.create_task", side_effect=_fake_create_task):
            await mm.initialize()

        assert len(tasks) == 0
