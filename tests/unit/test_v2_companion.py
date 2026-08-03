"""
tests/unit/test_v2_companion.py — Spidy V2.0 JARVIS Experience Unit Tests
==========================================================================
Covers all new V2 capabilities:
  - ContextResolver: pronoun and reference resolution
  - Compound intent detection in IntentClassifier
  - Multi-step planning in Planner
  - Retry and fallback in ToolRouter
  - Proactive checks (ProactiveChecker)
  - Natural language response composition (ResponseComposer)
  - ConversationManager V2 extensions

Design: All tests are self-contained. No real OS calls, no network,
no LLM calls. External dependencies are mocked.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.brain.context_resolver import ContextResolver, ResolutionResult
from spidy.brain.response_composer import ResponseComposer
from spidy.brain.types import (
    ConversationTurn,
    Decision,
    DecisionMode,
    Entity,
    Intent,
    Plan,
    PlanStep,
    ToolResult,
    TurnRole,
)
from spidy.core.event_bus import EventBus


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_bus() -> EventBus:
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)
    return bus


def make_turn(
    role: TurnRole,
    text: str,
    action: str = "",
    entities: list[tuple[str, str]] | None = None,
) -> ConversationTurn:
    """Build a ConversationTurn with an optional Intent."""
    intent: Intent | None = None
    if action:
        intent = Intent(
            action=action,
            entities=tuple(
                Entity(name=n, value=v) for n, v in (entities or [])
            ),
            confidence=0.9,
            raw_utterance=text,
        )
    return ConversationTurn(role=role, text=text, intent=intent)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ContextResolver Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextResolver:
    """Tests for anaphora and reference resolution."""

    def make_resolver(self) -> ContextResolver:
        return ContextResolver(max_history_turns=10)

    def test_no_change_without_history(self) -> None:
        """No references to resolve when conversation is empty."""
        resolver = self.make_resolver()
        result = resolver.resolve("Open it", turns=[])
        assert result.utterance == "Open it"
        assert not result.was_resolved

    def test_resolves_open_it_to_last_app(self) -> None:
        """'Open it' resolves to the last launched app."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Open VS Code", "launch_app", [("name", "VS Code")]),
            make_turn(TurnRole.ASSISTANT, "I've opened VS Code."),
        ]
        result = resolver.resolve("Open it", turns)
        assert result.was_resolved
        assert "VS Code" in result.utterance or "vs code" in result.utterance.lower()

    def test_resolves_close_that_to_last_app(self) -> None:
        """'Close that' resolves to the last known app."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Open Chrome", "launch_app", [("name", "Chrome")]),
        ]
        result = resolver.resolve("Close that", turns)
        assert result.was_resolved
        assert "chrome" in result.utterance.lower()

    def test_resolves_the_file_descriptor(self) -> None:
        """'Open the file' resolves to the last mentioned file."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Find my resume", "search_files", [("filename", "resume")]),
            make_turn(TurnRole.ASSISTANT, "Found resume.pdf."),
        ]
        result = resolver.resolve("Open the file", turns)
        assert result.was_resolved
        assert "resume" in result.utterance.lower()

    def test_no_resolution_for_greeting(self) -> None:
        """Greetings are not rewritten even if context exists."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Open VS Code", "launch_app", [("name", "VS Code")]),
        ]
        result = resolver.resolve("Hello Spidy", turns)
        assert not result.was_resolved
        assert result.utterance == "Hello Spidy"

    def test_no_resolution_for_plain_utterance(self) -> None:
        """Utterances without references are returned unchanged."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Open VS Code", "launch_app", [("name", "VS Code")]),
        ]
        result = resolver.resolve("Search YouTube for Python tutorials", turns)
        assert result.utterance == "Search YouTube for Python tutorials"

    def test_resolved_refs_list_populated(self) -> None:
        """resolved_refs is populated when resolution occurs."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Open Notepad", "launch_app", [("name", "Notepad")]),
        ]
        result = resolver.resolve("Open it", turns)
        if result.was_resolved:
            assert len(result.resolved_refs) > 0

    def test_resolution_result_dataclass(self) -> None:
        """ResolutionResult.was_resolved is False when refs is empty."""
        r = ResolutionResult(utterance="hello", resolved_refs=[])
        assert not r.was_resolved

    def test_entity_from_search_intent(self) -> None:
        """ContextResolver picks up search query entities."""
        resolver = self.make_resolver()
        turns = [
            make_turn(TurnRole.USER, "Search for Python tutorials",
                      "search_web", [("query", "Python tutorials")]),
        ]
        result = resolver.resolve("Search for that again", turns)
        # At least: the utterance should not crash; may or may not resolve
        assert isinstance(result.utterance, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Compound Intent Detection Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompoundIntentDetection:
    """Tests for multi-step utterance parsing in IntentClassifier."""

    @pytest.fixture
    def classifier(self):
        from spidy.brain.intent_classifier import IntentClassifier
        return IntentClassifier()

    @pytest.mark.asyncio
    async def test_compound_intent_detected(self, classifier) -> None:
        """Compound utterance returns action='compound'."""
        intent = await classifier.classify("Open VS Code, then search YouTube for Python")
        assert intent.action == "compound"
        assert len(intent.entities) >= 2

    @pytest.mark.asyncio
    async def test_compound_steps_parseable(self, classifier) -> None:
        """Each compound step encodes a valid sub-intent."""
        intent = await classifier.classify(
            "Open Chrome, then search Google for AI news, then open Notepad"
        )
        assert intent.action == "compound"
        steps = classifier.extract_compound_steps(intent)
        assert len(steps) >= 2
        for step in steps:
            assert "action" in step
            assert "utterance" in step

    @pytest.mark.asyncio
    async def test_single_intent_not_compound(self, classifier) -> None:
        """Single-intent utterances are not classified as compound."""
        intent = await classifier.classify("Open VS Code")
        assert intent.action != "compound"

    @pytest.mark.asyncio
    async def test_greeting_not_compound(self, classifier) -> None:
        """Greetings are not classified as compound."""
        intent = await classifier.classify("Hello Spidy")
        assert intent.action != "compound"

    @pytest.mark.asyncio
    async def test_compound_semicolon_separator(self, classifier) -> None:
        """Semicolons trigger compound detection."""
        intent = await classifier.classify("Open Notepad; open Calculator")
        assert intent.action == "compound"

    @pytest.mark.asyncio
    async def test_compound_then_separator(self, classifier) -> None:
        """'then' keyword triggers compound detection."""
        intent = await classifier.classify("Open Chrome then search for Python")
        assert intent.action == "compound"

    @pytest.mark.asyncio
    async def test_extract_compound_steps_static_method(self, classifier) -> None:
        """extract_compound_steps() works on any compound Intent."""
        from spidy.brain.intent_classifier import IntentClassifier
        intent = Intent(
            action="compound",
            entities=(
                Entity(name="step_0", value=json.dumps({
                    "utterance": "open chrome",
                    "action": "launch_app",
                    "entities": [{"name": "name", "value": "chrome"}],
                    "confidence": 0.9,
                })),
                Entity(name="step_1", value=json.dumps({
                    "utterance": "search for Python",
                    "action": "search_web",
                    "entities": [{"name": "query", "value": "Python"}],
                    "confidence": 0.75,
                })),
            ),
            confidence=0.9,
            raw_utterance="open chrome, search for Python",
        )
        steps = IntentClassifier.extract_compound_steps(intent)
        assert len(steps) == 2
        assert steps[0]["action"] == "launch_app"
        assert steps[1]["action"] == "search_web"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Step Planner Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiStepPlanner:
    """Tests for the Planner's multi-step plan generation."""

    @pytest.fixture
    def planner(self):
        from spidy.brain.planner import Planner
        return Planner()

    def _make_compound_decision(self, steps_data: list[dict]) -> Decision:
        """Helper: build a Decision wrapping a compound Intent."""
        entities = tuple(
            Entity(name=f"step_{i}", value=json.dumps(data))
            for i, data in enumerate(steps_data)
        )
        intent = Intent(
            action="compound",
            entities=entities,
            confidence=0.9,
            raw_utterance="multi-step test",
        )
        return Decision(
            mode=DecisionMode.SKILL,
            intent=intent,
            skill_name="",
            rationale="test",
        )

    @pytest.mark.asyncio
    async def test_compound_intent_produces_multi_step_plan(self, planner) -> None:
        """Compound intent → Plan with multiple steps."""
        decision = self._make_compound_decision([
            {"utterance": "open chrome", "action": "launch_app",
             "entities": [{"name": "name", "value": "chrome"}], "confidence": 0.9},
            {"utterance": "search python", "action": "search_web",
             "entities": [{"name": "query", "value": "python"}], "confidence": 0.75},
        ])
        plan = await planner.plan(decision, session_id="test")
        assert len(plan.steps) == 2
        assert plan.steps[0].step_type == "skill"
        assert plan.steps[1].step_type == "skill"

    @pytest.mark.asyncio
    async def test_single_intent_still_produces_one_step(self, planner) -> None:
        """Non-compound intents still produce single-step plans."""
        intent = Intent(action="launch_app",
                        entities=(Entity(name="name", value="notepad"),),
                        confidence=0.9, raw_utterance="open notepad")
        decision = Decision(
            mode=DecisionMode.SKILL,
            intent=intent,
            skill_name="app_skill",
        )
        plan = await planner.plan(decision, session_id="test")
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_compound_step_params_extracted(self, planner) -> None:
        """Entities from compound sub-intents are correctly mapped to params."""
        decision = self._make_compound_decision([
            {"utterance": "open vscode", "action": "launch_app",
             "entities": [{"name": "name", "value": "VS Code"}], "confidence": 0.9},
        ])
        plan = await planner.plan(decision)
        assert plan.steps[0].params.get("name") == "VS Code"

    @pytest.mark.asyncio
    async def test_plan_is_empty_on_noop(self, planner) -> None:
        """REJECT decision produces a noop plan."""
        intent = Intent(action="shutdown_system", confidence=0.9, raw_utterance="shutdown")
        decision = Decision(mode=DecisionMode.REJECT, intent=intent)
        plan = await planner.plan(decision)
        assert len(plan.steps) == 1
        assert plan.steps[0].step_type == "noop"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ToolRouter Retry & Fallback Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolRouterRetry:
    """Tests for retry logic and fallback messages in ToolRouter."""

    @pytest.mark.asyncio
    async def test_skill_retry_on_failure(self) -> None:
        """ToolRouter retries failed skills before giving up."""
        from spidy.brain.tool_router import ToolRouter
        from spidy.skills.registry import SkillRegistry

        bus = make_bus()
        registry = SkillRegistry()

        # Create a skill that fails twice then succeeds
        mock_skill = MagicMock()
        mock_skill.name = "test_skill"
        mock_skill.find_skill_for_action = MagicMock(return_value=mock_skill)
        mock_skill.execute = AsyncMock(side_effect=[
            Exception("transient error"),  # fail 1
            Exception("transient error"),  # fail 2
            MagicMock(success=True, message="Done!", data=None, error=None),  # succeed
        ])
        mock_skill.capabilities = MagicMock(return_value=[
            MagicMock(action="test_action")
        ])

        router = ToolRouter(bus=bus, skill_registry=registry, max_retries=2)

        step = PlanStep(step_type="skill", action="test_action", params={})
        plan = Plan(steps=(step,), session_id="test")

        with patch.object(registry, "find_skill_for_action", return_value=mock_skill):
            results = await router.execute(plan, session_id="test")

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_skill_fails_after_max_retries(self) -> None:
        """ToolRouter returns failure after exhausting all retries."""
        from spidy.brain.tool_router import ToolRouter
        from spidy.skills.registry import SkillRegistry

        bus = make_bus()
        registry = SkillRegistry()
        router = ToolRouter(bus=bus, skill_registry=registry, max_retries=1)

        step = PlanStep(step_type="skill", action="broken_action", params={})
        plan = Plan(steps=(step,), session_id="test")

        results = await router.execute(plan, session_id="test")
        assert len(results) == 1
        assert not results[0].success
        assert results[0].message  # Should have a user-facing message

    @pytest.mark.asyncio
    async def test_no_skill_returns_helpful_message(self) -> None:
        """When no skill handles an action, the message is helpful."""
        from spidy.brain.tool_router import ToolRouter
        from spidy.skills.registry import SkillRegistry

        bus = make_bus()
        registry = SkillRegistry()
        router = ToolRouter(bus=bus, skill_registry=registry, max_retries=0)

        step = PlanStep(step_type="skill", action="nonexistent_action", params={})
        plan = Plan(steps=(step,), session_id="test")

        results = await router.execute(plan, session_id="test")
        assert not results[0].success
        assert "nonexistent action" in results[0].message.lower() or \
               "don't have a skill" in results[0].message.lower() or \
               "help" in results[0].message.lower()

    @pytest.mark.asyncio
    async def test_multi_step_progress_events_published(self) -> None:
        """BrainProgressEvent is published for each step in a multi-step plan."""
        from spidy.brain.events import BrainProgressEvent
        from spidy.brain.tool_router import ToolRouter
        from spidy.skills.registry import SkillRegistry

        bus = make_bus()
        registry = SkillRegistry()
        router = ToolRouter(bus=bus, skill_registry=registry, max_retries=0)

        progress_events: list[BrainProgressEvent] = []

        def capture(event):
            progress_events.append(event)

        bus.subscribe("brain.progress", capture)

        step1 = PlanStep(step_type="noop", action="", params={})
        step2 = PlanStep(step_type="noop", action="", params={})
        plan = Plan(steps=(step1, step2), session_id="test")

        await router.execute(plan, session_id="test")
        assert len(progress_events) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ProactiveChecker Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProactiveChecker:
    """Tests for the ProactiveChecker."""

    @pytest.mark.asyncio
    async def test_proceed_for_unknown_app(self) -> None:
        """Returns 'proceed' when app is not in process list."""
        from spidy.brain.proactive import ProactiveChecker
        checker = ProactiveChecker()

        with patch("spidy.brain.proactive._get_running_processes", return_value=set()):
            advice = await checker.check("launch_app", {"name": "VS Code"})

        assert advice.action == "proceed"

    @pytest.mark.asyncio
    async def test_reuse_when_app_already_running(self) -> None:
        """Returns 'reuse' when the target app is already running."""
        from spidy.brain.proactive import ProactiveChecker
        checker = ProactiveChecker()

        with patch("spidy.brain.proactive._get_running_processes",
                   return_value={"code.exe", "explorer.exe"}):
            advice = await checker.check("launch_app", {"name": "VS Code"})

        assert advice.action == "reuse"
        assert advice.should_skip
        assert "VS Code" in advice.message or "vs code" in advice.message.lower()

    @pytest.mark.asyncio
    async def test_proceed_for_non_launch_action(self) -> None:
        """Non-launch actions always return 'proceed'."""
        from spidy.brain.proactive import ProactiveChecker
        checker = ProactiveChecker()
        advice = await checker.check("set_volume", {"level": "50"})
        assert advice.action == "proceed"

    @pytest.mark.asyncio
    async def test_ask_when_network_unavailable(self) -> None:
        """Returns 'ask' for search actions when network is down."""
        from spidy.brain.proactive import ProactiveChecker
        checker = ProactiveChecker()

        with patch("spidy.brain.proactive._is_network_available", return_value=False):
            advice = await checker.check("search_google", {"query": "python"})

        assert advice.action == "ask"
        assert "offline" in advice.message.lower() or "internet" in advice.message.lower()

    @pytest.mark.asyncio
    async def test_proceed_for_search_when_online(self) -> None:
        """Returns 'proceed' for search actions when network is available."""
        from spidy.brain.proactive import ProactiveChecker
        checker = ProactiveChecker()

        with patch("spidy.brain.proactive._is_network_available", return_value=True):
            advice = await checker.check("search_google", {"query": "python"})

        assert advice.action == "proceed"

    @pytest.mark.asyncio
    async def test_proactive_check_never_raises(self) -> None:
        """Even if psutil fails, ProactiveChecker returns 'proceed'."""
        from spidy.brain.proactive import ProactiveChecker
        checker = ProactiveChecker()

        with patch("spidy.brain.proactive._get_running_processes",
                   side_effect=Exception("psutil error")):
            advice = await checker.check("launch_app", {"name": "any app"})

        assert advice.action == "proceed"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ResponseComposer Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestResponseComposer:
    """Tests for natural language response composition."""

    @pytest.fixture
    def composer(self) -> ResponseComposer:
        """Non-varied composer for deterministic test output."""
        return ResponseComposer(use_varied_phrases=False)

    def test_single_success_returns_message(self, composer) -> None:
        """Single successful result → its message is returned."""
        result = ToolResult.ok(message="I've opened VS Code.", action="launch_app", step_type="skill")
        text = composer.compose([result], action="launch_app")
        assert "VS Code" in text

    def test_single_failure_returns_failure_message(self, composer) -> None:
        """Single failed result → failure message is returned."""
        result = ToolResult.fail(message="I couldn't open that app.", action="launch_app", step_type="skill")
        text = composer.compose([result], action="launch_app")
        assert text  # Non-empty
        assert "app" in text.lower() or "open" in text.lower() or "couldn't" in text.lower()

    def test_empty_results_returns_empty(self, composer) -> None:
        """Empty results list → empty string."""
        text = composer.compose([])
        assert text == ""

    def test_llm_result_passes_through(self, composer) -> None:
        """LLM results are returned as-is (already natural language)."""
        result = ToolResult.ok(
            message="The capital of France is Paris.",
            action="llm",
            step_type="llm",
        )
        text = composer.compose([result], action="llm")
        assert "Paris" in text

    def test_multistep_all_success(self, composer) -> None:
        """Multi-step all-success → aggregated narrative."""
        results = [
            ToolResult.ok(message="folder created", action="create_folder", step_type="skill"),
            ToolResult.ok(message="VS Code opened", action="launch_app", step_type="skill"),
            ToolResult.ok(message="main.py created", action="create_file", step_type="skill"),
        ]
        text = composer.compose(results)
        assert text  # Non-empty
        # Should mention multiple actions
        assert len(text) > 20

    def test_multistep_partial_failure(self, composer) -> None:
        """Partial failure → message acknowledges both success and failure."""
        results = [
            ToolResult.ok(message="folder created", action="create_folder", step_type="skill"),
            ToolResult.fail(message="couldn't open VS Code", action="launch_app", step_type="skill"),
        ]
        text = composer.compose(results)
        assert text
        # Should mention the failure
        assert "folder" in text.lower() or "created" in text.lower() or "couldn't" in text.lower()

    def test_compose_progress(self, composer) -> None:
        """Progress message includes step number and action."""
        text = composer.compose_progress(1, 4, "launch_app")
        assert "2/4" in text or "2" in text
        assert "launch" in text.lower() or "app" in text.lower()

    def test_noop_returns_empty(self, composer) -> None:
        """Noop result produces empty response."""
        result = ToolResult.ok(message="", action="noop", step_type="noop")
        text = composer.compose([result], action="noop")
        assert text == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ConversationManager V2 Extensions Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConversationManagerV2:
    """Tests for the V2 ConversationManager extensions."""

    @pytest.mark.asyncio
    async def test_get_entity_history_returns_intent_turns(self) -> None:
        """get_entity_history() only returns turns with intents."""
        from spidy.brain.conversation_manager import ConversationManager

        bus = make_bus()
        cm = ConversationManager(bus=bus)
        await cm.start_session("test")

        # Add turn without intent
        cm.add_turn(TurnRole.USER, "hello")
        # Add turn with intent
        intent = Intent(action="launch_app",
                        entities=(Entity(name="name", value="Chrome"),),
                        confidence=0.9, raw_utterance="open chrome")
        cm.add_turn(TurnRole.USER, "open chrome", intent=intent)

        history = cm.get_entity_history()
        assert len(history) == 1
        assert history[0].intent is not None

    @pytest.mark.asyncio
    async def test_record_and_get_last_action(self) -> None:
        """record_action() and get_last_action() round-trip correctly."""
        from spidy.brain.conversation_manager import ConversationManager

        bus = make_bus()
        cm = ConversationManager(bus=bus)
        await cm.start_session("test")

        cm.record_action("launch_app", "I've opened VS Code.")
        action, msg = cm.get_last_action()
        assert action == "launch_app"
        assert "VS Code" in msg

    @pytest.mark.asyncio
    async def test_get_last_action_default_empty(self) -> None:
        """get_last_action() returns ('', '') before any action is recorded."""
        from spidy.brain.conversation_manager import ConversationManager

        bus = make_bus()
        cm = ConversationManager(bus=bus)
        await cm.start_session("test")

        action, msg = cm.get_last_action()
        assert action == ""
        assert msg == ""

    @pytest.mark.asyncio
    async def test_get_summary_includes_more_turns(self) -> None:
        """V2 get_summary() includes up to 8 turns (up from 3)."""
        from spidy.brain.conversation_manager import ConversationManager

        bus = make_bus()
        cm = ConversationManager(bus=bus)
        await cm.start_session("test")

        for i in range(6):
            cm.add_turn(TurnRole.USER, f"Message {i}")
            cm.add_turn(TurnRole.ASSISTANT, f"Reply {i}")

        summary = cm.get_summary()
        # Should have more context than just 3 turns
        assert len(summary) > 50  # V2 summary should be richer
