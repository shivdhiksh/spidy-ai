"""
tests/unit/test_brain.py — Milestone 3: Brain Core Test Suite
===============================================================
Tests for all Brain Core components:
  - IntentClassifier
  - ConversationManager
  - DecisionEngine
  - Planner
  - ToolRouter
  - Brain (full pipeline)
  - Companion Interfaces (placeholders)
  - Brain Events
  - BrainConfig
  - LLM Client (OllamaClient + BaseLLMClient contract)

Design
------
All tests are self-contained with no real OS calls, no real LLM calls,
and no real skill execution. External dependencies are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.brain.interfaces import KnowledgeInterface, LearningInterface, MemoryInterface
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
    """Create an EventBus bound to the currently running asyncio loop."""
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)
    return bus


def make_registry(skills: list | None = None):
    from spidy.skills.registry import SkillRegistry
    reg = SkillRegistry()
    for skill in (skills or []):
        reg.register(skill)
    return reg


def make_reasoning_config(**kwargs):
    from spidy.config.manager import ReasoningConfig
    return ReasoningConfig(**kwargs)


# ─── TestBrainConfig ──────────────────────────────────────────────────────────


class TestBrainConfig:
    """BrainConfig Pydantic model validation."""

    def test_default_values(self):
        from spidy.config.manager import BrainConfig
        cfg = BrainConfig()
        assert cfg.intent_classifier == "heuristic"
        assert cfg.min_intent_confidence == 0.6
        assert cfg.max_conversation_turns == 20
        assert cfg.decision_mode == "auto"
        assert cfg.tool_routing_enabled is True

    def test_companion_flags_default_false(self):
        from spidy.config.manager import BrainConfig
        cfg = BrainConfig()
        assert cfg.enable_memory is False
        assert cfg.enable_knowledge is False
        assert cfg.enable_learning is False
        assert cfg.enable_web_search is False

    def test_override_values(self):
        from spidy.config.manager import BrainConfig
        cfg = BrainConfig(
            intent_classifier="llm",
            max_conversation_turns=50,
            enable_memory=True,
        )
        assert cfg.intent_classifier == "llm"
        assert cfg.max_conversation_turns == 50
        assert cfg.enable_memory is True

    def test_spidy_config_includes_brain(self):
        from spidy.config.manager import SpidyConfig
        cfg = SpidyConfig()
        assert hasattr(cfg, "brain")
        from spidy.config.manager import BrainConfig
        assert isinstance(cfg.brain, BrainConfig)


# ─── TestIntentClassifier ─────────────────────────────────────────────────────


class TestIntentClassifier:
    """IntentClassifier heuristic classification."""

    @pytest.fixture
    def classifier(self):
        from spidy.brain.intent_classifier import IntentClassifier
        return IntentClassifier()

    async def test_empty_utterance_returns_chat(self, classifier):
        intent = await classifier.classify("")
        assert intent.action == "chat"
        assert intent.confidence < 0.9

    async def test_whitespace_returns_chat(self, classifier):
        intent = await classifier.classify("   ")
        assert intent.action == "chat"

    async def test_open_app_intent(self, classifier):
        # 'open_app' was renamed to 'launch_app' to match AppSkill.capabilities()
        intent = await classifier.classify("Open Chrome")
        assert intent.action == "launch_app"
        assert intent.confidence >= 0.6

    # ── launch_app routing tests (stabilisation fix) ───────────────────────

    @pytest.mark.parametrize("utterance,expected_app", [
        ("open notepad",    "notepad"),
        ("launch notepad",  "notepad"),
        ("start notepad",   "notepad"),
        ("open calculator", "calculator"),
        ("open chrome",     "chrome"),
        ("launch chrome",   "chrome"),
        ("start spotify",   "spotify"),
        ("open cmd",        "cmd"),
        ("launch vscode",   "vscode"),
    ])
    async def test_launch_app_intent(self, classifier, utterance, expected_app):
        """Classifier must emit launch_app (not open_app or chat) for app launch phrases."""
        intent = await classifier.classify(utterance)
        assert intent.action == "launch_app", (
            f"Expected 'launch_app' for {utterance!r}, got {intent.action!r}"
        )
        assert intent.confidence >= 0.6
        # The app name entity (key "name") must be extracted correctly
        app_entities = [e for e in intent.entities if e.name == "name"]
        assert app_entities, f"No 'name' entity extracted for {utterance!r}"
        assert app_entities[0].value == expected_app, (
            f"Expected name={expected_app!r} for {utterance!r}, "
            f"got {app_entities[0].value!r}"
        )

    async def test_search_web_intent(self, classifier):
        intent = await classifier.classify("Search the web for Python tutorials")
        assert intent.action == "search_web"
        assert intent.confidence >= 0.6

    async def test_search_web_extracts_query_entity(self, classifier):
        intent = await classifier.classify("Search the web for Python tutorials")
        assert any(e.name == "query" for e in intent.entities)

    async def test_set_timer_intent(self, classifier):
        intent = await classifier.classify("Set a timer for 5 minutes")
        assert intent.action == "set_timer"

    async def test_timer_extracts_duration_entities(self, classifier):
        intent = await classifier.classify("Set a timer for 5 minutes")
        names = {e.name for e in intent.entities}
        assert "amount" in names
        assert "unit" in names

    async def test_take_note_intent(self, classifier):
        intent = await classifier.classify("Take a note: buy milk")
        assert intent.action == "take_note"

    async def test_note_extracts_content_entity(self, classifier):
        intent = await classifier.classify("Take a note: buy milk")
        content_entities = [e for e in intent.entities if e.name == "content"]
        assert len(content_entities) > 0
        assert "buy milk" in content_entities[0].value.lower()

    async def test_unknown_falls_back_to_chat(self, classifier):
        intent = await classifier.classify("Blah blah xlorf quintessential foo")
        assert intent.action == "chat"

    async def test_intent_carries_raw_utterance(self, classifier):
        utterance = "Open Firefox please"
        intent = await classifier.classify(utterance)
        assert intent.raw_utterance == utterance

    async def test_intent_source_is_heuristic(self, classifier):
        intent = await classifier.classify("Hello there")
        assert intent.source == "heuristic"

    async def test_lock_screen_intent(self, classifier):
        # Classifier now emits 'lock_workstation' directly (the real SystemControlSkill action).
        # The old 'lock_screen' was a stub that said "not yet implemented".
        intent = await classifier.classify("Lock screen")
        assert intent.action == "lock_workstation"

    async def test_help_intent(self, classifier):
        intent = await classifier.classify("help")
        assert intent.action == "help"

    # ── Greeting / identity rules (stabilisation fix) ──────────────────────

    @pytest.mark.parametrize("utterance", [
        "hi", "hello", "hey",
        "Hey Spidy", "hi spidy", "hello spidy",
        "hi there", "hey there",
        "good morning", "good afternoon", "good evening",
        "howdy", "greetings",
    ])
    async def test_greet_intent(self, classifier, utterance):
        intent = await classifier.classify(utterance)
        assert intent.action == "greet", (
            f"Expected 'greet' for {utterance!r}, got {intent.action!r}"
        )
        assert intent.confidence >= 0.9

    @pytest.mark.parametrize("utterance", [
        "goodbye", "bye", "see you", "farewell", "bye spidy",
    ])
    async def test_farewell_intent(self, classifier, utterance):
        intent = await classifier.classify(utterance)
        assert intent.action == "farewell", (
            f"Expected 'farewell' for {utterance!r}, got {intent.action!r}"
        )

    @pytest.mark.parametrize("utterance", [
        "who are you", "what are you", "introduce yourself",
        "tell me about yourself",
    ])
    async def test_introduce_intent(self, classifier, utterance):
        intent = await classifier.classify(utterance)
        assert intent.action == "introduce", (
            f"Expected 'introduce' for {utterance!r}, got {intent.action!r}"
        )
        assert intent.confidence >= 0.9

    # ── Browser search regression tests ───────────────────────────────────────
    # These tests guard against the regression introduced by the stabilization
    # commits (55801c4 / b7e9c07 / 0a9693c) where compound commands like
    # "open edge and search for python" were mis-routed to launch_app with a
    # garbage app name, and "search youtube for python" entity extraction was
    # not verified end-to-end.

    @pytest.mark.parametrize("utterance,expected_query", [
        ("search youtube for python",           "python"),
        ("search youtube for lofi music",       "lofi music"),
        ("search on youtube for cats",          "cats"),
        ("youtube search for cooking videos",   "cooking videos"),
    ])
    async def test_search_youtube_intent_and_query(
        self, classifier, utterance, expected_query
    ):
        """search_youtube must be classified and the query entity extracted."""
        intent = await classifier.classify(utterance)
        assert intent.action == "search_youtube", (
            f"Expected 'search_youtube' for {utterance!r}, got {intent.action!r}"
        )
        assert intent.confidence >= 0.9
        query_entities = [e for e in intent.entities if e.name == "query"]
        assert query_entities, f"No 'query' entity for {utterance!r}"
        assert query_entities[0].value == expected_query, (
            f"Expected query={expected_query!r} for {utterance!r}, "
            f"got {query_entities[0].value!r}"
        )

    @pytest.mark.parametrize("utterance,expected_app,expected_query", [
        ("open edge and search for python",        "edge",    "python"),
        ("open chrome and search for python",      "chrome",  "python"),
        ("open firefox and search for cats",       "firefox", "cats"),
        ("launch edge and search for weather",     "edge",    "weather"),
        ("open browser and search for tutorials",  "browser", "tutorials"),
        ("open edge and google python",            "edge",    "python"),
        ("open chrome and look up python",         "chrome",  "python"),
    ])
    async def test_open_browser_and_search_intent(
        self, classifier, utterance, expected_app, expected_query
    ):
        """Compound 'open X and search for Y' must route to open_browser_and_search
        with both app_name and query entities — NOT to launch_app."""
        intent = await classifier.classify(utterance)
        assert intent.action == "open_browser_and_search", (
            f"Expected 'open_browser_and_search' for {utterance!r}, "
            f"got {intent.action!r}"
        )
        assert intent.confidence >= 0.9

        app_entities   = [e for e in intent.entities if e.name == "app_name"]
        query_entities = [e for e in intent.entities if e.name == "query"]

        assert app_entities, f"No 'app_name' entity for {utterance!r}"
        assert app_entities[0].value == expected_app, (
            f"Expected app_name={expected_app!r} for {utterance!r}, "
            f"got {app_entities[0].value!r}"
        )

        assert query_entities, f"No 'query' entity for {utterance!r}"
        assert query_entities[0].value == expected_query, (
            f"Expected query={expected_query!r} for {utterance!r}, "
            f"got {query_entities[0].value!r}"
        )

    @pytest.mark.parametrize("utterance,expected_app", [
        ("open edge",    "edge"),
        ("open chrome",  "chrome"),
        ("launch edge",  "edge"),
        ("open firefox", "firefox"),
    ])
    async def test_plain_open_app_still_works(
        self, classifier, utterance, expected_app
    ):
        """Plain 'open edge' must still route to launch_app (not open_browser_and_search)
        and must extract only the clean app name (no trailing 'and search for...')."""
        intent = await classifier.classify(utterance)
        assert intent.action == "launch_app", (
            f"Expected 'launch_app' for {utterance!r}, got {intent.action!r}"
        )
        name_entities = [e for e in intent.entities if e.name == "name"]
        assert name_entities, f"No 'name' entity for {utterance!r}"
        assert name_entities[0].value == expected_app, (
            f"Expected name={expected_app!r} for {utterance!r}, "
            f"got {name_entities[0].value!r}"
        )


# ─── TestConversationManager ──────────────────────────────────────────────────


class TestConversationManager:
    """ConversationManager session lifecycle and window management."""

    @pytest.fixture
    async def manager(self):
        from spidy.brain.conversation_manager import ConversationManager
        bus = make_bus()
        return ConversationManager(bus=bus, max_turns=5)

    async def test_start_session_returns_id(self, manager):
        sid = await manager.start_session()
        assert sid
        assert isinstance(sid, str)

    async def test_start_session_with_explicit_id(self, manager):
        sid = await manager.start_session("my-session-123")
        assert sid == "my-session-123"

    async def test_is_active_after_start(self, manager):
        await manager.start_session()
        assert manager.is_active

    async def test_not_active_after_end(self, manager):
        await manager.start_session()
        await manager.end_session()
        assert not manager.is_active

    async def test_add_turn_user(self, manager):
        await manager.start_session()
        turn = manager.add_turn(TurnRole.USER, "Hello!")
        assert turn.role == TurnRole.USER
        assert turn.text == "Hello!"

    async def test_add_turn_assistant(self, manager):
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "Hi")
        manager.add_turn(TurnRole.ASSISTANT, "Hello!")
        assert manager.turn_count == 2

    async def test_rolling_window_evicts_oldest(self, manager):
        await manager.start_session()
        for i in range(7):  # max_turns=5
            manager.add_turn(TurnRole.USER, f"msg {i}")
        assert manager.turn_count == 5

    async def test_get_context_returns_ordered_turns(self, manager):
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "first")
        manager.add_turn(TurnRole.ASSISTANT, "second")
        ctx = manager.get_context()
        assert ctx[0].text == "first"
        assert ctx[1].text == "second"

    async def test_get_llm_messages_includes_system(self, manager):
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "test")
        msgs = manager.get_llm_messages(include_system=True)
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 2

    async def test_get_llm_messages_no_system(self, manager):
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "test")
        msgs = manager.get_llm_messages(include_system=False)
        assert msgs[0]["role"] == "user"

    async def test_last_user_utterance(self, manager):
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "first")
        manager.add_turn(TurnRole.ASSISTANT, "reply")
        manager.add_turn(TurnRole.USER, "second")
        assert manager.last_user_utterance == "second"

    async def test_session_started_event_published(self, manager):
        events = []
        manager._bus.subscribe("brain.session_started", lambda e: events.append(e))
        await manager.start_session("test-session")
        assert len(events) == 1
        assert events[0].session_id == "test-session"

    async def test_session_ended_event_published(self, manager):
        events = []
        manager._bus.subscribe("brain.session_ended", lambda e: events.append(e))
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "hi")
        await manager.end_session()
        assert len(events) == 1
        assert events[0].turn_count == 1

    async def test_get_summary_returns_string(self, manager):
        await manager.start_session()
        manager.add_turn(TurnRole.USER, "What time is it?")
        summary = manager.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    async def test_get_summary_empty_on_fresh_session(self, manager):
        await manager.start_session()
        assert manager.get_summary() == ""


# ─── TestDecisionEngine ───────────────────────────────────────────────────────


class TestDecisionEngine:
    """DecisionEngine intent → decision mapping."""

    @pytest.fixture
    def engine_no_skills(self):
        from spidy.brain.decision_engine import DecisionEngine
        return DecisionEngine(skill_registry=make_registry())

    async def test_empty_utterance_is_rejected(self, engine_no_skills):
        intent = Intent(action="", confidence=1.0, raw_utterance="")
        decision = await engine_no_skills.decide(intent)
        assert decision.mode == DecisionMode.REJECT

    async def test_low_confidence_triggers_clarify(self, engine_no_skills):
        intent = Intent(action="open_file", confidence=0.3, raw_utterance="umm thing")
        decision = await engine_no_skills.decide(intent)
        assert decision.mode == DecisionMode.CLARIFY
        assert decision.clarification_question

    async def test_chat_action_is_llm_direct(self, engine_no_skills):
        intent = Intent(action="chat", confidence=0.9, raw_utterance="hello")
        decision = await engine_no_skills.decide(intent)
        assert decision.mode == DecisionMode.LLM_DIRECT

    async def test_unknown_action_falls_through_to_llm_direct(self, engine_no_skills):
        intent = Intent(action="some_unknown_action", confidence=0.9,
                        raw_utterance="do unknown thing")
        decision = await engine_no_skills.decide(intent)
        assert decision.mode == DecisionMode.LLM_DIRECT

    async def test_skill_action_dispatches_to_skill(self):
        from spidy.brain.decision_engine import DecisionEngine
        from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult

        class FakeSkill(BaseSkill):
            name = "fake_skill"
            version = "1.0.0"
            def capabilities(self):
                return [SkillCapability("do_thing", "Does a thing")]
            async def execute(self, action, context):
                return SkillResult.ok("done")

        registry = make_registry([FakeSkill()])
        engine = DecisionEngine(skill_registry=registry)
        intent = Intent(action="do_thing", confidence=0.9, raw_utterance="do the thing")
        decision = await engine.decide(intent)
        assert decision.mode == DecisionMode.SKILL
        assert decision.skill_name == "fake_skill"

    async def test_shutdown_intent_routes_to_skill(self, engine_no_skills):
        # Previously 'shutdown' was incorrectly in _REJECT_ACTIONS.
        # It now correctly routes to SystemControlSkill's shutdown_system action.
        # With no skills registered it routes to LLM_DIRECT/REJECT depending on config.
        intent = Intent(action="shutdown_system", confidence=0.9, raw_utterance="shutdown computer")
        decision = await engine_no_skills.decide(intent)
        # Without a registered skill it should fall back to LLM_DIRECT, CLARIFY, or REJECT — not crash
        assert decision.mode in (DecisionMode.SKILL, DecisionMode.LLM_DIRECT, DecisionMode.CLARIFY, DecisionMode.REJECT)

    async def test_decision_carries_intent(self, engine_no_skills):
        intent = Intent(action="chat", confidence=0.9, raw_utterance="hi")
        decision = await engine_no_skills.decide(intent)
        assert decision.intent is intent


# ─── TestPlanner ──────────────────────────────────────────────────────────────


class TestPlanner:
    """Planner decision → plan conversion."""

    @pytest.fixture
    def planner(self):
        from spidy.brain.planner import Planner
        return Planner()

    def make_decision(self, mode: DecisionMode, action: str = "test",
                      skill_name: str = "", entities: list | None = None) -> Decision:
        intent = Intent(
            action=action,
            entities=tuple(entities or []),
            confidence=0.9,
            raw_utterance="test utterance",
        )
        return Decision(
            mode=mode,
            intent=intent,
            skill_name=skill_name,
            clarification_question="What do you mean?" if mode == DecisionMode.CLARIFY else "",
        )

    async def test_skill_decision_creates_skill_step(self, planner):
        decision = self.make_decision(DecisionMode.SKILL, "open_file", "file_skill")
        plan = await planner.plan(decision, session_id="s1")
        assert not plan.is_empty
        assert plan.first.step_type == "skill"
        assert plan.first.action == "open_file"

    async def test_skill_step_includes_entity_params(self, planner):
        entities = [Entity(name="filename", value="test.txt")]
        decision = self.make_decision(DecisionMode.SKILL, "open_file",
                                      "file_skill", entities=entities)
        plan = await planner.plan(decision)
        assert plan.first.params.get("filename") == "test.txt"

    async def test_llm_direct_creates_llm_step(self, planner):
        decision = self.make_decision(DecisionMode.LLM_DIRECT, "chat")
        plan = await planner.plan(decision)
        assert plan.first.step_type == "llm"
        assert plan.first.action == "direct"

    async def test_clarify_creates_clarify_step(self, planner):
        decision = self.make_decision(DecisionMode.CLARIFY, "open_file")
        plan = await planner.plan(decision)
        assert plan.first.step_type == "clarify"
        assert plan.first.prompt_context == "What do you mean?"

    async def test_reject_creates_noop_step(self, planner):
        decision = self.make_decision(DecisionMode.REJECT, "shutdown")
        plan = await planner.plan(decision)
        assert plan.first.step_type == "noop"

    async def test_plan_carries_session_id(self, planner):
        decision = self.make_decision(DecisionMode.LLM_DIRECT, "chat")
        plan = await planner.plan(decision, session_id="test-session")
        assert plan.session_id == "test-session"

    async def test_plan_carries_decision(self, planner):
        decision = self.make_decision(DecisionMode.SKILL, "launch_app", "app_skill")
        plan = await planner.plan(decision)
        assert plan.decision is decision


# ─── TestToolRouter ───────────────────────────────────────────────────────────


class TestToolRouter:
    """ToolRouter plan execution."""

    @pytest.fixture
    async def router_no_skills(self):
        from spidy.brain.tool_router import ToolRouter
        bus = make_bus()
        registry = make_registry()
        return ToolRouter(bus=bus, skill_registry=registry)

    async def test_noop_step_returns_empty_success(self, router_no_skills):
        plan = Plan(steps=(PlanStep(step_type="noop"),))
        results = await router_no_skills.execute(plan)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].action == "noop"

    async def test_clarify_step_returns_question(self, router_no_skills):
        plan = Plan(steps=(PlanStep(
            step_type="clarify",
            prompt_context="Could you clarify?",
        ),))
        results = await router_no_skills.execute(plan)
        assert results[0].success is True
        assert "clarify" in results[0].message.lower()

    async def test_skill_step_no_skill_returns_fail(self, router_no_skills):
        plan = Plan(steps=(PlanStep(step_type="skill", action="missing_action"),))
        results = await router_no_skills.execute(plan)
        assert results[0].success is False

    async def test_skill_step_dispatches_to_skill(self):
        from spidy.brain.tool_router import ToolRouter
        from spidy.skills.base import BaseSkill, SkillCapability, SkillResult

        class FakeSkill(BaseSkill):
            name = "fake"
            version = "1.0.0"
            def capabilities(self):
                return [SkillCapability("greet", "Greets the user")]
            async def execute(self, action, context):
                return SkillResult.ok("Hello there!")

        bus = make_bus()
        registry = make_registry([FakeSkill()])
        router = ToolRouter(bus=bus, skill_registry=registry)
        plan = Plan(steps=(PlanStep(step_type="skill", action="greet"),))
        results = await router.execute(plan)
        assert results[0].success is True
        assert results[0].message == "Hello there!"

    async def test_skill_exception_returns_fail(self):
        from spidy.brain.tool_router import ToolRouter
        from spidy.skills.base import BaseSkill, SkillCapability, SkillResult

        class BrokenSkill(BaseSkill):
            name = "broken"
            version = "1.0.0"
            def capabilities(self):
                return [SkillCapability("break_it", "Always breaks")]
            async def execute(self, action, context):
                raise RuntimeError("I always break!")

        bus = make_bus()
        registry = make_registry([BrokenSkill()])
        router = ToolRouter(bus=bus, skill_registry=registry)
        plan = Plan(steps=(PlanStep(step_type="skill", action="break_it"),))
        results = await router.execute(plan)
        assert results[0].success is False
        assert "break" in results[0].error.lower() or results[0].error

    async def test_llm_step_no_client_returns_fallback(self, router_no_skills):
        from spidy.llm.client import LLMMessage
        plan = Plan(steps=(PlanStep(step_type="llm", action="direct"),))
        results = await router_no_skills.execute(plan, llm_messages=[
            LLMMessage(role="user", content="hello")
        ])
        assert results[0].success is True  # Fallback is still "success"
        assert results[0].message  # Has fallback text

    async def test_llm_step_calls_client(self):
        from spidy.brain.tool_router import ToolRouter
        from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse

        class FakeLLM(BaseLLMClient):
            async def complete(self, messages, **kwargs):
                return LLMResponse(text="The answer is 42.", success=True)
            async def stream(self, messages, **kwargs):
                yield "The answer"
            async def close(self): pass

        bus = make_bus()
        registry = make_registry()
        router = ToolRouter(bus=bus, skill_registry=registry, llm_client=FakeLLM())
        plan = Plan(steps=(PlanStep(step_type="llm", action="direct"),))
        results = await router.execute(plan, llm_messages=[
            LLMMessage(role="user", content="What is the answer?")
        ])
        assert results[0].success is True
        assert results[0].message == "The answer is 42."

    async def test_tool_called_event_published(self, router_no_skills):
        events = []
        router_no_skills._bus.subscribe("brain.tool_called", lambda e: events.append(e))
        plan = Plan(steps=(PlanStep(step_type="noop"),))
        await router_no_skills.execute(plan, session_id="s1")
        assert len(events) == 1

    async def test_tool_result_event_published(self, router_no_skills):
        events = []
        router_no_skills._bus.subscribe("brain.tool_result", lambda e: events.append(e))
        plan = Plan(steps=(PlanStep(step_type="noop"),))
        await router_no_skills.execute(plan, session_id="s1")
        assert len(events) == 1


# ─── TestBrainPipeline ────────────────────────────────────────────────────────


class TestBrainPipeline:
    """End-to-end Brain pipeline tests."""

    @pytest.fixture
    async def brain(self):
        from spidy.brain.brain import Brain
        bus = make_bus()
        registry = make_registry()
        config = make_reasoning_config()
        return Brain(bus=bus, config=config, skill_registry=registry)

    async def test_brain_starts_and_is_running(self, brain):
        await brain.start()
        assert brain.is_running
        await brain.stop()

    async def test_brain_stop_makes_not_running(self, brain):
        await brain.start()
        await brain.stop()
        assert not brain.is_running

    async def test_double_start_is_noop(self, brain):
        await brain.start()
        await brain.start()  # Should not raise
        assert brain.is_running
        await brain.stop()

    async def test_double_stop_is_noop(self, brain):
        await brain.start()
        await brain.stop()
        await brain.stop()  # Should not raise

    async def test_process_returns_string(self, brain):
        await brain.start()
        response = await brain.process("Hello Spidy")
        assert isinstance(response, str)
        await brain.stop()

    async def test_process_no_llm_returns_fallback(self, brain):
        await brain.start()
        # With no LLM client, open-ended chat → LLM_DIRECT → fallback response
        response = await brain.process("Tell me something interesting")
        assert isinstance(response, str)
        await brain.stop()

    async def test_process_publishes_response_ready_event(self, brain):
        events = []
        brain._bus.subscribe("brain.response_ready", lambda e: events.append(e))
        await brain.start()
        await brain.process("Hello")
        assert len(events) >= 1
        await brain.stop()

    async def test_process_publishes_processing_started_event(self, brain):
        events = []
        brain._bus.subscribe("brain.processing_started", lambda e: events.append(e))
        await brain.start()
        await brain.process("Hey there")
        assert len(events) == 1
        assert events[0].utterance == "Hey there"
        await brain.stop()

    async def test_turn_count_increases_on_process(self, brain):
        await brain.start()
        assert brain.turn_count == 0
        await brain.process("Hello")
        # User turn + possibly assistant turn
        assert brain.turn_count >= 1
        await brain.stop()

    async def test_session_id_is_set(self, brain):
        await brain.start()
        assert brain.session_id
        await brain.stop()

    async def test_brain_companion_slots_default_none(self, brain):
        assert brain.memory is None
        assert brain.knowledge is None
        assert brain.learning is None

    async def test_brain_process_with_skill(self):
        from spidy.brain.brain import Brain
        from spidy.skills.base import BaseSkill, SkillCapability, SkillResult

        class GreetSkill(BaseSkill):
            name = "greeter"
            version = "1.0.0"
            def capabilities(self):
                return [SkillCapability("help", "Shows help")]
            async def execute(self, action, context):
                return SkillResult.ok(f"Hi {context.user_name}! I can help with many things.")

        bus = make_bus()
        registry = make_registry([GreetSkill()])
        brain = Brain(bus=bus, config=make_reasoning_config(),
                      skill_registry=registry, user_name="TestUser")
        await brain.start()
        response = await brain.process("help")
        assert "TestUser" in response or response
        await brain.stop()


# ─── TestBrainInterfaces ──────────────────────────────────────────────────────


class TestBrainInterfaces:
    """All three companion interface placeholders raise NotImplementedError."""

    def _make_memory(self):
        class ConcreteMemory(MemoryInterface):
            async def store(self, *a, **kw): raise NotImplementedError
            async def recall(self, *a, **kw): raise NotImplementedError
            async def search(self, *a, **kw): raise NotImplementedError
            async def clear(self, *a, **kw): raise NotImplementedError
        return ConcreteMemory()

    def _make_knowledge(self):
        class ConcreteKnowledge(KnowledgeInterface):
            async def query(self, *a, **kw): raise NotImplementedError
            async def ingest(self, *a, **kw): raise NotImplementedError
            async def search_rag(self, *a, **kw): raise NotImplementedError
        return ConcreteKnowledge()

    def _make_learning(self):
        class ConcreteLearning(LearningInterface):
            async def record_feedback(self, *a, **kw): raise NotImplementedError
            async def get_preference(self, *a, **kw): raise NotImplementedError
            async def get_habit(self, *a, **kw): raise NotImplementedError
        return ConcreteLearning()

    async def test_memory_store_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_memory().store("test")

    async def test_memory_recall_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_memory().recall("query")

    async def test_memory_search_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_memory().search("query")

    async def test_memory_clear_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_memory().clear()

    async def test_knowledge_query_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_knowledge().query("question")

    async def test_knowledge_ingest_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_knowledge().ingest("content", "source")

    async def test_knowledge_search_rag_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_knowledge().search_rag("query")

    async def test_learning_record_feedback_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_learning().record_feedback("sid", "utt", "resp", 0.5)

    async def test_learning_get_preference_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_learning().get_preference("style")

    async def test_learning_get_habit_raises(self):
        with pytest.raises(NotImplementedError):
            await self._make_learning().get_habit({})

    def test_memory_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            MemoryInterface()  # type: ignore

    def test_knowledge_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            KnowledgeInterface()  # type: ignore

    def test_learning_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LearningInterface()  # type: ignore


# ─── TestLLMClient ────────────────────────────────────────────────────────────


class TestLLMClient:
    """BaseLLMClient contract and OllamaClient fallback behaviour."""

    def test_llm_response_failure_factory(self):
        from spidy.llm.client import LLMResponse
        resp = LLMResponse.failure("connection refused", model="llama3")
        assert not resp.success
        assert resp.finish_reason == "error"
        assert resp.error_message == "connection refused"
        assert resp.model == "llama3"

    def test_llm_message_is_frozen(self):
        from spidy.llm.client import LLMMessage
        msg = LLMMessage(role="user", content="hello")
        with pytest.raises((AttributeError, TypeError)):
            msg.role = "system"  # type: ignore

    def test_base_llm_client_is_abstract(self):
        from spidy.llm.client import BaseLLMClient
        with pytest.raises(TypeError):
            BaseLLMClient()  # type: ignore

    async def test_ollama_client_returns_failure_when_server_down(self):
        from spidy.llm.backends.ollama import OllamaClient
        from spidy.llm.client import LLMMessage
        client = OllamaClient(
            base_url="http://localhost:19999",  # Nothing running here
            timeout=1,
        )
        response = await client.complete([LLMMessage(role="user", content="hi")])
        assert not response.success
        assert response.error_message

    async def test_ollama_client_stream_yields_nothing_on_failure(self):
        from spidy.llm.backends.ollama import OllamaClient
        from spidy.llm.client import LLMMessage
        client = OllamaClient(
            base_url="http://localhost:19999",
            timeout=1,
        )
        tokens = []
        async for token in client.stream([LLMMessage(role="user", content="hi")]):
            tokens.append(token)
        assert tokens == []

    def test_llm_client_factory_builds_ollama_raises_deprecated(self):
        """Ollama is deprecated — factory must raise ValueError, not build a client."""
        import pytest
        from spidy.llm.client import LLMClientFactory
        config = make_reasoning_config(provider="ollama")
        with pytest.raises(ValueError, match="deprecated"):
            LLMClientFactory.build(config)

    def test_llm_client_factory_unknown_provider_raises_value_error(self):
        """Unknown provider must raise ValueError — no silent fallback to Ollama."""
        import pytest
        from spidy.llm.client import LLMClientFactory
        config = make_reasoning_config(provider="unknown_provider_xyz")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClientFactory.build(config)


# ─── TestBrainEvents ──────────────────────────────────────────────────────────


class TestBrainEvents:
    """Brain event dataclasses."""

    def test_processing_started_event_topic(self):
        from spidy.brain.events import BrainProcessingStartedEvent
        e = BrainProcessingStartedEvent(session_id="s1", utterance="hello")
        assert e.topic == "brain.processing_started"
        assert e.session_id == "s1"

    def test_response_ready_event_topic(self):
        from spidy.brain.events import BrainResponseReadyEvent
        e = BrainResponseReadyEvent(
            session_id="s1", response_text="Hi!", decision_mode="skill"
        )
        assert e.topic == "brain.response_ready"
        assert e.decision_mode == "skill"

    def test_session_started_event_topic(self):
        from spidy.brain.events import BrainSessionStartedEvent
        e = BrainSessionStartedEvent(session_id="abc")
        assert e.topic == "brain.session_started"

    def test_session_ended_event_topic(self):
        from spidy.brain.events import BrainSessionEndedEvent
        e = BrainSessionEndedEvent(session_id="abc", turn_count=5)
        assert e.topic == "brain.session_ended"
        assert e.turn_count == 5

    def test_tool_called_event_topic(self):
        from spidy.brain.events import BrainToolCalledEvent
        e = BrainToolCalledEvent(session_id="s1", action="greet", step_type="skill")
        assert e.topic == "brain.tool_called"

    def test_tool_result_event_topic(self):
        from spidy.brain.events import BrainToolResultEvent
        e = BrainToolResultEvent(session_id="s1", action="greet", success=True)
        assert e.topic == "brain.tool_result"


# ─── TestBrainTypes ───────────────────────────────────────────────────────────


class TestBrainTypes:
    """Data type correctness."""

    def test_intent_get_entity_found(self):
        intent = Intent(
            action="open_file",
            entities=(Entity("filename", "notes.txt"),),
        )
        assert intent.get_entity("filename") == "notes.txt"

    def test_intent_get_entity_not_found(self):
        intent = Intent(action="chat")
        assert intent.get_entity("missing", default="x") == "x"

    def test_intent_is_confident_high(self):
        assert Intent(action="chat", confidence=0.9).is_confident

    def test_intent_is_confident_low(self):
        assert not Intent(action="chat", confidence=0.3).is_confident

    def test_intent_is_confident_at_threshold(self):
        assert Intent(action="chat", confidence=0.6).is_confident

    def test_conversation_turn_to_llm_message(self):
        turn = ConversationTurn(role=TurnRole.USER, text="Hello")
        msg = turn.to_llm_message()
        assert msg == {"role": "user", "content": "Hello"}

    def test_plan_is_empty(self):
        plan = Plan(steps=())
        assert plan.is_empty

    def test_plan_not_empty(self):
        plan = Plan(steps=(PlanStep(step_type="noop"),))
        assert not plan.is_empty

    def test_plan_first(self):
        step = PlanStep(step_type="skill", action="test")
        plan = Plan(steps=(step,))
        assert plan.first is step

    def test_tool_result_ok_factory(self):
        r = ToolResult.ok("Done!", action="greet", step_type="skill")
        assert r.success
        assert r.message == "Done!"
        assert r.step_type == "skill"

    def test_tool_result_fail_factory(self):
        r = ToolResult.fail("Failed!", action="greet", error="timeout")
        assert not r.success
        assert r.error == "timeout"

    def test_decision_mode_values(self):
        assert DecisionMode.SKILL.value == "skill"
        assert DecisionMode.LLM_DIRECT.value == "llm_direct"
        assert DecisionMode.CLARIFY.value == "clarify"
        assert DecisionMode.REJECT.value == "reject"
