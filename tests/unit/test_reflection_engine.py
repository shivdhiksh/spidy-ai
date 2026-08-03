"""
tests/unit/test_reflection_engine.py — Unit tests for ReflectionEngine
=======================================================================
Tests all reflection decision paths: CONTINUE, RETRY, ALTERNATIVE,
ASK_USER, ABORT, and the alternative utterance builder.
"""

from __future__ import annotations

import pytest

from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.types import ReflectionDecision, TaskRecord, TaskState


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_engine(max_retries: int = 2) -> ReflectionEngine:
    return ReflectionEngine(max_retries=max_retries)


def make_task(attempt: int = 1, extra: dict | None = None) -> TaskRecord:
    return TaskRecord(
        description="Test task",
        utterance="do something",
        state=TaskState.RUNNING,
        attempt=attempt,
        extra=extra or {},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUE decisions
# ═══════════════════════════════════════════════════════════════════════════════


class TestContinueDecisions:
    def test_clear_success_response_returns_continue(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, reason = engine.reflect(task, "I've opened VS Code for you.", True)
        assert decision == ReflectionDecision.CONTINUE

    def test_done_in_response_returns_continue(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "Done! All set.", True)
        assert decision == ReflectionDecision.CONTINUE

    def test_installed_in_response_returns_continue(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "I've installed Flask successfully.", True)
        assert decision == ReflectionDecision.CONTINUE

    def test_searching_returns_continue(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "Searching for Python tutorials.", True)
        assert decision == ReflectionDecision.CONTINUE

    def test_long_successful_response_returns_continue(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        # Even without explicit keyword, a long successful response = CONTINUE
        decision, _ = engine.reflect(task, "Here is a detailed response that is long enough.", True)
        assert decision == ReflectionDecision.CONTINUE

    def test_all_set_returns_continue(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "All set! Ready to go.", True)
        assert decision == ReflectionDecision.CONTINUE


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY decisions
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetryDecisions:
    def test_failure_with_retries_left_returns_retry(self) -> None:
        engine = make_engine(max_retries=2)
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "I couldn't open VS Code.", False)
        assert decision == ReflectionDecision.RETRY

    def test_timeout_returns_retry(self) -> None:
        engine = make_engine(max_retries=2)
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "The operation timed out.", False)
        assert decision == ReflectionDecision.RETRY

    def test_empty_response_with_retries_left_returns_retry(self) -> None:
        engine = make_engine(max_retries=2)
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "", False)
        assert decision == ReflectionDecision.RETRY

    def test_network_error_returns_retry(self) -> None:
        engine = make_engine(max_retries=2)
        task = make_task(attempt=1)
        decision, _ = engine.reflect(task, "Network connection temporarily unavailable.", False)
        assert decision == ReflectionDecision.RETRY


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE decisions
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlternativeDecisions:
    def test_exhausted_retries_returns_alternative(self) -> None:
        engine = make_engine(max_retries=2)
        # attempt=3 means all retries used
        task = make_task(attempt=3)
        decision, _ = engine.reflect(task, "I couldn't do that.", False)
        assert decision == ReflectionDecision.ALTERNATIVE

    def test_no_alternative_tried_yet_gives_alternative(self) -> None:
        engine = make_engine(max_retries=1)
        task = make_task(attempt=2)  # retries exhausted
        decision, _ = engine.reflect(task, "Failed.", False)
        assert decision == ReflectionDecision.ALTERNATIVE


# ═══════════════════════════════════════════════════════════════════════════════
# ABORT decisions
# ═══════════════════════════════════════════════════════════════════════════════


class TestAbortDecisions:
    def test_exhausted_retries_and_alternative_tried_returns_abort(self) -> None:
        engine = make_engine(max_retries=1)
        task = make_task(attempt=2, extra={"alternative_tried": True})
        decision, _ = engine.reflect(task, "Still failing.", False)
        assert decision == ReflectionDecision.ABORT


# ═══════════════════════════════════════════════════════════════════════════════
# ASK_USER decisions
# ═══════════════════════════════════════════════════════════════════════════════


class TestAskUserDecisions:
    def test_clarification_in_response_returns_ask_user(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, _ = engine.reflect(
            task,
            "I'm not sure what you mean. Could you clarify?",
            False,
        )
        assert decision == ReflectionDecision.ASK_USER

    def test_rephrase_request_returns_ask_user(self) -> None:
        engine = make_engine()
        task = make_task(attempt=1)
        decision, _ = engine.reflect(
            task,
            "Could you rephrase or be more specific?",
            False,
        )
        assert decision == ReflectionDecision.ASK_USER


# ═══════════════════════════════════════════════════════════════════════════════
# Reason string
# ═══════════════════════════════════════════════════════════════════════════════


class TestReasonString:
    def test_reason_is_non_empty(self) -> None:
        engine = make_engine()
        task = make_task()
        _, reason = engine.reflect(task, "Done!", True)
        assert reason
        assert isinstance(reason, str)

    def test_retry_reason_mentions_attempt(self) -> None:
        engine = make_engine(max_retries=2)
        task = make_task(attempt=1)
        _, reason = engine.reflect(task, "Failed.", False)
        assert "attempt" in reason.lower() or "retry" in reason.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Alternative utterance builder
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlternativeUtteranceBuilder:
    def _make_task_with_utterance(self, utterance: str) -> TaskRecord:
        return TaskRecord(
            description="test",
            utterance=utterance,
            state=TaskState.RUNNING,
            attempt=1,
        )

    def test_terminal_utterance_alternative(self) -> None:
        engine = make_engine()
        task = self._make_task_with_utterance("open a terminal")
        alt = engine.build_alternative_utterance(task)
        assert alt != task.utterance
        assert "command prompt" in alt.lower() or "powershell" in alt.lower()

    def test_vs_code_alternative(self) -> None:
        engine = make_engine()
        task = self._make_task_with_utterance("open vs code")
        alt = engine.build_alternative_utterance(task)
        assert alt != task.utterance
        assert "visual studio" in alt.lower() or "launch" in alt.lower()

    def test_unknown_utterance_gets_generic_prefix(self) -> None:
        engine = make_engine()
        task = self._make_task_with_utterance("do something completely unique xyz")
        alt = engine.build_alternative_utterance(task)
        assert alt.startswith("please try to")

    def test_flask_install_alternative(self) -> None:
        engine = make_engine()
        task = self._make_task_with_utterance("install flask using pip")
        alt = engine.build_alternative_utterance(task)
        assert "command line" in alt.lower() or "pip" in alt.lower()
