"""
Tests for Milestone 4 — SkillExecutor
======================================
Unit tests for permission-gated skill execution, retry logic,
timeout enforcement, and event publishing.
"""

from __future__ import annotations

import asyncio

import pytest

from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult
from spidy.skills.registry import SkillRegistry


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_skill(
    name: str = "fake_skill",
    actions: list[tuple[str, str]] | None = None,
    fail: bool = False,
    fail_times: int = 0,
    slow: bool = False,
) -> BaseSkill:
    """Build an inline BaseSkill for testing."""

    _actions = actions or [("do_thing", "T0")]
    _fail = fail
    _fail_times = fail_times
    _call_count = [0]

    # Use type() so we can set class-level string attributes correctly
    async def _execute(self, action, context):
        _call_count[0] += 1
        if slow:
            await asyncio.sleep(100)
        if _fail or _call_count[0] <= _fail_times:
            return SkillResult.fail(f"{action} failed on attempt {_call_count[0]}")
        return SkillResult.ok(f"{action} succeeded", action_taken=action)

    def _capabilities(self):
        return [
            SkillCapability(action=a, description=f"does {a}", permission_tier=tier)
            for a, tier in _actions
        ]

    FakeSkill = type(
        "FakeSkill",
        (BaseSkill,),
        {
            "name": name,
            "version": "1.0.0",
            "capabilities": _capabilities,
            "execute": _execute,
        },
    )
    instance = FakeSkill()
    instance._call_count = _call_count
    return instance



def _make_registry(*skills) -> SkillRegistry:
    reg = SkillRegistry()
    for skill in skills:
        reg.register(skill)
    return reg


def _make_executor(registry, permission_manager=None, bus=None, max_retries=2, timeout=5.0):
    from spidy.config.manager import ExecutorConfig
    from spidy.execution.executor import SkillExecutor
    config = ExecutorConfig(max_retries=max_retries, skill_timeout_seconds=timeout)
    return SkillExecutor(
        registry=registry,
        permission_manager=permission_manager,
        config=config,
        bus=bus,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Basic Execution
# ──────────────────────────────────────────────────────────────────────────────

class TestSkillExecutorBasic:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        skill = _make_skill("my_skill", [("do_thing", "T0")])
        registry = _make_registry(skill)
        executor = _make_executor(registry)
        result = await executor.execute("do_thing")
        assert result.success
        assert "succeeded" in result.message

    @pytest.mark.asyncio
    async def test_unknown_action_returns_failure(self):
        registry = _make_registry()
        executor = _make_executor(registry)
        result = await executor.execute("nonexistent_action")
        assert not result.success
        assert "No skill" in result.message

    @pytest.mark.asyncio
    async def test_params_passed_to_skill(self):
        class ParamSkill(BaseSkill):
            name = "param_skill"
            version = "1.0.0"

            def capabilities(self):
                return [SkillCapability("greet", "greet", "T0")]

            async def execute(self, action, context):
                name = context.get("name", "unknown")
                return SkillResult.ok(f"Hello {name}")

        registry = _make_registry(ParamSkill())
        executor = _make_executor(registry)
        result = await executor.execute("greet", params={"name": "Alice"})
        assert result.success
        assert "Alice" in result.message

    @pytest.mark.asyncio
    async def test_session_id_passed_to_context(self):
        captured = []

        class SessionSkill(BaseSkill):
            name = "session_skill"
            version = "1.0.0"

            def capabilities(self):
                return [SkillCapability("track", "track", "T0")]

            async def execute(self, action, context):
                captured.append(context.session_id)
                return SkillResult.ok("ok")

        registry = _make_registry(SessionSkill())
        executor = _make_executor(registry)
        await executor.execute("track", session_id="test-sess-123")
        assert captured == ["test-sess-123"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Retry Logic
# ──────────────────────────────────────────────────────────────────────────────

class TestSkillExecutorRetry:
    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self):
        # Fails first 2 times, succeeds on 3rd (attempt index = 2)
        skill = _make_skill("retry_skill", [("do_thing", "T0")], fail_times=2)
        registry = _make_registry(skill)
        executor = _make_executor(registry, max_retries=3, timeout=5.0)
        result = await executor.execute("do_thing")
        assert result.success
        assert skill._call_count[0] == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_failure(self):
        skill = _make_skill("always_fail", [("do_thing", "T0")], fail=True)
        registry = _make_registry(skill)
        executor = _make_executor(registry, max_retries=2)
        result = await executor.execute("do_thing")
        assert not result.success
        assert skill._call_count[0] == 3  # 1 attempt + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        skill = _make_skill("succeed_first", [("do_thing", "T0")])
        registry = _make_registry(skill)
        executor = _make_executor(registry, max_retries=3)
        await executor.execute("do_thing")
        assert skill._call_count[0] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. Timeout
# ──────────────────────────────────────────────────────────────────────────────

class TestSkillExecutorTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self):
        skill = _make_skill("slow_skill", [("slow_action", "T0")], slow=True)
        registry = _make_registry(skill)
        executor = _make_executor(registry, max_retries=0, timeout=0.05)
        result = await executor.execute("slow_action")
        assert not result.success
        assert "timed out" in result.message.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Permission Integration
# ──────────────────────────────────────────────────────────────────────────────

class TestSkillExecutorPermissions:
    @pytest.mark.asyncio
    async def test_t0_action_executes(self):
        from spidy.permissions.manager import PermissionManager
        from spidy.config.manager import PermissionsConfig

        skill = _make_skill("perm_skill", [("safe_action", "T0")])
        registry = _make_registry(skill)
        pm = PermissionManager(PermissionsConfig(audit_log_enabled=False))
        executor = _make_executor(registry, permission_manager=pm)
        result = await executor.execute("safe_action", tier="T0")
        assert result.success

    @pytest.mark.asyncio
    async def test_t2_action_blocked_by_permission_manager(self):
        from spidy.permissions.manager import PermissionManager
        from spidy.config.manager import PermissionsConfig

        skill = _make_skill("danger_skill", [("delete_all", "T2")])
        registry = _make_registry(skill)
        pm = PermissionManager(PermissionsConfig(audit_log_enabled=False))
        executor = _make_executor(registry, permission_manager=pm)
        result = await executor.execute("delete_all", tier="T2")
        assert not result.success
        assert "Permission denied" in result.message or "permission" in result.message.lower()

    @pytest.mark.asyncio
    async def test_no_permission_manager_allows_all(self):
        skill = _make_skill("danger_skill", [("delete_all", "T2")])
        registry = _make_registry(skill)
        executor = _make_executor(registry, permission_manager=None)
        result = await executor.execute("delete_all")
        assert result.success


# ──────────────────────────────────────────────────────────────────────────────
# 5. Event Publishing
# ──────────────────────────────────────────────────────────────────────────────

class TestSkillExecutorEvents:
    @pytest.mark.asyncio
    async def test_started_and_completed_events(self):
        from spidy.core.event_bus import EventBus
        from spidy.execution.events import SkillExecutionStartedEvent, SkillExecutionCompletedEvent

        bus = EventBus()
        events = []
        bus.subscribe("execution.started", lambda e: events.append(e))
        bus.subscribe("execution.completed", lambda e: events.append(e))

        skill = _make_skill("evt_skill", [("do_thing", "T0")])
        registry = _make_registry(skill)
        executor = _make_executor(registry, bus=bus)
        await executor.execute("do_thing")
        await asyncio.sleep(0)

        types = [type(e) for e in events]
        assert SkillExecutionStartedEvent in types
        assert SkillExecutionCompletedEvent in types

    @pytest.mark.asyncio
    async def test_failed_event_on_failure(self):
        from spidy.core.event_bus import EventBus
        from spidy.execution.events import SkillExecutionFailedEvent

        bus = EventBus()
        events = []
        bus.subscribe("execution.failed", lambda e: events.append(e))

        skill = _make_skill("fail_skill", [("bad_action", "T0")], fail=True)
        registry = _make_registry(skill)
        executor = _make_executor(registry, max_retries=0, bus=bus)
        await executor.execute("bad_action")
        await asyncio.sleep(0)

        assert any(isinstance(e, SkillExecutionFailedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_retry_event_published(self):
        from spidy.core.event_bus import EventBus
        from spidy.execution.events import SkillRetryEvent

        bus = EventBus()
        events = []
        bus.subscribe("execution.retry", lambda e: events.append(e))

        skill = _make_skill("retry_skill", [("do_thing", "T0")], fail_times=1)
        registry = _make_registry(skill)
        executor = _make_executor(registry, max_retries=2, bus=bus)
        await executor.execute("do_thing")
        await asyncio.sleep(0)

        assert any(isinstance(e, SkillRetryEvent) for e in events)

    @pytest.mark.asyncio
    async def test_no_bus_does_not_crash(self):
        skill = _make_skill()
        registry = _make_registry(skill)
        executor = _make_executor(registry, bus=None)
        result = await executor.execute("do_thing")
        assert result.success

    def test_execution_events_topics(self):
        from spidy.execution.events import (
            SkillExecutionCompletedEvent,
            SkillExecutionFailedEvent,
            SkillExecutionStartedEvent,
            SkillRetryEvent,
        )
        assert SkillExecutionStartedEvent.topic == "execution.started"
        assert SkillExecutionCompletedEvent.topic == "execution.completed"
        assert SkillExecutionFailedEvent.topic == "execution.failed"
        assert SkillRetryEvent.topic == "execution.retry"
