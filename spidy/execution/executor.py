"""
SkillExecutor — Permission-Gated Skill Execution Engine
========================================================
Sits between the Brain (ToolRouter) and the SkillRegistry.

Responsibilities
----------------
1. Pre-execution permission check via PermissionManager
2. Execute the skill via SkillRegistry.find_skill_for_action()
3. Retry on transient failures (up to max_retries)
4. Publish execution lifecycle events
5. Enforce per-skill timeouts

Design
------
The executor is stateless between calls. It receives a SkillContext
(containing action, params, session_id) and returns a SkillResult.

It does NOT perform intent classification or planning — that is the
Brain's job. It only executes what the Brain has already decided.

Usage
-----
    executor = SkillExecutor(
        registry=registry,
        permission_manager=permission_manager,
        config=config.executor,
        bus=bus,
    )
    result = await executor.execute(
        action="take_note",
        params={"content": "Buy milk"},
        tier="T1",
        session_id=session_id,
    )
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.skills.base import SkillContext, SkillResult

if TYPE_CHECKING:
    from spidy.config.manager import ExecutorConfig
    from spidy.core.event_bus import EventBus
    from spidy.permissions.manager import PermissionManager
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)


class SkillExecutor:
    """
    Permission-gated, retry-capable skill execution engine.

    Parameters
    ----------
    registry:
        SkillRegistry for action → skill lookup.
    permission_manager:
        PermissionManager for tier enforcement. May be None (no checks).
    config:
        ExecutorConfig with retry/timeout settings.
    bus:
        EventBus for publishing execution events. May be None.
    """

    def __init__(
        self,
        registry: "SkillRegistry",
        permission_manager: "PermissionManager | None" = None,
        config: "ExecutorConfig | None" = None,
        bus: "EventBus | None" = None,
    ) -> None:
        self._registry = registry
        self._permission_manager = permission_manager
        self._config = config
        self._bus = bus

    # ── Primary API ───────────────────────────────────────────────────────

    async def execute(
        self,
        action: str,
        params: dict | None = None,
        tier: str = "T0",
        session_id: str = "",
        user_name: str = "User",
        desktop_state: object = None,
        extra: dict | None = None,
    ) -> SkillResult:
        """
        Execute a skill action with permission checking and retry.

        Parameters
        ----------
        action:
            The action string (must match a registered skill's capability).
        params:
            Key-value parameters extracted by the Brain.
        tier:
            Permission tier declared for this action (T0–T3).
        session_id:
            Current conversation session ID.
        user_name:
            User name for personalised skill responses.
        desktop_state:
            Current DesktopStateSnapshot (may be None).
        extra:
            Additional context key-value pairs.

        Returns
        -------
        SkillResult
            Always returns a SkillResult — never raises.
        """
        params = params or {}
        extra = extra or {}

        # 1. Permission check
        if self._permission_manager is not None and self._config is not None:
            if getattr(self._config, "permission_checking_enabled", True):
                skill = self._registry.find_skill_for_action(action)
                # Resolve tier from capability if not explicitly provided
                effective_tier = tier
                if skill is not None:
                    for cap in skill.capabilities():
                        if cap.action == action:
                            effective_tier = cap.permission_tier
                            break

                allowed = await self._permission_manager.check(
                    action=action,
                    tier=effective_tier,
                    session_id=session_id,
                )
                if not allowed:
                    log.warning(
                        "SkillExecutor: action '{a}' denied by PermissionManager (tier={t})",
                        a=action,
                        t=effective_tier,
                    )
                    return SkillResult.fail(
                        f"Permission denied for action '{action}' (tier {effective_tier}). "
                        "This action requires user confirmation.",
                    )

        # 2. Look up skill
        skill = self._registry.find_skill_for_action(action)
        if skill is None:
            log.warning("SkillExecutor: no skill found for action '{a}'", a=action)
            return SkillResult.fail(f"No skill registered for action '{action}'.")

        # 3. Build context
        ctx = SkillContext(
            action=action,
            params=params,
            session_id=session_id,
            user_name=user_name,
            desktop_state=desktop_state,
            extra=extra,
        )

        # 4. Publish start event
        await self._publish_started(skill.name, action, session_id)

        # 5. Execute with retry
        max_retries = getattr(self._config, "max_retries", 2) if self._config else 2
        retry_delay = getattr(self._config, "retry_delay_seconds", 0.5) if self._config else 0.5
        timeout = getattr(self._config, "skill_timeout_seconds", 30.0) if self._config else 30.0

        result = await self._execute_with_retry(
            skill, action, ctx, max_retries, retry_delay, timeout, session_id
        )

        # 6. Publish completion event
        if result.success:
            await self._publish_completed(skill.name, action, session_id, success=True)
        else:
            await self._publish_failed(skill.name, action, session_id, str(result.error))

        return result

    # ── Internal ──────────────────────────────────────────────────────────

    async def _execute_with_retry(
        self,
        skill: object,
        action: str,
        ctx: SkillContext,
        max_retries: int,
        retry_delay: float,
        timeout: float,
        session_id: str,
    ) -> SkillResult:
        """Execute skill with retry loop and timeout."""
        from spidy.skills.base import BaseSkill
        assert isinstance(skill, BaseSkill)

        last_result = SkillResult.fail(f"Skill '{skill.name}' did not execute.")

        for attempt in range(max_retries + 1):
            if attempt > 0:
                await self._publish_retry(skill.name, action, attempt, session_id)
                await asyncio.sleep(retry_delay)

            try:
                last_result = await asyncio.wait_for(
                    skill.execute(action, ctx),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                last_result = SkillResult.fail(
                    f"Skill '{skill.name}' timed out after {timeout:.0f}s."
                )
                log.warning(
                    "SkillExecutor: '{s}' timed out on attempt {a}",
                    s=skill.name,
                    a=attempt + 1,
                )
            except Exception as exc:  # noqa: BLE001
                last_result = SkillResult.fail(
                    f"Skill '{skill.name}' raised: {exc}", error=exc
                )
                log.error(
                    "SkillExecutor: '{s}' raised exception: {exc}",
                    s=skill.name,
                    exc=exc,
                )

            if last_result.success:
                return last_result

        return last_result

    async def _publish_started(
        self, skill_name: str, action: str, session_id: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.execution.events import SkillExecutionStartedEvent
        await self._bus.publish(SkillExecutionStartedEvent(
            skill_name=skill_name, action=action, session_id=session_id
        ))

    async def _publish_completed(
        self, skill_name: str, action: str, session_id: str, success: bool
    ) -> None:
        if self._bus is None:
            return
        from spidy.execution.events import SkillExecutionCompletedEvent
        await self._bus.publish(SkillExecutionCompletedEvent(
            skill_name=skill_name, action=action, success=success, session_id=session_id
        ))

    async def _publish_failed(
        self, skill_name: str, action: str, session_id: str, error: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.execution.events import SkillExecutionFailedEvent
        await self._bus.publish(SkillExecutionFailedEvent(
            skill_name=skill_name, action=action, error=error, session_id=session_id
        ))

    async def _publish_retry(
        self, skill_name: str, action: str, attempt: int, session_id: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.execution.events import SkillRetryEvent
        await self._bus.publish(SkillRetryEvent(
            skill_name=skill_name, action=action, attempt=attempt, session_id=session_id
        ))
