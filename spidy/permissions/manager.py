"""
PermissionManager — Skill Permission Gate
==========================================
Enforces the Spidy permission tier system before every skill execution.

Permission Tiers
----------------
T0 — Always allowed. No confirmation. Read-only or fully safe actions.
     Examples: list files, get time, show help.

T1 — Allowed, but requires user awareness. Confirmed once per session.
     Examples: write a note, set a timer, change volume.

T2 — Requires explicit confirmation on every call.
     Examples: delete files, send emails, install packages.

T3 — Requires unlock (admin mode). Not confirmed automatically.
     Examples: system-level writes, network config, registry edits.

Policy
------
The ``PermissionsConfig.require_confirmation_for`` list overrides the
default tier. Any action listed there is treated as T2.

Audit Log
---------
All permission checks (grant and deny) are appended to a simple JSONL
audit log when ``PermissionsConfig.audit_log_enabled`` is True.

Usage
-----
    manager = PermissionManager(config.permissions, bus=bus)
    allowed = await manager.check("delete_file", tier="T2", session_id=sid)
    if not allowed:
        return SkillResult.fail("Permission denied for delete_file.")
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.config.manager import PermissionsConfig
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class PermissionTier(str, Enum):
    """Ordered permission tiers. Higher tier = more restrictive."""
    T0 = "T0"   # Always allow
    T1 = "T1"   # Allow with logging (once per session)
    T2 = "T2"   # Require explicit confirmation every time
    T3 = "T3"   # Require admin unlock

    @property
    def ordinal(self) -> int:
        return {"T0": 0, "T1": 1, "T2": 2, "T3": 3}[self.value]

    def __lt__(self, other: "PermissionTier") -> bool:
        return self.ordinal < other.ordinal

    def __le__(self, other: "PermissionTier") -> bool:
        return self.ordinal <= other.ordinal


class PermissionManager:
    """
    Enforces permission tiers for skill execution.

    Parameters
    ----------
    config:
        PermissionsConfig from SpidyConfig.
    bus:
        EventBus for publishing permission events. May be None.
    """

    def __init__(
        self,
        config: "PermissionsConfig",
        bus: "EventBus | None" = None,
    ) -> None:
        self._config = config
        self._bus = bus
        # T1 actions confirmed this session: {(action, session_id)}
        self._t1_confirmed: set[tuple[str, str]] = set()
        # T3 unlocked actions this session
        self._t3_unlocked: set[str] = set()
        # Pending T2 confirmations: key=(action, session_id) → Future[bool]
        self._pending_confirmations: dict[tuple[str, str], asyncio.Future] = {}

        # Subscribe to permission response events if bus is available
        if self._bus is not None:
            self._bus.subscribe("permission.response", self._on_permission_response)

    # ── Primary API ───────────────────────────────────────────────────────

    async def check(
        self,
        action: str,
        tier: str | PermissionTier = "T0",
        session_id: str = "",
        description: str = "",
    ) -> bool:
        """
        Check whether an action is permitted.

        Parameters
        ----------
        action:
            The action string (e.g. "delete_file", "set_volume").
        tier:
            The permission tier declared by the skill.
            May be overridden by PermissionsConfig.require_confirmation_for.
        session_id:
            Current session ID (used for T1 once-per-session tracking).
        description:
            Human-readable description shown in confirmation dialogs.

        Returns
        -------
        bool
            True = permitted. False = denied.
        """
        # Resolve and possibly override tier
        effective_tier = self._resolve_tier(action, tier)

        if not self._config.permission_checking_enabled if hasattr(self._config, "permission_checking_enabled") else False:
            await self._audit(action, effective_tier, granted=True, session_id=session_id)
            return True

        granted = await self._evaluate(action, effective_tier, session_id, description)
        await self._audit(action, effective_tier, granted=granted, session_id=session_id)
        await self._emit_event(action, effective_tier, granted, session_id, description)
        return granted

    def unlock_t3(self, action: str) -> None:
        """Unlock a T3 action for this session (requires admin confirmation)."""
        self._t3_unlocked.add(action)
        log.info("T3 action unlocked for this session: '{a}'", a=action)

    def reset_session(self, session_id: str) -> None:
        """Clear T1 session confirmations for a specific session."""
        self._t1_confirmed = {
            (a, sid) for (a, sid) in self._t1_confirmed if sid != session_id
        }

    def respond_to_confirmation(
        self, action: str, session_id: str, approved: bool
    ) -> None:
        """
        Resolve a pending T2 permission confirmation.

        Called by the UI overlay after the user approves or denies a
        permission dialog. This resolves the asyncio.Future that the
        PermissionManager is awaiting in ``_evaluate()``.

        Parameters
        ----------
        action:
            The action that was being confirmed.
        session_id:
            The session ID for the pending confirmation.
        approved:
            True if the user approved, False if denied.
        """
        key = (action, session_id)
        fut = self._pending_confirmations.get(key)
        if fut is not None and not fut.done():
            fut.set_result(approved)
            log.info(
                "Permission {'approved' if approved else 'denied'} for '{a}' (session={s})",
                a=action, s=session_id,
            )
        else:
            log.warning(
                "respond_to_confirmation: no pending confirmation for '{a}' (session={s})",
                a=action, s=session_id,
            )

    # ── Internal ──────────────────────────────────────────────────────────

    def _resolve_tier(
        self, action: str, declared_tier: str | PermissionTier
    ) -> PermissionTier:
        """
        Resolve effective tier, taking config overrides into account.

        If the action is in require_confirmation_for, bump to T2.
        """
        tier = PermissionTier(str(declared_tier)) if isinstance(declared_tier, str) else declared_tier

        if action in (self._config.require_confirmation_for or []):
            if tier < PermissionTier.T2:
                log.debug(
                    "Action '{a}' overridden to T2 by policy.",
                    a=action,
                )
                tier = PermissionTier.T2

        return tier

    async def _evaluate(
        self,
        action: str,
        tier: PermissionTier,
        session_id: str,
        description: str,
    ) -> bool:
        """Evaluate whether the action passes for the given tier."""
        if tier == PermissionTier.T0:
            return True

        if tier == PermissionTier.T1:
            # Allowed once per session without interactive confirmation
            if (action, session_id) in self._t1_confirmed:
                return True
            # Auto-approve T1 in M4 (interactive confirmation dialog is M5+)
            self._t1_confirmed.add((action, session_id))
            log.debug("T1 action auto-approved for session: '{a}'", a=action)
            return True

        if tier == PermissionTier.T2:
            # When no bus is available, no UI can respond to the confirmation dialog.
            # Immediately deny to avoid hanging (common in unit tests and CLI mode).
            if self._bus is None:
                log.warning(
                    "T2 action '{a}' denied instantly: no EventBus available "
                    "(no UI can respond to the confirmation request).",
                    a=action,
                )
                return False

            # M6: Publish a PermissionRequestedEvent and await user confirmation.
            # The UI overlay subscribes to this event and resolves the Future
            # via respond_to_confirmation() after user input.
            loop = asyncio.get_running_loop()
            key = (action, session_id)
            fut: asyncio.Future[bool] = loop.create_future()
            self._pending_confirmations[key] = fut


            # Emit the dedicated PermissionRequestedEvent so the UI shows a dialog
            from spidy.permissions.events import PermissionRequestedEvent
            await self._bus.publish(PermissionRequestedEvent(
                action=action,
                tier=tier.value,
                description=description or f"Allow '{action}'?",
                session_id=session_id,
            ))

            # Also emit the generic denied event (UI can update state)
            await self._emit_event(
                action, tier, granted=False, session_id=session_id,
                description=description or f"Allow '{action}'?",
            )

            timeout = getattr(self._config, "t2_confirmation_timeout_seconds", 30.0)
            try:
                # Wait for the user to respond via respond_to_confirmation()
                confirmed = await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "T2 permission request for '{a}' timed out after {t}s — denying.",
                    a=action, t=timeout,
                )
                confirmed = False
            finally:
                self._pending_confirmations.pop(key, None)

            return confirmed


        if tier == PermissionTier.T3:
            if action in self._t3_unlocked:
                return True
            log.warning(
                "T3 action '{a}' is locked. Call permission_manager.unlock_t3('{a}') first.",
                a=action,
            )
            return False

        return False

    async def _audit(
        self,
        action: str,
        tier: PermissionTier,
        granted: bool,
        session_id: str,
    ) -> None:
        """Append a line to the JSONL audit log."""
        if not getattr(self._config, "audit_log_enabled", False):
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "tier": tier.value,
            "granted": granted,
            "session_id": session_id,
        }
        audit_file = getattr(self._config, "audit_log_file", "spidy_audit.log")
        try:
            await asyncio.to_thread(
                Path(audit_file).open("a", encoding="utf-8").write,
                json.dumps(record) + "\n",
            )
        except Exception:  # noqa: BLE001
            pass  # Audit failures must never crash the app

    async def _on_permission_response(self, event: object) -> None:
        """EventBus subscriber: resolves pending T2 Future from the UI."""
        from spidy.permissions.events import PermissionResponseEvent
        if not isinstance(event, PermissionResponseEvent):
            return
        self.respond_to_confirmation(
            action=event.action,
            session_id=event.session_id,
            approved=event.approved,
        )

    async def _emit_event(
        self,
        action: str,
        tier: PermissionTier,
        granted: bool,
        session_id: str,
        description: str = "",
    ) -> None:
        """Publish permission event to the EventBus."""
        if self._bus is None:
            return
        from spidy.permissions.events import PermissionDeniedEvent, PermissionGrantedEvent
        if granted:
            await self._bus.publish(PermissionGrantedEvent(
                action=action, tier=tier.value, session_id=session_id,
            ))
        else:
            await self._bus.publish(PermissionDeniedEvent(
                action=action,
                tier=tier.value,
                reason=f"Tier {tier.value} requires user confirmation.",
                session_id=session_id,
            ))
