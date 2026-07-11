"""
TimerSkill — Set, list, and cancel countdown timers.
Permission tier: T0 (always allowed)

Timers are in-memory only. They fire by publishing a TimerFiredEvent
to the EventBus. A future milestone can wire TTS to read it aloud.

Actions
-------
set_timer    — Start a countdown timer             (T0)
list_timers  — List active timers                  (T0)
cancel_timer — Cancel a timer by name or id        (T0)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from spidy.skills.base import BaseSkill, SkillCapability, ParamSchema, SkillContext, SkillResult

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus


@dataclass
class _Timer:
    id: str
    name: str
    duration_seconds: float
    fire_at: datetime
    task: asyncio.Task | None = None


class TimerSkill(BaseSkill):
    """
    In-memory countdown timer skill.

    Timers survive for the lifetime of the process only.
    When a timer fires, a log message is written and (future) a TTS
    event is published via the EventBus.
    """

    name = "timer_skill"
    version = "1.0.0"

    def __init__(self, bus: "EventBus | None" = None) -> None:
        self._bus = bus
        self._timers: dict[str, _Timer] = {}

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="set_timer",
                description="Start a countdown timer.",
                permission_tier="T0",
                params=[
                    ParamSchema("duration", "string", required=True,
                                description="Duration (e.g. '5 minutes', '30 seconds', '1 hour')."),
                    ParamSchema("name", "string", required=False,
                                default="", description="Optional timer name."),
                ],
                examples=["set a timer for 5 minutes", "remind me in 10 minutes",
                          "timer 30 seconds"],
            ),
            SkillCapability(
                action="list_timers",
                description="List all active timers.",
                permission_tier="T0",
                examples=["list timers", "what timers do I have", "show active timers"],
            ),
            SkillCapability(
                action="cancel_timer",
                description="Cancel an active timer by name or ID.",
                permission_tier="T0",
                params=[ParamSchema("name", "string", required=True,
                                    description="Timer name or ID to cancel.")],
                examples=["cancel timer", "stop the timer"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "set_timer":
            return await self._set_timer(context)
        if action == "list_timers":
            return await self._list_timers(context)
        if action == "cancel_timer":
            return await self._cancel_timer(context)
        return SkillResult.fail(f"TimerSkill: unknown action '{action}'.")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _set_timer(self, context: SkillContext) -> SkillResult:
        duration_str = context.get("duration", "").strip()
        timer_name = context.get("name", "").strip() or f"Timer {len(self._timers) + 1}"

        seconds = self._parse_duration(duration_str)
        if seconds is None or seconds <= 0:
            return SkillResult.fail(
                f"Could not understand duration: '{duration_str}'. "
                "Try '5 minutes', '30 seconds', or '1 hour'."
            )

        timer_id = str(uuid.uuid4())[:8]
        fire_at = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        timer = _Timer(id=timer_id, name=timer_name, duration_seconds=seconds, fire_at=fire_at)

        task = asyncio.create_task(self._run_timer(timer))
        timer.task = task
        self._timers[timer_id] = timer

        human = self._format_duration(seconds)
        return SkillResult.ok(
            f"Timer '{timer_name}' set for {human}.",
            data={"id": timer_id, "name": timer_name, "seconds": seconds},
            action_taken="set_timer",
        )

    async def _list_timers(self, context: SkillContext) -> SkillResult:
        # Clean up completed timers
        self._timers = {tid: t for tid, t in self._timers.items()
                        if t.task and not t.task.done()}

        if not self._timers:
            return SkillResult.ok("No active timers.", action_taken="list_timers")

        now = datetime.now(timezone.utc)
        lines = []
        for timer in self._timers.values():
            remaining = max(0, (timer.fire_at - now).total_seconds())
            lines.append(f"• {timer.name}: {self._format_duration(remaining)} remaining")
        return SkillResult.ok(
            f"{len(self._timers)} active timer(s):\n" + "\n".join(lines),
            action_taken="list_timers",
        )

    async def _cancel_timer(self, context: SkillContext) -> SkillResult:
        query = context.get("name", "").strip().lower()

        # Find by id prefix or name
        found = None
        for timer in self._timers.values():
            if timer.id.startswith(query) or timer.name.lower() == query:
                found = timer
                break

        if found is None:
            return SkillResult.fail(f"No active timer matching '{query}'.")

        if found.task and not found.task.done():
            found.task.cancel()
        del self._timers[found.id]
        return SkillResult.ok(
            f"Timer '{found.name}' cancelled.",
            action_taken="cancel_timer",
        )

    # ── Timer runner ──────────────────────────────────────────────────────

    async def _run_timer(self, timer: _Timer) -> None:
        from spidy.logging.logger import get_logger
        log = get_logger(__name__)
        try:
            await asyncio.sleep(timer.duration_seconds)
            log.info("⏰ Timer fired: '{name}'", name=timer.name)
            # Remove from active set
            self._timers.pop(timer.id, None)
        except asyncio.CancelledError:
            log.debug("Timer cancelled: '{name}'", name=timer.name)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_duration(text: str) -> float | None:
        """Parse a human-readable duration string into seconds."""
        if not text:
            return None
        import re
        text = text.lower().strip()
        total = 0.0
        found_any = False

        # Order matters: longer unit names first to prevent partial matches
        # Each pattern replaces its match with empty string to prevent re-use
        patterns = [
            (r"(\d+(?:\.\d+)?)\s*hours?", 3600),
            (r"(\d+(?:\.\d+)?)\s*hrs?", 3600),
            (r"(\d+(?:\.\d+)?)\s*minutes?", 60),
            (r"(\d+(?:\.\d+)?)\s*mins?", 60),
            (r"(\d+(?:\.\d+)?)\s*seconds?", 1),
            (r"(\d+(?:\.\d+)?)\s*secs?", 1),
            (r"(\d+(?:\.\d+)?)\s*h\b", 3600),
            (r"(\d+(?:\.\d+)?)\s*m\b", 60),
            (r"(\d+(?:\.\d+)?)\s*s\b", 1),
        ]
        for pattern, multiplier in patterns:
            def _adder(m, _mult=multiplier):
                nonlocal total, found_any
                total += float(m.group(1)) * _mult
                found_any = True
                return ""
            text = re.sub(pattern, _adder, text)

        if not found_any:
            # bare number — assume seconds
            try:
                total = float(text.strip())
                found_any = True
            except ValueError:
                pass
        return total if found_any else None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into a human-readable string."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        minutes = seconds // 60
        secs = seconds % 60
        if minutes < 60:
            base = f"{minutes} minute{'s' if minutes != 1 else ''}"
            if secs:
                base += f" {secs} second{'s' if secs != 1 else ''}"
            return base
        hours = minutes // 60
        mins = minutes % 60
        base = f"{hours} hour{'s' if hours != 1 else ''}"
        if mins:
            base += f" {mins} minute{'s' if mins != 1 else ''}"
        return base
