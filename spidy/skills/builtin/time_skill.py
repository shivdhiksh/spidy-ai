"""
TimeSkill — Current time, date, and day information.
Permission tier: T0 (always allowed)
"""

from __future__ import annotations

from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult
from spidy.utils.datetime_utils import get_local_datetime

_MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class TimeSkill(BaseSkill):
    """
    Provides current time, date, and day-of-week information.

    All operations resolve to the authoritative configured or system local timezone.
    No network calls, no permissions required.
    """

    name = "time_skill"
    version = "1.0.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="get_time",
                description="Tell the current time.",
                permission_tier="T0",
                examples=[
                    "what time is it", "what time is it now", "tell me the time",
                    "current time", "what is the time", "what's the time",
                ],
            ),
            SkillCapability(
                action="get_date",
                description="Tell today's date.",
                permission_tier="T0",
                examples=["what's today's date", "what date is it", "today's date", "what is the date"],
            ),
            SkillCapability(
                action="get_day",
                description="Tell what day of the week it is.",
                permission_tier="T0",
                examples=["what day is it", "what day of the week", "what day of the week is it"],
            ),
            SkillCapability(
                action="get_datetime",
                description="Tell the full current date and time.",
                permission_tier="T0",
                examples=["what is the date and time", "date and time", "what is the current time and date"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        tz_override = context.get("timezone") if context else None
        now = get_local_datetime(tz_override)

        if action == "get_time":
            hour = now.hour
            minute = now.minute
            period = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            msg = f"The current time is {display_hour}:{minute:02d} {period}."
            return SkillResult.ok(
                msg,
                data={
                    "time": now.strftime("%H:%M"),
                    "timezone": str(now.tzinfo),
                    "iso": now.isoformat(),
                },
                action_taken=action,
            )

        if action == "get_date":
            day_name = _DAYS[now.weekday()]
            month_name = _MONTHS[now.month]
            msg = f"Today is {day_name}, {month_name} {now.day}, {now.year}."
            return SkillResult.ok(
                msg,
                data={
                    "date": now.strftime("%Y-%m-%d"),
                    "timezone": str(now.tzinfo),
                    "iso": now.isoformat(),
                },
                action_taken=action,
            )

        if action == "get_day":
            day_name = _DAYS[now.weekday()]
            msg = f"Today is {day_name}."
            return SkillResult.ok(
                msg,
                data={
                    "day": day_name,
                    "timezone": str(now.tzinfo),
                },
                action_taken=action,
            )

        if action == "get_datetime":
            day_name = _DAYS[now.weekday()]
            month_name = _MONTHS[now.month]
            hour = now.hour
            period = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            msg = (
                f"It's {day_name}, {month_name} {now.day}, {now.year}, "
                f"{display_hour}:{now.minute:02d} {period}."
            )
            return SkillResult.ok(
                msg,
                data={
                    "datetime": now.isoformat(),
                    "timezone": str(now.tzinfo),
                },
                action_taken=action,
            )

        return SkillResult.fail(f"TimeSkill: unknown action '{action}'.")
