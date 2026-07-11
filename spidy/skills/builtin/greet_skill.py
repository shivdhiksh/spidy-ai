"""
GreetSkill — Personalised greetings based on time of day.
Permission tier: T0 (always allowed)
"""

from __future__ import annotations

from datetime import datetime

from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult


def _time_of_day_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 21:
        return "Good evening"
    return "Good night"


class GreetSkill(BaseSkill):
    """
    Generates personalised greeting responses.

    Uses the configured user_name from SkillContext.
    Greeting text adapts to the current time of day.
    """

    name = "greet_skill"
    version = "1.0.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="greet",
                description="Say hello or greet the user.",
                permission_tier="T0",
                examples=["hello", "hi", "good morning", "hey spidy", "greetings"],
            ),
            SkillCapability(
                action="farewell",
                description="Say goodbye to the user.",
                permission_tier="T0",
                examples=["goodbye", "bye", "see you", "farewell", "later"],
            ),
            SkillCapability(
                action="introduce",
                description="Introduce Spidy — what it is and what it does.",
                permission_tier="T0",
                examples=["who are you", "what are you", "introduce yourself"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        name = context.user_name or "User"

        if action == "greet":
            greeting = _time_of_day_greeting()
            msg = f"{greeting}, {name}! How can I help you today?"
            return SkillResult.ok(msg, action_taken=action)

        if action == "farewell":
            msg = f"Goodbye, {name}! Have a wonderful day. I'll be here when you need me."
            return SkillResult.ok(msg, action_taken=action)

        if action == "introduce":
            msg = (
                f"Hi {name}, I'm Spidy — your personal AI companion. "
                "I can help you manage notes, set timers, answer questions, "
                "control your computer, and much more. "
                "I learn your preferences over time to serve you better. "
                "Just ask me anything!"
            )
            return SkillResult.ok(msg, action_taken=action)

        return SkillResult.fail(f"GreetSkill: unknown action '{action}'.")
