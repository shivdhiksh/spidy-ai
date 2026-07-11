"""
HelpSkill — Lists all registered Spidy capabilities.
Permission tier: T0 (always allowed)
"""

from __future__ import annotations

from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult


class HelpSkill(BaseSkill):
    """
    Tells the user what Spidy can do.

    When asked "what can you do?" or "help", this skill queries the
    SkillRegistry for all registered capabilities and returns a summary.

    The registry is injected at construction time to avoid global state.
    """

    name = "help_skill"
    version = "1.0.0"

    def __init__(self, registry: object = None) -> None:
        # Accept registry injection; if None, returns a static message
        self._registry = registry

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="show_help",
                description="List everything Spidy can do.",
                permission_tier="T0",
                examples=["what can you do", "help", "show commands"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action != "show_help":
            return SkillResult.fail(f"HelpSkill: unknown action '{action}'.")

        if self._registry is None:
            return SkillResult.ok(
                "I'm Spidy, your AI companion! I can answer questions, "
                "manage notes, set timers, and much more. "
                "More skills will be added in future updates.",
                action_taken="show_help",
            )

        # Build capability list from registry
        try:
            all_caps = self._registry.all_capabilities()
            if not all_caps:
                return SkillResult.ok(
                    "No skills are currently registered.",
                    action_taken="show_help",
                )
            lines = [f"• {cap.description} ({cap.action})" for cap in all_caps[:20]]
            summary = "Here's what I can do:\n" + "\n".join(lines)
            if len(all_caps) > 20:
                summary += f"\n...and {len(all_caps) - 20} more capabilities."
            return SkillResult.ok(summary, action_taken="show_help")
        except Exception as exc:
            return SkillResult.ok(
                "I can help you with many tasks! Ask me anything.",
                action_taken="show_help",
            )
