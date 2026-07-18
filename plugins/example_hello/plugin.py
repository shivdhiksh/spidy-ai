"""
Example Hello Plugin
====================
Reference implementation of a Spidy plugin.

This plugin registers a single skill (``HelloPluginSkill``) that responds
to "hello plugin" utterances. It demonstrates:

1. Subclassing ``BasePlugin``
2. Using ``PluginContext`` to register a skill
3. Subscribing to an EventBus topic
4. Reading plugin-specific config

To adapt this for your own plugin:
- Copy this directory to ``plugins/my_plugin/``
- Update ``plugin.yaml`` with your metadata
- Replace ``HelloPluginSkill`` with your own ``BaseSkill`` subclass
"""

from __future__ import annotations

from spidy.plugins.interfaces import BasePlugin, PluginContext
from spidy.skills.base import (
    BaseSkill,
    ParamSchema,
    SkillCapability,
    SkillContext,
    SkillResult,
)


class HelloPluginSkill(BaseSkill):
    """
    A minimal skill that says hello.

    This is the simplest possible Spidy skill — it demonstrates how
    plugin-provided skills are indistinguishable from built-in skills
    once registered.
    """

    name = "hello_plugin_skill"
    version = "1.0.0"

    def __init__(self, greeting: str = "Hello from the Plugin Marketplace!") -> None:
        self._greeting = greeting

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="hello_plugin",
                description="Say hello from the example plugin.",
                permission_tier="T0",
                params=[
                    ParamSchema(
                        "name", "string", required=False,
                        description="Optional name to greet.",
                    ),
                ],
                examples=[
                    "hello plugin",
                    "say hello from the plugin",
                    "greet me plugin",
                ],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action != "hello_plugin":
            return SkillResult.fail(f"HelloPluginSkill: unknown action '{action}'.")

        recipient = context.get("name", "")
        if recipient:
            msg = f"{self._greeting} Nice to meet you, {recipient}!"
        else:
            msg = self._greeting

        return SkillResult.ok(
            msg,
            data={"greeting": msg},
            action_taken="hello_plugin",
        )


class HelloPlugin(BasePlugin):
    """
    Reference ``BasePlugin`` implementation.

    Registers ``HelloPluginSkill`` and subscribes to ``plugin.enabled``
    events to log when any plugin activates.
    """

    name = "example_hello"
    version = "1.0.0"
    description = "Reference plugin that adds a 'say hello' skill."

    def __init__(self) -> None:
        self._skill: HelloPluginSkill | None = None

    async def setup(self, context: PluginContext) -> None:
        """Register the HelloPluginSkill and an EventBus subscription."""
        greeting = context.config.get("greeting", "Hello from the Plugin Marketplace!")
        self._skill = HelloPluginSkill(greeting=greeting)
        context.register_skill(self._skill)

        # Subscribe to plugin.enabled events to demonstrate EventBus integration
        context.subscribe("plugin.enabled", self._on_plugin_enabled)

        context.logger.info(
            "HelloPlugin ready. Greeting: '{g}'",
            g=greeting,
        )

    async def teardown(self) -> None:
        """No external resources to release — PluginContext handles cleanup."""
        self._skill = None

    async def _on_plugin_enabled(self, event: object) -> None:
        """Log when any plugin is enabled (EventBus subscription demo)."""
        name = getattr(event, "plugin_name", "unknown")
        # Just log — don't do anything expensive in an event handler
