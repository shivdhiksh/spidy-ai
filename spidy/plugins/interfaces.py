"""
Plugin Interfaces — BasePlugin ABC and PluginContext
=====================================================
Defines the contracts that all third-party plugins must satisfy.

BasePlugin
----------
Every plugin must subclass ``BasePlugin`` and implement:
- ``setup(context)``   — called once on enable; register skills/events here
- ``teardown()``       — called once on disable; do cleanup here

PluginContext
-------------
The **only** object a plugin may use to interact with Spidy.
It provides four safe extension points:

    context.register_skill(skill)      → adds to SkillRegistry
    context.subscribe(topic, handler)  → subscribes to EventBus
    context.register_command(cmd)      → registers a text command
    context.logger                     → scoped logger for this plugin

PluginContext tracks every registration and reverses them automatically
when ``teardown()`` is called — so disable is always clean.

Design
------
- No plugin has direct access to Brain, Memory, LLM, or SpidyCore.
- A crashing plugin is always sandboxed — it cannot take down Spidy.
- Skills registered by plugins use the same SkillCapability / BaseSkill
  contract as built-in skills; the Brain sees no difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus
    from spidy.plugins.types import CommandRegistration
    from spidy.skills.base import BaseSkill
    from spidy.skills.registry import SkillRegistry


# ─── BasePlugin ───────────────────────────────────────────────────────────────


class BasePlugin(ABC):
    """
    Abstract base class for all Spidy plugins.

    Plugin authors subclass this and implement ``setup`` and ``teardown``.
    The class-level attributes (``name``, ``version``, ``description``) are
    used as fallbacks if the manifest doesn't declare them.

    Example
    -------
        class WeatherPlugin(BasePlugin):
            name = "weather_plugin"
            version = "1.0.0"
            description = "Adds a weather lookup skill."

            async def setup(self, context: PluginContext) -> None:
                context.register_skill(WeatherSkill(api_key=context.config["api_key"]))

            async def teardown(self) -> None:
                pass  # SkillRegistry cleanup is handled by PluginContext
    """

    # Subclasses MAY override these
    name: ClassVar[str] = "base_plugin"
    version: ClassVar[str] = "0.0.0"
    description: ClassVar[str] = ""

    @abstractmethod
    async def setup(self, context: "PluginContext") -> None:
        """
        Activate the plugin.

        Called once by ``PluginManager.enable()``.
        Register skills, subscribe to events, and register commands here.

        Parameters
        ----------
        context:
            The ``PluginContext`` for this plugin session. The only object
            the plugin may use to interact with Spidy.
        """
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """
        Deactivate the plugin.

        Called once by ``PluginManager.disable()``.
        Release external resources (network connections, file handles).
        Skill / EventBus cleanup is handled automatically by ``PluginContext``.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', v{self.version})"


# ─── PluginContext ─────────────────────────────────────────────────────────────


@dataclass
class PluginContext:
    """
    Safe interaction surface passed to ``BasePlugin.setup()``.

    A plugin may ONLY interact with Spidy through this object.
    Every registration is tracked so it can be cleanly reversed when the
    plugin is disabled, without any coordination from the plugin itself.

    Attributes
    ----------
    plugin_name:
        The name of the owning plugin (used in log messages).
    config:
        Plugin-specific configuration dict from ``plugin.yaml``.

    Methods
    -------
    register_skill(skill)      → SkillRegistry.register()
    subscribe(topic, handler)  → EventBus.subscribe()
    register_command(cmd)      → CommandRegistry (tracked internally)
    cleanup()                  → Reverse all registrations (called on disable)
    """

    plugin_name: str
    config: dict[str, Any]

    # Internal — injected by PluginManager; not exposed to plugins
    _skill_registry: "SkillRegistry"
    _bus: "EventBus"
    _logger: Any   # structlog / loguru logger

    # Tracking: every registration that must be reversed on disable
    _registered_skills: list["BaseSkill"] = field(default_factory=list)
    _subscriptions: list[tuple[str, Callable]] = field(default_factory=list)
    _commands: list["CommandRegistration"] = field(default_factory=list)

    # ── Public API ────────────────────────────────────────────────────────

    def register_skill(self, skill: "BaseSkill") -> None:
        """
        Register a skill with Spidy's SkillRegistry.

        The skill will be visible to the Brain's DecisionEngine and
        ToolRouter immediately. It is removed when the plugin is disabled.

        Parameters
        ----------
        skill:
            A ``BaseSkill`` instance. Must implement ``capabilities()``
            and ``execute()``.
        """
        self._skill_registry.register(skill, overwrite=True)
        self._registered_skills.append(skill)
        self._logger.info(
            "Plugin '{p}' registered skill '{s}'",
            p=self.plugin_name,
            s=skill.name,
        )

    def subscribe(self, topic: str, handler: Callable) -> None:
        """
        Subscribe to an EventBus topic.

        The subscription is removed when the plugin is disabled.

        Parameters
        ----------
        topic:
            EventBus topic string (e.g. ``"voice.user_spoke"``).
        handler:
            Async callable ``(event: Event) -> None``.
        """
        self._bus.subscribe(topic, handler)
        self._subscriptions.append((topic, handler))
        self._logger.debug(
            "Plugin '{p}' subscribed to '{t}'",
            p=self.plugin_name,
            t=topic,
        )

    def register_command(self, cmd: "CommandRegistration") -> None:
        """
        Register a plugin-provided text command.

        Commands are direct-dispatch shortcuts that bypass the Intent
        Classifier. They are tracked here for cleanup on disable.

        Parameters
        ----------
        cmd:
            A ``CommandRegistration`` instance created by the plugin.
        """
        self._commands.append(cmd)
        self._logger.debug(
            "Plugin '{p}' registered command '{c}'",
            p=self.plugin_name,
            c=cmd.name,
        )

    @property
    def logger(self) -> Any:
        """Return the plugin-scoped logger."""
        return self._logger

    @property
    def registered_skill_names(self) -> list[str]:
        """Names of skills currently registered by this plugin."""
        return [s.name for s in self._registered_skills]

    @property
    def subscribed_topics(self) -> list[str]:
        """Topics currently subscribed by this plugin."""
        return [t for t, _ in self._subscriptions]

    @property
    def registered_commands(self) -> list["CommandRegistration"]:
        """Commands currently registered by this plugin."""
        return list(self._commands)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """
        Reverse all registrations made during ``setup()``.

        Called automatically by ``PluginManager.disable()`` — plugins
        should NOT call this themselves.
        """
        for skill in list(self._registered_skills):
            try:
                self._skill_registry.unregister(skill.name)
                self._logger.debug(
                    "Plugin '{p}': unregistered skill '{s}'",
                    p=self.plugin_name,
                    s=skill.name,
                )
            except Exception:  # noqa: BLE001
                pass

        for topic, handler in list(self._subscriptions):
            try:
                self._bus.unsubscribe(topic, handler)
                self._logger.debug(
                    "Plugin '{p}': unsubscribed from '{t}'",
                    p=self.plugin_name,
                    t=topic,
                )
            except Exception:  # noqa: BLE001
                pass

        self._registered_skills.clear()
        self._subscriptions.clear()
        self._commands.clear()
