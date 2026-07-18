"""
PluginManager — Top-Level Plugin Orchestrator
=============================================
The single object that SpidyCore uses to manage the entire plugin lifecycle.

Responsibilities
----------------
1. **Discovery** — scan the plugins directory for valid plugin directories.
2. **Installation** — load manifests, register with ``PluginRegistry``.
3. **Activation**  — instantiate plugins, call ``setup()``, inject ``PluginContext``.
4. **Deactivation** — call ``teardown()``, clean up skills and event handlers.
5. **Uninstallation** — remove from registry entirely.
6. **Event publishing** — emit typed events for each lifecycle transition.

Integration with SpidyCore
--------------------------
    # In SpidyCore._run() — after Brain is started:
    self._plugin_mgr = PluginManager(...)
    await self._plugin_mgr.initialize()

    # In SpidyCore._stop() — before Brain is stopped:
    await self._plugin_mgr.teardown()

Design
------
- All plugin calls go through ``PluginSandbox`` — no exception from a plugin
  can crash SpidyCore.
- ``PluginManager`` is NOT a singleton. SpidyCore owns the one instance.
- Plugins are enabled in discovery order (alphabetical within each directory).
- A plugin that fails ``setup()`` is marked ERROR; other plugins still load.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger
from spidy.plugins.interfaces import PluginContext
from spidy.plugins.loader import PluginLoader
from spidy.plugins.registry import PluginRegistry
from spidy.plugins.sandbox import PluginSandbox
from spidy.plugins.types import PluginError, PluginInfo, PluginState

if TYPE_CHECKING:
    from spidy.config.manager import PluginsConfig
    from spidy.core.event_bus import EventBus
    from spidy.plugins.interfaces import BasePlugin
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)


class PluginManager:
    """
    Orchestrates the full plugin lifecycle for SpidyCore.

    Parameters
    ----------
    config:
        ``PluginsConfig`` from ``SpidyConfig.plugins``. Controls whether
        auto-discovery is enabled and which directory to scan.
    bus:
        Application ``EventBus`` for lifecycle event publishing.
    skill_registry:
        Spidy's ``SkillRegistry`` — passed to each ``PluginContext`` so
        plugins can register skills.
    plugins_dir:
        Root directory to scan for plugin subdirectories.
    """

    def __init__(
        self,
        config: "PluginsConfig",
        bus: "EventBus",
        skill_registry: "SkillRegistry",
        plugins_dir: Path,
    ) -> None:
        self._config = config
        self._bus = bus
        self._skill_registry = skill_registry
        self._plugins_dir = plugins_dir

        self._registry = PluginRegistry()
        # Active plugin instances: name → BasePlugin
        self._instances: dict[str, "BasePlugin"] = {}
        # Active plugin contexts: name → PluginContext
        self._contexts: dict[str, PluginContext] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Discover, install, and enable all plugins.

        Called once by SpidyCore after Brain starts.
        Failures in individual plugins are isolated — Spidy always starts.
        """
        if not self._config.enabled:
            log.info("PluginManager: plugins disabled by config.")
            return

        if not self._config.auto_discover:
            log.info("PluginManager: auto-discovery disabled.")
            return

        plugin_dirs = PluginLoader.discover_plugin_dirs(self._plugins_dir)
        log.info(
            "PluginManager: discovered {n} plugin director{s} in '{d}'",
            n=len(plugin_dirs),
            s="ies" if len(plugin_dirs) != 1 else "y",
            d=self._plugins_dir,
        )

        for plugin_dir in plugin_dirs:
            await self._load_one(plugin_dir)

        enabled = self._registry.list_by_state(PluginState.ENABLED)
        log.info(
            "PluginManager: {n} plugin{s} active.",
            n=len(enabled),
            s="s" if len(enabled) != 1 else "",
        )

    async def teardown(self) -> None:
        """
        Disable all active plugins.

        Called by SpidyCore during shutdown. Each plugin's ``teardown()``
        is called and its skill/event registrations are reversed.
        """
        names = list(self._instances.keys())
        for name in names:
            await self.disable(name)
        log.info("PluginManager: all plugins disabled.")

    # ── Public plugin control API ──────────────────────────────────────────

    async def enable(self, name: str) -> bool:
        """
        Enable a previously installed plugin.

        Instantiates the plugin class, calls ``setup()``, and registers
        its skills and event subscriptions.

        Parameters
        ----------
        name:
            Plugin name (must be installed in the registry).

        Returns
        -------
        bool
            True on success; False if the plugin is unknown or setup failed.
        """
        from spidy.plugins.events import PluginEnabledEvent, PluginErrorEvent

        info = self._registry.get(name)
        if info is None:
            log.warning("PluginManager.enable: '{name}' is not installed.", name=name)
            return False

        if info.state == PluginState.ENABLED:
            log.debug("Plugin '{name}' is already enabled.", name=name)
            return True

        # Instantiate
        try:
            plugin = PluginLoader.instantiate(info.manifest, info.plugin_dir)
        except PluginError as exc:
            self._registry.set_error(name, str(exc))
            await self._publish_error(name, str(exc), "instantiate")
            return False

        # Build context
        ctx = self._make_context(info)

        # Call setup() through sandbox
        error = await PluginSandbox.safe_setup(plugin, ctx)
        if error:
            ctx.cleanup()  # reverse any partial registrations
            self._registry.set_error(name, error)
            await self._publish_error(name, error, "setup")
            return False

        # All good — record instance and context
        self._instances[name] = plugin
        self._contexts[name] = ctx
        self._registry.set_state(name, PluginState.ENABLED)

        await self._bus.publish(PluginEnabledEvent(
            plugin_name=name,
            version=info.manifest.version,
        ))
        log.info(
            "Plugin enabled: '{name}' v{ver}",
            name=name,
            ver=info.manifest.version,
        )
        return True

    async def disable(self, name: str) -> bool:
        """
        Disable an active plugin.

        Calls ``teardown()``, then removes skills and event subscriptions
        via the ``PluginContext.cleanup()`` mechanism.

        Parameters
        ----------
        name:
            Plugin name (must be currently enabled).

        Returns
        -------
        bool
            True if the plugin was active and is now disabled; False otherwise.
        """
        from spidy.plugins.events import PluginDisabledEvent

        plugin = self._instances.get(name)
        ctx = self._contexts.get(name)
        if plugin is None:
            log.debug("PluginManager.disable: '{name}' is not active.", name=name)
            return False

        # Call teardown() — errors are logged but do not prevent cleanup
        await PluginSandbox.safe_teardown(plugin)

        # Reverse registrations
        if ctx is not None:
            ctx.cleanup()

        # Remove from active tracking
        self._instances.pop(name, None)
        self._contexts.pop(name, None)

        info = self._registry.get(name)
        if info is not None and info.state != PluginState.ERROR:
            self._registry.set_state(name, PluginState.DISABLED)

        await self._bus.publish(PluginDisabledEvent(plugin_name=name))
        log.info("Plugin disabled: '{name}'", name=name)
        return True

    async def uninstall(self, name: str) -> bool:
        """
        Disable and completely remove a plugin from the registry.

        Parameters
        ----------
        name:
            Plugin name to remove.

        Returns
        -------
        bool
            True if found and removed; False if the plugin was unknown.
        """
        from spidy.plugins.events import PluginUninstalledEvent

        if not self._registry.is_installed(name):
            log.debug("PluginManager.uninstall: '{name}' not installed.", name=name)
            return False

        await self.disable(name)
        self._registry.uninstall(name)

        await self._bus.publish(PluginUninstalledEvent(plugin_name=name))
        log.info("Plugin uninstalled: '{name}'", name=name)
        return True

    async def reload(self, name: str) -> bool:
        """
        Disable then re-enable a plugin (hot-reload).

        Parameters
        ----------
        name:
            Plugin name to reload.

        Returns
        -------
        bool
            True if the plugin is now active after the reload.
        """
        await self.disable(name)
        return await self.enable(name)

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def registry(self) -> PluginRegistry:
        """The underlying ``PluginRegistry``."""
        return self._registry

    def is_enabled(self, name: str) -> bool:
        """True if the named plugin is currently active."""
        return name in self._instances

    def all_plugin_info(self) -> list[PluginInfo]:
        """All registered plugins regardless of state."""
        return self._registry.all_plugins()

    def get_context(self, name: str) -> PluginContext | None:
        """Return the ``PluginContext`` for an active plugin, or None."""
        return self._contexts.get(name)

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _load_one(self, plugin_dir: Path) -> None:
        """Attempt to install and (if enabled) activate one plugin directory."""
        from spidy.plugins.events import (
            PluginDiscoveredEvent,
            PluginInstalledEvent,
            PluginErrorEvent,
        )

        await self._bus.publish(PluginDiscoveredEvent(
            plugin_name=plugin_dir.name,
            plugin_dir=str(plugin_dir),
        ))

        # Load manifest
        try:
            manifest = PluginLoader.load_manifest(plugin_dir)
        except PluginError as exc:
            log.warning(
                "PluginManager: skipping '{d}' — bad manifest: {e}",
                d=plugin_dir.name,
                e=exc,
            )
            await self._bus.publish(PluginErrorEvent(
                plugin_name=plugin_dir.name,
                error=str(exc),
                stage="install",
            ))
            return

        # Install into registry
        try:
            self._registry.install(manifest, plugin_dir, overwrite=False)
        except PluginError as exc:
            log.warning("PluginManager: could not install '{n}': {e}",
                        n=manifest.name, e=exc)
            return

        await self._bus.publish(PluginInstalledEvent(
            plugin_name=manifest.name,
            version=manifest.version,
            author=manifest.author,
        ))

        # Enable if the manifest says to
        if manifest.enabled:
            await self.enable(manifest.name)

    def _make_context(self, info: PluginInfo) -> PluginContext:
        """Build a fresh ``PluginContext`` for the given plugin."""
        return PluginContext(
            plugin_name=info.name,
            config=dict(info.manifest.config),
            _skill_registry=self._skill_registry,
            _bus=self._bus,
            _logger=get_logger(f"plugin.{info.name}"),
        )

    async def _publish_error(
        self, name: str, error: str, stage: str
    ) -> None:
        from spidy.plugins.events import PluginErrorEvent
        await self._bus.publish(PluginErrorEvent(
            plugin_name=name,
            error=error,
            stage=stage,
        ))
