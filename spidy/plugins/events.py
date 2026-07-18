"""
Plugin Events — Typed EventBus Events for the Plugin Marketplace
================================================================
Every plugin lifecycle transition publishes a typed event on the EventBus.
This lets the UI, logging layer, and other modules react without coupling
to the plugin internals.

Topics
------
    plugin.discovered    PluginDiscoveredEvent
    plugin.installed     PluginInstalledEvent
    plugin.enabled       PluginEnabledEvent
    plugin.disabled      PluginDisabledEvent
    plugin.uninstalled   PluginUninstalledEvent
    plugin.error         PluginErrorEvent
"""

from __future__ import annotations

from dataclasses import dataclass

from spidy.core.event_bus import Event


# ─── Plugin Lifecycle Events ──────────────────────────────────────────────────


@dataclass
class PluginDiscoveredEvent(Event):
    """Published when a plugin directory is found during discovery."""

    topic = "plugin.discovered"
    plugin_name: str = ""
    plugin_dir: str = ""


@dataclass
class PluginInstalledEvent(Event):
    """Published when a plugin manifest is validated and registered."""

    topic = "plugin.installed"
    plugin_name: str = ""
    version: str = ""
    author: str = ""


@dataclass
class PluginEnabledEvent(Event):
    """Published when a plugin's setup() completes successfully."""

    topic = "plugin.enabled"
    plugin_name: str = ""
    version: str = ""


@dataclass
class PluginDisabledEvent(Event):
    """Published when a plugin's teardown() completes and registrations are removed."""

    topic = "plugin.disabled"
    plugin_name: str = ""


@dataclass
class PluginUninstalledEvent(Event):
    """Published when a plugin is removed from the PluginRegistry."""

    topic = "plugin.uninstalled"
    plugin_name: str = ""


@dataclass
class PluginErrorEvent(Event):
    """Published when a plugin fails at any lifecycle stage."""

    topic = "plugin.error"
    plugin_name: str = ""
    error: str = ""
    stage: str = ""   # "install" | "enable" | "disable" | "setup" | "teardown"
