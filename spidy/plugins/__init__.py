"""
Spidy Plugin Marketplace — Public API
=====================================
Milestone 12: Plugin architecture with discovery, metadata, loading/unloading,
full lifecycle management, and sandboxing.

Usage
-----
    from spidy.plugins import PluginManager, PluginRegistry, BasePlugin, PluginContext

    manager = PluginManager(
        config=settings.plugins,
        bus=bus,
        skill_registry=skill_registry,
        plugins_dir=Path("plugins"),
    )
    await manager.initialize()
"""

from spidy.plugins.interfaces import BasePlugin, PluginContext
from spidy.plugins.manager import PluginManager
from spidy.plugins.registry import PluginRegistry
from spidy.plugins.types import (
    CommandRegistration,
    PluginError,
    PluginInfo,
    PluginManifest,
    PluginState,
)

__all__ = [
    "BasePlugin",
    "CommandRegistration",
    "PluginContext",
    "PluginError",
    "PluginInfo",
    "PluginManifest",
    "PluginManager",
    "PluginRegistry",
    "PluginState",
]
