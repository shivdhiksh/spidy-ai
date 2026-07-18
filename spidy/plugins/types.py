"""
Plugin Types — Core Data Structures for the Plugin Marketplace
==============================================================
Shared vocabulary used across all plugin subsystems.

Design
------
- ``PluginManifest``    : Immutable declaration parsed from plugin.yaml
- ``PluginState``       : Enum for the plugin lifecycle state machine
- ``PluginInfo``        : Mutable tracking record held by the PluginRegistry
- ``PluginError``       : Sentinel exception type for plugin faults
- ``CommandRegistration``: A text command registered by a plugin

State Machine
-------------
    DISCOVERED ──► INSTALLED ──► ENABLED ◄──► DISABLED
                                    │
                                    └──► UNINSTALLED
    (any state can transition to ERROR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ─── Lifecycle State ──────────────────────────────────────────────────────────


class PluginState(str, Enum):
    """Lifecycle state of a plugin managed by the PluginRegistry."""

    DISCOVERED = "discovered"
    """Manifest found on disk; not yet validated or installed."""

    INSTALLED = "installed"
    """Manifest validated and registered; not yet activated."""

    ENABLED = "enabled"
    """Plugin class instantiated and ``setup()`` called successfully."""

    DISABLED = "disabled"
    """``teardown()`` called; skills/subscriptions removed from Spidy."""

    UNINSTALLED = "uninstalled"
    """Removed from the PluginRegistry entirely."""

    ERROR = "error"
    """Plugin failed at some point in the lifecycle."""


# ─── Manifest ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PluginManifest:
    """
    Immutable declaration parsed from a plugin's ``plugin.yaml``.

    Attributes
    ----------
    name:
        Unique snake_case identifier. Must not clash with existing plugins.
    version:
        Semantic version string (e.g. ``"1.0.0"``).
    description:
        Human-readable one-line description shown in the marketplace.
    author:
        Plugin author name or organisation.
    entry_point:
        Filename (relative to the plugin directory) that contains the
        plugin class. Typically ``"plugin.py"``.
    class_name:
        Python class inside ``entry_point`` that subclasses ``BasePlugin``.
    enabled:
        Whether to activate this plugin at startup. Users can flip this
        without uninstalling.
    permissions:
        Permission tiers this plugin declares it needs (advisory for now).
    tags:
        Freeform tags for marketplace categorisation (e.g. ``["productivity"]``).
    min_spidy_version:
        Minimum Spidy version required. Checked at install time.
    config:
        Plugin-specific configuration dict from the manifest.
    """

    name: str
    version: str
    description: str
    author: str
    entry_point: str
    class_name: str
    enabled: bool = True
    permissions: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    min_spidy_version: str = "0.1.0"
    config: dict[str, Any] = field(default_factory=dict)

    # ── Validators ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("PluginManifest: 'name' must not be empty.")
        if not self.version or not self.version.strip():
            raise ValueError("PluginManifest: 'version' must not be empty.")
        if not self.entry_point or not self.entry_point.strip():
            raise ValueError("PluginManifest: 'entry_point' must not be empty.")
        if not self.class_name or not self.class_name.strip():
            raise ValueError("PluginManifest: 'class_name' must not be empty.")
        # Normalise name: only allow alphanumeric + underscore
        import re
        if not re.match(r"^[a-z][a-z0-9_]*$", self.name):
            raise ValueError(
                f"PluginManifest: 'name' must be snake_case (got '{self.name}')."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        """
        Build a PluginManifest from a raw YAML-parsed dict.

        Raises ``PluginError`` if required fields are missing.
        """
        required = ("name", "version", "entry_point", "class_name")
        missing = [k for k in required if not data.get(k)]
        if missing:
            raise PluginError(
                f"plugin.yaml is missing required fields: {missing}"
            )

        permissions = tuple(data.get("permissions") or [])
        tags = tuple(data.get("tags") or [])
        config = dict(data.get("config") or {})

        return cls(
            name=str(data["name"]).strip(),
            version=str(data.get("version", "0.0.0")).strip(),
            description=str(data.get("description", "")).strip(),
            author=str(data.get("author", "Unknown")).strip(),
            entry_point=str(data["entry_point"]).strip(),
            class_name=str(data["class_name"]).strip(),
            enabled=bool(data.get("enabled", True)),
            permissions=permissions,
            tags=tags,
            min_spidy_version=str(data.get("min_spidy_version", "0.1.0")).strip(),
            config=config,
        )


# ─── Plugin Info ──────────────────────────────────────────────────────────────


@dataclass
class PluginInfo:
    """
    Mutable tracking record held by the PluginRegistry.

    Wraps a ``PluginManifest`` with runtime state and error information.
    """

    manifest: PluginManifest
    state: PluginState
    plugin_dir: Path
    error: str = ""

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def is_active(self) -> bool:
        return self.state == PluginState.ENABLED

    def __repr__(self) -> str:
        return (
            f"PluginInfo(name='{self.name}', "
            f"v{self.manifest.version}, state={self.state.value})"
        )


# ─── Command Registration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommandRegistration:
    """
    A text command registered by a plugin through ``PluginContext``.

    Commands are short user-facing keywords (e.g. ``"!weather"``) that
    bypass the Intent Classifier and route directly to a handler.

    Attributes
    ----------
    name:
        Primary command keyword (e.g. ``"hello"``).
    description:
        Short description shown in the help listing.
    handler:
        Async callable ``(utterance: str) -> str`` invoked when the command
        is matched. Must return the response text.
    aliases:
        Alternative keywords that map to the same handler.
    plugin_name:
        Owning plugin (set automatically by PluginContext).
    """

    name: str
    description: str
    handler: Any  # Callable[[str], Awaitable[str]]
    aliases: tuple[str, ...] = field(default_factory=tuple)
    plugin_name: str = ""


# ─── Error ────────────────────────────────────────────────────────────────────


class PluginError(Exception):
    """
    Raised when a plugin operation fails.

    This exception is always caught by the PluginSandbox — it must
    never propagate to the Brain or SpidyCore.
    """
