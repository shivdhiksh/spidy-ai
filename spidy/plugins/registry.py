"""
PluginRegistry — Plugin Catalog and State Machine
==================================================
Maintains the canonical catalog of all known plugins and their lifecycle
states. It is the single source of truth for what plugins exist, what
state they are in, and why they failed (if they did).

State Machine
-------------
    DISCOVERED ──► INSTALLED ──► ENABLED ◄──► DISABLED
                       │              │
                       └──────────────┴──► UNINSTALLED
    (any state can transition to ERROR on failure)

Design
------
- The registry is pure state — it does not import modules or call plugins.
- All mutations are protected by a re-entrant lock (thread-safe).
- Duplicate plugin names raise ``PluginError`` unless ``overwrite=True``.
- ``PluginInfo`` objects are returned by reference; callers must not mutate
  them. The registry mutates them via controlled ``set_state``/``set_error``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from spidy.logging.logger import get_logger
from spidy.plugins.types import PluginError, PluginInfo, PluginManifest, PluginState

log = get_logger(__name__)


class PluginRegistry:
    """
    Thread-safe catalog of all managed plugins.

    Usage
    -----
        registry = PluginRegistry()
        info = registry.install(manifest, plugin_dir)
        registry.set_state(manifest.name, PluginState.ENABLED)

        plugin = registry.get("my_plugin")
        all_active = registry.list_by_state(PluginState.ENABLED)
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}
        self._lock = threading.RLock()

    # ── Installation ──────────────────────────────────────────────────────

    def install(
        self,
        manifest: PluginManifest,
        plugin_dir: Path,
        overwrite: bool = False,
    ) -> PluginInfo:
        """
        Register a plugin manifest in the catalog.

        Parameters
        ----------
        manifest:
            Validated ``PluginManifest`` from ``PluginLoader.load_manifest()``.
        plugin_dir:
            Absolute path to the plugin directory.
        overwrite:
            If True, replace an existing entry with the same name.

        Returns
        -------
        PluginInfo
            The newly created tracking record.

        Raises
        ------
        PluginError
            If a plugin with the same name is already registered and
            ``overwrite=False``.
        """
        with self._lock:
            name = manifest.name
            if name in self._plugins and not overwrite:
                raise PluginError(
                    f"Plugin '{name}' is already installed. "
                    "Use overwrite=True to replace it."
                )

            info = PluginInfo(
                manifest=manifest,
                state=PluginState.INSTALLED,
                plugin_dir=plugin_dir,
            )
            self._plugins[name] = info
            log.info(
                "Plugin installed: '{name}' v{version} by {author}",
                name=name,
                version=manifest.version,
                author=manifest.author,
            )
            return info

    def uninstall(self, name: str) -> bool:
        """
        Remove a plugin from the catalog entirely.

        Parameters
        ----------
        name:
            Plugin name to remove.

        Returns
        -------
        bool
            True if found and removed; False if not found.
        """
        with self._lock:
            if name not in self._plugins:
                log.debug("PluginRegistry.uninstall: '{name}' not found.", name=name)
                return False
            del self._plugins[name]
            log.info("Plugin uninstalled: '{name}'", name=name)
            return True

    # ── State mutations ───────────────────────────────────────────────────

    def set_state(self, name: str, state: PluginState) -> None:
        """
        Transition a plugin to a new lifecycle state.

        Parameters
        ----------
        name:
            Plugin name.
        state:
            Target ``PluginState``.

        Raises
        ------
        PluginError
            If the plugin is not registered.
        """
        with self._lock:
            info = self._plugins.get(name)
            if info is None:
                raise PluginError(
                    f"PluginRegistry.set_state: '{name}' is not registered."
                )
            old = info.state
            info.state = state
            log.debug(
                "Plugin '{name}': {old} → {new}",
                name=name,
                old=old.value,
                new=state.value,
            )

    def set_error(self, name: str, error: str) -> None:
        """
        Mark a plugin as failed and record the error message.

        Parameters
        ----------
        name:
            Plugin name.
        error:
            Human-readable description of the failure.
        """
        with self._lock:
            info = self._plugins.get(name)
            if info is None:
                return
            info.state = PluginState.ERROR
            info.error = error
            log.warning(
                "Plugin '{name}' entered ERROR state: {error}",
                name=name,
                error=error,
            )

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, name: str) -> PluginInfo | None:
        """Return the ``PluginInfo`` for a plugin, or ``None`` if not registered."""
        with self._lock:
            return self._plugins.get(name)

    def all_plugins(self) -> list[PluginInfo]:
        """Return all registered plugins (any state)."""
        with self._lock:
            return list(self._plugins.values())

    def list_by_state(self, state: PluginState) -> list[PluginInfo]:
        """Return all plugins in a given lifecycle state."""
        with self._lock:
            return [p for p in self._plugins.values() if p.state == state]

    def is_installed(self, name: str) -> bool:
        """True if the plugin is registered (any state)."""
        with self._lock:
            return name in self._plugins

    def is_enabled(self, name: str) -> bool:
        """True if the plugin is in the ENABLED state."""
        with self._lock:
            info = self._plugins.get(name)
            return info is not None and info.state == PluginState.ENABLED

    # ── Dunder ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._plugins

    def __repr__(self) -> str:
        with self._lock:
            names = list(self._plugins.keys())
        return f"PluginRegistry({names})"
