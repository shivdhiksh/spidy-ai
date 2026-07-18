"""
PluginLoader — Manifest Parsing and Plugin Instantiation
=========================================================
Responsible for:
1. Reading and validating ``plugin.yaml`` from a plugin directory.
2. Dynamically importing the plugin module via ``importlib``.
3. Instantiating the plugin class and returning a ``BasePlugin`` instance.

Design
------
- All failures raise ``PluginError`` (never raw ImportError or FileNotFoundError).
- Plugin modules are loaded into an isolated ``spec`` — they are not added
  to ``sys.modules`` under a public name that other code could import.
- The loader is stateless: call the static methods directly.

Security note
-------------
``importlib.util.spec_from_file_location`` executes arbitrary Python code.
The ``PluginSandbox`` wraps every loader call to catch exceptions. The sandbox
does NOT prevent a determined malicious plugin from accessing the filesystem;
full OS-level sandboxing is a future enhancement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spidy.plugins.interfaces import BasePlugin
    from spidy.plugins.types import PluginManifest

from spidy.plugins.types import PluginError


class PluginLoader:
    """
    Stateless helper that reads manifests and instantiates plugins.

    Methods
    -------
    load_manifest(plugin_dir)
        Parse ``plugin.yaml`` and return a validated ``PluginManifest``.
    instantiate(manifest, plugin_dir)
        Import the plugin module, locate the class, and return an instance.
    """

    # ── Manifest ──────────────────────────────────────────────────────────

    @staticmethod
    def load_manifest(plugin_dir: Path) -> "PluginManifest":
        """
        Parse and validate a ``plugin.yaml`` file.

        Parameters
        ----------
        plugin_dir:
            Directory that contains the plugin (must have ``plugin.yaml``).

        Returns
        -------
        PluginManifest
            Validated, immutable manifest.

        Raises
        ------
        PluginError
            If the file is missing, unparseable, or fails validation.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise PluginError(
                "PyYAML is not installed. Cannot load plugin manifests."
            ) from exc

        manifest_path = plugin_dir / "plugin.yaml"
        if not manifest_path.exists():
            raise PluginError(
                f"No 'plugin.yaml' found in '{plugin_dir}'. "
                "Every plugin directory must contain a plugin.yaml manifest."
            )

        try:
            with manifest_path.open(encoding="utf-8") as fh:
                data: dict[str, Any] = yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            raise PluginError(
                f"Failed to parse '{manifest_path}': {exc}"
            ) from exc

        try:
            from spidy.plugins.types import PluginManifest
            return PluginManifest.from_dict(data)
        except (PluginError, ValueError) as exc:
            raise PluginError(
                f"Manifest validation failed for '{manifest_path}': {exc}"
            ) from exc

    # ── Instantiation ─────────────────────────────────────────────────────

    @staticmethod
    def instantiate(
        manifest: "PluginManifest",
        plugin_dir: Path,
    ) -> "BasePlugin":
        """
        Dynamically import the plugin module and instantiate the plugin class.

        Parameters
        ----------
        manifest:
            Validated manifest that declares ``entry_point`` and ``class_name``.
        plugin_dir:
            Directory containing the plugin files.

        Returns
        -------
        BasePlugin
            A fresh plugin instance. ``setup()`` has NOT been called yet.

        Raises
        ------
        PluginError
            If the entry point does not exist, the import fails, or the
            class is not found / is not a BasePlugin subclass.
        """
        from spidy.plugins.interfaces import BasePlugin

        entry_path = plugin_dir / manifest.entry_point
        if not entry_path.exists():
            raise PluginError(
                f"Plugin '{manifest.name}': entry point '{manifest.entry_point}' "
                f"not found in '{plugin_dir}'."
            )

        # Use a unique module name per plugin to avoid collisions
        module_name = f"_spidy_plugin_{manifest.name}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, entry_path)
            if spec is None or spec.loader is None:
                raise PluginError(
                    f"Plugin '{manifest.name}': could not create module spec "
                    f"from '{entry_path}'."
                )
            module = importlib.util.module_from_spec(spec)
            # Register temporarily so relative imports inside the plugin work
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            finally:
                # Remove from sys.modules — plugins must not be importable by others
                sys.modules.pop(module_name, None)
        except PluginError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PluginError(
                f"Plugin '{manifest.name}': failed to import '{entry_path}': {exc}"
            ) from exc

        cls = getattr(module, manifest.class_name, None)
        if cls is None:
            raise PluginError(
                f"Plugin '{manifest.name}': class '{manifest.class_name}' "
                f"not found in '{entry_path}'."
            )

        if not (isinstance(cls, type) and issubclass(cls, BasePlugin)):
            raise PluginError(
                f"Plugin '{manifest.name}': '{manifest.class_name}' must be "
                f"a subclass of BasePlugin (got {type(cls).__name__})."
            )

        try:
            instance = cls()
        except Exception as exc:  # noqa: BLE001
            raise PluginError(
                f"Plugin '{manifest.name}': failed to instantiate "
                f"'{manifest.class_name}': {exc}"
            ) from exc

        return instance

    # ── Discovery helper ──────────────────────────────────────────────────

    @staticmethod
    def discover_plugin_dirs(plugins_root: Path) -> list[Path]:
        """
        Return all subdirectories of ``plugins_root`` that contain a
        ``plugin.yaml`` file.

        Parameters
        ----------
        plugins_root:
            The root directory to scan (e.g. ``%APPDATA%/Spidy/plugins``).

        Returns
        -------
        list[Path]
            Sorted list of plugin directories.
        """
        if not plugins_root.exists():
            return []

        found: list[Path] = []
        for child in plugins_root.iterdir():
            if child.is_dir() and (child / "plugin.yaml").exists():
                found.append(child)

        return sorted(found)
