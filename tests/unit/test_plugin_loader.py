"""
Unit tests for spidy.plugins.loader
Tests PluginLoader.load_manifest() and PluginLoader.instantiate().
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from spidy.plugins.loader import PluginLoader
from spidy.plugins.types import PluginError


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_plugin_dir(
    tmp_path: Path,
    manifest_content: str = "",
    plugin_content: str = "",
    subdir: str = "my_plugin",
) -> Path:
    """Helper: create a plugin directory with manifest and plugin.py."""
    plugin_dir = tmp_path / subdir
    plugin_dir.mkdir(parents=True)
    if manifest_content:
        (plugin_dir / "plugin.yaml").write_text(manifest_content, encoding="utf-8")
    if plugin_content:
        (plugin_dir / "plugin.py").write_text(plugin_content, encoding="utf-8")
    return plugin_dir


VALID_MANIFEST = textwrap.dedent("""\
    name: my_plugin
    version: "1.0.0"
    description: "A test plugin"
    author: "Test"
    entry_point: plugin.py
    class_name: MyPlugin
    enabled: true
""")

VALID_PLUGIN = textwrap.dedent("""\
    from spidy.plugins.interfaces import BasePlugin, PluginContext

    class MyPlugin(BasePlugin):
        name = "my_plugin"
        version = "1.0.0"

        async def setup(self, context: PluginContext) -> None:
            pass

        async def teardown(self) -> None:
            pass
""")


# ─── load_manifest ────────────────────────────────────────────────────────────


class TestLoadManifest:
    def test_valid_manifest(self, tmp_path: Path) -> None:
        plugin_dir = make_plugin_dir(tmp_path, manifest_content=VALID_MANIFEST)
        manifest = PluginLoader.load_manifest(plugin_dir)
        assert manifest.name == "my_plugin"
        assert manifest.version == "1.0.0"
        assert manifest.author == "Test"
        assert manifest.enabled is True

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "empty_plugin"
        plugin_dir.mkdir()
        with pytest.raises(PluginError, match="No 'plugin.yaml'"):
            PluginLoader.load_manifest(plugin_dir)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content="name: [invalid: yaml:\n  bad: - nesting",
        )
        with pytest.raises(PluginError, match="Failed to parse"):
            PluginLoader.load_manifest(plugin_dir)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content="name: my_plugin\nversion: '1.0.0'\n",  # missing entry_point + class_name
        )
        with pytest.raises(PluginError, match="missing required fields"):
            PluginLoader.load_manifest(plugin_dir)

    def test_disabled_plugin_loaded(self, tmp_path: Path) -> None:
        content = VALID_MANIFEST + "enabled: false\n"
        plugin_dir = make_plugin_dir(tmp_path, manifest_content=content)
        manifest = PluginLoader.load_manifest(plugin_dir)
        assert manifest.enabled is False

    def test_with_config_block(self, tmp_path: Path) -> None:
        content = VALID_MANIFEST + "config:\n  api_key: secret\n"
        plugin_dir = make_plugin_dir(tmp_path, manifest_content=content)
        manifest = PluginLoader.load_manifest(plugin_dir)
        assert manifest.config.get("api_key") == "secret"

    def test_with_tags_and_permissions(self, tmp_path: Path) -> None:
        content = VALID_MANIFEST + "tags:\n  - tool\n  - demo\npermissions:\n  - T1\n"
        plugin_dir = make_plugin_dir(tmp_path, manifest_content=content)
        manifest = PluginLoader.load_manifest(plugin_dir)
        assert "tool" in manifest.tags
        assert "T1" in manifest.permissions


# ─── instantiate ─────────────────────────────────────────────────────────────


class TestInstantiate:
    def test_valid_plugin(self, tmp_path: Path) -> None:
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content=VALID_MANIFEST,
            plugin_content=VALID_PLUGIN,
        )
        manifest = PluginLoader.load_manifest(plugin_dir)
        plugin = PluginLoader.instantiate(manifest, plugin_dir)
        assert plugin is not None

    def test_plugin_is_base_plugin_instance(self, tmp_path: Path) -> None:
        from spidy.plugins.interfaces import BasePlugin

        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content=VALID_MANIFEST,
            plugin_content=VALID_PLUGIN,
        )
        manifest = PluginLoader.load_manifest(plugin_dir)
        plugin = PluginLoader.instantiate(manifest, plugin_dir)
        assert isinstance(plugin, BasePlugin)

    def test_missing_entry_point_raises(self, tmp_path: Path) -> None:
        plugin_dir = make_plugin_dir(tmp_path, manifest_content=VALID_MANIFEST)
        # No plugin.py created
        manifest = PluginLoader.load_manifest(plugin_dir)
        with pytest.raises(PluginError, match="entry point"):
            PluginLoader.instantiate(manifest, plugin_dir)

    def test_wrong_class_name_raises(self, tmp_path: Path) -> None:
        bad_manifest = VALID_MANIFEST.replace("class_name: MyPlugin", "class_name: WrongClass")
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content=bad_manifest,
            plugin_content=VALID_PLUGIN,
        )
        manifest = PluginLoader.load_manifest(plugin_dir)
        with pytest.raises(PluginError, match="not found"):
            PluginLoader.instantiate(manifest, plugin_dir)

    def test_class_not_baseplugin_raises(self, tmp_path: Path) -> None:
        bad_plugin = textwrap.dedent("""\
            class MyPlugin:
                pass  # Does NOT subclass BasePlugin
        """)
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content=VALID_MANIFEST,
            plugin_content=bad_plugin,
        )
        manifest = PluginLoader.load_manifest(plugin_dir)
        with pytest.raises(PluginError, match="subclass of BasePlugin"):
            PluginLoader.instantiate(manifest, plugin_dir)

    def test_import_error_raises_plugin_error(self, tmp_path: Path) -> None:
        bad_plugin = textwrap.dedent("""\
            import this_module_does_not_exist_12345
            class MyPlugin:
                pass
        """)
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content=VALID_MANIFEST,
            plugin_content=bad_plugin,
        )
        manifest = PluginLoader.load_manifest(plugin_dir)
        with pytest.raises(PluginError, match="failed to import"):
            PluginLoader.instantiate(manifest, plugin_dir)

    def test_module_not_added_to_sys_modules(self, tmp_path: Path) -> None:
        import sys
        plugin_dir = make_plugin_dir(
            tmp_path,
            manifest_content=VALID_MANIFEST,
            plugin_content=VALID_PLUGIN,
        )
        manifest = PluginLoader.load_manifest(plugin_dir)
        PluginLoader.instantiate(manifest, plugin_dir)
        assert "_spidy_plugin_my_plugin" not in sys.modules


# ─── discover_plugin_dirs ─────────────────────────────────────────────────────


class TestDiscoverPluginDirs:
    def test_empty_directory(self, tmp_path: Path) -> None:
        result = PluginLoader.discover_plugin_dirs(tmp_path)
        assert result == []

    def test_missing_root(self, tmp_path: Path) -> None:
        result = PluginLoader.discover_plugin_dirs(tmp_path / "nonexistent")
        assert result == []

    def test_discovers_valid_plugin(self, tmp_path: Path) -> None:
        make_plugin_dir(tmp_path, manifest_content=VALID_MANIFEST)
        result = PluginLoader.discover_plugin_dirs(tmp_path)
        assert len(result) == 1
        assert result[0].name == "my_plugin"

    def test_ignores_dirs_without_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "no_manifest").mkdir()
        make_plugin_dir(tmp_path, manifest_content=VALID_MANIFEST)
        result = PluginLoader.discover_plugin_dirs(tmp_path)
        assert len(result) == 1

    def test_ignores_files(self, tmp_path: Path) -> None:
        (tmp_path / "some_file.txt").write_text("hello")
        result = PluginLoader.discover_plugin_dirs(tmp_path)
        assert result == []

    def test_sorted_results(self, tmp_path: Path) -> None:
        for name in ("z_plugin", "a_plugin", "m_plugin"):
            make_plugin_dir(tmp_path, manifest_content=VALID_MANIFEST.replace("my_plugin", name), subdir=name)
        result = PluginLoader.discover_plugin_dirs(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)
