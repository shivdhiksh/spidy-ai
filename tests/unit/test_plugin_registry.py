"""
Unit tests for spidy.plugins.registry
Tests the full PluginRegistry state machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spidy.plugins.registry import PluginRegistry
from spidy.plugins.types import PluginError, PluginInfo, PluginManifest, PluginState


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_manifest(name: str = "test_plugin", enabled: bool = True) -> PluginManifest:
    return PluginManifest(
        name=name,
        version="1.0.0",
        description="Test",
        author="Dev",
        entry_point="plugin.py",
        class_name="TestPlugin",
        enabled=enabled,
    )


# ─── Install ─────────────────────────────────────────────────────────────────


class TestPluginRegistryInstall:
    def test_install_returns_info(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        manifest = make_manifest()
        info = reg.install(manifest, tmp_path)
        assert isinstance(info, PluginInfo)
        assert info.name == "test_plugin"
        assert info.state == PluginState.INSTALLED

    def test_install_sets_dir(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        info = reg.install(make_manifest(), tmp_path)
        assert info.plugin_dir == tmp_path

    def test_install_duplicate_raises(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        with pytest.raises(PluginError, match="already installed"):
            reg.install(make_manifest(), tmp_path)

    def test_install_overwrite(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(name="test_plugin"), tmp_path)
        new_manifest = make_manifest(name="test_plugin")
        info = reg.install(new_manifest, tmp_path, overwrite=True)
        assert info.name == "test_plugin"

    def test_install_multiple_plugins(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest("plugin_a"), tmp_path)
        reg.install(make_manifest("plugin_b"), tmp_path)
        reg.install(make_manifest("plugin_c"), tmp_path)
        assert len(reg) == 3

    def test_contains_after_install(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        assert "test_plugin" in reg

    def test_not_contains_before_install(self) -> None:
        reg = PluginRegistry()
        assert "test_plugin" not in reg


# ─── Uninstall ───────────────────────────────────────────────────────────────


class TestPluginRegistryUninstall:
    def test_uninstall_returns_true(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        result = reg.uninstall("test_plugin")
        assert result is True

    def test_uninstall_removes_from_registry(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        reg.uninstall("test_plugin")
        assert "test_plugin" not in reg
        assert len(reg) == 0

    def test_uninstall_not_found_returns_false(self) -> None:
        reg = PluginRegistry()
        result = reg.uninstall("no_such_plugin")
        assert result is False


# ─── State transitions ────────────────────────────────────────────────────────


class TestPluginRegistryStateTransitions:
    def test_set_state(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        reg.set_state("test_plugin", PluginState.ENABLED)
        assert reg.get("test_plugin").state == PluginState.ENABLED

    def test_set_state_disabled(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        reg.set_state("test_plugin", PluginState.ENABLED)
        reg.set_state("test_plugin", PluginState.DISABLED)
        assert reg.get("test_plugin").state == PluginState.DISABLED

    def test_set_state_unknown_raises(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(PluginError, match="not registered"):
            reg.set_state("unknown", PluginState.ENABLED)

    def test_set_error(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        reg.set_error("test_plugin", "something exploded")
        info = reg.get("test_plugin")
        assert info.state == PluginState.ERROR
        assert info.error == "something exploded"

    def test_set_error_unknown_noop(self) -> None:
        reg = PluginRegistry()
        # Should not raise
        reg.set_error("unknown", "err")


# ─── Queries ──────────────────────────────────────────────────────────────────


class TestPluginRegistryQueries:
    def test_get_returns_info(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        info = reg.get("test_plugin")
        assert info is not None
        assert info.name == "test_plugin"

    def test_get_unknown_returns_none(self) -> None:
        reg = PluginRegistry()
        assert reg.get("no_such") is None

    def test_all_plugins_empty(self) -> None:
        reg = PluginRegistry()
        assert reg.all_plugins() == []

    def test_all_plugins_returns_all(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        for name in ("a_plugin", "b_plugin", "c_plugin"):
            reg.install(make_manifest(name), tmp_path)
        all_p = reg.all_plugins()
        assert len(all_p) == 3

    def test_list_by_state(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest("enabled_plugin"), tmp_path)
        reg.install(make_manifest("disabled_plugin"), tmp_path)
        reg.set_state("enabled_plugin", PluginState.ENABLED)
        reg.set_state("disabled_plugin", PluginState.DISABLED)

        enabled = reg.list_by_state(PluginState.ENABLED)
        disabled = reg.list_by_state(PluginState.DISABLED)
        assert len(enabled) == 1
        assert enabled[0].name == "enabled_plugin"
        assert len(disabled) == 1

    def test_is_installed(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        assert reg.is_installed("test_plugin") is False
        reg.install(make_manifest(), tmp_path)
        assert reg.is_installed("test_plugin") is True

    def test_is_enabled(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        assert reg.is_enabled("test_plugin") is False
        reg.set_state("test_plugin", PluginState.ENABLED)
        assert reg.is_enabled("test_plugin") is True

    def test_len(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        assert len(reg) == 0
        reg.install(make_manifest("a_plugin"), tmp_path)
        assert len(reg) == 1
        reg.install(make_manifest("b_plugin"), tmp_path)
        assert len(reg) == 2

    def test_repr(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        reg.install(make_manifest(), tmp_path)
        assert "test_plugin" in repr(reg)
