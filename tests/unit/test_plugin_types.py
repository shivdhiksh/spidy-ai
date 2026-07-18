"""
Unit tests for spidy.plugins.types
Tests PluginManifest, PluginState, PluginInfo, PluginError, CommandRegistration.
"""

from __future__ import annotations

import pytest

from spidy.plugins.types import (
    CommandRegistration,
    PluginError,
    PluginInfo,
    PluginManifest,
    PluginState,
)


# ─── PluginState ──────────────────────────────────────────────────────────────


class TestPluginState:
    def test_all_states_defined(self) -> None:
        states = {s.value for s in PluginState}
        assert "discovered" in states
        assert "installed" in states
        assert "enabled" in states
        assert "disabled" in states
        assert "uninstalled" in states
        assert "error" in states

    def test_is_string_enum(self) -> None:
        assert PluginState.ENABLED == "enabled"
        assert PluginState.DISABLED == "disabled"

    def test_state_str_repr(self) -> None:
        assert str(PluginState.ENABLED) == "PluginState.ENABLED"


# ─── PluginManifest ───────────────────────────────────────────────────────────


class TestPluginManifest:
    def _make(self, **kwargs) -> PluginManifest:
        defaults = {
            "name": "test_plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "author": "Test Author",
            "entry_point": "plugin.py",
            "class_name": "TestPlugin",
        }
        defaults.update(kwargs)
        return PluginManifest(**defaults)

    def test_valid_manifest(self) -> None:
        m = self._make()
        assert m.name == "test_plugin"
        assert m.version == "1.0.0"
        assert m.enabled is True

    def test_defaults(self) -> None:
        m = self._make()
        assert m.permissions == ()
        assert m.tags == ()
        assert m.config == {}
        assert m.min_spidy_version == "0.1.0"

    def test_frozen(self) -> None:
        m = self._make()
        with pytest.raises((AttributeError, TypeError)):
            m.name = "new_name"  # type: ignore

    def test_name_validation_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self._make(name="")

    def test_name_validation_not_snake_case(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            self._make(name="My Plugin")  # spaces not allowed

    def test_name_validation_starts_with_number(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            self._make(name="1plugin")

    def test_name_validation_valid_snake_case(self) -> None:
        # These should not raise
        self._make(name="hello_world")
        self._make(name="my_plugin_v2")
        self._make(name="a")

    def test_version_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self._make(version="")

    def test_entry_point_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self._make(entry_point="")

    def test_class_name_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self._make(class_name="")

    def test_from_dict_valid(self) -> None:
        data = {
            "name": "my_plugin",
            "version": "2.0.0",
            "description": "My plugin",
            "author": "Dev",
            "entry_point": "plugin.py",
            "class_name": "MyPlugin",
            "enabled": True,
            "permissions": ["T1"],
            "tags": ["tool"],
            "config": {"key": "value"},
        }
        m = PluginManifest.from_dict(data)
        assert m.name == "my_plugin"
        assert m.version == "2.0.0"
        assert m.permissions == ("T1",)
        assert m.tags == ("tool",)
        assert m.config == {"key": "value"}

    def test_from_dict_missing_required(self) -> None:
        with pytest.raises(PluginError, match="missing required fields"):
            PluginManifest.from_dict({"name": "test_plugin"})

    def test_from_dict_defaults_optional(self) -> None:
        data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "entry_point": "plugin.py",
            "class_name": "TestPlugin",
        }
        m = PluginManifest.from_dict(data)
        assert m.description == ""
        assert m.author == "Unknown"
        assert m.enabled is True

    def test_from_dict_enabled_false(self) -> None:
        data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "entry_point": "plugin.py",
            "class_name": "TestPlugin",
            "enabled": False,
        }
        m = PluginManifest.from_dict(data)
        assert m.enabled is False


# ─── PluginInfo ───────────────────────────────────────────────────────────────


class TestPluginInfo:
    def _make_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="info_plugin",
            version="1.0.0",
            description="",
            author="Dev",
            entry_point="plugin.py",
            class_name="InfoPlugin",
        )

    def test_is_active_when_enabled(self, tmp_path) -> None:
        info = PluginInfo(
            manifest=self._make_manifest(),
            state=PluginState.ENABLED,
            plugin_dir=tmp_path,
        )
        assert info.is_active is True

    def test_is_not_active_when_disabled(self, tmp_path) -> None:
        info = PluginInfo(
            manifest=self._make_manifest(),
            state=PluginState.DISABLED,
            plugin_dir=tmp_path,
        )
        assert info.is_active is False

    def test_name_property(self, tmp_path) -> None:
        info = PluginInfo(
            manifest=self._make_manifest(),
            state=PluginState.INSTALLED,
            plugin_dir=tmp_path,
        )
        assert info.name == "info_plugin"

    def test_error_field_mutable(self, tmp_path) -> None:
        info = PluginInfo(
            manifest=self._make_manifest(),
            state=PluginState.ERROR,
            plugin_dir=tmp_path,
        )
        info.error = "something went wrong"
        assert info.error == "something went wrong"

    def test_repr(self, tmp_path) -> None:
        info = PluginInfo(
            manifest=self._make_manifest(),
            state=PluginState.INSTALLED,
            plugin_dir=tmp_path,
        )
        r = repr(info)
        assert "info_plugin" in r
        assert "installed" in r


# ─── PluginError ──────────────────────────────────────────────────────────────


class TestPluginError:
    def test_is_exception(self) -> None:
        err = PluginError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_can_be_caught(self) -> None:
        with pytest.raises(PluginError, match="plugin failed"):
            raise PluginError("plugin failed")


# ─── CommandRegistration ──────────────────────────────────────────────────────


class TestCommandRegistration:
    def test_basic_construction(self) -> None:
        async def handler(s: str) -> str:
            return s

        cmd = CommandRegistration(
            name="hello",
            description="Say hello",
            handler=handler,
        )
        assert cmd.name == "hello"
        assert cmd.aliases == ()
        assert cmd.plugin_name == ""

    def test_with_aliases(self) -> None:
        async def handler(s: str) -> str:
            return s

        cmd = CommandRegistration(
            name="hi",
            description="",
            handler=handler,
            aliases=("hey", "greet"),
            plugin_name="test_plugin",
        )
        assert "hey" in cmd.aliases
        assert cmd.plugin_name == "test_plugin"

    def test_frozen(self) -> None:
        async def handler(s: str) -> str:
            return s

        cmd = CommandRegistration(name="cmd", description="", handler=handler)
        with pytest.raises((AttributeError, TypeError)):
            cmd.name = "other"  # type: ignore
