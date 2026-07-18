"""
Integration tests for spidy.plugins.manager
Tests the full PluginManager lifecycle: discovery → install → enable → disable → uninstall.
Uses temporary directories with real plugin files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from spidy.core.event_bus import EventBus
from spidy.plugins.manager import PluginManager
from spidy.plugins.types import PluginState
from spidy.skills.registry import SkillRegistry


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(enabled: bool = True, auto_discover: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.auto_discover = auto_discover
    cfg.sandbox = True
    return cfg


def _make_plugin_dir(
    tmp_path: Path,
    name: str = "test_plugin",
    enabled_in_manifest: bool = True,
    crash_setup: bool = False,
) -> Path:
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(parents=True)

    manifest = textwrap.dedent(f"""\
        name: {name}
        version: "1.0.0"
        description: "Integration test plugin"
        author: "Test"
        entry_point: plugin.py
        class_name: TestPlugin
        enabled: {'true' if enabled_in_manifest else 'false'}
    """)
    (plugin_dir / "plugin.yaml").write_text(manifest, encoding="utf-8")

    if crash_setup:
        plugin_code = textwrap.dedent("""\
            from spidy.plugins.interfaces import BasePlugin, PluginContext

            class TestPlugin(BasePlugin):
                name = "crashing_plugin"

                async def setup(self, context: PluginContext) -> None:
                    raise RuntimeError("Intentional crash in setup!")

                async def teardown(self) -> None:
                    pass
        """)
    else:
        plugin_code = textwrap.dedent(f"""\
            from spidy.plugins.interfaces import BasePlugin, PluginContext

            class TestPlugin(BasePlugin):
                name = "{name}"

                async def setup(self, context: PluginContext) -> None:
                    pass  # No-op

                async def teardown(self) -> None:
                    pass
        """)
    (plugin_dir / "plugin.py").write_text(plugin_code, encoding="utf-8")
    return plugin_dir


async def _make_manager(config, plugins_dir: Path) -> PluginManager:
    bus = EventBus()
    loop = __import__("asyncio").get_event_loop()
    bus.set_loop(loop)
    skill_registry = SkillRegistry()
    return PluginManager(
        config=config,
        bus=bus,
        skill_registry=skill_registry,
        plugins_dir=plugins_dir,
    )


# ─── Initialization ───────────────────────────────────────────────────────────


class TestPluginManagerInitialize:
    @pytest.mark.asyncio
    async def test_no_plugins_dir(self, tmp_path: Path) -> None:
        """Empty plugins dir → no crash, no plugins loaded."""
        config = _make_config()
        mgr = await _make_manager(config, tmp_path / "empty")
        await mgr.initialize()
        assert len(mgr.all_plugin_info()) == 0

    @pytest.mark.asyncio
    async def test_disabled_by_config(self, tmp_path: Path) -> None:
        """If plugins disabled, nothing loads even if dir has plugins."""
        _make_plugin_dir(tmp_path)
        config = _make_config(enabled=False)
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        assert len(mgr.all_plugin_info()) == 0

    @pytest.mark.asyncio
    async def test_auto_discover_disabled(self, tmp_path: Path) -> None:
        """If auto_discover disabled, nothing loads."""
        _make_plugin_dir(tmp_path)
        config = _make_config(auto_discover=False)
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        assert len(mgr.all_plugin_info()) == 0

    @pytest.mark.asyncio
    async def test_valid_plugin_loaded(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        assert len(mgr.all_plugin_info()) == 1
        assert mgr.is_enabled("test_plugin")

    @pytest.mark.asyncio
    async def test_plugin_not_enabled_in_manifest(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path, enabled_in_manifest=False)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        info = mgr.registry.get("test_plugin")
        assert info is not None
        assert info.state == PluginState.INSTALLED  # Not enabled

    @pytest.mark.asyncio
    async def test_multiple_plugins_loaded(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path, name="plugin_a")
        _make_plugin_dir(tmp_path, name="plugin_b")
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        assert mgr.is_enabled("plugin_a")
        assert mgr.is_enabled("plugin_b")

    @pytest.mark.asyncio
    async def test_crashing_plugin_does_not_stop_others(self, tmp_path: Path) -> None:
        """A plugin that crashes in setup should not prevent other plugins from loading."""
        _make_plugin_dir(tmp_path, name="good_plugin")
        _make_plugin_dir(tmp_path, name="crashing_plugin", crash_setup=True)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        # good_plugin must be enabled
        assert mgr.is_enabled("good_plugin")
        # crashing_plugin must be in ERROR state
        info = mgr.registry.get("crashing_plugin")
        assert info is not None
        assert info.state == PluginState.ERROR


# ─── Enable / Disable ─────────────────────────────────────────────────────────


class TestPluginManagerEnableDisable:
    @pytest.mark.asyncio
    async def test_enable_unknown_returns_false(self, tmp_path: Path) -> None:
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        result = await mgr.enable("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_enable_already_enabled_returns_true(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        result = await mgr.enable("test_plugin")  # Already enabled
        assert result is True

    @pytest.mark.asyncio
    async def test_disable_active_plugin(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        assert mgr.is_enabled("test_plugin")
        result = await mgr.disable("test_plugin")
        assert result is True
        assert not mgr.is_enabled("test_plugin")

    @pytest.mark.asyncio
    async def test_disable_inactive_returns_false(self, tmp_path: Path) -> None:
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        result = await mgr.disable("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_enable_after_disable(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        await mgr.disable("test_plugin")
        result = await mgr.enable("test_plugin")
        assert result is True
        assert mgr.is_enabled("test_plugin")


# ─── Uninstall ───────────────────────────────────────────────────────────────


class TestPluginManagerUninstall:
    @pytest.mark.asyncio
    async def test_uninstall_active_plugin(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path)
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        result = await mgr.uninstall("test_plugin")
        assert result is True
        assert not mgr.registry.is_installed("test_plugin")

    @pytest.mark.asyncio
    async def test_uninstall_unknown_returns_false(self, tmp_path: Path) -> None:
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        result = await mgr.uninstall("nonexistent")
        assert result is False


# ─── Teardown ────────────────────────────────────────────────────────────────


class TestPluginManagerTeardown:
    @pytest.mark.asyncio
    async def test_teardown_disables_all(self, tmp_path: Path) -> None:
        _make_plugin_dir(tmp_path, name="plugin_a")
        _make_plugin_dir(tmp_path, name="plugin_b")
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        await mgr.initialize()
        assert mgr.is_enabled("plugin_a")
        assert mgr.is_enabled("plugin_b")

        await mgr.teardown()
        assert not mgr.is_enabled("plugin_a")
        assert not mgr.is_enabled("plugin_b")

    @pytest.mark.asyncio
    async def test_teardown_empty_is_safe(self, tmp_path: Path) -> None:
        config = _make_config()
        mgr = await _make_manager(config, tmp_path)
        # No plugins loaded — teardown should be a no-op
        await mgr.teardown()


# ─── Skill Registration ───────────────────────────────────────────────────────


class TestPluginManagerSkillRegistration:
    @pytest.mark.asyncio
    async def test_plugin_skill_registered_in_skill_registry(self, tmp_path: Path) -> None:
        """Plugin-registered skills must be visible in the SkillRegistry."""
        plugin_dir = tmp_path / "skill_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(textwrap.dedent("""\
            name: skill_plugin
            version: "1.0.0"
            description: ""
            author: "Test"
            entry_point: plugin.py
            class_name: SkillPlugin
            enabled: true
        """), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(textwrap.dedent("""\
            from spidy.plugins.interfaces import BasePlugin, PluginContext
            from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult

            class PluginProvidedSkill(BaseSkill):
                name = "plugin_test_action_skill"
                async def execute(self, action, context):
                    return SkillResult.ok("plugin skill response")
                def capabilities(self):
                    return [SkillCapability(action="plugin_test_action", description="A plugin action", permission_tier="T0")]

            class SkillPlugin(BasePlugin):
                name = "skill_plugin"
                async def setup(self, context: PluginContext) -> None:
                    context.register_skill(PluginProvidedSkill())
                async def teardown(self) -> None:
                    pass
        """), encoding="utf-8")

        bus = EventBus()
        skill_registry = SkillRegistry()
        config = _make_config()
        mgr = PluginManager(
            config=config,
            bus=bus,
            skill_registry=skill_registry,
            plugins_dir=tmp_path,
        )
        await mgr.initialize()

        # The skill must be visible to the Brain
        skill = skill_registry.find_skill_for_action("plugin_test_action")
        assert skill is not None
        assert skill.name == "plugin_test_action_skill"

    @pytest.mark.asyncio
    async def test_skill_removed_on_disable(self, tmp_path: Path) -> None:
        """When a plugin is disabled, its skills must be removed from the registry."""
        plugin_dir = tmp_path / "removable_skill_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(textwrap.dedent("""\
            name: removable_skill_plugin
            version: "1.0.0"
            description: ""
            author: "Test"
            entry_point: plugin.py
            class_name: RemovablePlugin
            enabled: true
        """), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(textwrap.dedent("""\
            from spidy.plugins.interfaces import BasePlugin, PluginContext
            from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult

            class RemovableSkill(BaseSkill):
                name = "removable_plugin_skill"
                async def execute(self, action, context):
                    return SkillResult.ok("ok")
                def capabilities(self):
                    return [SkillCapability(action="removable_action", description="Removable", permission_tier="T0")]

            class RemovablePlugin(BasePlugin):
                name = "removable_skill_plugin"
                async def setup(self, context: PluginContext) -> None:
                    context.register_skill(RemovableSkill())
                async def teardown(self) -> None:
                    pass
        """), encoding="utf-8")

        bus = EventBus()
        skill_registry = SkillRegistry()
        config = _make_config()
        mgr = PluginManager(
            config=config,
            bus=bus,
            skill_registry=skill_registry,
            plugins_dir=tmp_path,
        )
        await mgr.initialize()

        # Skill is present after enable
        assert skill_registry.find_skill_for_action("removable_action") is not None

        # Disable the plugin
        await mgr.disable("removable_skill_plugin")

        # Skill must be gone
        assert skill_registry.find_skill_for_action("removable_action") is None


# ─── Events ──────────────────────────────────────────────────────────────────


class TestPluginManagerEvents:
    @pytest.mark.asyncio
    async def test_events_published_during_lifecycle(self, tmp_path: Path) -> None:
        """Check that enabled/disabled events are published on the bus."""
        _make_plugin_dir(tmp_path)
        config = _make_config()
        bus = EventBus()
        skill_registry = SkillRegistry()

        received_topics: list[str] = []

        async def catch_all(event) -> None:
            received_topics.append(event.topic)

        for topic in (
            "plugin.discovered",
            "plugin.installed",
            "plugin.enabled",
            "plugin.disabled",
        ):
            bus.subscribe(topic, catch_all)

        mgr = PluginManager(
            config=config,
            bus=bus,
            skill_registry=skill_registry,
            plugins_dir=tmp_path,
        )
        await mgr.initialize()
        await mgr.disable("test_plugin")

        assert "plugin.discovered" in received_topics
        assert "plugin.installed" in received_topics
        assert "plugin.enabled" in received_topics
        assert "plugin.disabled" in received_topics
