"""
Unit tests for spidy.plugins.events
Tests that each plugin event has the correct topic and fields.
"""

from __future__ import annotations

import pytest

from spidy.plugins.events import (
    PluginDiscoveredEvent,
    PluginDisabledEvent,
    PluginEnabledEvent,
    PluginErrorEvent,
    PluginInstalledEvent,
    PluginUninstalledEvent,
)


class TestPluginEvents:
    """All plugin events must have correct topics and inherit from Event."""

    def test_discovered_topic(self) -> None:
        e = PluginDiscoveredEvent(plugin_name="p", plugin_dir="/p")
        assert e.topic == "plugin.discovered"
        assert e.plugin_name == "p"
        assert e.plugin_dir == "/p"

    def test_installed_topic(self) -> None:
        e = PluginInstalledEvent(plugin_name="p", version="1.0", author="Dev")
        assert e.topic == "plugin.installed"
        assert e.version == "1.0"
        assert e.author == "Dev"

    def test_enabled_topic(self) -> None:
        e = PluginEnabledEvent(plugin_name="p", version="1.0")
        assert e.topic == "plugin.enabled"

    def test_disabled_topic(self) -> None:
        e = PluginDisabledEvent(plugin_name="p")
        assert e.topic == "plugin.disabled"

    def test_uninstalled_topic(self) -> None:
        e = PluginUninstalledEvent(plugin_name="p")
        assert e.topic == "plugin.uninstalled"

    def test_error_topic(self) -> None:
        e = PluginErrorEvent(plugin_name="p", error="boom", stage="setup")
        assert e.topic == "plugin.error"
        assert e.error == "boom"
        assert e.stage == "setup"

    def test_events_are_event_subclasses(self) -> None:
        from spidy.core.event_bus import Event

        for cls in (
            PluginDiscoveredEvent,
            PluginInstalledEvent,
            PluginEnabledEvent,
            PluginDisabledEvent,
            PluginUninstalledEvent,
            PluginErrorEvent,
        ):
            instance = cls()
            assert isinstance(instance, Event), f"{cls.__name__} must subclass Event"

    def test_events_default_empty_fields(self) -> None:
        e = PluginDiscoveredEvent()
        assert e.plugin_name == ""
        assert e.plugin_dir == ""

    def test_error_event_stages(self) -> None:
        for stage in ("install", "enable", "disable", "setup", "teardown"):
            e = PluginErrorEvent(plugin_name="p", error="err", stage=stage)
            assert e.stage == stage
