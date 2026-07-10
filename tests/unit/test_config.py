"""
Test suite for spidy.config.manager
=====================================
Tests ConfigManager loading, validation, defaults, and env overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from spidy.config.manager import ConfigManager, SpidyConfig


class TestConfigManagerLoad:
    """ConfigManager.load() should validate and return SpidyConfig."""

    def test_load_valid_config(self, tmp_path: Path):
        """Loading a complete valid config should return a SpidyConfig."""
        cfg = {"app": {"user_name": "TestUser", "debug": True}}
        cfg_file = tmp_path / "spidy_config.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")

        mgr = ConfigManager(config_path=cfg_file)
        settings = mgr.load()

        assert isinstance(settings, SpidyConfig)
        assert settings.app.user_name == "TestUser"
        assert settings.app.debug is True

    def test_load_missing_file_uses_defaults(self, tmp_path: Path):
        """If config file does not exist, all defaults should be applied."""
        mgr = ConfigManager(config_path=tmp_path / "nonexistent.yaml")
        settings = mgr.load()

        assert isinstance(settings, SpidyConfig)
        assert settings.app.user_name == "User"  # default
        assert settings.app.debug is False       # default

    def test_load_invalid_yaml_raises(self, tmp_path: Path):
        """Malformed YAML should raise ValueError with a helpful message."""
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("app: {bad: yaml: content: :", encoding="utf-8")

        mgr = ConfigManager(config_path=cfg_file)
        with pytest.raises(ValueError, match="Failed to parse"):
            mgr.load()

    def test_load_invalid_log_level_raises(self, tmp_path: Path):
        """Invalid log_level value should fail Pydantic validation."""
        cfg = {"app": {"log_level": "VERBOSE"}}
        cfg_file = tmp_path / "spidy_config.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")

        mgr = ConfigManager(config_path=cfg_file)
        with pytest.raises(ValueError, match="Config validation failed"):
            mgr.load()


class TestConfigManagerSettings:
    """settings property should guard against premature access."""

    def test_settings_before_load_raises(self, tmp_path: Path):
        """Accessing settings before load() should raise RuntimeError."""
        mgr = ConfigManager(config_path=tmp_path / "x.yaml")
        with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
            _ = mgr.settings

    def test_settings_after_load_returns_config(self, tmp_path: Path):
        """settings property should return the loaded config after load()."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("", encoding="utf-8")  # empty → all defaults

        mgr = ConfigManager(config_path=cfg_file)
        mgr.load()
        assert mgr.settings.app.name == "Spidy"


class TestConfigManagerEnvOverrides:
    """Environment variables with SPIDY__ prefix should override config."""

    def test_env_override_top_level_string(self, tmp_path: Path, monkeypatch):
        """SPIDY__APP__USER_NAME should override app.user_name."""
        monkeypatch.setenv("SPIDY__APP__USER_NAME", "EnvUser")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("", encoding="utf-8")

        mgr = ConfigManager(config_path=cfg_file)
        settings = mgr.load()
        assert settings.app.user_name == "EnvUser"

    def test_env_override_boolean(self, tmp_path: Path, monkeypatch):
        """SPIDY__APP__DEBUG should override app.debug."""
        monkeypatch.setenv("SPIDY__APP__DEBUG", "true")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("", encoding="utf-8")

        mgr = ConfigManager(config_path=cfg_file)
        # Pydantic coerces string "true" → True for bool fields
        settings = mgr.load()
        # Note: pydantic v2 coerces string "true" → True
        assert settings.app.debug is True


class TestConfigManagerReload:
    """reload() should re-read the file from disk."""

    def test_reload_picks_up_changes(self, tmp_path: Path):
        """After writing a new config, reload() should return updated values."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump({"app": {"user_name": "First"}}), encoding="utf-8")

        mgr = ConfigManager(config_path=cfg_file)
        mgr.load()
        assert mgr.settings.app.user_name == "First"

        # Update the file
        cfg_file.write_text(yaml.dump({"app": {"user_name": "Second"}}), encoding="utf-8")
        mgr.reload()
        assert mgr.settings.app.user_name == "Second"
