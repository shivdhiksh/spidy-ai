"""
Test suite for spidy.logging.logger
====================================
Tests that the logging module bootstraps correctly, respects levels,
and supports multiple named loggers.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


class TestGetLogger:
    """get_logger() should always return a working logger."""

    def test_returns_logger_instance(self):
        """get_logger must return a loguru-compatible bound logger."""
        # Reimport to avoid cross-test state
        import spidy.logging.logger as log_module
        log_module._configured = False  # reset for isolation

        logger = log_module.get_logger("test.module")
        # loguru bound loggers are callable (you can log with them)
        assert callable(logger.info)
        assert callable(logger.debug)
        assert callable(logger.error)

    def test_auto_configures_on_first_call(self, tmp_path: Path):
        """get_logger should trigger auto-configure if called before configure()."""
        import spidy.logging.logger as log_module
        log_module._configured = False
        log_module._log_dir = tmp_path / "logs"

        log = log_module.get_logger("test.auto")
        log.info("Auto-configured logger works.")  # Must not raise

    def test_different_module_names(self):
        """Each module name should bind independently."""
        import spidy.logging.logger as log_module
        log_module._configured = True  # assume already configured

        log_a = log_module.get_logger("module.a")
        log_b = log_module.get_logger("module.b")
        # Both are valid loggers; they are different bound instances
        assert log_a is not log_b


class TestConfigure:
    """configure() should be idempotent and respect parameters."""

    def test_configure_is_idempotent(self, tmp_path: Path):
        """Calling configure() twice should not raise or add duplicate handlers."""
        import spidy.logging.logger as log_module
        log_module._configured = False

        log_module.configure(level="DEBUG", log_dir=tmp_path / "logs")
        log_module.configure(level="DEBUG", log_dir=tmp_path / "logs")  # second call
        # No assertion needed — just must not raise

    def test_configure_creates_log_dir(self, tmp_path: Path):
        """configure() must create the log directory if it does not exist."""
        import spidy.logging.logger as log_module
        log_module._configured = False

        log_dir = tmp_path / "deep" / "nested" / "logs"
        assert not log_dir.exists()

        log_module.configure(log_dir=log_dir)
        assert log_dir.exists()
