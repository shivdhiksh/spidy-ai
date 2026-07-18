"""
Unit tests for spidy.plugins.sandbox
Tests that PluginSandbox correctly isolates exceptions and timeouts.
"""

from __future__ import annotations

import asyncio

import pytest

from spidy.plugins.sandbox import PluginSandbox


# ─── Minimal plugin stub ──────────────────────────────────────────────────────


class _SuccessPlugin:
    """Minimal plugin that setup/teardown succeed."""
    name = "success_plugin"

    async def setup(self, context) -> None:
        pass  # No-op

    async def teardown(self) -> None:
        pass


class _RaisingPlugin:
    """Plugin whose setup always raises."""
    name = "raising_plugin"

    async def setup(self, context) -> None:
        raise RuntimeError("Setup exploded!")

    async def teardown(self) -> None:
        raise ValueError("Teardown exploded!")


class _TeardownRaisingPlugin:
    """Plugin whose teardown raises but setup succeeds."""
    name = "teardown_raising_plugin"

    async def setup(self, context) -> None:
        pass

    async def teardown(self) -> None:
        raise RuntimeError("Teardown error!")


class _SlowSetupPlugin:
    """Plugin that sleeps too long in setup."""
    name = "slow_plugin"

    async def setup(self, context) -> None:
        await asyncio.sleep(999)  # will timeout

    async def teardown(self) -> None:
        pass


# ─── safe_setup ──────────────────────────────────────────────────────────────


class TestSafeSetup:
    @pytest.mark.asyncio
    async def test_success_returns_none(self) -> None:
        plugin = _SuccessPlugin()
        result = await PluginSandbox.safe_setup(plugin, context=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_error_string(self) -> None:
        plugin = _RaisingPlugin()
        result = await PluginSandbox.safe_setup(plugin, context=None)
        assert result is not None
        assert isinstance(result, str)
        assert "Setup exploded" in result or "raising_plugin" in result

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self) -> None:
        plugin = _RaisingPlugin()
        # Must not raise — sandbox catches everything
        try:
            result = await PluginSandbox.safe_setup(plugin, context=None)
        except Exception as exc:
            pytest.fail(f"PluginSandbox.safe_setup propagated: {exc}")

    @pytest.mark.asyncio
    async def test_timeout_returns_error_string(self) -> None:
        # Override the timeout for test speed
        plugin = _SlowSetupPlugin()
        # Monkeypatch: we can't easily set timeout=0.01 without modifying sandbox
        # so we test that the existing 30s timeout catches asyncio.TimeoutError gracefully
        # by mocking asyncio.wait_for to timeout immediately
        original_wait_for = asyncio.wait_for

        async def fast_timeout(coro, timeout):
            raise asyncio.TimeoutError

        import spidy.plugins.sandbox as sb_module
        original = sb_module.asyncio.wait_for
        sb_module.asyncio.wait_for = fast_timeout

        try:
            result = await PluginSandbox.safe_setup(plugin, context=None)
            assert result is not None
            assert "timed out" in result.lower()
        finally:
            sb_module.asyncio.wait_for = original


# ─── safe_teardown ────────────────────────────────────────────────────────────


class TestSafeTeardown:
    @pytest.mark.asyncio
    async def test_success_returns_none(self) -> None:
        plugin = _SuccessPlugin()
        result = await PluginSandbox.safe_teardown(plugin)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_error_string(self) -> None:
        plugin = _TeardownRaisingPlugin()
        result = await PluginSandbox.safe_teardown(plugin)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self) -> None:
        plugin = _RaisingPlugin()
        try:
            result = await PluginSandbox.safe_teardown(plugin)
        except Exception as exc:
            pytest.fail(f"PluginSandbox.safe_teardown propagated: {exc}")

    @pytest.mark.asyncio
    async def test_timeout_returns_error_string(self) -> None:
        import spidy.plugins.sandbox as sb_module

        async def fast_timeout(coro, timeout):
            raise asyncio.TimeoutError

        original = sb_module.asyncio.wait_for
        sb_module.asyncio.wait_for = fast_timeout

        try:
            plugin = _SlowSetupPlugin()
            result = await PluginSandbox.safe_teardown(plugin)
            assert result is not None
            assert "timed out" in result.lower()
        finally:
            sb_module.asyncio.wait_for = original


# ─── safe_call ───────────────────────────────────────────────────────────────


class TestSafeCall:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        async def good():
            pass

        ok, err = await PluginSandbox.safe_call("p", good(), label="test")
        assert ok is True
        assert err == ""

    @pytest.mark.asyncio
    async def test_exception(self) -> None:
        async def bad():
            raise RuntimeError("oops")

        ok, err = await PluginSandbox.safe_call("p", bad(), label="test")
        assert ok is False
        assert "oops" in err

    @pytest.mark.asyncio
    async def test_not_awaitable(self) -> None:
        ok, err = await PluginSandbox.safe_call("p", "not_a_coro", label="test")
        assert ok is False
        assert "not a coroutine" in err.lower()

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        async def slow():
            await asyncio.sleep(999)

        ok, err = await PluginSandbox.safe_call("p", slow(), label="test", timeout=0.01)
        assert ok is False
        assert "timed out" in err.lower()
