"""
tests/unit/test_openrouter_provider.py — OpenRouter Provider Integration Tests
===============================================================================
Acceptance criteria for the Ollama-removal / OpenRouter-addition milestone:

 1. NVIDIA is the primary provider in default config.
 2. OpenRouter is the fallback provider in default config.
 3. Ollama is never selected by default (with either or both cloud keys present).
 4. Ollama is never instantiated by LLMClientFactory.build_router().
 5. Ollama health check is NOT called at startup (no localhost:11434 connection).
 6. NVIDIA failure → OpenRouter fallback succeeds.
 7. Both providers fail → LLMResponse.failure().
 8. OPENROUTER_API_KEY never appears in logs.
 9. SPIDY can start without Ollama (no localhost:11434 in logs).
10. Local skills do not call the LLM.
11. Existing NVIDIA code path is unaffected.
12. LLMRouter backward compatibility: works with any BaseLLMClient list.

All tests are pure unit tests — no real network calls.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.llm.client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMMessage,
    LLMResponse,
)
from spidy.llm.router import LLMRouter


# ─── Helpers ──────────────────────────────────────────────────────────────────

_MSG = [LLMMessage(role="user", content="Hello, what is 2+2?")]
_FAKE_NVIDIA_KEY = "nvapi-testNVIDIAkey1234567890"
_FAKE_OR_KEY = "sk-or-testOPENROUTERkey1234567890"


def _ok(text: str = "response", provider: str = "provider") -> LLMResponse:
    return LLMResponse(text=text, model=f"{provider}-model", success=True)


def _fail(error: str = "error") -> LLMResponse:
    return LLMResponse.failure(error=error)


class _FakeClient(BaseLLMClient):
    """Controllable fake LLM client."""

    def __init__(self, name: str, response: LLMResponse) -> None:
        self.provider_name = name
        self._response = response
        self.call_count = 0

    async def complete(self, messages, temperature=None, max_tokens=None):
        self.call_count += 1
        return self._response

    async def stream(self, messages, temperature=None, max_tokens=None):
        return
        yield  # make it a generator

    async def close(self):
        pass


# ─── 1 & 2. Default provider priority: nvidia → openrouter ───────────────────

class TestDefaultProviderChain:
    """With NVIDIA + OPENROUTER keys, chain must be [nvidia, openrouter]."""

    def test_nvidia_is_primary(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": _FAKE_NVIDIA_KEY,
            "OPENROUTER_API_KEY": _FAKE_OR_KEY,
        }):
            providers = _default_llm_providers()
        assert providers[0].name == "nvidia"

    def test_openrouter_is_fallback(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": _FAKE_NVIDIA_KEY,
            "OPENROUTER_API_KEY": _FAKE_OR_KEY,
        }):
            providers = _default_llm_providers()
        assert len(providers) == 2
        assert providers[1].name == "openrouter"

    def test_only_openrouter_key_returns_openrouter_only(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": _FAKE_OR_KEY,
        }, clear=True):
            os.environ.pop("NVIDIA_API_KEY", None)
            providers = _default_llm_providers()
        assert len(providers) == 1
        assert providers[0].name == "openrouter"

    def test_neither_key_returns_nvidia_only(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVIDIA_API_KEY", None)
            os.environ.pop("OPENROUTER_API_KEY", None)
            providers = _default_llm_providers()
        assert len(providers) == 1
        assert providers[0].name == "nvidia"  # graceful fallback, health_check returns False


# ─── 3. Ollama never selected by default ─────────────────────────────────────

class TestOllamaNotSelectedByDefault:
    """Ollama must not appear in any default provider list."""

    def test_no_keys_no_ollama(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVIDIA_API_KEY", None)
            os.environ.pop("OPENROUTER_API_KEY", None)
            providers = _default_llm_providers()
        names = [p.name for p in providers]
        assert "ollama" not in names

    def test_nvidia_key_no_ollama(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_NVIDIA_KEY}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            providers = _default_llm_providers()
        names = [p.name for p in providers]
        assert "ollama" not in names

    def test_both_keys_no_ollama(self):
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": _FAKE_NVIDIA_KEY,
            "OPENROUTER_API_KEY": _FAKE_OR_KEY,
        }):
            providers = _default_llm_providers()
        names = [p.name for p in providers]
        assert "ollama" not in names


# ─── 4. Ollama never instantiated by build_router() ──────────────────────────

class TestOllamaNotInstantiated:
    """LLMClientFactory.build_router() must never produce an OllamaClient."""

    def test_build_router_with_nvidia_key_never_creates_ollama(self):
        from spidy.config.manager import MultiLLMConfig
        from spidy.llm.backends.ollama import OllamaClient

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": _FAKE_NVIDIA_KEY,
            "OPENROUTER_API_KEY": _FAKE_OR_KEY,
        }):
            cfg = MultiLLMConfig()
        client = LLMClientFactory.build_router(cfg)
        # Must not be an OllamaClient — it should be an LLMRouter or NvidiaClient
        assert not isinstance(client, OllamaClient)

    def test_build_router_direct_ollama_config_raises(self):
        """Passing ollama explicitly to build_router should raise ValueError."""
        from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
        with pytest.raises(ValueError, match="deprecated|No LLM providers"):
            cfg = MultiLLMConfig(
                providers=[LLMProviderConfig(name="ollama", model="llama3.2:3b")]
            )
            # After deprecation filter, no providers remain → ValueError
            LLMClientFactory.build_router(cfg)


# ─── 5. No localhost:11434 call at startup ────────────────────────────────────

class TestNoOllamaStartupCall:
    """Startup must not connect to localhost:11434."""

    @pytest.mark.asyncio
    async def test_log_llm_provider_chain_does_not_touch_localhost(self):
        """
        _log_llm_provider_chain() must not open any connection to localhost:11434.
        We verify by ensuring urllib.request.urlopen is never called with that URL.
        """
        import urllib.request
        from spidy.config.manager import (
            ConfigManager, LLMProviderConfig, MultiLLMConfig,
        )
        from spidy.core.app import SpidyCore

        # Patch the entire ConfigManager.load() to return a controlled settings object
        mock_settings = MagicMock()
        mock_settings.llm.providers = [
            LLMProviderConfig(name="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b", enabled=True),
            LLMProviderConfig(name="openrouter", model="nvidia/nemotron-3-ultra-550b-a55b:free", enabled=True),
        ]

        app = SpidyCore.__new__(SpidyCore)
        app._settings = mock_settings

        opened_urls: list[str] = []

        def mock_urlopen(url, *args, **kwargs):
            opened_urls.append(str(url))
            raise Exception("Should not open any URL during startup log")

        with patch.object(urllib.request, "urlopen", side_effect=mock_urlopen):
            await app._log_llm_provider_chain()

        # No URL should have been opened
        localhost_calls = [u for u in opened_urls if "localhost" in u or "11434" in u]
        assert localhost_calls == [], (
            f"_log_llm_provider_chain() made unexpected network calls: {localhost_calls}"
        )


# ─── 6. NVIDIA failure → OpenRouter fallback ─────────────────────────────────

class TestNvidiaFailoverToOpenRouter:
    """When NVIDIA fails, OpenRouter must be used."""

    @pytest.mark.asyncio
    async def test_nvidia_http_error_falls_back_to_openrouter(self):
        nvidia = _FakeClient("nvidia", _fail("HTTP 503"))
        openrouter = _FakeClient("openrouter", _ok("OpenRouter response", "openrouter"))
        router = LLMRouter(providers=[nvidia, openrouter])
        result = await router.complete(_MSG)
        assert result.success
        assert nvidia.call_count == 1
        assert openrouter.call_count == 1
        assert "OpenRouter response" in result.text

    @pytest.mark.asyncio
    async def test_nvidia_timeout_falls_back_to_openrouter(self):
        nvidia = _FakeClient("nvidia", _fail("Connection timed out"))
        openrouter = _FakeClient("openrouter", _ok("From OpenRouter"))
        router = LLMRouter(providers=[nvidia, openrouter])
        result = await router.complete(_MSG)
        assert result.success
        assert openrouter.call_count == 1

    @pytest.mark.asyncio
    async def test_nvidia_quota_falls_back_to_openrouter(self):
        nvidia = _FakeClient("nvidia", _fail("HTTP 429: quota exceeded"))
        openrouter = _FakeClient("openrouter", _ok("Quota relief from OpenRouter"))
        router = LLMRouter(providers=[nvidia, openrouter])
        result = await router.complete(_MSG)
        assert result.success

    @pytest.mark.asyncio
    async def test_nvidia_serves_first_when_healthy(self):
        nvidia = _FakeClient("nvidia", _ok("NVIDIA answer"))
        openrouter = _FakeClient("openrouter", _ok("OpenRouter answer"))
        router = LLMRouter(providers=[nvidia, openrouter])
        result = await router.complete(_MSG)
        assert nvidia.call_count == 1
        assert openrouter.call_count == 0
        assert result.text == "NVIDIA answer"


# ─── 7. Both fail → LLMResponse.failure() ────────────────────────────────────

class TestBothProvidersFail:
    @pytest.mark.asyncio
    async def test_nvidia_and_openrouter_both_fail(self):
        nvidia = _FakeClient("nvidia", _fail("nvidia down"))
        openrouter = _FakeClient("openrouter", _fail("openrouter down"))
        router = LLMRouter(providers=[nvidia, openrouter])
        result = await router.complete(_MSG)
        assert not result.success
        assert result.text == ""


# ─── 8. OPENROUTER_API_KEY never in logs ─────────────────────────────────────

class TestOpenRouterKeyNotInLogs:
    """The OPENROUTER_API_KEY value must never appear in any log record."""

    def _collect_logs(self, fn):
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            fn()
        finally:
            root.removeHandler(handler)
        return records

    def test_key_not_in_log_on_instantiation(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        secret = "sk-or-SUPER_SECRET_XYZ_12345"
        records = self._collect_logs(lambda: OpenRouterClient(api_key=secret))
        for rec in records:
            msg = rec.getMessage()
            assert secret not in msg, f"API key leaked in log: {msg!r}"

    def test_key_not_in_log_from_env(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        secret = "sk-or-ENV_SECRET_ABC_67890"
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": secret}):
                OpenRouterClient()
        finally:
            root.removeHandler(handler)

        for rec in records:
            msg = rec.getMessage()
            assert secret not in msg, f"API key leaked in log: {msg!r}"

    def test_key_not_in_default_providers_log(self):
        from spidy.config.manager import _default_llm_providers
        secret = "sk-or-CONFIG_LOG_TEST_99999"
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": secret}, clear=True):
                os.environ.pop("NVIDIA_API_KEY", None)
                _default_llm_providers()
        finally:
            root.removeHandler(handler)

        for rec in records:
            msg = rec.getMessage()
            assert secret not in msg, f"API key leaked in log: {msg!r}"


# ─── 9. SPIDY starts without Ollama installed ─────────────────────────────────

class TestSpidyStartsWithoutOllama:
    """Starting SPIDY must not require Ollama to be installed or running."""

    def test_openrouter_client_importable_without_ollama(self):
        """OpenRouterClient should import successfully regardless of Ollama."""
        from spidy.llm.backends.openrouter import OpenRouterClient
        assert OpenRouterClient.provider_name == "openrouter"

    def test_default_config_importable_without_ollama(self):
        """MultiLLMConfig with NVIDIA key should not touch Ollama."""
        from spidy.config.manager import MultiLLMConfig
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_NVIDIA_KEY}):
            cfg = MultiLLMConfig()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names


# ─── 10. Local skills do not call the LLM ─────────────────────────────────────

class TestLocalSkillNoLLMCallOpenRouter:
    @pytest.mark.asyncio
    async def test_skill_step_does_not_call_openrouter(self):
        from spidy.brain.tool_router import ToolRouter
        from spidy.brain.types import Plan, PlanStep
        from spidy.core.event_bus import EventBus
        from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult
        from spidy.skills.registry import SkillRegistry

        bus = EventBus()
        registry = SkillRegistry()

        class FakeSkill(BaseSkill):
            name = "test_skill"
            description = "test"
            def capabilities(self):
                return [SkillCapability(action="test_action", description="Test")]
            async def execute(self, action: str, ctx: SkillContext) -> SkillResult:
                return SkillResult(success=True, message="done")

        registry.register(FakeSkill())
        llm_client = MagicMock(spec=BaseLLMClient)
        llm_client.complete = AsyncMock()
        router = ToolRouter(bus=bus, skill_registry=registry, llm_client=llm_client)
        plan = Plan(steps=[PlanStep(step_type="skill", action="test_action", params={})])
        await router.execute(plan, session_id="test")
        llm_client.complete.assert_not_called()


# ─── 11. OpenRouterClient basics ──────────────────────────────────────────────

class TestOpenRouterClientBasics:
    """Smoke tests for the OpenRouterClient class itself."""

    def test_provider_name(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        assert OpenRouterClient.provider_name == "openrouter"

    def test_default_base_url(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="sk-or-test")
        assert "openrouter.ai" in client._base_url

    def test_default_model(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="sk-or-test")
        assert "nemotron" in client._model.lower()

    def test_reads_env_var(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": _FAKE_OR_KEY}):
            client = OpenRouterClient()
        assert client._api_key == _FAKE_OR_KEY

    def test_constructor_key_takes_precedence(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env"}):
            client = OpenRouterClient(api_key="sk-or-constructor")
        assert client._api_key == "sk-or-constructor"

    @pytest.mark.asyncio
    async def test_health_check_no_key_returns_false(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            client = OpenRouterClient(api_key="")
        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_key_returns_true(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key=_FAKE_OR_KEY)
        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_complete_success(self):
        import urllib.error
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key=_FAKE_OR_KEY)
        response_data = {
            "choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        with patch.object(client, "_post", return_value=response_data):
            result = await client.complete(_MSG)
        assert result.success
        assert result.text == "4"
        assert result.usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_complete_http_error_returns_failure(self):
        import urllib.error
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key=_FAKE_OR_KEY)
        err = urllib.error.HTTPError(url="x", code=502, msg="Bad Gateway", hdrs=None, fp=None)
        with patch.object(client, "_post", side_effect=err):
            result = await client.complete(_MSG)
        assert not result.success
        assert "502" in result.error_message

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key=_FAKE_OR_KEY)
        chunks = [
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
        ]
        with patch.object(client, "_post_stream", return_value=chunks):
            tokens = [tok async for tok in client.stream(_MSG)]
        assert "".join(tokens) == "Hello"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        from spidy.llm.backends.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key=_FAKE_OR_KEY)
        await client.close()  # Must not raise


# ─── 12. LLMRouter backward compatibility ─────────────────────────────────────

class TestLLMRouterBackwardCompat:
    """LLMRouter must work with any list of BaseLLMClient instances."""

    @pytest.mark.asyncio
    async def test_router_works_with_openrouter_only(self):
        openrouter = _FakeClient("openrouter", _ok("solo openrouter"))
        router = LLMRouter(providers=[openrouter])
        result = await router.complete(_MSG)
        assert result.success
        assert result.text == "solo openrouter"

    @pytest.mark.asyncio
    async def test_router_works_with_mixed_providers(self):
        nvidia = _FakeClient("nvidia", _ok("nvidia"))
        openrouter = _FakeClient("openrouter", _ok("openrouter"))
        router = LLMRouter(providers=[nvidia, openrouter])
        result = await router.complete(_MSG)
        assert result.text == "nvidia"  # primary wins

    @pytest.mark.asyncio
    async def test_router_reset_failures(self):
        openrouter = _FakeClient("openrouter", _fail("down"))
        router = LLMRouter(providers=[openrouter], failure_threshold=1)
        await router.complete(_MSG)
        assert router._failure_counts[0] >= 1
        router.reset_failures()
        assert router._failure_counts[0] == 0
