"""
tests/unit/test_nvidia_nim.py — NVIDIA NIM Integration Tests
=============================================================
Verifies that:

1. NvidiaClient exists with the correct provider_name.
2. health_check() → False when no key, True when key is set.
3. LLMClientFactory builds NvidiaClient for "nvidia" and "nim".
4. LLMRouter with [nvidia, ollama] tries NVIDIA first for every LLM intent
   (chat, factual, coding, studying).
5. NVIDIA failure (any reason) → Ollama fallback succeeds.
6. Missing NVIDIA_API_KEY → config defaults to Ollama only.
7. Local skill steps do NOT invoke the LLM at all.
8. The API key value never appears in log output.
9. Timeout on NVIDIA → falls back to Ollama.
10. LLMClientFactory.build_router returns LLMRouter when nvidia+ollama configured.

All tests are pure unit tests — no real network calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.error
import urllib.request
import unittest.mock as mock
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

_MSG = [LLMMessage(role="user", content="Hello, what is Python?")]

_FAKE_KEY = "nvapi-testkey1234567890"


def _ok(text: str = "NVIDIA response", provider: str = "nvidia") -> LLMResponse:
    return LLMResponse(text=text, model=f"{provider}-model", success=True)


def _fail(error: str = "NVIDIA NIM unavailable") -> LLMResponse:
    return LLMResponse.failure(error=error)


class _FakeClient(BaseLLMClient):
    """Controllable fake LLM client that tracks how many times it was called."""

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


# ─── 1. NvidiaClient instantiation ───────────────────────────────────────────

class TestNvidiaClientBasics:
    """NvidiaClient module-level smoke tests."""

    def test_provider_name(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        assert NvidiaClient.provider_name == "nvidia"

    def test_instantiates_without_key(self):
        """Should instantiate even without a key (health_check will be False)."""
        from spidy.llm.backends.nvidia import NvidiaClient
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_API_KEY", None)
            client = NvidiaClient(api_key="")
        assert client is not None

    def test_instantiates_with_constructor_key(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        client = NvidiaClient(api_key=_FAKE_KEY)
        # Key is present internally but never exposed
        assert client._api_key == _FAKE_KEY

    def test_default_model(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        client = NvidiaClient(api_key=_FAKE_KEY)
        assert client._model == "meta/llama-3.1-8b-instruct"

    def test_default_base_url(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        client = NvidiaClient(api_key=_FAKE_KEY)
        assert "integrate.api.nvidia.com" in client._base_url

    def test_reads_env_var(self):
        """NvidiaClient should pick up NVIDIA_API_KEY from environment."""
        from spidy.llm.backends.nvidia import NvidiaClient
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_KEY}):
            client = NvidiaClient()
        assert client._api_key == _FAKE_KEY

    def test_constructor_key_takes_precedence_over_env(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        other_key = "nvapi-other"
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_KEY}):
            client = NvidiaClient(api_key=other_key)
        assert client._api_key == other_key


# ─── 2. health_check ─────────────────────────────────────────────────────────

class TestNvidiaHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_no_key_returns_false(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_API_KEY", None)
            client = NvidiaClient(api_key="")
        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_key_returns_true(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        client = NvidiaClient(api_key=_FAKE_KEY)
        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_env_key_returns_true(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_KEY}):
            client = NvidiaClient()
        result = await client.health_check()
        assert result is True


# ─── 3. Factory ──────────────────────────────────────────────────────────────

class TestLLMClientFactoryNvidia:
    """LLMClientFactory should build NvidiaClient for "nvidia" and "nim"."""

    def _cfg(self, name: str):
        from spidy.config.manager import ReasoningConfig
        return ReasoningConfig(provider=name, api_key=_FAKE_KEY)

    def test_build_nvidia(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        client = LLMClientFactory.build(self._cfg("nvidia"))
        assert isinstance(client, NvidiaClient)

    def test_build_nim_alias(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        client = LLMClientFactory.build(self._cfg("nim"))
        assert isinstance(client, NvidiaClient)

    def test_build_router_nvidia_ollama_returns_router(self):
        from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
        cfg = MultiLLMConfig(
            providers=[
                LLMProviderConfig(name="nvidia", model="meta/llama-3.1-8b-instruct",
                                   api_key=_FAKE_KEY),
                LLMProviderConfig(name="ollama", model="llama3.2:3b"),
            ]
        )
        client = LLMClientFactory.build_router(cfg)
        assert isinstance(client, LLMRouter)

    def test_build_router_single_nvidia_returns_nvidia_client(self):
        from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
        from spidy.llm.backends.nvidia import NvidiaClient
        cfg = MultiLLMConfig(
            providers=[
                LLMProviderConfig(name="nvidia", model="meta/llama-3.1-8b-instruct",
                                   api_key=_FAKE_KEY),
            ]
        )
        client = LLMClientFactory.build_router(cfg)
        assert isinstance(client, NvidiaClient)


# ─── 4. Routing behaviour: NVIDIA first for all LLM intents ──────────────────

class TestNvidiaIsTriedFirst:
    """
    The router must attempt NVIDIA (providers[0]) before Ollama for every
    LLM request — regardless of the question category.
    """

    def _router(self, nvidia_response: LLMResponse, ollama_response: LLMResponse):
        nvidia = _FakeClient("nvidia", nvidia_response)
        ollama = _FakeClient("ollama", ollama_response)
        router = LLMRouter(providers=[nvidia, ollama])
        return router, nvidia, ollama

    @pytest.mark.asyncio
    async def test_chat_tries_nvidia_first(self):
        msgs = [LLMMessage(role="user", content="How are you?")]
        router, nvidia, ollama = self._router(_ok("Hi!"), _ok("Hi from ollama"))
        result = await router.complete(msgs)
        assert nvidia.call_count == 1
        assert ollama.call_count == 0
        assert result.text == "Hi!"

    @pytest.mark.asyncio
    async def test_factual_tries_nvidia_first(self):
        msgs = [LLMMessage(role="user", content="What is the speed of light?")]
        router, nvidia, ollama = self._router(_ok("299,792 km/s"), _ok("from ollama"))
        result = await router.complete(msgs)
        assert nvidia.call_count == 1
        assert ollama.call_count == 0
        assert "299,792" in result.text

    @pytest.mark.asyncio
    async def test_coding_tries_nvidia_first(self):
        msgs = [LLMMessage(role="user", content="Write a Python hello world.")]
        router, nvidia, ollama = self._router(_ok("print('hello')"), _ok("from ollama"))
        result = await router.complete(msgs)
        assert nvidia.call_count == 1
        assert ollama.call_count == 0

    @pytest.mark.asyncio
    async def test_studying_tries_nvidia_first(self):
        msgs = [LLMMessage(role="user", content="Explain photosynthesis.")]
        router, nvidia, ollama = self._router(
            _ok("Plants convert sunlight..."), _ok("from ollama")
        )
        result = await router.complete(msgs)
        assert nvidia.call_count == 1
        assert ollama.call_count == 0

    @pytest.mark.asyncio
    async def test_reasoning_tries_nvidia_first(self):
        msgs = [LLMMessage(role="user", content="Solve this logical puzzle.")]
        router, nvidia, ollama = self._router(_ok("The answer is A."), _ok("from ollama"))
        result = await router.complete(msgs)
        assert nvidia.call_count == 1
        assert ollama.call_count == 0

    @pytest.mark.asyncio
    async def test_debugging_tries_nvidia_first(self):
        msgs = [LLMMessage(role="user", content="Why does my code raise TypeError?")]
        router, nvidia, ollama = self._router(_ok("Because..."), _ok("from ollama"))
        result = await router.complete(msgs)
        assert nvidia.call_count == 1
        assert ollama.call_count == 0


# ─── 5. Fallback on NVIDIA failure ───────────────────────────────────────────

class TestNvidiaFallback:
    """When NVIDIA fails for any reason, Ollama must succeed."""

    def _router(self, nvidia_response: LLMResponse):
        nvidia = _FakeClient("nvidia", nvidia_response)
        ollama = _FakeClient("ollama", _ok("Ollama fallback response"))
        router = LLMRouter(providers=[nvidia, ollama])
        return router, nvidia, ollama

    @pytest.mark.asyncio
    async def test_nvidia_http_error_falls_back_to_ollama(self):
        router, nvidia, ollama = self._router(_fail("HTTP 503 Service Unavailable"))
        result = await router.complete(_MSG)
        assert result.success
        assert nvidia.call_count == 1
        assert ollama.call_count == 1
        assert "Ollama fallback" in result.text

    @pytest.mark.asyncio
    async def test_nvidia_timeout_falls_back_to_ollama(self):
        router, nvidia, ollama = self._router(_fail("Connection timed out"))
        result = await router.complete(_MSG)
        assert result.success
        assert ollama.call_count == 1

    @pytest.mark.asyncio
    async def test_nvidia_quota_error_falls_back_to_ollama(self):
        router, nvidia, ollama = self._router(_fail("HTTP 429: quota exceeded"))
        result = await router.complete(_MSG)
        assert result.success
        assert ollama.call_count == 1

    @pytest.mark.asyncio
    async def test_nvidia_network_failure_falls_back_to_ollama(self):
        router, nvidia, ollama = self._router(_fail("Network unreachable"))
        result = await router.complete(_MSG)
        assert result.success
        assert ollama.call_count == 1

    @pytest.mark.asyncio
    async def test_nvidia_unavailable_model_falls_back_to_ollama(self):
        router, nvidia, ollama = self._router(_fail("Model not available"))
        result = await router.complete(_MSG)
        assert result.success
        assert ollama.call_count == 1

    @pytest.mark.asyncio
    async def test_both_fail_returns_failure(self):
        nvidia = _FakeClient("nvidia", _fail("nvidia down"))
        ollama = _FakeClient("ollama", _fail("ollama down"))
        router = LLMRouter(providers=[nvidia, ollama])
        result = await router.complete(_MSG)
        assert not result.success

    @pytest.mark.asyncio
    async def test_fallback_result_is_successful(self):
        """The returned LLMResponse must be success=True when Ollama saves us."""
        router, nvidia, ollama = self._router(_fail("any error"))
        result = await router.complete(_MSG)
        assert result.success is True
        assert result.text != ""


# ─── 6. Missing NVIDIA_API_KEY → Ollama only ─────────────────────────────────

class TestMissingNvidiaKey:
    """When NVIDIA_API_KEY is not set, config must default to Ollama only."""

    def test_default_providers_no_key(self):
        """_default_llm_providers() returns only Ollama when key is absent."""
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_API_KEY", None)
            providers = _default_llm_providers()
        assert len(providers) == 1
        assert providers[0].name == "ollama"

    def test_default_providers_with_key(self):
        """_default_llm_providers() returns [nvidia, ollama] when key present."""
        from spidy.config.manager import _default_llm_providers
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_KEY}):
            providers = _default_llm_providers()
        assert len(providers) == 2
        assert providers[0].name == "nvidia"
        assert providers[1].name == "ollama"

    def test_multiLLMConfig_no_key_uses_ollama(self):
        from spidy.config.manager import MultiLLMConfig
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_API_KEY", None)
            cfg = MultiLLMConfig()
        assert len(cfg.providers) == 1
        assert cfg.providers[0].name == "ollama"

    def test_multiLLMConfig_with_key_uses_nvidia_first(self):
        from spidy.config.manager import MultiLLMConfig
        with patch.dict(os.environ, {"NVIDIA_API_KEY": _FAKE_KEY}):
            cfg = MultiLLMConfig()
        assert cfg.providers[0].name == "nvidia"
        assert cfg.providers[1].name == "ollama"

    @pytest.mark.asyncio
    async def test_router_no_key_uses_ollama(self):
        """End-to-end: config → factory → router without key uses Ollama."""
        from spidy.config.manager import MultiLLMConfig
        from spidy.llm.backends.ollama import OllamaClient
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_API_KEY", None)
            cfg = MultiLLMConfig()
        client = LLMClientFactory.build_router(cfg)
        # Single provider → direct OllamaClient (no router overhead)
        assert isinstance(client, OllamaClient)


# ─── 7. Local skill → LLM never called ───────────────────────────────────────

class TestLocalSkillNoLLMCall:
    """
    Deterministic skills (open_app, minimize_window, set_timer, etc.)
    must be handled by the skill layer without touching the LLM.
    """

    @pytest.mark.asyncio
    async def test_skill_step_does_not_call_llm(self):
        """
        ToolRouter routes 'skill' step_type directly to the SkillRegistry,
        completely bypassing the LLM client.
        """
        from spidy.brain.tool_router import ToolRouter
        from spidy.brain.types import Plan, PlanStep
        from spidy.core.event_bus import EventBus
        from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult
        from spidy.skills.registry import SkillRegistry

        bus = EventBus()
        registry = SkillRegistry()

        # Build a minimal real BaseSkill subclass so capabilities() works
        class FakeAppSkill(BaseSkill):
            name = "app_skill"
            description = "fake app skill"

            def capabilities(self):
                return [SkillCapability(action="open_app", description="Open an application")]

            async def execute(self, action: str, ctx: SkillContext) -> SkillResult:
                return SkillResult(success=True, message="Chrome opened.")

        registry.register(FakeAppSkill())

        llm_client = MagicMock(spec=BaseLLMClient)
        llm_client.complete = AsyncMock()

        router = ToolRouter(bus=bus, skill_registry=registry, llm_client=llm_client)

        plan = Plan(steps=[PlanStep(step_type="skill", action="open_app", params={})])
        await router.execute(plan, session_id="test")

        # LLM must NOT have been called at all
        llm_client.complete.assert_not_called()


    @pytest.mark.asyncio
    async def test_noop_step_does_not_call_llm(self):
        from spidy.brain.tool_router import ToolRouter
        from spidy.brain.types import Plan, PlanStep
        from spidy.core.event_bus import EventBus
        from spidy.skills.registry import SkillRegistry

        bus = EventBus()
        registry = SkillRegistry()
        llm_client = MagicMock(spec=BaseLLMClient)
        llm_client.complete = AsyncMock()

        router = ToolRouter(bus=bus, skill_registry=registry, llm_client=llm_client)
        plan = Plan(steps=[PlanStep(step_type="noop", action="noop", params={})])
        await router.execute(plan, session_id="test")
        llm_client.complete.assert_not_called()


# ─── 8. API key never appears in logs ────────────────────────────────────────

class TestApiKeyNotInLogs:
    """
    The NVIDIA API key value must never appear in any log record —
    not at DEBUG, INFO, WARNING, or ERROR level.
    """

    def _collect_log_records(self, fn):
        """Run fn() and return all log records emitted during it."""
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
        from spidy.llm.backends.nvidia import NvidiaClient
        secret_key = "nvapi-SUPER_SECRET_KEY_XYZ"
        records = self._collect_log_records(lambda: NvidiaClient(api_key=secret_key))
        for rec in records:
            msg = rec.getMessage()
            assert secret_key not in msg, (
                f"API key appeared in log record: {msg!r}"
            )

    def test_key_not_in_log_on_env_init(self):
        from spidy.llm.backends.nvidia import NvidiaClient
        secret_key = "nvapi-ENV_SECRET_KEY_ABC"
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with patch.dict(os.environ, {"NVIDIA_API_KEY": secret_key}):
                NvidiaClient()
        finally:
            root.removeHandler(handler)

        for rec in records:
            msg = rec.getMessage()
            assert secret_key not in msg, (
                f"API key appeared in log record: {msg!r}"
            )

    def test_key_not_in_config_providers_log(self):
        """_default_llm_providers() must not log the key value."""
        from spidy.config.manager import _default_llm_providers
        secret_key = "nvapi-CONFIG_LOG_SECRET"
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with patch.dict(os.environ, {"NVIDIA_API_KEY": secret_key}):
                _default_llm_providers()
        finally:
            root.removeHandler(handler)

        for rec in records:
            msg = rec.getMessage()
            assert secret_key not in msg, (
                f"API key appeared in log record: {msg!r}"
            )


# ─── 9. NvidiaClient complete() — mocked HTTP ─────────────────────────────────

class TestNvidiaClientComplete:
    """Verify NvidiaClient.complete() uses the inherited OpenAI HTTP code."""

    def _make_client(self) -> "NvidiaClient":
        from spidy.llm.backends.nvidia import NvidiaClient
        return NvidiaClient(api_key=_FAKE_KEY)

    @pytest.mark.asyncio
    async def test_complete_success(self):
        client = self._make_client()
        response_data = {
            "choices": [{"message": {"content": "Hello from NIM!"}, "finish_reason": "stop"}],
            "model": "meta/llama-3.1-8b-instruct",
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
        }
        with patch.object(client, "_post", return_value=response_data):
            result = await client.complete(_MSG)
        assert result.success
        assert result.text == "Hello from NIM!"
        assert result.usage.total_tokens == 13

    @pytest.mark.asyncio
    async def test_complete_http_error_returns_failure(self):
        client = self._make_client()
        err = urllib.error.HTTPError(
            url="x", code=503, msg="Service Unavailable", hdrs=None, fp=None
        )
        with patch.object(client, "_post", side_effect=err):
            result = await client.complete(_MSG)
        assert not result.success
        assert "503" in result.error_message

    @pytest.mark.asyncio
    async def test_complete_timeout_returns_failure(self):
        client = self._make_client()
        with patch.object(client, "_post", side_effect=TimeoutError("timed out")):
            result = await client.complete(_MSG)
        assert not result.success

    @pytest.mark.asyncio
    async def test_complete_url_error_returns_failure(self):
        client = self._make_client()
        with patch.object(
            client, "_post", side_effect=urllib.error.URLError("name or service not known")
        ):
            result = await client.complete(_MSG)
        assert not result.success

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        client = self._make_client()
        chunks = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo NIM"}}]},
        ]
        with patch.object(client, "_post_stream", return_value=chunks):
            tokens = [tok async for tok in client.stream(_MSG)]
        assert "".join(tokens) == "Hello NIM"

    @pytest.mark.asyncio
    async def test_stream_error_yields_nothing(self):
        client = self._make_client()
        with patch.object(
            client, "_post_stream", side_effect=urllib.error.URLError("refused")
        ):
            tokens = [tok async for tok in client.stream(_MSG)]
        assert tokens == []


# ─── 10. Router failure_threshold & skip behaviour ───────────────────────────

class TestRouterThresholdWithNvidia:

    @pytest.mark.asyncio
    async def test_nvidia_skipped_after_threshold(self):
        """After 3 failures, NVIDIA is skipped and Ollama serves all requests."""
        nvidia = _FakeClient("nvidia", _fail("always fails"))
        ollama = _FakeClient("ollama", _ok("from ollama"))
        router = LLMRouter(providers=[nvidia, ollama], failure_threshold=3)

        # Three failures exhaust the threshold
        for _ in range(3):
            result = await router.complete(_MSG)
            assert result.success  # ollama saved it each time

        assert nvidia.call_count == 3
        nvidia_before = nvidia.call_count

        # 4th call: nvidia skipped entirely
        result = await router.complete(_MSG)
        assert result.success
        assert nvidia.call_count == nvidia_before  # not incremented

    @pytest.mark.asyncio
    async def test_reset_failures_re_enables_nvidia(self):
        nvidia = _FakeClient("nvidia", _fail())
        ollama = _FakeClient("ollama", _ok())
        router = LLMRouter(providers=[nvidia, ollama], failure_threshold=1)

        # Exhaust threshold
        await router.complete(_MSG)
        assert router._failure_counts[0] >= 1

        # Reset
        router.reset_failures()
        assert router._failure_counts[0] == 0

        # Make nvidia succeed now
        nvidia._response = _ok("nvidia back")
        result = await router.complete(_MSG)
        assert result.text == "nvidia back"
