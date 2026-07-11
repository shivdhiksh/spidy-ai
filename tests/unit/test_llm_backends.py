"""
Tests for Milestone 4 — LLM Multi-Backend Layer
================================================
All tests are pure unit tests. No real network calls.
Backends are tested by mocking urllib at the call site.
"""

from __future__ import annotations

import asyncio
import json
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.llm.client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMMessage,
    LLMResponse,
    LLMUsage,
)
from spidy.llm.router import LLMRouter


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_MSG = [LLMMessage(role="user", content="Hello")]
_SYSTEM_MSG = [
    LLMMessage(role="system", content="You are Spidy."),
    LLMMessage(role="user", content="Hello"),
]


def _ok_response(text: str = "Hi there!", model: str = "test") -> LLMResponse:
    return LLMResponse(text=text, model=model, success=True)


def _fail_response(error: str = "oops") -> LLMResponse:
    return LLMResponse.failure(error=error)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LLMMessage, LLMUsage, LLMResponse
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMDataTypes:
    def test_message_is_frozen(self):
        msg = LLMMessage(role="user", content="hello")
        with pytest.raises((AttributeError, TypeError)):
            msg.role = "assistant"  # type: ignore

    def test_response_failure_factory(self):
        r = LLMResponse.failure("network error", model="gpt-4")
        assert not r.success
        assert r.finish_reason == "error"
        assert "network error" in r.error_message
        assert r.model == "gpt-4"
        assert r.text == ""

    def test_response_success_defaults(self):
        r = LLMResponse(text="hello")
        assert r.success
        assert r.finish_reason == "stop"

    def test_usage_defaults(self):
        u = LLMUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. OpenAIClient
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIClient:
    def _make_client(self, api_key: str = "sk-test"):
        from spidy.llm.backends.openai import OpenAIClient
        return OpenAIClient(api_key=api_key, model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_complete_success(self):
        client = self._make_client()
        response_data = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        with patch.object(client, "_post", return_value=response_data):
            result = await client.complete(_MSG)
        assert result.success
        assert result.text == "Hello!"
        assert result.model == "gpt-4o-mini"
        assert result.usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_complete_http_error(self):
        import urllib.error
        client = self._make_client()
        err = urllib.error.HTTPError(url="x", code=401, msg="Unauthorized", hdrs=None, fp=None)
        with patch.object(client, "_post", side_effect=err):
            result = await client.complete(_MSG)
        assert not result.success
        assert "401" in result.error_message

    @pytest.mark.asyncio
    async def test_complete_url_error(self):
        import urllib.error
        client = self._make_client()
        with patch.object(client, "_post", side_effect=urllib.error.URLError("refused")):
            result = await client.complete(_MSG)
        assert not result.success
        assert "unreachable" in result.error_message.lower() or "refused" in result.error_message

    @pytest.mark.asyncio
    async def test_complete_unexpected_exception(self):
        client = self._make_client()
        with patch.object(client, "_post", side_effect=RuntimeError("boom")):
            result = await client.complete(_MSG)
        assert not result.success

    @pytest.mark.asyncio
    async def test_provider_name(self):
        from spidy.llm.backends.openai import OpenAIClient
        assert OpenAIClient.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self):
        from spidy.llm.backends.openai import OpenAIClient
        client = OpenAIClient(api_key="")
        healthy = await client.health_check()
        assert not healthy

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        client = self._make_client()
        await client.close()  # Must not raise

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        client = self._make_client()
        chunks = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
        ]
        with patch.object(client, "_post_stream", return_value=chunks):
            tokens = []
            async for tok in client.stream(_MSG):
                tokens.append(tok)
        assert "".join(tokens) == "Hello"

    @pytest.mark.asyncio
    async def test_stream_error_yields_nothing(self):
        import urllib.error
        client = self._make_client()
        with patch.object(client, "_post_stream", side_effect=urllib.error.URLError("x")):
            tokens = []
            async for tok in client.stream(_MSG):
                tokens.append(tok)
        assert tokens == []


# ──────────────────────────────────────────────────────────────────────────────
# 3. ClaudeClient
# ──────────────────────────────────────────────────────────────────────────────

class TestClaudeClient:
    def _make_client(self):
        from spidy.llm.backends.claude import ClaudeClient
        return ClaudeClient(api_key="test-key")

    @pytest.mark.asyncio
    async def test_complete_success(self):
        client = self._make_client()
        response_data = {
            "content": [{"type": "text", "text": "Hi from Claude!"}],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with patch.object(client, "_post", return_value=response_data):
            result = await client.complete(_MSG)
        assert result.success
        assert result.text == "Hi from Claude!"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_system_message_extraction(self):
        from spidy.llm.backends.claude import ClaudeClient
        system, chat = ClaudeClient._split_system(_SYSTEM_MSG)
        assert system == "You are Spidy."
        assert len(chat) == 1
        assert chat[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_consecutive_same_role_merging(self):
        from spidy.llm.backends.claude import ClaudeClient
        messages = [
            LLMMessage(role="user", content="First"),
            LLMMessage(role="user", content="Second"),
        ]
        _, chat = ClaudeClient._split_system(messages)
        assert len(chat) == 1
        assert "First" in chat[0]["content"]
        assert "Second" in chat[0]["content"]

    @pytest.mark.asyncio
    async def test_http_error_returns_failure(self):
        import urllib.error
        client = self._make_client()
        err = urllib.error.HTTPError(url="x", code=529, msg="Overloaded", hdrs=None, fp=None)
        with patch.object(client, "_post", side_effect=err):
            result = await client.complete(_MSG)
        assert not result.success
        assert "529" in result.error_message

    @pytest.mark.asyncio
    async def test_provider_name(self):
        from spidy.llm.backends.claude import ClaudeClient
        assert ClaudeClient.provider_name == "claude"

    @pytest.mark.asyncio
    async def test_health_check_has_key(self):
        client = self._make_client()
        assert await client.health_check()

    @pytest.mark.asyncio
    async def test_health_check_no_key(self):
        from spidy.llm.backends.claude import ClaudeClient
        client = ClaudeClient(api_key="")
        assert not await client.health_check()


# ──────────────────────────────────────────────────────────────────────────────
# 4. GeminiClient
# ──────────────────────────────────────────────────────────────────────────────

class TestGeminiClient:
    def _make_client(self):
        from spidy.llm.backends.gemini import GeminiClient
        return GeminiClient(api_key="test-key")

    @pytest.mark.asyncio
    async def test_complete_success(self):
        client = self._make_client()
        response_data = {
            "candidates": [{
                "content": {"parts": [{"text": "Hi from Gemini!"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4},
        }
        with patch.object(client, "_post", return_value=response_data):
            result = await client.complete(_MSG)
        assert result.success
        assert result.text == "Hi from Gemini!"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 4

    @pytest.mark.asyncio
    async def test_message_conversion(self):
        from spidy.llm.backends.gemini import GeminiClient
        system, contents = GeminiClient._convert_messages(_SYSTEM_MSG)
        assert system == "You are Spidy."
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_assistant_maps_to_model_role(self):
        from spidy.llm.backends.gemini import GeminiClient
        msgs = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi!"),
        ]
        _, contents = GeminiClient._convert_messages(msgs)
        assert contents[1]["role"] == "model"

    @pytest.mark.asyncio
    async def test_http_error_returns_failure(self):
        import urllib.error
        client = self._make_client()
        err = urllib.error.HTTPError(url="x", code=403, msg="Forbidden", hdrs=None, fp=None)
        with patch.object(client, "_post", side_effect=err):
            result = await client.complete(_MSG)
        assert not result.success
        assert "403" in result.error_message

    @pytest.mark.asyncio
    async def test_provider_name(self):
        from spidy.llm.backends.gemini import GeminiClient
        assert GeminiClient.provider_name == "gemini"

    @pytest.mark.asyncio
    async def test_url_includes_api_key(self):
        client = self._make_client()
        url = client._url("/models/gemini:generate")
        assert "key=test-key" in url

    @pytest.mark.asyncio
    async def test_url_no_key(self):
        from spidy.llm.backends.gemini import GeminiClient
        client = GeminiClient(api_key="")
        url = client._url("/models/gemini:generate")
        assert "key=" not in url


# ──────────────────────────────────────────────────────────────────────────────
# 5. LLMRouter
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMRouter:
    def _fake_client(self, name: str, response: LLMResponse) -> BaseLLMClient:
        """Create a fake BaseLLMClient that always returns a fixed response."""
        class FakeClient(BaseLLMClient):
            provider_name = name

            async def complete(self, messages, temperature=None, max_tokens=None):
                return response

            async def stream(self, messages, temperature=None, max_tokens=None):
                return; yield

            async def close(self):
                pass

        return FakeClient()

    @pytest.mark.asyncio
    async def test_router_uses_primary(self):
        primary = self._fake_client("primary", _ok_response("from primary"))
        fallback = self._fake_client("fallback", _ok_response("from fallback"))
        router = LLMRouter(providers=[primary, fallback])
        result = await router.complete(_MSG)
        assert result.text == "from primary"

    @pytest.mark.asyncio
    async def test_router_falls_back_on_failure(self):
        primary = self._fake_client("primary", _fail_response("primary failed"))
        fallback = self._fake_client("fallback", _ok_response("from fallback"))
        router = LLMRouter(providers=[primary, fallback])
        result = await router.complete(_MSG)
        assert result.text == "from fallback"
        assert result.success

    @pytest.mark.asyncio
    async def test_router_returns_failure_if_all_fail(self):
        p1 = self._fake_client("p1", _fail_response("p1 fail"))
        p2 = self._fake_client("p2", _fail_response("p2 fail"))
        router = LLMRouter(providers=[p1, p2])
        result = await router.complete(_MSG)
        assert not result.success

    @pytest.mark.asyncio
    async def test_router_no_auto_fallback(self):
        p1 = self._fake_client("p1", _fail_response("p1 fail"))
        p2 = self._fake_client("p2", _ok_response("p2 ok"))
        router = LLMRouter(providers=[p1, p2], auto_fallback=False)
        result = await router.complete(_MSG)
        assert not result.success

    @pytest.mark.asyncio
    async def test_router_skips_threshold_exceeded(self):
        p1 = self._fake_client("p1", _fail_response("fail"))
        p2 = self._fake_client("p2", _ok_response("ok"))
        router = LLMRouter(providers=[p1, p2], failure_threshold=1)
        # First call: p1 fails → p2 succeeds, p1 count = 1
        r1 = await router.complete(_MSG)
        assert r1.success
        # Second call: p1 skipped (count=1 >= threshold=1) → p2
        r2 = await router.complete(_MSG)
        assert r2.success

    @pytest.mark.asyncio
    async def test_reset_failures(self):
        p1 = self._fake_client("p1", _fail_response())
        router = LLMRouter(providers=[p1], failure_threshold=1)
        await router.complete(_MSG)  # p1 count = 1
        router.reset_failures()
        assert router._failure_counts[0] == 0

    @pytest.mark.asyncio
    async def test_router_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            LLMRouter(providers=[])

    @pytest.mark.asyncio
    async def test_router_publishes_failover_event(self):
        from spidy.core.event_bus import EventBus
        from spidy.llm.events import LLMProviderSwitchedEvent

        bus = EventBus()
        events = []
        bus.subscribe("llm.provider_switched", lambda e: events.append(e))

        p1 = self._fake_client("p1", _fail_response())
        p2 = self._fake_client("p2", _ok_response())
        router = LLMRouter(providers=[p1, p2], bus=bus)
        await router.complete(_MSG)

        # Give async tasks time to complete
        await asyncio.sleep(0)
        assert any(isinstance(e, LLMProviderSwitchedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_provider_count(self):
        p1 = self._fake_client("p1", _ok_response())
        p2 = self._fake_client("p2", _ok_response())
        router = LLMRouter(providers=[p1, p2])
        assert router.provider_count == 2

    @pytest.mark.asyncio
    async def test_primary_provider_property(self):
        p1 = self._fake_client("p1", _ok_response())
        router = LLMRouter(providers=[p1])
        assert router.primary_provider is p1

    @pytest.mark.asyncio
    async def test_close_calls_all_providers(self):
        closed: list[str] = []

        class TrackingClient(BaseLLMClient):
            provider_name = "tracking"
            def __init__(self, tag: str):
                self._tag = tag
            async def complete(self, m, temperature=None, max_tokens=None):
                return _ok_response()
            async def stream(self, m, temperature=None, max_tokens=None):
                return; yield
            async def close(self):
                closed.append(self._tag)

        router = LLMRouter(providers=[TrackingClient("a"), TrackingClient("b")])
        await router.close()
        assert "a" in closed and "b" in closed


# ──────────────────────────────────────────────────────────────────────────────
# 6. LLMClientFactory
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMClientFactory:
    def _make_reasoning_config(self, provider: str = "ollama"):
        from spidy.config.manager import ReasoningConfig
        return ReasoningConfig(provider=provider, model="test-model")

    def test_build_ollama(self):
        from spidy.llm.backends.ollama import OllamaClient
        cfg = self._make_reasoning_config("ollama")
        client = LLMClientFactory.build(cfg)
        assert isinstance(client, OllamaClient)

    def test_build_openai(self):
        from spidy.llm.backends.openai import OpenAIClient
        cfg = self._make_reasoning_config("openai")
        client = LLMClientFactory.build(cfg)
        assert isinstance(client, OpenAIClient)

    def test_build_claude(self):
        from spidy.llm.backends.claude import ClaudeClient
        cfg = self._make_reasoning_config("claude")
        client = LLMClientFactory.build(cfg)
        assert isinstance(client, ClaudeClient)

    def test_build_anthropic_alias(self):
        from spidy.llm.backends.claude import ClaudeClient
        cfg = self._make_reasoning_config("anthropic")
        client = LLMClientFactory.build(cfg)
        assert isinstance(client, ClaudeClient)

    def test_build_gemini(self):
        from spidy.llm.backends.gemini import GeminiClient
        cfg = self._make_reasoning_config("gemini")
        client = LLMClientFactory.build(cfg)
        assert isinstance(client, GeminiClient)

    def test_build_unknown_falls_back_to_ollama(self):
        from spidy.llm.backends.ollama import OllamaClient
        cfg = self._make_reasoning_config("unknown_provider")
        client = LLMClientFactory.build(cfg)
        assert isinstance(client, OllamaClient)

    def test_build_router_single_provider(self):
        from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
        from spidy.llm.backends.ollama import OllamaClient
        cfg = MultiLLMConfig(
            providers=[LLMProviderConfig(name="ollama", model="llama3.2:3b")]
        )
        client = LLMClientFactory.build_router(cfg)
        assert isinstance(client, OllamaClient)

    def test_build_router_multiple_providers(self):
        from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
        cfg = MultiLLMConfig(
            providers=[
                LLMProviderConfig(name="ollama", model="llama3.2:3b"),
                LLMProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test"),
            ]
        )
        client = LLMClientFactory.build_router(cfg)
        assert isinstance(client, LLMRouter)

    def test_build_router_no_enabled_providers(self):
        from spidy.config.manager import LLMProviderConfig, MultiLLMConfig
        from spidy.llm.backends.ollama import OllamaClient
        cfg = MultiLLMConfig(
            providers=[LLMProviderConfig(name="ollama", enabled=False)]
        )
        client = LLMClientFactory.build_router(cfg)
        assert isinstance(client, OllamaClient)


# ──────────────────────────────────────────────────────────────────────────────
# 7. LLM Events
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMEvents:
    def test_event_topics(self):
        from spidy.llm.events import (
            LLMProviderFailedEvent,
            LLMProviderSwitchedEvent,
            LLMRequestStartedEvent,
            LLMResponseReadyEvent,
        )
        assert LLMRequestStartedEvent.topic == "llm.request_started"
        assert LLMResponseReadyEvent.topic == "llm.response_ready"
        assert LLMProviderFailedEvent.topic == "llm.provider_failed"
        assert LLMProviderSwitchedEvent.topic == "llm.provider_switched"

    def test_event_instantiation(self):
        from spidy.llm.events import LLMProviderSwitchedEvent
        e = LLMProviderSwitchedEvent(from_provider="ollama", to_provider="openai")
        assert e.from_provider == "ollama"
        assert e.to_provider == "openai"
