"""
Unit tests for LLM failover behavior, bounded timeouts, and deterministic fast-paths.
"""

import asyncio
import pytest

from spidy.brain.intent_classifier import IntentClassifier
from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse
from spidy.llm.router import LLMRouter


class SlowLLMClient(BaseLLMClient):
    provider_name = "slow_provider"

    def __init__(self, delay: float):
        self.delay = delay

    async def complete(self, messages, temperature=None, max_tokens=None):
        await asyncio.sleep(self.delay)
        return LLMResponse(text="Slow response", model="slow-model")

    async def stream(self, messages, temperature=None, max_tokens=None):
        yield "slow"

    async def close(self):
        pass


class FastLLMClient(BaseLLMClient):
    provider_name = "fast_provider"

    async def complete(self, messages, temperature=None, max_tokens=None):
        return LLMResponse(text="Fast response", model="fast-model")

    async def stream(self, messages, temperature=None, max_tokens=None):
        yield "fast"

    async def close(self):
        pass


class TestLLMFailoverUnit:
    @pytest.mark.asyncio
    async def test_router_times_out_primary_and_switches_to_fallback(self):
        """When primary provider exceeds bounded timeout, router immediately fails over to fallback."""
        primary = SlowLLMClient(delay=10.0)  # Would hang
        fallback = FastLLMClient()

        router = LLMRouter(
            providers=[primary, fallback],
            auto_fallback=True,
            provider_timeout=0.1,  # 100ms bounded timeout for test
        )

        messages = [LLMMessage(role="user", content="Hello")]
        resp = await router.complete(messages)

        assert resp.success is True
        assert resp.text == "Fast response"
        assert resp.model == "fast-model"
        assert router._failure_counts[0] == 1
        assert router._failure_counts[1] == 0

    @pytest.mark.asyncio
    async def test_router_fails_if_all_providers_time_out(self):
        """If all providers exceed bounded timeout, router returns structured failure response."""
        p1 = SlowLLMClient(delay=1.0)
        p2 = SlowLLMClient(delay=1.0)

        router = LLMRouter(
            providers=[p1, p2],
            auto_fallback=True,
            provider_timeout=0.05,
        )

        messages = [LLMMessage(role="user", content="Test")]
        resp = await router.complete(messages)

        assert resp.success is False
        assert "timed out" in resp.error_message.lower()

    @pytest.mark.asyncio
    async def test_deterministic_youtube_playback_bypasses_llm(self):
        """Verify IntentClassifier resolves play_youtube_video without LLM."""
        classifier = IntentClassifier()
        utterances = [
            "play any one video random which you like",
            "play any video",
            "play random video",
            "play a random video",
            "play a video",
            "play the first video",
            "play youtube video",
        ]

        for u in utterances:
            intent = await classifier.classify(u)
            assert intent.action == "play_youtube_video", f"Failed for '{u}'"
            assert intent.confidence >= 0.85
