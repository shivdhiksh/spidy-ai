"""
LLM Client — Multi-Backend Abstraction Layer
=============================================
Abstract interface and data types shared by all LLM backends.

Supported providers (current and planned):
  Ollama    (local, default)        → spidy.llm.backends.ollama
  OpenAI    (cloud, Milestone 4+)   → spidy.llm.backends.openai
  Claude    (cloud, Milestone 4+)   → spidy.llm.backends.claude
  Gemini    (cloud, Milestone 4+)   → spidy.llm.backends.gemini

Design principles
-----------------
- All backends implement ``BaseLLMClient``
- Brain and other modules import only ``BaseLLMClient`` — never a backend
- ``LLMClientFactory.build(config)`` returns the right client from config
- All backends handle their own errors and return ``LLMResponse.failure()``
  instead of raising — the Brain never crashes from a failed LLM call
- Streaming is supported by every backend (for future real-time TTS)

Usage
-----
    from spidy.llm.client import LLMClientFactory, LLMMessage

    client = LLMClientFactory.build(settings.reasoning)
    response = await client.complete([
        LLMMessage(role="system", content="You are Spidy, a helpful AI."),
        LLMMessage(role="user", content="Open Chrome"),
    ])
    print(response.text)   # "Opening Chrome..."
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.config.manager import ReasoningConfig

log = get_logger(__name__)


# ─── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMMessage:
    """A single message in an LLM conversation (OpenAI-style format)."""
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMUsage:
    """Token usage statistics for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Result of a single LLM completion call."""
    text: str
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: str = "stop"     # "stop" | "length" | "error"
    success: bool = True
    error_message: str = ""

    @classmethod
    def failure(cls, error: str, model: str = "") -> "LLMResponse":
        """Construct a failed response (LLM unavailable, API error, etc.)."""
        return cls(
            text="",
            model=model,
            finish_reason="error",
            success=False,
            error_message=error,
        )


# ─── Abstract Client ──────────────────────────────────────────────────────────


class BaseLLMClient(abc.ABC):
    """
    Abstract base class for all LLM backends.

    Subclasses must implement:
      ``complete()`` — single-shot completion (most common use case)
      ``stream()``   — streaming completion (token by token, for real-time TTS)
      ``close()``    — release resources (HTTP sessions, etc.)

    Error contract
    --------------
    Neither ``complete()`` nor ``stream()`` should raise exceptions to the
    caller. All errors must be caught internally and reflected via
    ``LLMResponse.failure()`` or by yielding nothing from ``stream()``.
    """

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send messages and return a complete response.

        Never raises — returns ``LLMResponse.failure()`` on error.
        """
        ...

    @abc.abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Send messages and yield response tokens as they arrive.

        Never raises — yields nothing on error and logs the failure.
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Release resources (HTTP session, GPU handles, etc.)."""
        ...

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


# ─── Factory ──────────────────────────────────────────────────────────────────


class LLMClientFactory:
    """Build the correct LLM backend from config."""

    @staticmethod
    def build(config: "ReasoningConfig") -> BaseLLMClient:
        """
        Construct and return the LLM client for the configured provider.

        Supported providers
        -------------------
        ``"ollama"``   — Local Ollama server (default)
        ``"openai"``   — OpenAI API (Milestone 4+)
        ``"claude"``   — Anthropic Claude API (Milestone 4+)
        ``"gemini"``   — Google Gemini API (Milestone 4+)

        Falls back to OllamaClient for unknown providers with a warning.
        """
        provider = (config.provider or "ollama").lower()

        if provider == "ollama":
            from spidy.llm.backends.ollama import OllamaClient
            return OllamaClient(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
            )

        # Future backends — planned for Milestone 4+
        future_providers = {
            "openai": "OpenAI",
            "claude": "Claude (Anthropic)",
            "anthropic": "Claude (Anthropic)",
            "gemini": "Google Gemini",
        }
        if provider in future_providers:
            log.warning(
                "{name} backend is not yet implemented. Falling back to Ollama. "
                "Multi-provider LLM support is planned for Milestone 4.",
                name=future_providers[provider],
            )
        else:
            log.warning(
                "Unknown LLM provider '{p}'. Falling back to Ollama.",
                p=provider,
            )

        from spidy.llm.backends.ollama import OllamaClient
        return OllamaClient(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )
