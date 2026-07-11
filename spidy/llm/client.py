"""
LLM Client — Multi-Backend Abstraction Layer
=============================================
Abstract interface and data types shared by all LLM backends.

Supported providers:
  Ollama  (local, default)  → spidy.llm.backends.ollama
  OpenAI  (cloud)           → spidy.llm.backends.openai
  Claude  (cloud)           → spidy.llm.backends.claude
  Gemini  (cloud)           → spidy.llm.backends.gemini

Design principles
-----------------
- All backends implement ``BaseLLMClient``
- Brain and other modules import only ``BaseLLMClient`` — never a backend
- ``LLMClientFactory.build(config)`` returns the right client from config
- All backends catch their own errors and return ``LLMResponse.failure()``
- ``LLMClientFactory.build_router(config)`` builds a failover-capable router

Usage
-----
    from spidy.llm.client import LLMClientFactory, LLMMessage

    client = LLMClientFactory.build(settings.reasoning)
    response = await client.complete([
        LLMMessage(role="system", content="You are Spidy."),
        LLMMessage(role="user", content="Open Chrome"),
    ])
    print(response.text)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.config.manager import LLMProviderConfig, MultiLLMConfig, ReasoningConfig

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
      ``complete()``      — single-shot completion
      ``stream()``        — streaming completion (token by token)
      ``close()``         — release resources (HTTP sessions, etc.)

    Subclasses should set:
      ``provider_name``   — class-level string identifying the provider

    Error contract
    --------------
    Neither ``complete()`` nor ``stream()`` should raise to the caller.
    All errors must be caught internally and reflected via
    ``LLMResponse.failure()`` or by yielding nothing from ``stream()``.
    """

    provider_name: str = "base"

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages and return a complete response. Never raises."""
        ...

    @abc.abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens. Never raises; yields nothing on error."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Release resources (HTTP session, GPU handles, etc.)."""
        ...

    async def health_check(self) -> bool:
        """
        Check whether this provider is reachable and configured.

        Returns True by default. Backends override for real checks.
        """
        return True

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


# ─── Factory ──────────────────────────────────────────────────────────────────


class LLMClientFactory:
    """Build LLM clients from configuration."""

    @staticmethod
    def build(config: "ReasoningConfig") -> BaseLLMClient:
        """
        Build a single-provider client from a ReasoningConfig.

        Supports: ollama | openai | claude | anthropic | gemini
        """
        return LLMClientFactory._build_provider(
            name=(config.provider or "ollama").lower(),
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key or "",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    def build_from_provider_config(cfg: "LLMProviderConfig") -> BaseLLMClient:
        """Build a single-provider client from a LLMProviderConfig entry."""
        return LLMClientFactory._build_provider(
            name=cfg.name.lower(),
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_seconds,
        )

    @staticmethod
    def build_router(
        config: "MultiLLMConfig",
        bus: object = None,
        session_id: str = "",
    ) -> "BaseLLMClient":
        """
        Build an LLMRouter from a MultiLLMConfig.

        If only one provider is enabled, returns it directly (no router
        overhead). If zero providers are enabled, falls back to OllamaClient.
        """
        from spidy.llm.router import LLMRouter

        enabled = [p for p in config.providers if p.enabled]
        if not enabled:
            log.warning("No LLM providers enabled — using default OllamaClient.")
            from spidy.llm.backends.ollama import OllamaClient
            return OllamaClient()

        clients = [LLMClientFactory.build_from_provider_config(p) for p in enabled]

        if len(clients) == 1:
            return clients[0]

        return LLMRouter(
            providers=clients,
            bus=bus,
            auto_fallback=config.auto_fallback,
            failure_threshold=config.failure_threshold,
            session_id=session_id,
        )

    @staticmethod
    def _build_provider(
        name: str,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> BaseLLMClient:
        """Internal: instantiate one provider client by name."""
        if name == "ollama":
            from spidy.llm.backends.ollama import OllamaClient
            return OllamaClient(
                base_url=base_url or "http://localhost:11434",
                model=model or "llama3.2:3b",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        if name == "openai":
            from spidy.llm.backends.openai import OpenAIClient
            return OpenAIClient(
                api_key=api_key,
                model=model or "gpt-4o-mini",
                base_url=base_url or "https://api.openai.com/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        if name in ("claude", "anthropic"):
            from spidy.llm.backends.claude import ClaudeClient
            return ClaudeClient(
                api_key=api_key,
                model=model or "claude-3-5-haiku-20241022",
                base_url=base_url or "https://api.anthropic.com/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        if name == "gemini":
            from spidy.llm.backends.gemini import GeminiClient
            return GeminiClient(
                api_key=api_key,
                model=model or "gemini-1.5-flash",
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        log.warning("Unknown LLM provider '{p}'. Falling back to Ollama.", p=name)
        from spidy.llm.backends.ollama import OllamaClient
        return OllamaClient(
            base_url=base_url or "http://localhost:11434",
            model=model or "llama3.2:3b",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
