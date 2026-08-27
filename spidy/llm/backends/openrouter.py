"""
OpenRouterClient — OpenRouter Cloud LLM Backend
================================================
Connects to the OpenRouter API, which exposes an OpenAI-compatible
``/v1/chat/completions`` endpoint supporting hundreds of cloud models.

API reference: https://openrouter.ai/docs

Default model : ``nvidia/nemotron-3-ultra-550b-a55b:free``
               (free-tier, matches the NIM model family for consistency)
Auth          : ``Authorization: Bearer <api_key>``
               Falls back to the ``OPENROUTER_API_KEY`` environment variable.
Base URL      : ``https://openrouter.ai/api/v1``

Why extend OpenAIClient?
------------------------
OpenRouter mirrors the OpenAI API schema, so all HTTP transport, streaming,
and error-handling logic in ``OpenAIClient`` works unchanged. Only the
default ``base_url``, ``model``, and env-var name differ.

Failure handling
----------------
Returns ``LLMResponse.failure()`` on any error — never raises to caller.
If the API key is absent, ``health_check()`` returns False immediately
(no network call).

Security
--------
- The API key is **never** logged at any level.
- Reading from env var keeps the key out of source code and config files.
- ``health_check()`` only checks key presence; it never logs or exposes
  the key value.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from spidy.llm.backends.openai import OpenAIClient
from spidy.llm.client import LLMMessage, LLMResponse
from spidy.logging.logger import get_logger

from pathlib import Path

log = get_logger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
_ENV_KEY = "OPENROUTER_API_KEY"


class OpenRouterClient(OpenAIClient):
    """
    LLM backend for the OpenRouter Chat Completions API.

    Inherits all HTTP transport and streaming logic from ``OpenAIClient``.
    Only the defaults for ``base_url``, ``model``, and env-var key name differ.

    Parameters
    ----------
    api_key:
        OpenRouter API key.  Falls back to the ``OPENROUTER_API_KEY`` env var.
        Leave empty to use only the env var (recommended).
    model:
        OpenRouter model slug, e.g.
        ``"nvidia/nemotron-3-ultra-550b-a55b:free"``,
        ``"mistralai/mistral-7b-instruct:free"``,
        ``"anthropic/claude-3-haiku"``.
    base_url:
        OpenRouter API base URL.  Change only if using a self-hosted proxy.
    temperature:
        Sampling temperature (0.0–2.0).
    max_tokens:
        Maximum tokens to generate.
    timeout:
        Request timeout in seconds.  Cloud inference can be slow for large
        models; 60 s is a safe default.
    """

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        # Resolve key: constructor arg → env var.
        # We intentionally do NOT log the key value at any level.
        resolved_key = api_key or os.environ.get(_ENV_KEY, "")

        super().__init__(
            api_key=resolved_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # Explicitly enforce OpenRouter key — never fall back to OPENAI_API_KEY in base class
        self._api_key = resolved_key

        # Log presence only — never the value.
        if resolved_key:
            log.debug(
                "OpenRouterClient: API key detected (length={n}).", n=len(resolved_key)
            )
        else:
            log.debug(
                "OpenRouterClient: No API key found in constructor or {env}.",
                env=_ENV_KEY,
            )

    # ── Public API (inherited from OpenAIClient) ──────────────────────────
    # complete(), stream(), close() all inherited — nothing to override.

    async def health_check(self) -> bool:
        """
        Return True iff the API key is set.

        We avoid making a real network call here because:
        - The OpenRouter catalog endpoint adds latency to every startup.
        - We only want to know if the key exists (key presence != key validity,
          but a missing key guarantees failure at call time anyway).
        """
        has_key = bool(self._api_key)
        if not has_key:
            log.debug(
                "OpenRouterClient.health_check: no API key — provider unavailable."
            )
        return has_key
