"""
OpenAIClient — OpenAI Chat Completions Backend
===============================================
Connects to the OpenAI API using stdlib urllib only.
No `openai` package required.

API reference: https://platform.openai.com/docs/api-reference/chat

Default model: gpt-4o-mini (fast, cheap, capable)
Auth:          Authorization: Bearer <api_key>
               Also supports OPENAI_API_KEY env var fallback.

Failure handling
----------------
Returns LLMResponse.failure() on any error — never raises to caller.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import AsyncIterator

from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse, LLMUsage
from spidy.logging.logger import get_logger

log = get_logger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(BaseLLMClient):
    """
    LLM backend for the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    model:
        Model name (e.g. "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo").
    base_url:
        API base URL. Override for Azure OpenAI or compatible endpoints.
    temperature:
        Sampling temperature (0.0–2.0).
    max_tokens:
        Maximum tokens to generate.
    timeout:
        Request timeout in seconds.
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Single-shot completion via OpenAI /v1/chat/completions."""
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": False,
        }
        try:
            result = await asyncio.to_thread(self._post, "/chat/completions", payload)
            choice = result.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            usage_data = result.get("usage", {})
            return LLMResponse(
                text=text,
                model=result.get("model", self._model),
                usage=LLMUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
                success=True,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            msg = f"OpenAI HTTP {exc.code}: {body[:200]}"
            log.warning("OpenAIClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except urllib.error.URLError as exc:
            msg = f"OpenAI unreachable: {exc}"
            log.warning("OpenAIClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except Exception as exc:  # noqa: BLE001
            msg = f"OpenAIClient unexpected error: {exc}"
            log.error("OpenAIClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming via OpenAI SSE (server-sent events)."""
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
        }
        try:
            chunks = await asyncio.to_thread(self._post_stream, "/chat/completions", payload)
            for chunk in chunks:
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenAIClient stream error: {exc}", exc=exc)
            return

    async def close(self) -> None:
        """No persistent connection to close."""

    async def health_check(self) -> bool:
        """Return True if the API key is set and the endpoint is reachable."""
        if not self._api_key:
            return False
        try:
            await asyncio.to_thread(self._get, "/models")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_stream(self, path: str, payload: dict) -> list[dict]:
        """Collect SSE chunks from OpenAI streaming endpoint."""
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        chunks: list[dict] = []
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for line in resp:
                line = line.strip().decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    payload_str = line[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunks.append(json.loads(payload_str))
                    except json.JSONDecodeError:
                        pass
        return chunks
