"""
ClaudeClient — Anthropic Messages API Backend
==============================================
Connects to the Anthropic Claude API using stdlib urllib only.
No `anthropic` package required.

API reference: https://docs.anthropic.com/en/api/messages

Default model: claude-3-5-haiku-20241022 (fast, affordable)
Auth:          x-api-key header + anthropic-version header
               Also supports ANTHROPIC_API_KEY env var fallback.

Message mapping
---------------
Anthropic uses a "system" top-level field (not a message role).
user/assistant alternate roles in the messages array.
The OpenAI-style system message is extracted and sent separately.

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

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_DEFAULT_MODEL = "claude-3-5-haiku-20241022"
_API_VERSION = "2023-06-01"


class ClaudeClient(BaseLLMClient):
    """
    LLM backend for the Anthropic Claude Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model:
        Model name (e.g. "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022").
    base_url:
        API base URL.
    temperature:
        Sampling temperature (0.0–1.0 for Claude).
    max_tokens:
        Maximum tokens to generate (required by Anthropic API).
    timeout:
        Request timeout in seconds.
    """

    provider_name = "claude"

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
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
        """Single-shot completion via Anthropic /v1/messages."""
        system_prompt, chat_messages = self._split_system(messages)

        payload: dict = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            result = await asyncio.to_thread(self._post, "/messages", payload)
            content_blocks = result.get("content", [])
            text = "".join(
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            )
            usage_data = result.get("usage", {})
            return LLMResponse(
                text=text,
                model=result.get("model", self._model),
                usage=LLMUsage(
                    prompt_tokens=usage_data.get("input_tokens", 0),
                    completion_tokens=usage_data.get("output_tokens", 0),
                    total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                ),
                finish_reason=result.get("stop_reason", "stop"),
                success=True,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            msg = f"Claude HTTP {exc.code}: {body[:200]}"
            log.warning("ClaudeClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except urllib.error.URLError as exc:
            msg = f"Claude unreachable: {exc}"
            log.warning("ClaudeClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except Exception as exc:  # noqa: BLE001
            msg = f"ClaudeClient unexpected error: {exc}"
            log.error("ClaudeClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming via Anthropic SSE."""
        system_prompt, chat_messages = self._split_system(messages)
        payload: dict = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            chunks = await asyncio.to_thread(self._post_stream, "/messages", payload)
            for chunk in chunks:
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {}).get("text", "")
                    if delta:
                        yield delta
        except Exception as exc:  # noqa: BLE001
            log.warning("ClaudeClient stream error: {exc}", exc=exc)
            return

    async def close(self) -> None:
        """No persistent connection to close."""

    async def health_check(self) -> bool:
        """Return True if API key is set."""
        return bool(self._api_key)

    # ── Internal ──────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "anthropic-version": _API_VERSION,
        }
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    @staticmethod
    def _split_system(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict]]:
        """
        Separate system messages from the chat history.

        Anthropic requires system content as a top-level field,
        not as a message in the messages array.
        """
        system_parts: list[str] = []
        chat: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                # Claude requires alternating user/assistant turns.
                # Merge consecutive same-role messages if needed.
                role = "user" if msg.role == "user" else "assistant"
                if chat and chat[-1]["role"] == role:
                    chat[-1]["content"] += "\n" + msg.content
                else:
                    chat.append({"role": role, "content": msg.content})
        return "\n".join(system_parts), chat

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_stream(self, path: str, payload: dict) -> list[dict]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        chunks: list[dict] = []
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for line in resp:
                line = line.strip().decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    payload_str = line[6:]
                    try:
                        chunks.append(json.loads(payload_str))
                    except json.JSONDecodeError:
                        pass
        return chunks
