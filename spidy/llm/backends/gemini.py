"""
GeminiClient — Google Generative Language API Backend
======================================================
Connects to the Google Gemini API using stdlib urllib only.
No `google-generativeai` package required.

API reference: https://ai.google.dev/api/generate-content

Default model: gemini-1.5-flash (fast, free tier available)
Auth:          ?key=<api_key> query parameter
               Also supports GEMINI_API_KEY / GOOGLE_API_KEY env var fallback.

Message mapping
---------------
Gemini uses "user"/"model" roles (not "assistant").
System prompts are injected as "system_instruction" in the request body.

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
from urllib.parse import urlencode

from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse, LLMUsage
from spidy.logging.logger import get_logger

log = get_logger(__name__)

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiClient(BaseLLMClient):
    """
    LLM backend for the Google Gemini Generative Language API.

    Parameters
    ----------
    api_key:
        Google API key. Falls back to GEMINI_API_KEY or GOOGLE_API_KEY env var.
    model:
        Model name (e.g. "gemini-1.5-flash", "gemini-1.5-pro").
    base_url:
        API base URL.
    temperature:
        Sampling temperature (0.0–2.0).
    max_tokens:
        Maximum tokens to generate (mapped to maxOutputTokens).
    timeout:
        Request timeout in seconds.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
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
        """Single-shot completion via Gemini generateContent."""
        system_prompt, contents = self._convert_messages(messages)

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self._temperature,
                "maxOutputTokens": max_tokens if max_tokens is not None else self._max_tokens,
            },
        }
        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        path = f"/models/{self._model}:generateContent"
        try:
            result = await asyncio.to_thread(self._post, path, payload)
            candidates = result.get("candidates", [{}])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)

            usage_meta = result.get("usageMetadata", {})
            prompt_tokens = usage_meta.get("promptTokenCount", 0)
            completion_tokens = usage_meta.get("candidatesTokenCount", 0)

            finish_reason = candidates[0].get("finishReason", "STOP") if candidates else "STOP"

            return LLMResponse(
                text=text,
                model=self._model,
                usage=LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                finish_reason=finish_reason.lower(),
                success=True,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            msg = f"Gemini HTTP {exc.code}: {body[:200]}"
            log.warning("GeminiClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except urllib.error.URLError as exc:
            msg = f"Gemini unreachable: {exc}"
            log.warning("GeminiClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except Exception as exc:  # noqa: BLE001
            msg = f"GeminiClient unexpected error: {exc}"
            log.error("GeminiClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming via Gemini streamGenerateContent (NDJSON)."""
        system_prompt, contents = self._convert_messages(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self._temperature,
                "maxOutputTokens": max_tokens if max_tokens is not None else self._max_tokens,
            },
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        path = f"/models/{self._model}:streamGenerateContent"
        try:
            chunks = await asyncio.to_thread(self._post_stream, path, payload)
            for chunk in chunks:
                candidates = chunk.get("candidates", [{}])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        if text:
                            yield text
        except Exception as exc:  # noqa: BLE001
            log.warning("GeminiClient stream error: {exc}", exc=exc)
            return

    async def close(self) -> None:
        """No persistent connection to close."""

    async def health_check(self) -> bool:
        """Return True if API key is set."""
        return bool(self._api_key)

    # ── Internal ──────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        """Build URL with API key query parameter."""
        qs = urlencode({"key": self._api_key}) if self._api_key else ""
        return f"{self._base_url}{path}?{qs}" if qs else f"{self._base_url}{path}"

    @staticmethod
    def _convert_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini contents format.

        Returns (system_prompt, contents_list).
        Gemini uses "user"/"model" roles; system messages become system_instruction.
        """
        system_parts: list[str] = []
        contents: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        return "\n".join(system_parts), contents

    def _post(self, path: str, payload: dict) -> dict:
        url = self._url(path)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_stream(self, path: str, payload: dict) -> list[dict]:
        url = self._url(path)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[dict] = []
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")
            # Gemini streaming returns a JSON array; each element is a chunk
            try:
                items = json.loads(body)
                if isinstance(items, list):
                    chunks = items
                else:
                    chunks = [items]
            except json.JSONDecodeError:
                # Try NDJSON line-by-line
                for line in body.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return chunks
