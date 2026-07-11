"""
OllamaClient — Local Ollama LLM Backend
=========================================
Connects to a locally running Ollama server and calls the chat endpoint.

Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md

Endpoint used:  POST /api/chat
Model format:   e.g. "llama3.2:3b", "mistral:7b", "gemma2:2b"

Zero extra dependencies
-----------------------
This client uses Python's built-in ``urllib`` + ``asyncio.to_thread()``
so it works without installing any additional packages.
Install the ``brain`` extras for the OpenAI-compatible client:
    pip install -e ".[brain]"

Failure handling
----------------
If Ollama is not running, ``complete()`` returns ``LLMResponse.failure()``
with a descriptive message. The Brain detects ``success=False`` and falls
back to a canned response. This never raises to the caller.

Streaming
---------
``stream()`` calls the same endpoint with ``"stream": true`` and yields
partial tokens as they arrive. If the server is unavailable, yields nothing.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import AsyncIterator

from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse, LLMUsage
from spidy.logging.logger import get_logger

log = get_logger(__name__)

# Ollama's default local server address
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient(BaseLLMClient):
    """
    LLM backend for a locally running Ollama server.

    Parameters
    ----------
    base_url:
        Ollama server URL. Defaults to http://localhost:11434.
    model:
        Model name to use (e.g. "llama3.2:3b").
    temperature:
        Sampling temperature (0.0–2.0). Lower = more deterministic.
    max_tokens:
        Maximum tokens to generate. Passed as ``num_predict`` to Ollama.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
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
        """Single-shot completion via Ollama /api/chat."""
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens if max_tokens is not None else self._max_tokens

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
        }

        try:
            result = await asyncio.to_thread(self._post, "/api/chat", payload)
            message = result.get("message", {})
            text = message.get("content", "")
            usage = result.get("usage", {})
            return LLMResponse(
                text=text,
                model=result.get("model", self._model),
                usage=LLMUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
                finish_reason=result.get("done_reason", "stop"),
                success=True,
            )
        except urllib.error.URLError as exc:
            msg = (
                f"Ollama is not reachable at {self._base_url}. "
                f"Is 'ollama serve' running? ({exc})"
            )
            log.warning("OllamaClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)
        except Exception as exc:  # noqa: BLE001
            msg = f"OllamaClient unexpected error: {exc}"
            log.error("OllamaClient: {msg}", msg=msg)
            return LLMResponse.failure(msg, model=self._model)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming completion via Ollama /api/chat with stream=true."""
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens if max_tokens is not None else self._max_tokens

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
        }

        try:
            chunks = await asyncio.to_thread(self._post_stream, "/api/chat", payload)
            for chunk in chunks:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception as exc:  # noqa: BLE001
            log.warning("OllamaClient stream error: {exc}", exc=exc)
            return  # Yield nothing — caller handles empty stream

    async def close(self) -> None:
        """No persistent connection to close for urllib-based client."""

    # ── Internal ──────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        """
        Make a synchronous POST request to the Ollama API.
        Runs inside asyncio.to_thread().
        """
        url = f"{self._base_url}{path}"
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
        """
        Make a streaming POST request and collect all chunks.
        Runs inside asyncio.to_thread().
        Returns list of parsed JSON objects (one per line from NDJSON response).
        """
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[dict] = []
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for line in resp:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line.decode("utf-8")))
                    except json.JSONDecodeError:
                        pass
        return chunks
