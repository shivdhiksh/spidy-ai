"""
LLMRouter — Multi-Provider Failover Router
==========================================
Wraps an ordered list of BaseLLMClient instances and provides
automatic failover when the primary provider fails.

Architecture
------------
    LLMRouter
    ├── providers[0]  ← primary (tried first)
    ├── providers[1]  ← first fallback
    └── providers[2]  ← second fallback

On every ``complete()`` call:
1. Try primary provider
2. If response.success=False and auto_fallback=True, try next provider
3. Emit LLMProviderSwitchedEvent when switching
4. If all providers fail, return LLMResponse.failure()

Provider health tracking
------------------------
The router tracks consecutive failure counts per provider.
Once a provider reaches failure_threshold, it is skipped for the
current session to avoid repeated timeouts.

Usage
-----
    router = LLMRouter(
        providers=[ollama_client, openai_client],
        bus=bus,
        auto_fallback=True,
    )
    response = await router.complete(messages)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from spidy.llm.client import BaseLLMClient, LLMMessage, LLMResponse
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

_ROUTER_MODEL = "router"


class LLMRouter(BaseLLMClient):
    """
    Multi-provider LLM router with automatic failover.

    Implements ``BaseLLMClient`` so it is a drop-in replacement anywhere
    a single client is expected.

    Parameters
    ----------
    providers:
        Ordered list of LLM clients. Index 0 is the primary provider.
    bus:
        EventBus for publishing failover events. May be None (no events).
    auto_fallback:
        If True, automatically try the next provider on failure.
    failure_threshold:
        Number of consecutive failures before skipping a provider.
    session_id:
        Session ID attached to emitted events.
    """

    provider_name = "router"

    def __init__(
        self,
        providers: list[BaseLLMClient],
        bus: "EventBus | None" = None,
        auto_fallback: bool = True,
        failure_threshold: int = 3,
        session_id: str = "",
    ) -> None:
        if not providers:
            raise ValueError("LLMRouter requires at least one provider.")
        self._providers = list(providers)
        self._bus = bus
        self._auto_fallback = auto_fallback
        self._failure_threshold = failure_threshold
        self._session_id = session_id
        # Track consecutive failures per provider index
        self._failure_counts: dict[int, int] = {i: 0 for i in range(len(providers))}

    # ── Public API ────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Attempt completion across providers in order.

        Returns the first successful response, or failure if all fail.
        """
        from spidy.llm.events import LLMProviderFailedEvent, LLMProviderSwitchedEvent

        last_error = "All LLM providers failed."
        active_provider_idx: int | None = None

        for idx, provider in enumerate(self._providers):
            # Skip providers that have exceeded failure threshold
            if self._failure_counts.get(idx, 0) >= self._failure_threshold:
                log.debug(
                    "Skipping provider '{p}' (failure threshold reached).",
                    p=getattr(provider, "provider_name", str(idx)),
                )
                continue

            # Notify of provider switch
            if active_provider_idx is not None and self._bus:
                prev_name = getattr(self._providers[active_provider_idx], "provider_name", "unknown")
                curr_name = getattr(provider, "provider_name", "unknown")
                await self._bus.publish(LLMProviderSwitchedEvent(
                    from_provider=prev_name,
                    to_provider=curr_name,
                    session_id=self._session_id,
                ))

            active_provider_idx = idx
            provider_name = getattr(provider, "provider_name", f"provider_{idx}")

            log.debug(
                "LLMRouter: trying provider '{p}' (idx={i})",
                p=provider_name,
                i=idx,
            )

            response = await provider.complete(messages, temperature=temperature, max_tokens=max_tokens)

            if response.success:
                # Reset failure count on success
                self._failure_counts[idx] = 0
                log.debug("LLMRouter: '{p}' succeeded.", p=provider_name)
                return response

            # Provider failed
            self._failure_counts[idx] = self._failure_counts.get(idx, 0) + 1
            last_error = response.error_message
            log.warning(
                "LLMRouter: provider '{p}' failed (failures={n}): {err}",
                p=provider_name,
                n=self._failure_counts[idx],
                err=last_error[:100],
            )
            if self._bus:
                await self._bus.publish(LLMProviderFailedEvent(
                    provider=provider_name,
                    model=response.model,
                    error_message=last_error,
                    session_id=self._session_id,
                ))

            if not self._auto_fallback:
                break

        log.error("LLMRouter: all providers failed. Last error: {err}", err=last_error)
        return LLMResponse.failure(last_error, model=_ROUTER_MODEL)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream from the first available provider.

        Falls back to non-streaming complete() if the primary stream fails.
        """
        for idx, provider in enumerate(self._providers):
            if self._failure_counts.get(idx, 0) >= self._failure_threshold:
                continue
            provider_name = getattr(provider, "provider_name", f"provider_{idx}")
            try:
                had_output = False
                async for token in provider.stream(messages, temperature=temperature, max_tokens=max_tokens):
                    had_output = True
                    yield token
                if had_output:
                    self._failure_counts[idx] = 0
                    return
            except Exception as exc:  # noqa: BLE001
                log.warning("LLMRouter stream: provider '{p}' failed: {exc}", p=provider_name, exc=exc)
                self._failure_counts[idx] = self._failure_counts.get(idx, 0) + 1
            if not self._auto_fallback:
                return

    async def close(self) -> None:
        """Close all managed providers."""
        for provider in self._providers:
            try:
                await provider.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("LLMRouter.close: error closing provider: {exc}", exc=exc)

    async def health_check(self) -> dict[str, bool]:
        """
        Ping all providers and return a health report.

        Returns
        -------
        dict[str, bool]
            Mapping of provider_name → is_healthy.
        """
        results: dict[str, bool] = {}
        for provider in self._providers:
            name = getattr(provider, "provider_name", str(provider))
            if hasattr(provider, "health_check"):
                try:
                    healthy = await provider.health_check()
                    results[name] = healthy
                except Exception:  # noqa: BLE001
                    results[name] = False
            else:
                results[name] = True  # Assume healthy if no health_check method
        return results

    def reset_failures(self) -> None:
        """Reset all failure counters (e.g. after user re-enables a provider)."""
        self._failure_counts = {i: 0 for i in range(len(self._providers))}

    @property
    def primary_provider(self) -> BaseLLMClient | None:
        """Return the first non-failed provider, or None."""
        for idx, provider in enumerate(self._providers):
            if self._failure_counts.get(idx, 0) < self._failure_threshold:
                return provider
        return None

    @property
    def provider_count(self) -> int:
        return len(self._providers)
