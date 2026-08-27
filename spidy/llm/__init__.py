# spidy/llm package
from spidy.llm.client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMMessage,
    LLMResponse,
    LLMUsage,
)
from spidy.llm.router import LLMRouter
from spidy.llm.backends.openrouter import OpenRouterClient
from spidy.llm.health import (
    OllamaHealthReport,
    run_health_check,
    run_health_check_async,
    print_health_report,
    maybe_pull_model,
)

__all__ = [
    "BaseLLMClient",
    "LLMClientFactory",
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "LLMRouter",
    "OpenRouterClient",
    # Retained for backward compatibility (health.py is stdlib-only)
    "OllamaHealthReport",
    "run_health_check",
    "run_health_check_async",
    "print_health_report",
    "maybe_pull_model",
]
