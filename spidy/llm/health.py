"""
Ollama Health Check — Startup Diagnostics & Onboarding
=======================================================
Runs four sequential checks at startup and produces a structured report.
Never raises; all errors are captured and returned as report fields.

Checks (in order)
-----------------
1. Ollama binary installed   — ``shutil.which("ollama")``
2. Ollama server running     — ``GET <base_url>/api/tags`` (no extra deps)
3. Configured model available — scans ``/api/tags`` response for the model
4. LLM round-trip success    — tiny completion via OllamaClient

Public API
----------
    from spidy.llm.health import run_health_check_async, print_health_report

    report = await run_health_check_async(settings.reasoning)
    print_health_report(report)

    if not report.all_ok:
        # handle degraded state

Design constraints
------------------
- Zero extra dependencies (stdlib only: shutil, urllib, json, asyncio)
- Never raises to the caller — all exceptions are caught and recorded
- Sync helper ``run_health_check()`` available for tests / non-async callers
- Pull helper ``maybe_pull_model()`` accepts a ``confirm`` callable so tests
  can inject ``lambda: True`` without touching stdin
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.config.manager import ReasoningConfig

log = get_logger(__name__)


# ─── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class OllamaHealthReport:
    """
    Structured result of the four-stage Ollama health check.

    Attributes
    ----------
    binary_installed:
        True if the ``ollama`` executable is found on PATH.
    binary_version:
        Version string from ``ollama --version``, or empty string.
    server_running:
        True if the Ollama HTTP server responded to ``GET /api/tags``.
    model_available:
        True if the configured model name is present in the server's model list.
    available_models:
        List of model names reported by the server (empty if server is down).
    configured_model:
        The model name from ``ReasoningConfig.model``.
    base_url:
        The Ollama base URL from ``ReasoningConfig.base_url``.
    llm_connected:
        True if a round-trip completion succeeded (non-empty text returned).
    errors:
        Human-readable error messages for each failed check.
    """

    binary_installed: bool = False
    binary_version: str = ""
    server_running: bool = False
    model_available: bool = False
    available_models: list[str] = field(default_factory=list)
    configured_model: str = ""
    base_url: str = "http://localhost:11434"
    llm_connected: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """True only when all four checks passed."""
        return (
            self.binary_installed
            and self.server_running
            and self.model_available
            and self.llm_connected
        )

    @property
    def usable(self) -> bool:
        """
        True when the Brain can actually respond to requests.

        Slightly more lenient than ``all_ok``: the binary being absent
        does not prevent using an already-running remote server, but
        both server_running and model_available must be True.
        """
        return self.server_running and self.model_available


# ─── Individual Check Functions ───────────────────────────────────────────────


def _check_binary_installed() -> tuple[bool, str]:
    """
    Check 1: Is the ``ollama`` binary on PATH?

    Returns (found: bool, version: str).
    """
    path = shutil.which("ollama")
    if path is None:
        return False, ""

    # Try to get the version (best-effort; ignore failures)
    try:
        import subprocess  # noqa: PLC0415
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip() or result.stderr.strip()
        # e.g. "ollama version 0.3.14"
        version = version.split("\n")[0]
    except Exception:  # noqa: BLE001
        version = "(version unknown)"

    return True, version


def _check_server_running(base_url: str) -> tuple[bool, list[str]]:
    """
    Check 2: Is the Ollama HTTP server reachable?

    Calls ``GET <base_url>/api/tags`` and parses the JSON model list.

    Returns (running: bool, model_names: list[str]).
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        models = body.get("models", [])
        names = [m.get("name", "") for m in models if m.get("name")]
        return True, names
    except urllib.error.URLError:
        return False, []
    except Exception:  # noqa: BLE001
        return False, []


def _check_model_available(configured_model: str, available_models: list[str]) -> bool:
    """
    Check 3: Is the configured model in the available-models list?

    Ollama model names can include tags (``llama3.2:3b``). We do an exact
    match first, then a prefix match so ``llama3.2`` matches ``llama3.2:3b``.
    """
    if not configured_model:
        return False
    if configured_model in available_models:
        return True
    # Prefix match: "llama3.2" matches "llama3.2:latest", "llama3.2:3b", …
    prefix = configured_model.split(":")[0]
    return any(m.startswith(prefix) for m in available_models)


def _check_llm_connection(base_url: str, model: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Check 4: Can we get a real response from the LLM?

    Sends a tiny single-message completion (``"ping"``) and checks that
    the response is non-empty. Uses the same ``urllib`` path as OllamaClient.

    Returns (success: bool, error_message: str).
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "options": {"num_predict": 5},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("message", {}).get("content", "")
        if text:
            return True, ""
        return False, "LLM returned an empty response."
    except urllib.error.URLError as exc:
        return False, f"Connection failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}"


# ─── Main Entry Points ────────────────────────────────────────────────────────


def run_health_check(config: "ReasoningConfig") -> OllamaHealthReport:
    """
    Run all four Ollama health checks synchronously.

    Safe to call from any context. Never raises.

    Parameters
    ----------
    config:
        The ``ReasoningConfig`` object from ``SpidyConfig.reasoning``.
        Only Ollama-provider checks are meaningful; for other providers
        the report is returned with ``all_ok=False`` and a note in ``errors``.
    """
    report = OllamaHealthReport(
        configured_model=config.model,
        base_url=config.base_url,
    )

    # Only run Ollama-specific checks for the Ollama provider
    provider = getattr(config, "provider", "ollama").lower()
    if provider != "ollama":
        report.errors.append(
            f"Health check only supports Ollama provider (configured: {provider})."
        )
        # Still mark server_running=True so startup proceeds without warnings
        report.server_running = True
        report.model_available = True
        report.llm_connected = True
        return report

    # Check 1: Binary
    installed, version = _check_binary_installed()
    report.binary_installed = installed
    report.binary_version = version
    if not installed:
        report.errors.append(
            "Ollama binary not found on PATH. "
            "Download from https://ollama.com/download"
        )

    # Check 2: Server
    running, available_models = _check_server_running(config.base_url)
    report.server_running = running
    report.available_models = available_models
    if not running:
        report.errors.append(
            f"Ollama server not reachable at {config.base_url}. "
            "Run: ollama serve"
        )
        # No point checking model/LLM if server is down
        return report

    # Check 3: Model
    model_ok = _check_model_available(config.model, available_models)
    report.model_available = model_ok
    if not model_ok:
        report.errors.append(
            f"Model '{config.model}' is not installed. "
            f"Run: ollama pull {config.model}"
        )

    # Check 4: LLM round-trip (only if model is available)
    if model_ok:
        try:
            configured_timeout = int(getattr(config, "timeout_seconds", 30))
        except (TypeError, ValueError):
            configured_timeout = 30
        llm_ok, llm_err = _check_llm_connection(
            config.base_url, config.model, timeout=max(configured_timeout, 30)
        )
        report.llm_connected = llm_ok
        if not llm_ok:
            report.errors.append(f"LLM round-trip failed: {llm_err}")
    else:
        report.llm_connected = False

    return report


async def run_health_check_async(config: "ReasoningConfig") -> OllamaHealthReport:
    """
    Async wrapper: runs ``run_health_check()`` in a thread pool so the
    asyncio event loop is not blocked during I/O.
    """
    return await asyncio.to_thread(run_health_check, config)


# ─── Console Output ───────────────────────────────────────────────────────────

# ANSI colour helpers (gracefully disabled when terminal doesn't support them)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"


def _coloured(text: str, colour: str) -> str:
    """Apply ANSI colour only when stdout supports it."""
    if sys.stdout.isatty():
        return f"{colour}{text}{_RESET}"
    return text


def _ok(label: str) -> str:
    return f"  {_coloured('✓', _GREEN)} {label}"


def _warn(label: str) -> str:
    return f"  {_coloured('!', _YELLOW)} {label}"


def _fail(label: str) -> str:
    return f"  {_coloured('✗', _RED)} {label}"


def print_health_report(report: OllamaHealthReport) -> None:
    """
    Print a concise, colour-coded health-check banner to stdout.

    Designed to appear at the very top of the startup sequence so users
    immediately see the Ollama status without digging through logs.
    """
    # On Windows the default stdout encoding is often cp1252 (charmap), which
    # cannot encode the Unicode glyphs used below (─, —, ✓, ✗, →, …).
    # Reconfigure to UTF-8 so the banner prints correctly.  errors='replace'
    # substitutes '?' for any still-unencodable character on legacy consoles.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — non-fatal; proceed with current encoding
            pass

    divider = _coloured("─" * 56, _CYAN)
    print(f"\n{divider}")
    print(_coloured("  Spidy — Startup Health Check", _BOLD))
    print(divider)


    # Check 1: Binary
    if report.binary_installed:
        ver = f" ({report.binary_version})" if report.binary_version else ""
        print(_ok(f"Ollama installed{ver}"))
    else:
        print(_fail("Ollama binary not found on PATH"))

    # Check 2: Server
    if report.server_running:
        model_count = len(report.available_models)
        print(_ok(f"Ollama server running  [{model_count} model(s) installed]"))
    else:
        print(_fail(f"Ollama server not reachable at {report.base_url}"))

    # Check 3: Model
    if report.model_available:
        print(_ok(f"Model available        [{report.configured_model}]"))
    elif report.server_running:
        print(_fail(f"Model not installed    [{report.configured_model}]"))
    else:
        print(_warn(f"Model check skipped    [server is down]"))

    # Check 4: LLM round-trip
    if report.llm_connected:
        print(_ok("LLM connection         [round-trip OK]"))
    elif report.model_available:
        print(_fail("LLM connection failed  [check logs for details]"))
    else:
        print(_warn("LLM connection skipped [model unavailable]"))

    print(divider)

    if report.all_ok:
        print(_coloured("  ✓ All systems go — Brain is ready.\n", _GREEN))
    else:
        print(_coloured("  ! Issues detected — see messages below.\n", _YELLOW))
        for err in report.errors:
            print(f"    {_coloured('→', _YELLOW)} {err}")
        print()

        # Actionable fix guidance
        if not report.binary_installed:
            _print_install_instructions()
        elif not report.server_running:
            _print_server_instructions()
        elif not report.model_available:
            _print_model_instructions(report.configured_model)

    print(divider + "\n")


def _print_install_instructions() -> None:
    """Print Ollama installation instructions for the current platform."""
    print(_coloured("  HOW TO INSTALL OLLAMA:", _BOLD))
    if sys.platform == "win32":
        print("    1. Download: https://ollama.com/download/windows")
        print("    2. Run the installer and follow the prompts.")
        print("    3. Re-launch Spidy.")
    elif sys.platform == "darwin":
        print("    1. Download: https://ollama.com/download/mac")
        print("    2. Or via Homebrew: brew install ollama")
        print("    3. Re-launch Spidy.")
    else:
        print("    1. Run: curl -fsSL https://ollama.com/install.sh | sh")
        print("    2. Re-launch Spidy.")
    print()


def _print_server_instructions() -> None:
    """Print instructions for starting the Ollama server."""
    print(_coloured("  HOW TO START THE OLLAMA SERVER:", _BOLD))
    print("    Open a terminal and run:")
    print(_coloured("      ollama serve", _CYAN))
    print("    Leave it running, then re-launch Spidy.")
    print()


def _print_model_instructions(model: str) -> None:
    """Print instructions for pulling the missing model."""
    print(_coloured(f"  HOW TO INSTALL MODEL '{model}':", _BOLD))
    print("    Open a terminal and run:")
    print(_coloured(f"      ollama pull {model}", _CYAN))
    print("    This will download the model (may take a few minutes).")
    print()


# ─── Model Auto-Pull ─────────────────────────────────────────────────────────


def maybe_pull_model(
    model: str,
    base_url: str,
    confirm: Callable[[], bool] | None = None,
) -> bool:
    """
    Offer to pull the missing Ollama model.

    Parameters
    ----------
    model:
        The model name to pull (e.g. ``"llama3.2:3b"``).
    base_url:
        Ollama server base URL (must be reachable).
    confirm:
        A callable that returns True if the user wants to pull.
        If None, prompts the user via stdin.
        Inject ``lambda: True`` in automated tests.

    Returns
    -------
    bool
        True if the pull was attempted and completed without error,
        False if the user declined or an error occurred.
    """
    if confirm is None:
        # Interactive stdin prompt
        try:
            ans = input(
                f"\n  Download '{model}' from Ollama now? "
                "(This may take a few minutes.) [y/N]: "
            ).strip().lower()
            user_confirmed = ans in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            user_confirmed = False
    else:
        user_confirmed = confirm()

    if not user_confirmed:
        print(f"  Skipped. Run manually: ollama pull {model}")
        return False

    print(f"\n  Pulling '{model}' — please wait…")
    url = f"{base_url.rstrip('/')}/api/pull"
    payload = {"name": model, "stream": True}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                status = chunk.get("status", "")
                # Print progress without flooding the console
                if status and status not in {"pulling manifest"}:
                    completed = chunk.get("completed", 0)
                    total = chunk.get("total", 0)
                    if total > 0:
                        pct = int(completed / total * 100)
                        print(f"\r  {status}: {pct}%", end="", flush=True)
                    else:
                        print(f"\r  {status}", end="", flush=True)
        print(f"\n  ✓ '{model}' pulled successfully.\n")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ✗ Pull failed: {exc}")
        log.error("maybe_pull_model: pull failed for {model}: {exc}", model=model, exc=exc)
        return False
