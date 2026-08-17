"""
Spidy Logging Module
====================
Centralised, structured, rotating logger built on loguru.

Usage
-----
    from spidy.logging.logger import get_logger

    log = get_logger(__name__)
    log.info("Spidy started")
    log.debug("Processing input: {input}", input=text)

Design notes
------------
- Every module gets its own named logger via get_logger().
- Logs go to stderr (coloured) AND a rotating file.
- Log file location comes from ConfigManager once available; before that,
  we use a sensible fallback so logging works during bootstrap.
- We bind 'module' to every record for structured filtering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

# ─── Internal state ──────────────────────────────────────────────────────────
_configured = False
_bootstrapped = False   # True only after bootstrap (not full configure)
_log_dir: Path = Path(os.environ.get("APPDATA", ".")) / "Spidy" / "logs"


def configure(
    level: str = "INFO",
    log_dir: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    fmt: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{extra[module]:<30} | "
        "{message}"
    ),
    colorize: bool = True,
    backtrace: bool = True,
    diagnose: bool = False,
) -> None:
    """
    Configure the global Spidy logger.

    This must be called once at application startup (from SpidyCore).
    Calling it more than once is safe — subsequent calls are no-ops unless
    *force=True* is passed (reserved for tests).

    Parameters
    ----------
    level:
        Minimum log level to capture. One of DEBUG, INFO, WARNING, ERROR,
        CRITICAL.
    log_dir:
        Directory where rotating log files are written. Defaults to
        %APPDATA%/Spidy/logs.
    rotation:
        Loguru rotation policy, e.g. "10 MB" or "1 day".
    retention:
        Loguru retention policy, e.g. "30 days" or "7 days".
    fmt:
        Loguru format string. The default includes a 'module' extra field
        that every logger binds automatically.
    colorize:
        Whether to emit ANSI colour codes on stderr.
    backtrace:
        Whether to extend tracebacks to show the full call stack.
    diagnose:
        Whether to show local variable values in tracebacks. Disable in
        production (may expose sensitive data).
    """
    global _configured, _bootstrapped, _log_dir

    # BUG 5 FIX: Only skip if we already ran a FULL configure() — not just
    # a bootstrap.  The bootstrap sets _bootstrapped=True but leaves
    # _configured=False so the real call from SpidyCore always wins.
    if _configured:
        return

    if log_dir is not None:
        _log_dir = log_dir

    _log_dir.mkdir(parents=True, exist_ok=True)
    log_file = _log_dir / "spidy_{time:YYYY-MM-DD}.log"

    # Remove ALL existing handlers (including any bootstrap handler)
    logger.remove()

    # ── Stderr sink (coloured, human-readable) ────────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=colorize,
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=True,       # thread-safe non-blocking writes
    )

    # ── File sink (rotating, structured) ─────────────────────────────────
    logger.add(
        str(log_file),
        level=level,
        format=fmt,
        rotation=rotation,
        retention=retention,
        backtrace=backtrace,
        diagnose=False,     # never write variable values to disk
        enqueue=True,
        encoding="utf-8",
    )

    _configured = True
    _bootstrapped = True
    logger.bind(module="spidy.logging").info(
        "Logger configured | level={level} | log_dir={dir}",
        level=level,
        dir=str(_log_dir),
    )


def _bootstrap() -> None:
    """
    Minimal bootstrap logger for import-time use only.

    BUG 5 FIX: Uses enqueue=False to avoid spawning background threads
    before the asyncio event loop starts.  _configured is left False so
    the real configure() call from SpidyCore always overrides this.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    # Remove the default loguru stderr handler (handler #0)
    try:
        logger.remove()
    except Exception:  # noqa: BLE001
        pass
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level:<8} | "
            "{extra[module]:<30} | "
            "{message}"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
        enqueue=False,  # No background thread during early import
    )
    _bootstrapped = True


def get_logger(module_name: str):
    """
    Return a module-scoped logger.

    The returned object is a loguru logger with 'module' pre-bound to
    *module_name*. This appears in every log record, making it easy to
    filter by component.

    Parameters
    ----------
    module_name:
        Typically __name__ of the calling module.

    Returns
    -------
    loguru.Logger
        A bound logger instance. Stateless — can be stored at module level.

    Example
    -------
        log = get_logger(__name__)
        log.info("Ready")
    """
    if not _bootstrapped:
        # BUG 5 FIX: Use lightweight bootstrap (no enqueue, no file sink)
        # so early import-time log calls don't trigger background threads
        # before the asyncio loop is running.  The real configure() from
        # SpidyCore will replace these handlers with the full setup.
        _bootstrap()

    return logger.bind(module=module_name)

