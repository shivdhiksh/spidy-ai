"""
Spidy — Application Entry Point
================================
Run with:
    python -m spidy.main
or (after pip install -e .):
    spidy
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def main() -> None:
    """
    Production entry point.

    Determines the config path, creates SpidyCore, and runs it.
    Handles top-level exceptions with a clear error message.
    """
    # Delayed import so logging is not configured at import time
    from spidy.core.app import SpidyCore

    # Allow overriding config path via first CLI argument
    config_path: Path | None = None
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if not config_path.exists():
            print(f"[Spidy] Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

    core = SpidyCore(config_path=config_path)

    try:
        asyncio.run(core.start())
    except KeyboardInterrupt:
        # Clean exit on CTRL+C when signal handler fallback is used
        print("\n[Spidy] Interrupted. Exiting.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[Spidy] Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
