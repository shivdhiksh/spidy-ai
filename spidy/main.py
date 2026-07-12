"""
Spidy — Application Entry Point
================================
Run with:
    python -m spidy.main
or (after pip install -e .):
    spidy

Flags
-----
    --text-mode     Skip voice pipeline; open an interactive text REPL.
                    Useful for testing without audio hardware.

    <config_path>   Optional first positional argument: path to a custom
                    YAML config file (overrides the default
                    config/spidy_config.yaml).

Examples
--------
    spidy                                    # production, voice enabled
    spidy --text-mode                        # REPL, no voice required
    spidy config/dev_config.yaml             # custom config, production
    spidy config/dev_config.yaml --text-mode # custom config + REPL
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def main() -> None:
    """
    Production entry point.

    Parses CLI arguments, creates SpidyCore, and runs it.
    Handles top-level exceptions with a clear error message.
    """
    # Delayed import so logging is not configured at import time
    from spidy.core.app import SpidyCore

    # ── Parse CLI arguments ────────────────────────────────────────────────
    args = sys.argv[1:]
    text_mode = False
    config_path: Path | None = None

    for arg in args:
        if arg == "--text-mode":
            text_mode = True
        elif arg.startswith("--"):
            print(f"[Spidy] Unknown flag: {arg}", file=sys.stderr)
            print("Usage: spidy [config_path] [--text-mode]", file=sys.stderr)
            sys.exit(1)
        else:
            # First non-flag argument is treated as config path
            if config_path is None:
                config_path = Path(arg)
                if not config_path.exists():
                    print(
                        f"[Spidy] Error: Config file not found: {config_path}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    core = SpidyCore(config_path=config_path, text_mode=text_mode)

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
