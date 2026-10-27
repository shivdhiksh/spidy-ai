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

import argparse
import asyncio
import os
import sys
from pathlib import Path


_CRASH_LOG = None  # kept open for faulthandler for the life of the process


def _exit_now(code: int) -> None:
    """Flush logs and terminate the process even if non-daemon threads are alive."""
    try:
        from loguru import logger
        logger.complete()
        logger.remove()
    except Exception:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:  # noqa: BLE001
            pass
    from spidy.core.watchdog import kill_self
    kill_self(code)   # TerminateProcess: a normal exit can hang in a DLL's detach code


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spidy — Autonomous Desktop AI Companion")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to YAML config file")
    parser.add_argument("--text-mode", action="store_true", help="Skip voice pipeline and open interactive text REPL")
    parser.add_argument("--background", "--hidden", dest="background", action="store_true", help="Start Spidy in background with HUD hidden")
    parser.add_argument("--quit", action="store_true", help="Ask the running Spidy to close itself, then exit")
    parser.add_argument("--install-autostart", action="store_true", help="Register Spidy in HKCU Run to start on Windows login")
    parser.add_argument("--uninstall-autostart", action="store_true", help="Remove Spidy from HKCU Run")
    parser.add_argument("--status-autostart", action="store_true", help="Check Windows login autostart registration status")
    return parser


def main() -> None:
    """
    Production entry point.

    Parses CLI arguments, creates SpidyCore, and runs it.
    Handles top-level exceptions with a clear error message.
    """
    # Delayed import so logging is not configured at import time
    from spidy.core.app import SpidyCore

    parser = _build_cli_parser()
    opts = parser.parse_args()

    if opts.quit:
        from spidy.core.single_instance import SingleInstanceGuard
        sent = SingleInstanceGuard().notify_existing_instance("QUIT")
        print("[Spidy] Quit sent to the running instance." if sent else "[Spidy] No running Spidy found.")
        sys.exit(0 if sent else 1)

    if opts.install_autostart:
        from spidy.desktop.autostart import enable_autostart
        ok = enable_autostart(background=True)
        print(f"[Spidy] Autostart registration: {'SUCCESS' if ok else 'FAILED'}")
        sys.exit(0 if ok else 1)

    if opts.uninstall_autostart:
        from spidy.desktop.autostart import disable_autostart
        ok = disable_autostart()
        print(f"[Spidy] Autostart unregistration: {'SUCCESS' if ok else 'FAILED'}")
        sys.exit(0 if ok else 1)

    if opts.status_autostart:
        from spidy.desktop.autostart import is_autostart_enabled, get_autostart_command
        enabled = is_autostart_enabled()
        cmd = get_autostart_command()
        print(f"[Spidy] Autostart enabled: {enabled} (cmd: {cmd})")
        sys.exit(0)

    config_path = Path(opts.config_path) if opts.config_path else None
    if config_path and not config_path.exists():
        print(f"[Spidy] Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Config, models, plugins and prompts are all referenced relative to the project
    # root. Windows starts login items from System32, so pin the working directory
    # (after resolving any user-supplied config path against the caller's cwd).
    if config_path:
        config_path = config_path.resolve()
    from spidy.config.manager import PROJECT_ROOT
    os.chdir(PROJECT_ROOT)

    # Native crashes (CUDA, onnxruntime, PortAudio, Qt) kill the process without a
    # Python traceback, and pythonw has no console. Dump every thread's stack to a file.
    try:
        import faulthandler
        crash_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "Spidy" / "logs"
        crash_dir.mkdir(parents=True, exist_ok=True)
        global _CRASH_LOG
        _CRASH_LOG = open(crash_dir / "spidy_crash.log", "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_CRASH_LOG, all_threads=True)
    except Exception:  # noqa: BLE001 — diagnostics must never stop Spidy starting
        pass

    from spidy.core.watchdog import enable as _enable_watchdog
    _enable_watchdog()   # real app only: a stalled shutdown is force-closed (tests never arm it)

    core = SpidyCore(
        config_path=config_path,
        text_mode=opts.text_mode,
        start_hidden=opts.background,
        single_instance=True,
    )

    try:
        asyncio.run(core.start())
        # Shutdown finished. Native audio/GPU/ONNX worker threads can keep a process
        # alive after "close" (it vanished from the screen but kept listening), so
        # leave explicitly once logs are flushed.
        _exit_now(0)
    except KeyboardInterrupt:
        # Clean exit on CTRL+C when signal handler fallback is used
        print("\n[Spidy] Interrupted. Exiting.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[Spidy] Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
