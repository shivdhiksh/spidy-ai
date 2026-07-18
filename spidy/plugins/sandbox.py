"""
PluginSandbox — Exception Isolation for Plugin Execution
=========================================================
Every interaction with a plugin's ``setup()`` and ``teardown()`` goes through
this sandbox. Its only job is to ensure that a misbehaving plugin **never**
propagates an exception to SpidyCore or the Brain.

Design
------
- All sandbox methods are ``async`` (plugins may do async I/O in setup).
- On failure the sandbox returns an error string, not ``None``.
- On success it returns ``None``.
- No exception class from the plugin leaks out — everything is serialised
  to a string and wrapped in a ``PluginError`` at the manager level.

Future Work
-----------
Real OS-level sandboxing (e.g. Windows job objects, AppContainers) is
deferred to a later milestone. For now the sandbox provides exception
isolation, which prevents Spidy from crashing due to plugin bugs.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spidy.plugins.interfaces import BasePlugin, PluginContext

from spidy.logging.logger import get_logger

log = get_logger(__name__)


class PluginSandbox:
    """
    Exception-isolation wrapper around plugin lifecycle calls.

    All methods are static — the sandbox is stateless.

    Return convention
    -----------------
    - ``None``  → success
    - ``str``   → error message (plugin failed but Spidy is unaffected)
    """

    @staticmethod
    async def safe_setup(
        plugin: "BasePlugin",
        context: "PluginContext",
    ) -> str | None:
        """
        Call ``plugin.setup(context)`` inside an exception guard.

        Parameters
        ----------
        plugin:
            The plugin instance to set up.
        context:
            The ``PluginContext`` to pass to ``setup()``.

        Returns
        -------
        None
            On success.
        str
            Error description on failure.
        """
        try:
            await asyncio.wait_for(
                plugin.setup(context),
                timeout=30.0,  # plugin setup must complete within 30 s
            )
            return None
        except asyncio.TimeoutError:
            error = f"Plugin '{plugin.name}' timed out during setup (>30 s)."
            log.warning(error)
            return error
        except Exception as exc:  # noqa: BLE001
            error = f"Plugin '{plugin.name}' raised during setup: {exc}"
            log.warning(error)
            return error

    @staticmethod
    async def safe_teardown(plugin: "BasePlugin") -> str | None:
        """
        Call ``plugin.teardown()`` inside an exception guard.

        Parameters
        ----------
        plugin:
            The plugin instance to tear down.

        Returns
        -------
        None
            On success.
        str
            Error description on failure.
        """
        try:
            await asyncio.wait_for(
                plugin.teardown(),
                timeout=10.0,  # teardown must complete within 10 s
            )
            return None
        except asyncio.TimeoutError:
            error = f"Plugin '{plugin.name}' timed out during teardown (>10 s)."
            log.warning(error)
            return error
        except Exception as exc:  # noqa: BLE001
            error = f"Plugin '{plugin.name}' raised during teardown: {exc}"
            log.warning(error)
            return error

    @staticmethod
    async def safe_call(
        plugin_name: str,
        coro: object,  # Coroutine
        label: str = "call",
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """
        Execute an arbitrary coroutine from a plugin with a timeout guard.

        Used by PluginContext for plugin-initiated async operations.

        Returns
        -------
        (True, "")
            On success.
        (False, error_message)
            On failure or timeout.
        """
        import inspect
        if not inspect.isawaitable(coro):
            return False, f"Plugin '{plugin_name}' {label}: not a coroutine."
        try:
            await asyncio.wait_for(coro, timeout=timeout)  # type: ignore[arg-type]
            return True, ""
        except asyncio.TimeoutError:
            msg = f"Plugin '{plugin_name}' {label} timed out (>{timeout:.0f} s)."
            log.warning(msg)
            return False, msg
        except Exception as exc:  # noqa: BLE001
            msg = f"Plugin '{plugin_name}' {label} raised: {exc}"
            log.warning(msg)
            return False, msg
