"""
SystemControlSkill — Windows System Control Agent
==================================================
Provides real system control: volume, brightness, lock, sleep, shutdown,
restart, and recycle bin management.

Permission Tiers
----------------
T1 — Reversible system actions (volume, brightness, lock)
T2 — Session-disruptive actions (sleep, empty recycle bin, close_app)
T3 — Destructive / irreversible (shutdown, restart)

Actions
-------
set_volume         — Set or adjust system volume via pycaw          (T1)
set_brightness     — Set screen brightness (if hardware supports)   (T1)
lock_workstation   — Lock the current Windows session               (T1)
sleep_system       — Put the computer to sleep                      (T2)
shutdown_system    — Shutdown the computer                          (T3)
restart_system     — Restart the computer                           (T3)
empty_recycle_bin  — Empty the Windows Recycle Bin                  (T2)

Architecture
------------
- volume: pycaw (COM-based) with graceful stub fallback.
- brightness: screen_brightness_control library with WMI fallback.
- lock: ctypes.windll.user32.LockWorkStation().
- sleep: ctypes.windll.powrprof.SetSuspendState().
- shutdown/restart: subprocess + shutdown.exe.
- recycle bin: winshell.recycle_bin().empty().
- All Windows-specific imports are guarded with try/except.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.skills.base import (
    BaseSkill,
    ParamSchema,
    SkillCapability,
    SkillContext,
    SkillResult,
)

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)


class SystemControlSkill(BaseSkill):
    """
    Real system control agent for Windows.

    This skill provides the full implementations of actions that were stubs
    in the original SystemSkill (set_volume, lock_workstation) and adds
    new capabilities for brightness, sleep, shutdown, restart, and recycle bin.

    Parameters
    ----------
    bus:
        EventBus for publishing system control events. May be None.
    """

    name = "system_control_skill"
    version = "1.0.0"

    def __init__(self, bus: "EventBus | None" = None) -> None:
        self._bus = bus

    # ── Capabilities ──────────────────────────────────────────────────────

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="set_volume",
                description="Set, increase, decrease, or mute the system volume.",
                permission_tier="T1",
                params=[
                    ParamSchema("level", "int", required=False,
                                description="Target volume level 0–100."),
                    ParamSchema("direction", "string", required=False,
                                description="'up', 'down', 'mute', or 'unmute'."),
                    ParamSchema("step", "int", required=False, default=10,
                                description="Step size for up/down adjustments."),
                ],
                examples=[
                    "set volume to 50", "volume up", "turn volume down",
                    "mute", "unmute", "increase volume", "lower the volume",
                    "volume 75", "set volume to maximum",
                ],
            ),
            SkillCapability(
                action="set_brightness",
                description="Set the screen brightness level (if hardware supports it).",
                permission_tier="T1",
                params=[
                    ParamSchema("level", "int", required=True,
                                description="Brightness level 0–100."),
                ],
                examples=[
                    "set brightness to 50", "increase brightness",
                    "lower screen brightness", "dim the screen",
                    "brightness 80",
                ],
            ),
            SkillCapability(
                action="lock_workstation",
                description="Lock the current Windows session immediately.",
                permission_tier="T1",
                examples=[
                    "lock my computer", "lock the screen", "lock workstation",
                    "lock session", "secure the computer",
                ],
            ),
            SkillCapability(
                action="sleep_system",
                description="Put the computer into sleep (suspend) mode.",
                permission_tier="T2",
                examples=[
                    "sleep", "put computer to sleep", "suspend",
                    "sleep mode", "hibernate", "go to sleep",
                ],
            ),
            SkillCapability(
                action="shutdown_system",
                description="Shut down the computer.",
                permission_tier="T3",
                params=[
                    ParamSchema("delay", "int", required=False, default=0,
                                description="Delay in seconds before shutdown (0 = immediate)."),
                ],
                examples=[
                    "shutdown", "shut down the computer", "power off",
                    "turn off computer", "shutdown in 60 seconds",
                ],
            ),
            SkillCapability(
                action="restart_system",
                description="Restart the computer.",
                permission_tier="T3",
                params=[
                    ParamSchema("delay", "int", required=False, default=0,
                                description="Delay in seconds before restart (0 = immediate)."),
                ],
                examples=[
                    "restart", "reboot", "restart the computer",
                    "reboot system", "restart in 30 seconds",
                ],
            ),
            SkillCapability(
                action="empty_recycle_bin",
                description="Permanently empty the Windows Recycle Bin.",
                permission_tier="T2",
                examples=[
                    "empty recycle bin", "clear recycle bin",
                    "empty trash", "delete recycle bin contents",
                ],
            ),
        ]

    # ── Dispatch ──────────────────────────────────────────────────────────

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "set_volume":
            return await self._set_volume(context)
        if action == "set_brightness":
            return await self._set_brightness(context)
        if action == "lock_workstation":
            return await self._lock_workstation(context)
        if action == "sleep_system":
            return await self._sleep_system(context)
        if action == "shutdown_system":
            return await self._shutdown_system(context)
        if action == "restart_system":
            return await self._restart_system(context)
        if action == "empty_recycle_bin":
            return await self._empty_recycle_bin(context)
        return SkillResult.fail(f"SystemControlSkill: unknown action '{action}'.")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _set_volume(self, context: SkillContext) -> SkillResult:
        level_raw = context.get("level")
        direction: str = (context.get("direction") or "").lower().strip()
        step = int(context.get("step") or 10)
        step = max(1, min(step, 50))

        try:
            new_level, muted = await asyncio.to_thread(
                self._do_set_volume, level_raw, direction, step
            )
        except RuntimeError as exc:
            return SkillResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to set volume: {exc}", error=exc
            )

        await self._emit_volume_changed(new_level, muted, context.session_id)

        if muted:
            return SkillResult.ok("Muted.", data={"level": new_level, "muted": True},
                                  action_taken="set_volume")
        if direction == "unmute":
            return SkillResult.ok(
                f"Unmuted. Volume at {new_level}%.",
                data={"level": new_level, "muted": False},
                action_taken="set_volume",
            )
        return SkillResult.ok(
            f"Volume set to {new_level}%.",
            data={"level": new_level, "muted": False},
            action_taken="set_volume",
        )

    async def _set_brightness(self, context: SkillContext) -> SkillResult:
        level_raw = context.get("level")
        if level_raw is None:
            return SkillResult.fail(
                "SystemControlSkill: 'level' parameter is required for set_brightness."
            )

        try:
            level = int(level_raw)
            level = max(0, min(100, level))
        except (ValueError, TypeError):
            return SkillResult.fail(
                "SystemControlSkill: 'level' must be an integer 0–100."
            )

        try:
            actual_level = await asyncio.to_thread(self._do_set_brightness, level)
        except RuntimeError as exc:
            return SkillResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to set brightness: {exc}", error=exc
            )

        await self._emit_brightness_changed(actual_level, context.session_id)
        return SkillResult.ok(
            f"Brightness set to {actual_level}%.",
            data={"level": actual_level},
            action_taken="set_brightness",
        )

    async def _lock_workstation(self, context: SkillContext) -> SkillResult:
        try:
            success = await asyncio.to_thread(self._do_lock_workstation)
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to lock workstation: {exc}", error=exc
            )

        if not success:
            return SkillResult.fail(
                "SystemControlSkill: LockWorkStation returned failure. "
                "This may not be supported in the current session."
            )

        await self._emit_workstation_locked(context.session_id)
        return SkillResult.ok(
            "Workstation locked.",
            action_taken="lock_workstation",
        )

    async def _sleep_system(self, context: SkillContext) -> SkillResult:
        try:
            await asyncio.to_thread(self._do_sleep)
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to initiate sleep: {exc}", error=exc
            )

        await self._emit_sleep_initiated(context.session_id)
        return SkillResult.ok(
            "System going to sleep.",
            action_taken="sleep_system",
        )

    async def _shutdown_system(self, context: SkillContext) -> SkillResult:
        delay = int(context.get("delay") or 0)
        delay = max(0, min(delay, 3600))  # Cap at 1 hour

        try:
            await asyncio.to_thread(self._do_shutdown, delay, restart=False)
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to initiate shutdown: {exc}", error=exc
            )

        await self._emit_shutdown_initiated(delay, context.session_id)
        if delay > 0:
            return SkillResult.ok(
                f"System will shut down in {delay} seconds. "
                f"Run 'shutdown /a' to abort.",
                data={"delay": delay},
                action_taken="shutdown_system",
            )
        return SkillResult.ok("Shutting down now.", action_taken="shutdown_system")

    async def _restart_system(self, context: SkillContext) -> SkillResult:
        delay = int(context.get("delay") or 0)
        delay = max(0, min(delay, 3600))

        try:
            await asyncio.to_thread(self._do_shutdown, delay, restart=True)
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to initiate restart: {exc}", error=exc
            )

        await self._emit_restart_initiated(delay, context.session_id)
        if delay > 0:
            return SkillResult.ok(
                f"System will restart in {delay} seconds. "
                f"Run 'shutdown /a' to abort.",
                data={"delay": delay},
                action_taken="restart_system",
            )
        return SkillResult.ok("Restarting now.", action_taken="restart_system")

    async def _empty_recycle_bin(self, context: SkillContext) -> SkillResult:
        try:
            await asyncio.to_thread(self._do_empty_recycle_bin)
        except RuntimeError as exc:
            return SkillResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"SystemControlSkill: failed to empty recycle bin: {exc}", error=exc
            )

        await self._emit_recycle_bin_emptied(context.session_id)
        return SkillResult.ok(
            "Recycle Bin emptied.",
            action_taken="empty_recycle_bin",
        )

    # ── Internal Windows implementations ─────────────────────────────────

    @staticmethod
    def _do_set_volume(
        level_raw: int | None,
        direction: str,
        step: int,
    ) -> tuple[int, bool]:
        """
        Set system volume using pycaw (Windows Audio Session API).

        Returns (new_level_0_100, is_muted).
        Raises RuntimeError if pycaw is not available.
        """
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
            from comtypes import CLSCTX_ALL  # type: ignore
        except ImportError:
            raise RuntimeError(
                "SystemControlSkill: pycaw is not installed. "
                "Run: pip install pycaw"
            )

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)

        # Current state
        current_scalar = volume.GetMasterVolumeLevelScalar()
        current_level = round(current_scalar * 100)
        is_muted = volume.GetMute()

        if direction == "mute":
            volume.SetMute(True, None)
            return current_level, True

        if direction == "unmute":
            volume.SetMute(False, None)
            return current_level, False

        # Calculate target level
        if level_raw is not None:
            target = max(0, min(100, int(level_raw)))
        elif direction == "up":
            target = min(100, current_level + step)
        elif direction == "down":
            target = max(0, current_level - step)
        else:
            # No direction, no level — just return current
            return current_level, bool(is_muted)

        scalar = target / 100.0
        volume.SetMasterVolumeLevelScalar(scalar, None)
        if is_muted and target > 0:
            volume.SetMute(False, None)

        return target, False

    @staticmethod
    def _do_set_brightness(level: int) -> int:
        """
        Set screen brightness using screen_brightness_control.
        Falls back to WMI on failure.
        Returns the actual level set.
        Raises RuntimeError if not supported.
        """
        try:
            import screen_brightness_control as sbc  # type: ignore
            sbc.set_brightness(level)
            result = sbc.get_brightness()
            if isinstance(result, list):
                return result[0] if result else level
            return int(result)
        except ImportError:
            pass  # Try WMI fallback

        # WMI fallback
        try:
            import wmi  # type: ignore
            w = wmi.WMI(namespace="wmi")
            for monitor in w.WmiMonitorBrightnessMethods():
                monitor.WmiSetBrightness(level, 0)
            return level
        except Exception:
            pass

        raise RuntimeError(
            "SystemControlSkill: brightness control is not supported on this hardware. "
            "Install screen_brightness_control: pip install screen-brightness-control"
        )

    @staticmethod
    def _do_lock_workstation() -> bool:
        """Lock the Windows workstation. Returns True on success."""
        try:
            import ctypes
            result = ctypes.windll.user32.LockWorkStation()
            return bool(result)
        except AttributeError:
            raise RuntimeError(
                "SystemControlSkill: lock_workstation is only supported on Windows."
            )

    @staticmethod
    def _do_sleep() -> None:
        """Put the system into sleep (suspend) mode."""
        try:
            import ctypes
            # SetSuspendState(Hibernate=False, ForceCritical=False, DisableWakeEvent=False)
            ctypes.windll.powrprof.SetSuspendState(False, False, False)
        except AttributeError:
            raise RuntimeError(
                "SystemControlSkill: sleep_system is only supported on Windows."
            )

    @staticmethod
    def _do_shutdown(delay: int, restart: bool) -> None:
        """Execute shutdown or restart via shutdown.exe."""
        flag = "/r" if restart else "/s"
        cmd = ["shutdown", flag, "/t", str(delay)]
        if delay == 0:
            cmd += ["/f"]  # Force close apps immediately
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"SystemControlSkill: shutdown command failed: {exc.stderr.decode()}"
            ) from exc

    @staticmethod
    def _do_empty_recycle_bin() -> None:
        """Empty the Windows Recycle Bin using winshell."""
        try:
            import winshell  # type: ignore
            winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=False)
        except ImportError:
            # Fallback: use SHEmptyRecycleBin via ctypes
            try:
                import ctypes
                # SHEmptyRecycleBinW(hwnd=None, pszRootPath=None, dwFlags=7)
                # SHERB_NOCONFIRMATION|SHERB_NOPROGRESSUI|SHERB_NOSOUND = 7
                result = ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 7)
                if result not in (0, -2147418113):  # S_OK or already empty
                    raise RuntimeError(
                        f"SystemControlSkill: SHEmptyRecycleBinW returned {result:#x}"
                    )
            except AttributeError:
                raise RuntimeError(
                    "SystemControlSkill: empty_recycle_bin is only supported on Windows."
                )

    # ── EventBus publishers ───────────────────────────────────────────────

    async def _emit_volume_changed(
        self, level: int, muted: bool, session_id: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import VolumeChangedEvent
        await self._bus.publish(VolumeChangedEvent(
            level=level, muted=muted, session_id=session_id
        ))

    async def _emit_brightness_changed(self, level: int, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import BrightnessChangedEvent
        await self._bus.publish(BrightnessChangedEvent(level=level, session_id=session_id))

    async def _emit_workstation_locked(self, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import WorkstationLockedEvent
        await self._bus.publish(WorkstationLockedEvent(session_id=session_id))

    async def _emit_sleep_initiated(self, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import SystemSleepInitiatedEvent
        await self._bus.publish(SystemSleepInitiatedEvent(session_id=session_id))

    async def _emit_shutdown_initiated(self, delay: int, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import SystemShutdownInitiatedEvent
        await self._bus.publish(SystemShutdownInitiatedEvent(
            delay_seconds=delay, session_id=session_id
        ))

    async def _emit_restart_initiated(self, delay: int, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import SystemRestartInitiatedEvent
        await self._bus.publish(SystemRestartInitiatedEvent(
            delay_seconds=delay, session_id=session_id
        ))

    async def _emit_recycle_bin_emptied(self, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import RecycleBinEmptiedEvent
        await self._bus.publish(RecycleBinEmptiedEvent(session_id=session_id))
