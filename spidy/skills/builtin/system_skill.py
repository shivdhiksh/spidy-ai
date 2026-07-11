"""
SystemSkill — Safe system information and control actions.
Permission tiers: T0 for info, T1 for control.

Actions
-------
get_system_info  — CPU, RAM, battery, platform info   (T0)
set_volume       — Change system volume (stub, M7+)   (T1)
lock_screen      — Lock the Windows session (stub)    (T1)
"""

from __future__ import annotations

import platform
from typing import Any

from spidy.skills.base import BaseSkill, SkillCapability, ParamSchema, SkillContext, SkillResult


class SystemSkill(BaseSkill):
    """
    Provides system information and basic control actions.

    Volume control and screen lock are stubs in M4.
    Full implementation requires pyaudio / win32 in M7 (Desktop Agent).
    System info is live and uses psutil when available.
    """

    name = "system_skill"
    version = "1.0.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="get_system_info",
                description="Get current CPU, memory, battery, and platform information.",
                permission_tier="T0",
                examples=["system info", "how is my computer doing",
                          "cpu usage", "battery level", "ram usage"],
            ),
            SkillCapability(
                action="set_volume",
                description="Set or change the system volume.",
                permission_tier="T1",
                params=[ParamSchema("level", "int", required=False,
                                    description="Volume level 0–100. Omit to mute/unmute.")],
                examples=["set volume to 50", "volume up", "mute", "unmute",
                          "turn volume down"],
            ),
            SkillCapability(
                action="lock_screen",
                description="Lock the current Windows session.",
                permission_tier="T1",
                examples=["lock screen", "lock my computer", "lock the session"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "get_system_info":
            return await self._get_system_info(context)
        if action == "set_volume":
            return await self._set_volume(context)
        if action == "lock_screen":
            return await self._lock_screen(context)
        return SkillResult.fail(f"SystemSkill: unknown action '{action}'.")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _get_system_info(self, _context: SkillContext) -> SkillResult:
        info: dict[str, Any] = {
            "platform": platform.system(),
            "platform_version": platform.version()[:50],
            "python": platform.python_version(),
        }
        lines: list[str] = [
            f"Platform: {info['platform']} {platform.release()}",
        ]

        try:
            import psutil

            # CPU
            cpu_pct = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count(logical=True)
            info["cpu_percent"] = cpu_pct
            info["cpu_count"] = cpu_count
            lines.append(f"CPU: {cpu_pct}% ({cpu_count} logical cores)")

            # RAM
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / 1024**3
            ram_total_gb = ram.total / 1024**3
            info["ram_percent"] = ram.percent
            info["ram_used_gb"] = round(ram_used_gb, 2)
            info["ram_total_gb"] = round(ram_total_gb, 2)
            lines.append(
                f"RAM: {ram.percent}% used ({ram_used_gb:.1f} / {ram_total_gb:.1f} GB)"
            )

            # Battery
            battery = psutil.sensors_battery()
            if battery:
                charging = "charging" if battery.power_plugged else "on battery"
                info["battery_percent"] = round(battery.percent)
                info["battery_charging"] = battery.power_plugged
                lines.append(f"Battery: {battery.percent:.0f}% ({charging})")

        except ImportError:
            lines.append("(psutil not installed — limited system info)")

        msg = "System status:\n" + "\n".join(lines)
        return SkillResult.ok(msg, data=info, action_taken="get_system_info")

    async def _set_volume(self, context: SkillContext) -> SkillResult:
        # Stub — full implementation requires pycaw / win32 (Milestone 7)
        level = context.get("level")
        if level is not None:
            try:
                level = int(level)
                level = max(0, min(100, level))
                msg = (
                    f"Volume control is not yet implemented (coming in Milestone 7). "
                    f"Requested level: {level}%"
                )
            except (ValueError, TypeError):
                msg = "Volume control is not yet implemented (coming in Milestone 7)."
        else:
            msg = "Volume control is not yet implemented (coming in Milestone 7)."
        return SkillResult.ok(msg, action_taken="set_volume")

    async def _lock_screen(self, _context: SkillContext) -> SkillResult:
        # Stub — full implementation uses win32api.LockWorkStation() (Milestone 7)
        return SkillResult.ok(
            "Screen lock is not yet implemented (coming in Milestone 7).",
            action_taken="lock_screen",
        )
