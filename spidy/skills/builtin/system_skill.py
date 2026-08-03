"""
SystemSkill — System information.
Permission tier: T0.

Actions
-------
get_system_info  — CPU, RAM, battery, platform info   (T0)

Note
----
Volume control and screen lock are provided by SystemControlSkill (desktop package).
This skill only handles read-only system introspection.
"""

from __future__ import annotations

import platform
from typing import Any

from spidy.skills.base import BaseSkill, SkillCapability, SkillContext, SkillResult


class SystemSkill(BaseSkill):
    """
    Provides live system information.

    Volume control and screen lock were originally stubs here; they are
    now fully implemented in SystemControlSkill (spidy.skills.desktop).
    This skill only exposes get_system_info to avoid action name conflicts.
    """

    name = "system_skill"
    version = "1.1.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="get_system_info",
                description="Get current CPU, memory, battery, and platform information.",
                permission_tier="T0",
                examples=["system info", "how is my computer doing",
                          "cpu usage", "battery level", "ram usage",
                          "check system status", "what's my memory usage"],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "get_system_info":
            return await self._get_system_info(context)
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
