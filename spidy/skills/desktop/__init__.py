"""
Desktop Skills Package
======================
Loader for all Milestone 6 Desktop & File Agent skills.

Usage
-----
    from spidy.skills.desktop import register_desktop_skills

    registry = SkillRegistry()
    register_desktop_skills(registry, config=settings.skills, bus=bus)

Skills Registered
-----------------
- file_skill         — File/folder search, open, reveal (T0/T1)
- app_skill          — Launch, detect, focus, close apps (T0/T1/T2)
- system_control_skill — Volume, brightness, lock, sleep, shutdown (T1/T2/T3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spidy.config.manager import SkillsConfig
    from spidy.core.event_bus import EventBus
    from spidy.skills.registry import SkillRegistry


def register_desktop_skills(
    registry: "SkillRegistry",
    config: "SkillsConfig | None" = None,
    bus: "EventBus | None" = None,
) -> list[str]:
    """
    Register all enabled Desktop & File Agent skills.

    Parameters
    ----------
    registry:
        SkillRegistry instance to register skills into.
    config:
        SkillsConfig from SpidyConfig. If None, all skills are enabled.
    bus:
        EventBus for skills that emit events.

    Returns
    -------
    list[str]
        Names of skills that were successfully registered.
    """
    from spidy.logging.logger import get_logger
    log = get_logger(__name__)

    registered: list[str] = []

    def _enabled(flag: str) -> bool:
        if config is None:
            return True
        return getattr(config, flag, True)

    def _int_opt(flag: str, default: int) -> int:
        if config is None:
            return default
        return int(getattr(config, flag, default))

    # ── FileSkill ─────────────────────────────────────────────────────────
    if _enabled("file_skill_enabled"):
        try:
            from spidy.skills.desktop.file_skill import FileSkill
            max_results = _int_opt("desktop_file_search_max_results", 50)
            search_root = getattr(config, "desktop_file_search_root", None) if config else None
            registry.register(FileSkill(bus=bus, search_root=search_root,
                                        max_results=max_results))
            registered.append("file_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register FileSkill: {exc}", exc=exc)

    # ── AppSkill ──────────────────────────────────────────────────────────
    if _enabled("app_skill_enabled"):
        try:
            from spidy.skills.desktop.app_skill import AppSkill
            registry.register(AppSkill(bus=bus))
            registered.append("app_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register AppSkill: {exc}", exc=exc)

    # ── SystemControlSkill ────────────────────────────────────────────────
    if _enabled("system_control_skill_enabled"):
        try:
            from spidy.skills.desktop.system_control_skill import SystemControlSkill
            registry.register(SystemControlSkill(bus=bus))
            registered.append("system_control_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register SystemControlSkill: {exc}", exc=exc)

    # ── ComputerControlSkill ──────────────────────────────────────────────
    if _enabled("computer_control_skill_enabled"):
        try:
            from spidy.skills.desktop.computer_control_skill import ComputerControlSkill
            registry.register(ComputerControlSkill(bus=bus))
            registered.append("computer_control_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register ComputerControlSkill: {exc}", exc=exc)

    log.info(
        "Desktop skills registered: {skills}",
        skills=registered,
    )
    return registered


__all__ = ["register_desktop_skills"]

