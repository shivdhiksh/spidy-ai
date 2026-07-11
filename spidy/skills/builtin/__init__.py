"""
Built-in Skills Package
=======================
Convenience loader for all built-in Spidy skills.

Usage
-----
    from spidy.skills.builtin import register_builtin_skills

    registry = SkillRegistry()
    register_builtin_skills(registry, config=settings.skills)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spidy.config.manager import SkillsConfig
    from spidy.skills.registry import SkillRegistry


def register_builtin_skills(
    registry: "SkillRegistry",
    config: "SkillsConfig | None" = None,
    bus: object = None,
    notes_file: str = "notes.jsonl",
) -> list[str]:
    """
    Register all enabled built-in skills with the given SkillRegistry.

    Parameters
    ----------
    registry:
        SkillRegistry instance to register skills into.
    config:
        SkillsConfig from SpidyConfig. If None, all skills are enabled.
    bus:
        EventBus for skills that emit events (e.g. TimerSkill).
    notes_file:
        Path to the JSONL notes file for NoteSkill.

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

    if _enabled("help_skill_enabled"):
        try:
            from spidy.skills.builtin.help_skill import HelpSkill
            registry.register(HelpSkill(registry=registry))
            registered.append("help_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register HelpSkill: {exc}", exc=exc)

    if _enabled("time_skill_enabled"):
        try:
            from spidy.skills.builtin.time_skill import TimeSkill
            registry.register(TimeSkill())
            registered.append("time_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register TimeSkill: {exc}", exc=exc)

    if _enabled("greet_skill_enabled"):
        try:
            from spidy.skills.builtin.greet_skill import GreetSkill
            registry.register(GreetSkill())
            registered.append("greet_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register GreetSkill: {exc}", exc=exc)

    if _enabled("note_skill_enabled"):
        try:
            from spidy.skills.builtin.note_skill import NoteSkill
            nf = getattr(config, "notes_file", notes_file) if config else notes_file
            registry.register(NoteSkill(notes_file=nf))
            registered.append("note_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register NoteSkill: {exc}", exc=exc)

    if _enabled("timer_skill_enabled"):
        try:
            from spidy.skills.builtin.timer_skill import TimerSkill
            registry.register(TimerSkill(bus=bus))
            registered.append("timer_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register TimerSkill: {exc}", exc=exc)

    if _enabled("system_skill_enabled"):
        try:
            from spidy.skills.builtin.system_skill import SystemSkill
            registry.register(SystemSkill())
            registered.append("system_skill")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to register SystemSkill: {exc}", exc=exc)

    log.info(
        "Built-in skills registered: {skills}",
        skills=registered,
    )
    return registered


__all__ = ["register_builtin_skills"]
