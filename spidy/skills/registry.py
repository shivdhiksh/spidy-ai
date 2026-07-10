"""
SkillRegistry — Central Capability Catalog
==========================================
Maintains the set of all loaded Skills and provides fast lookups
for the Planner (Brain) to match intents to executable actions.

Architecture
------------
The registry follows a publish/subscribe model:
  1. Skills register themselves at startup
  2. The Planner queries the registry to discover capabilities
  3. The Executor looks up the concrete Skill for a given action

Design decisions
----------------
- Skills are stored in a dict keyed by name (O(1) lookup)
- Capability → Skill reverse mapping is built at registration time
  for fast intent dispatch
- Thread-safe: uses a RLock so the Brain can query during async execution
- No skill has direct access to any other skill (enforced by registry API)

This module has zero dependencies on voice, UI, or Brain code.
It is a pure data structure.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.skills.base import BaseSkill, SkillCapability

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class SkillRegistry:
    """
    Central catalog of all registered Skills.

    Usage
    -----
        registry = SkillRegistry()
        registry.register(MySkill())

        skill = registry.find_skill_for_action("open_browser")
        if skill:
            result = await skill.execute("open_browser", ctx)

    Methods
    -------
    register(skill)
        Add a skill. Raises if a skill with the same name is already
        registered (prevents silent overwrites in production).

    unregister(skill_name)
        Remove a skill by name. No-op if not registered.

    find_skill_for_action(action)
        Return the Skill that supports this action, or None.

    all_capabilities()
        Return all SkillCapabilities across all registered skills.

    all_skills()
        Return all registered skill instances.
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}
        # action → skill name mapping for O(1) dispatch
        self._action_map: dict[str, str] = {}
        self._lock = threading.RLock()

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, skill: BaseSkill, overwrite: bool = False) -> None:
        """
        Register a skill with the registry.

        Parameters
        ----------
        skill:
            An instance of a BaseSkill subclass. Must be fully initialised.
        overwrite:
            If True, silently replace any existing skill with the same name.
            Default: False (raises on duplicate).

        Raises
        ------
        ValueError
            If a skill with the same name is already registered and
            overwrite=False.
        """
        with self._lock:
            name = skill.name
            if name in self._skills and not overwrite:
                raise ValueError(
                    f"Skill '{name}' is already registered. "
                    f"Use overwrite=True to replace it."
                )

            self._skills[name] = skill

            # Build reverse action→skill mapping
            for cap in skill.capabilities():
                if cap.action in self._action_map and not overwrite:
                    existing_skill = self._action_map[cap.action]
                    log.warning(
                        "Action '{a}' from skill '{new}' shadows existing "
                        "skill '{old}'.",
                        a=cap.action,
                        new=name,
                        old=existing_skill,
                    )
                self._action_map[cap.action] = name

            log.info(
                "Skill registered: '{name}' | actions={actions}",
                name=name,
                actions=[c.action for c in skill.capabilities()],
            )

    def unregister(self, skill_name: str) -> None:
        """
        Remove a skill and all its actions from the registry.

        Parameters
        ----------
        skill_name:
            The name attribute of the skill to remove.
        """
        with self._lock:
            skill = self._skills.pop(skill_name, None)
            if skill is None:
                log.debug("unregister: skill '{name}' not found.", name=skill_name)
                return

            # Remove its actions from the reverse map
            actions_to_remove = [
                action
                for action, sname in self._action_map.items()
                if sname == skill_name
            ]
            for action in actions_to_remove:
                del self._action_map[action]

            log.info("Skill unregistered: '{name}'", name=skill_name)

    # ── Queries ───────────────────────────────────────────────────────────

    def find_skill_for_action(self, action: str) -> BaseSkill | None:
        """
        Return the Skill that handles the given action.

        Parameters
        ----------
        action:
            The action name (e.g. ``"open_browser"``, ``"read_file"``).

        Returns
        -------
        BaseSkill | None
            The skill instance, or None if no skill handles this action.
        """
        with self._lock:
            skill_name = self._action_map.get(action)
            if skill_name is None:
                return None
            return self._skills.get(skill_name)

    def find_skill_by_name(self, name: str) -> BaseSkill | None:
        """Return a skill by its registered name, or None."""
        with self._lock:
            return self._skills.get(name)

    def all_capabilities(self) -> list[SkillCapability]:
        """
        Return all capabilities across all registered skills.

        Used by the Planner to build its intent-matching table.
        """
        with self._lock:
            caps: list[SkillCapability] = []
            for skill in self._skills.values():
                caps.extend(skill.capabilities())
            return caps

    def all_skills(self) -> list[BaseSkill]:
        """Return all registered skill instances."""
        with self._lock:
            return list(self._skills.values())

    def __len__(self) -> int:
        """Number of registered skills."""
        with self._lock:
            return len(self._skills)

    def __contains__(self, skill_name: str) -> bool:
        """True if a skill with this name is registered."""
        with self._lock:
            return skill_name in self._skills

    def __repr__(self) -> str:
        with self._lock:
            names = list(self._skills.keys())
        return f"SkillRegistry({names})"
