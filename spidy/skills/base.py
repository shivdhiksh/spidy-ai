"""
BaseSkill — Contract for All Spidy Skills
==========================================
Every capability Spidy can perform is implemented as a Skill.

Architecture
------------
- Skills are stateless, focused units of capability.
- They never call other modules directly — only through the SkillContext.
- All exceptions are caught and returned in SkillResult — never propagated.
- Skills declare their capabilities up-front so the SkillRegistry can
  build a searchable index without importing every skill eagerly.

The skill system replaces the flat "agents" model. Key differences:

  Agents (v1.0):         Skills (v2.0):
  ─────────────          ─────────────
  Ad-hoc dispatch        Formal capability registry
  No param schema        Typed param declarations
  No permission tier     Permission tier per capability
  No lifecycle hooks     on_load / on_unload hooks
  No version             Versioned (skill.version)
  Monolithic             Composable (Brain picks combinations)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


# ─── Supporting data structures ───────────────────────────────────────────────


@dataclass(frozen=True)
class ParamSchema:
    """Declares a single parameter for a skill capability."""
    name: str
    type: str                   # "string" | "bool" | "int" | "float" | "path"
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class SkillCapability:
    """
    Declares one action a skill can perform.

    Used by SkillRegistry to build the CapabilityIndex.
    Used by the Planner to know what Spidy can do.
    Used by PermissionManager to apply the correct tier.
    """
    action: str                         # e.g. "search_files"
    description: str                    # human-readable, used in LLM prompts
    permission_tier: str = "T0"         # T0 | T1 | T2 | T3
    params: list[ParamSchema] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)  # example utterances


@dataclass
class SkillContext:
    """
    Execution context passed to every skill.execute() call.

    Skills receive this instead of accessing global state directly.
    This makes skills testable — inject a fake context in tests.

    Attributes
    ----------
    action:
        The action string being requested.
    params:
        Key-value parameters extracted by the Brain.
    session_id:
        Current conversation session ID.
    desktop_state:
        Current desktop snapshot (active app, window, clipboard, etc.).
        May be None if ContextAwareness is not yet initialised.
    user_name:
        The user's name from config (for personalised messages).
    """
    action: str
    params: dict[str, Any]
    session_id: str = ""
    desktop_state: Any = None       # DesktopStateSnapshot | None
    user_name: str = "User"
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience: get a param by key with a default."""
        return self.params.get(key, default)


@dataclass
class SkillResult:
    """
    Standardised result returned by every skill execution.

    Skills MUST always return a SkillResult — never raise to the caller.
    The Executor and Verifier inspect this to decide on retry/fallback.

    Attributes
    ----------
    success:
        Whether the action completed without error.
    message:
        Human-readable summary shown to the user via TTS + UI.
    data:
        Optional structured payload (e.g. file list, screenshot path,
        generated code). The Brain uses this for follow-up reasoning.
    error:
        The exception if the action failed. For logging and retry decisions.
    action_taken:
        Description of what was actually done (may differ from intent
        if partial execution occurred).
    """
    success: bool
    message: str
    data: Any = None
    error: Exception | None = None
    action_taken: str = ""

    @classmethod
    def ok(cls, message: str, data: Any = None, action_taken: str = "") -> "SkillResult":
        """Convenience constructor for successful results."""
        return cls(success=True, message=message, data=data, action_taken=action_taken)

    @classmethod
    def fail(cls, message: str, error: Exception | None = None) -> "SkillResult":
        """Convenience constructor for failed results."""
        return cls(success=False, message=message, error=error)


# ─── BaseSkill ABC ────────────────────────────────────────────────────────────


class BaseSkill(ABC):
    """
    Abstract base class for all Spidy skills.

    Subclasses must define:
    - ``name``: class-level string (snake_case, unique across all skills)
    - ``version``: class-level semantic version string
    - ``capabilities()``: list of SkillCapability declarations
    - ``execute()``: perform the requested action

    Lifecycle
    ---------
    1. SkillRegistry calls ``on_load()`` when registering the skill.
    2. ``execute()`` is called by the Executor for each action.
    3. SkillRegistry calls ``on_unload()`` at application shutdown.

    Example
    -------
        class MySkill(BaseSkill):
            name = "my_skill"
            version = "1.0.0"

            def capabilities(self):
                return [
                    SkillCapability("do_thing", "Does a thing", "T0",
                                    params=[ParamSchema("target", "string")])
                ]

            async def execute(self, action: str, context: SkillContext) -> SkillResult:
                if action == "do_thing":
                    target = context.get("target")
                    return SkillResult.ok(f"Did the thing to {target}")
                return SkillResult.fail(f"Unknown action: {action}")
    """

    # Subclasses MUST define these at class level
    name: ClassVar[str] = "base_skill"
    version: ClassVar[str] = "1.0.0"

    @abstractmethod
    def capabilities(self) -> list[SkillCapability]:
        """
        Declare all actions this skill can perform.

        Called once at registration time. The returned list is indexed
        by the CapabilityIndex for fast action → skill lookup.

        Returns
        -------
        list[SkillCapability]
            One entry per action this skill supports.
        """
        ...

    @abstractmethod
    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        """
        Execute the requested action.

        Parameters
        ----------
        action:
            The action string (must match one in self.capabilities()).
        context:
            Execution context: params, session state, desktop snapshot.

        Returns
        -------
        SkillResult
            Always returns a SkillResult — never raises to the caller.
            Internal exceptions must be caught and wrapped.
        """
        ...

    async def on_load(self) -> None:
        """
        Called once when the skill is registered.

        Use for: loading models, opening connections, warming up caches.
        """

    async def on_unload(self) -> None:
        """
        Called once at application shutdown.

        Use for: closing connections, releasing GPU memory, flushing state.
        """

    def capability_names(self) -> list[str]:
        """Return just the action name strings for quick lookup."""
        return [c.action for c in self.capabilities()]

    def supports(self, action: str) -> bool:
        """Return True if this skill handles the given action."""
        return action in self.capability_names()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', v{self.version})"
