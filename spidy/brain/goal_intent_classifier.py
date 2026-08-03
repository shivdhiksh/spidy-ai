"""
GoalIntentClassifier — Intent → Execution Route Decision (Milestone 13.1)
==========================================================================
Given a classified Intent, decides whether to route the request to:

  "goal"  → brain.run_goal(utterance)   — AutonomousAgent multi-step pipeline
  "chat"  → brain.process(utterance)    — Single-step Brain pipeline (existing)

Design rationale
----------------
The existing Brain pipeline (brain.process) handles single-step executable
commands perfectly well via the ToolRouter + skill system. The AutonomousAgent
(brain.run_goal) is used when the request implies:

  1. Multiple sequential steps   (e.g. "Create a React project")
  2. Desktop automation          (e.g. "Create a folder on the Desktop")
  3. File-system operations      (e.g. "Find all PDF files in Downloads")
  4. Browser automation          (e.g. "Search YouTube for Python tutorials")
  5. System/app control          (e.g. "Open VS Code", "Open Chrome")
  6. Multi-intent compound goals (e.g. "Open Edge and search for Python")

Conversational intents (greet, chat, calculate, note, help) stay in
brain.process() because the Brain already handles them perfectly.

Rule: When in doubt, route to "chat" (conservative). This avoids accidentally
sending a greeting like "Hi Spidy" through the agent loop.

Usage
-----
    classifier = GoalIntentClassifier()
    route = classifier.classify(intent)          # "goal" or "chat"
    is_goal = classifier.is_executable(intent)   # bool

    # Shorthand function (no instantiation required):
    from spidy.brain.goal_intent_classifier import is_goal_intent
    if is_goal_intent(intent):
        response = await brain.run_goal(utterance)
    else:
        response = await brain.process(utterance)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.brain.types import Intent

log = get_logger(__name__)

# ─── Action Sets ──────────────────────────────────────────────────────────────
#
# GOAL_ACTIONS: actions that map to real-world, executable desktop operations.
# These are routed to brain.run_goal() so the AutonomousAgent can plan,
# execute, monitor, and recover from each step.
#
GOAL_ACTIONS: frozenset[str] = frozenset({
    # Application control
    "launch_app",
    "close_app",
    "bring_app_to_foreground",
    "detect_running_apps",

    # File system
    "search_files",
    "search_folders",
    "open_file",
    "open_folder",
    "reveal_in_explorer",
    "list_recent_files",
    "create_folder",
    "delete_folder",
    "rename_folder",
    "move_file",
    "copy_file",
    "delete_file",
    "read_file",

    # Browser & web
    "search_youtube",
    "search_google",
    "search_web",
    "search_bing",
    "search_duckduckgo",
    "search_wikipedia",
    "search_maps",
    "open_url",
    "open_browser_and_search",
    "read_webpage",
    "summarize_webpage",
    "open_project_in_browser",

    # System control
    "set_volume",
    "set_brightness",
    "lock_workstation",
    "sleep_system",
    "shutdown_system",
    "restart_system",
    "empty_recycle_bin",
    "get_system_info",
    "take_screenshot",

    # Project / workspace
    "open_project",
    "create_project",
    "run_command",
    "open_terminal",

    # Compound (multi-step)
    "compound",

    # Goal management
    "cancel_goal",
})

# CHAT_ACTIONS: actions that are purely conversational or handled entirely by
# the single-step Brain pipeline without needing autonomous planning.
CHAT_ACTIONS: frozenset[str] = frozenset({
    "chat",
    "greet",
    "farewell",
    "introduce",
    "help",
    "calculate",
    "take_note",
    "read_notes",
    "find_note",
    "set_timer",
    "set_alarm",
    "set_reminder",
    "get_clipboard",
    "get_time",
})


class GoalIntentClassifier:
    """
    Routes a classified Intent to either the AutonomousAgent or the
    single-step Brain pipeline.

    Parameters
    ----------
    min_confidence:
        Intents below this confidence threshold are always sent to
        brain.process() (conservative — avoids misrouting ambiguous input).
    """

    def __init__(self, min_confidence: float = 0.65) -> None:
        self._min_confidence = min_confidence

    def classify(self, intent: "Intent") -> str:
        """
        Return "goal" or "chat" for the given intent.

        Parameters
        ----------
        intent:
            A classified Intent from IntentClassifier.

        Returns
        -------
        str
            ``"goal"`` if the intent should be routed to AutonomousAgent.
            ``"chat"`` if the intent should be handled by brain.process().
        """
        # Low-confidence intents → chat (never trust ambiguous classification)
        if intent.confidence < self._min_confidence:
            log.debug(
                "GoalIntentClassifier: low confidence ({c:.0%}) → chat",
                c=intent.confidence,
            )
            return "chat"

        action = intent.action

        # Explicit goal action
        if action in GOAL_ACTIONS:
            log.debug(
                "GoalIntentClassifier: '{a}' → goal",
                a=action,
            )
            return "goal"

        # Explicit chat action
        if action in CHAT_ACTIONS:
            log.debug(
                "GoalIntentClassifier: '{a}' → chat",
                a=action,
            )
            return "chat"

        # Unknown action: conservatively route to chat
        log.debug(
            "GoalIntentClassifier: unknown action '{a}' → chat (conservative)",
            a=action,
        )
        return "chat"

    def is_executable(self, intent: "Intent") -> bool:
        """Return True if this intent should be executed via AutonomousAgent."""
        return self.classify(intent) == "goal"


# ─── Module-level convenience ─────────────────────────────────────────────────

_default_classifier = GoalIntentClassifier()


def is_goal_intent(intent: "Intent") -> bool:
    """
    Module-level shorthand: return True if *intent* should go to run_goal().

    Creates a single shared GoalIntentClassifier instance on first call.
    """
    return _default_classifier.is_executable(intent)


def get_route(intent: "Intent") -> str:
    """Return 'goal' or 'chat' for the given intent (module-level shorthand)."""
    return _default_classifier.classify(intent)
