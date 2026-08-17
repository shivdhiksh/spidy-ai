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
    # NOTE: "search_web" has been intentionally moved to CHAT_ACTIONS (P1-2).
    # It is a generic/unresolved search intent that should be answered via the
    # fast LLM-direct path in brain.process(), not via AutonomousAgent planning.
    # Explicit named-engine actions below DO trigger browser automation.
    "search_youtube",
    "search_google",
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
    # P1-2: search_web moved here from GOAL_ACTIONS — it is a generic/unresolved
    # search intent.  The Brain pipeline's DecisionEngine routes it to LLM_DIRECT
    # or to BrowserSkill in a single step via brain.process(), which is far faster
    # than the AutonomousAgent multi-step planning loop.
    "search_web",
})

# ─── Factual-question fast-path ───────────────────────────────────────────────
#
# P1-3: Interrogative utterance prefixes that almost always express a factual
# question best answered by the LLM directly (brain.process → LLM_DIRECT).
# These are checked in GoalIntentClassifier.classify() before the action-set
# lookup as a defence-in-depth measure, so that even if a future code change
# were to re-introduce search_web in GOAL_ACTIONS, simple factual questions
# would still bypass the AutonomousAgent.
#
# Important safety valve: the fast-path is NOT applied when the classifier
# reports a clearly non-search GOAL_ACTION with high confidence (≥ 0.8), so
# that utterances like "what is the volume set to" that fire set_volume, or
# imperative-disguised-as-question patterns that map to real desktop actions
# are still executed correctly.
#
_FACTUAL_QUESTION_PREFIXES: frozenset[str] = frozenset({
    "what is", "what are", "what was", "what were",
    "who is", "who are", "who was",
    "where is", "where are",
    "when is", "when did", "when was",
    "why is", "why are", "why does",
    "how does", "how do",
})

# Search-family actions where the fast-path CAN still override even at high
# confidence, because an LLM-direct answer is always faster and more appropriate
# for a factual question than launching browser automation.
_SEARCH_ACTIONS: frozenset[str] = frozenset({
    "search_web", "search_google", "search_bing",
    "search_duckduckgo", "search_wikipedia", "search_maps",
    "search_youtube", "open_browser_and_search",
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
        raw_utterance = intent.raw_utterance
        raw = raw_utterance.lower().strip() if isinstance(raw_utterance, str) else ""

        # ── P1-3: Factual-question fast-path ──────────────────────────────────
        # If the utterance starts with an interrogative prefix AND the classified
        # action is either a search action OR not a clearly high-confidence
        # GOAL_ACTION (e.g. launch_app, create_folder), route to chat so the
        # LLM can answer directly without spawning the AutonomousAgent.
        if any(raw.startswith(p) for p in _FACTUAL_QUESTION_PREFIXES):
            # Only apply fast-path when:
            #   (a) The action is a search/web action (always fast-path), OR
            #   (b) The action is NOT in GOAL_ACTIONS (unknown → chat anyway), OR
            #   (c) The action IS in GOAL_ACTIONS but confidence < 0.8 (uncertain)
            # Do NOT apply when the action is a clearly recognized non-search
            # GOAL_ACTION at high confidence (e.g. "what is the volume" → set_volume)
            is_search = action in _SEARCH_ACTIONS
            is_certain_goal = (action in GOAL_ACTIONS) and (intent.confidence >= 0.8)
            should_fast_path = is_search or not is_certain_goal
            if should_fast_path:
                log.debug(
                    "GoalIntentClassifier: factual-question fast-path "
                    "('{a}', conf={c:.0%}) → chat",
                    a=action,
                    c=intent.confidence,
                )
                return "chat"

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
