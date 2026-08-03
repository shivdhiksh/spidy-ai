"""
ResponseComposer — Natural Language Response Builder
=====================================================
Converts structured ToolResult lists into human-sounding, non-robotic
responses that feel like they're coming from a real AI companion.

Philosophy
----------
Spidy should feel like JARVIS, not a command-line tool. Every response
should:
  - Be warm and natural, not terse or robotic
  - Vary phrasing so it doesn't feel scripted
  - Provide context ("I've opened VS Code for you")
  - Be honest about failures ("I couldn't find that file")
  - For multi-step tasks, narrate the whole story at once

Design
------
- Stateless: all context comes via parameters
- No LLM dependency: template-based with controlled variation
- Handles: single success, single failure, multi-step success,
           partial failure, clarification, progress reporting
- All public methods are synchronous (pure string operations)

Usage
-----
    from spidy.brain.response_composer import ResponseComposer

    composer = ResponseComposer()
    text = composer.compose(results, action="launch_app")
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spidy.brain.types import ToolResult


# ─── Phrase Libraries ─────────────────────────────────────────────────────────

_SUCCESS_OPENERS = [
    "Done!",
    "All set!",
    "Got it!",
    "Sure thing!",
    "Of course!",
    "Consider it done.",
    "No problem!",
    "On it!",
]

_FAILURE_OPENERS = [
    "Hmm,",
    "Sorry,",
    "Unfortunately,",
    "I ran into an issue —",
    "There was a problem —",
]

_PARTIAL_OPENERS = [
    "I managed to do most of it, but —",
    "Almost there —",
    "Mostly done, but there's one issue:",
    "I completed some steps, but hit a snag:",
]

_CONFIRMATION_VERBS: dict[str, str] = {
    "launch_app": "I've opened",
    "open_app": "I've opened",
    "close_app": "I've closed",
    "search_web": "I've searched the web for",
    "search_google": "I've searched Google for",
    "search_youtube": "I've searched YouTube for",
    "search_bing": "I've searched Bing for",
    "search_duckduckgo": "I've searched DuckDuckGo for",
    "search_wikipedia": "I've looked up on Wikipedia:",
    "search_maps": "I've opened maps for",
    "open_url": "I've opened",
    "open_file": "I've opened",
    "open_folder": "I've opened",
    "search_files": "I searched for",
    "take_screenshot": "I've taken a screenshot.",
    "take_note": "I've saved your note.",
    "read_notes": "Here are your notes:",
    "set_timer": "Timer set!",
    "set_alarm": "Alarm set!",
    "set_volume": "Volume adjusted.",
    "set_brightness": "Brightness adjusted.",
    "lock_workstation": "Screen locked.",
    "sleep_system": "Putting the computer to sleep.",
    "shutdown_system": "Shutting down...",
    "restart_system": "Restarting now...",
    "empty_recycle_bin": "Recycle Bin emptied.",
    "get_system_info": "Here's your system info:",
    "calculate": "The answer is:",
    "get_clipboard": "Here's what's in your clipboard:",
    # M13.2 additions
    "show_desktop": "Desktop shown.",
    "create_folder": "I've created the folder.",
    "rename_folder": "I've renamed the folder.",
    "delete_folder": "I've deleted the folder.",
    # Conversational / no-op
    "greet": "",
    "farewell": "",
    "introduce": "",
    "chat": "",
    "llm": "",
    "clarify": "",
    "noop": "",
}


def _opener(pool: list[str]) -> str:
    """Pick a random opener from a pool."""
    return random.choice(pool)  # noqa: S311 — not cryptographic


# ─── Public API ───────────────────────────────────────────────────────────────


class ResponseComposer:
    """
    Builds human-sounding responses from ToolResult lists.

    Parameters
    ----------
    use_varied_phrases:
        If True (default), varies openers to avoid repetition.
        Set False in tests for deterministic output.
    """

    def __init__(self, use_varied_phrases: bool = True) -> None:
        self._vary = use_varied_phrases

    def compose(
        self,
        results: list["ToolResult"],
        action: str = "",
    ) -> str:
        """
        Compose a final response from all step results.

        Parameters
        ----------
        results:
            List of ToolResults from the ToolRouter.
        action:
            Primary action of the first step (for verb selection).

        Returns
        -------
        str
            Human-readable response text. Empty string if no output needed.
        """
        if not results:
            return ""

        # Single result
        if len(results) == 1:
            return self._compose_single(results[0], action)

        # Multi-step results
        return self._compose_multistep(results)

    def compose_progress(
        self,
        step_index: int,
        total_steps: int,
        current_action: str,
    ) -> str:
        """
        Build a progress message for a long-running multi-step task.

        Parameters
        ----------
        step_index:
            Zero-based index of the step currently executing.
        total_steps:
            Total number of steps in the plan.
        current_action:
            Human-readable description of what's happening now.

        Returns
        -------
        str
            E.g. "Working on it... (step 2 of 4)"
        """
        step_num = step_index + 1
        desc = _readable_action(current_action)
        return f"Working on it... ({step_num}/{total_steps}) {desc}"

    def compose_proactive(self, advice_message: str) -> str:
        """Wrap a proactive check message in natural language."""
        return advice_message  # Already human-readable from ProactiveAdvice

    def compose_clarification(self, question: str) -> str:
        """Wrap a clarification question naturally."""
        return question  # Questions are authored already; no wrapping needed

    # ── Private helpers ────────────────────────────────────────────────────

    def _compose_single(self, result: "ToolResult", action: str) -> str:
        """Build response for a single ToolResult."""
        # LLM / clarify / noop — return message as-is (already natural)
        if result.step_type in ("llm", "clarify", "noop"):
            return result.message

        if result.success:
            return self._success_message(result, action)
        else:
            return self._failure_message(result, action)

    def _compose_multistep(self, results: list["ToolResult"]) -> str:
        """Build a unified narrative from multiple step results."""
        successes = [r for r in results if r.success and r.message]
        failures = [r for r in results if not r.success]

        # All succeeded
        if not failures:
            return self._multistep_all_success(successes)

        # All failed
        if not successes:
            first_failure = failures[0]
            return self._failure_message(first_failure, first_failure.action)

        # Partial: some succeeded, some failed
        return self._multistep_partial(successes, failures)

    def _success_message(self, result: "ToolResult", action: str) -> str:
        """Build a success message for a skill result."""
        # If the skill already produced a rich message, use it
        if result.message and not _is_raw_message(result.message):
            return result.message

        verb = _CONFIRMATION_VERBS.get(action or result.action, "")
        if verb:
            base = f"{verb} {result.message}" if result.message else verb
            return base.strip(". ").rstrip() + "."

        # Fallback: prepend a natural opener
        if result.message:
            if self._vary:
                opener = _opener(_SUCCESS_OPENERS)
                return f"{opener} {result.message}"
            return result.message

        return _opener(_SUCCESS_OPENERS) if self._vary else "Done."

    def _failure_message(self, result: "ToolResult", action: str) -> str:
        """Build a failure message with a suggestion if possible."""
        msg = result.message
        if msg:
            # Strip internal skill prefixes before presenting to the user
            msg = _sanitize_message(msg)
            return msg

        opener = _opener(_FAILURE_OPENERS) if self._vary else "Sorry,"
        desc = _readable_action(action or result.action)
        suggestion = _suggest_alternative(action or result.action)
        msg = f"{opener} I couldn't {desc}."
        if suggestion:
            msg += f" {suggestion}"
        return msg

    def _multistep_all_success(self, successes: list["ToolResult"]) -> str:
        """Compose a single narrative for all-successful multi-step results."""
        # Collect unique meaningful messages
        messages: list[str] = []
        seen: set[str] = set()
        for r in successes:
            verb = _CONFIRMATION_VERBS.get(r.action, "")
            msg = r.message.strip()
            key = verb + msg
            if key and key not in seen:
                seen.add(key)
                if verb and not _is_raw_message(msg):
                    messages.append(f"{verb} {msg}".strip(".").rstrip())
                elif msg:
                    messages.append(msg.rstrip(".").rstrip())

        if not messages:
            return "All done!" if not self._vary else _opener(_SUCCESS_OPENERS)

        if len(messages) == 1:
            return messages[0] + "."

        # Two items: "I've done A and B."
        if len(messages) == 2:
            return f"{messages[0]}, and {messages[1].lower()}."

        # Three or more: "I've done A, B, and C."
        all_but_last = ", ".join(m.lower() for m in messages[:-1])
        last = messages[-1].lower()
        return f"All done! I've {all_but_last}, and {last}."

    def _multistep_partial(
        self,
        successes: list["ToolResult"],
        failures: list["ToolResult"],
    ) -> str:
        """Compose a partial-success narrative."""
        opener = _opener(_PARTIAL_OPENERS) if self._vary else "Partially done —"
        success_str = ", ".join(
            (_CONFIRMATION_VERBS.get(r.action, "completed " + r.action) + " " + r.message).strip()
            for r in successes
            if r.message
        ) or "some steps succeeded"

        failure_str = failures[0].message or f"couldn't {_readable_action(failures[0].action)}"

        return f"{opener} {success_str}. However, {failure_str.lower()}"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _readable_action(action: str) -> str:
    """Convert a snake_case action name to readable lowercase."""
    return action.replace("_", " ").lower() if action else "do that"


def _is_raw_message(msg: str) -> bool:
    """
    Return True if the message looks like an unprocessed internal string
    (e.g. 'no skill found', error stacks, or raw skill prefix messages)
    rather than user-facing text.

    These are NOT shown verbatim to the user; instead they are sanitized
    by _sanitize_message() before display.
    """
    msg_lower = msg.lower()
    raw_indicators = [
        "traceback",
        "error:",
        "exception",
        "not found",
        "no skill",
        # Internal skill prefixes — these should never reach the user raw
        "appskill:",
        "fileskill:",
        "systemcontrolskill:",
        "browserskill:",
        "browseragent:",
    ]
    return any(ind in msg_lower for ind in raw_indicators)


# Maps known internal skill error prefixes to clean human-readable replacements.
_SKILL_PREFIX_MAP: list[tuple[str, str]] = [
    ("AppSkill: 'name' parameter is required for bring_app_to_foreground.",
     "I couldn't determine which application to bring to the foreground. "
     "Try: 'switch to Chrome' or 'bring up Notepad'."),
    ("AppSkill: 'name' parameter is required for launch_app.",
     "I need to know which application to open. Try: 'open Chrome' or 'open Notepad'."),
    ("AppSkill: 'name' parameter is required for close_app.",
     "I need to know which application to close. Try: 'close Chrome'."),
    ("AppSkill:", "I couldn't complete the app action."),
    ("FileSkill:", "I couldn't complete the file operation."),
    ("SystemControlSkill:", "I couldn't complete that system action."),
    ("BrowserSkill:", "I couldn't complete the browser action."),
    ("BrowserAgent:", "I couldn't complete the browser action."),
]


def _sanitize_message(msg: str) -> str:
    """
    Remove internal skill prefixes from error messages.

    Iterates from most-specific to most-generic entries in _SKILL_PREFIX_MAP.
    For specific full-phrase matches, returns the clean_replacement directly.
    For generic prefix-only matches (e.g. "AppSkill:"), strips the prefix and
    returns the remainder so the actual error reason is preserved.
    """
    if not msg:
        return msg

    stripped = msg.strip()
    for raw_prefix, clean_replacement in _SKILL_PREFIX_MAP:
        if stripped.startswith(raw_prefix):
            # Generic prefix only (ends with ":") → strip and return remainder
            if raw_prefix.endswith(":"):
                remainder = stripped[len(raw_prefix):].strip()
                return remainder if remainder else clean_replacement
            # Full specific phrase → return the curated clean replacement
            return clean_replacement

    return msg



def _suggest_alternative(action: str) -> str:
    """Return a helpful suggestion for a failed action."""
    suggestions: dict[str, str] = {
        "launch_app": "You can try opening it manually, or say 'help' for what I can do.",
        "open_file": "Would you like me to search for it instead?",
        "search_files": "I could search in a different location if you tell me where.",
        "search_web": "Make sure you're connected to the internet and try again.",
        "open_url": "Check that the URL is correct and you're online.",
        "set_volume": "You can adjust volume manually from the taskbar.",
        "set_brightness": "Try adjusting brightness from the system settings.",
    }
    return suggestions.get(action, "")
