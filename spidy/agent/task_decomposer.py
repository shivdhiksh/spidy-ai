"""
TaskDecomposer -- Goal -> Task List Conversion (Milestones 13 + 18)
====================================================================
Converts a high-level goal description into an ordered list of
executable TaskRecords that the ExecutionLoop can run step by step.

M18 addition: Structured action contracts
------------------------------------------
Where possible, the decomposer now attaches a structured ``action`` dict
to each TaskRecord so that ExecutionLoop can route directly to the
appropriate skill without re-parsing through Brain.

Key improvement: search queries are extracted verbatim from the user's
goal and stored in action.query, so they are never mangled by Brain's
intent classifier.

  Example for "search YouTube for Python tutorials":
    action = {
        "skill": "browser",
        "action": "search",
        "target": "youtube",
        "query": "Python tutorials",   # preserved exactly
        "expected_outcome": "YouTube search results for Python tutorials are visible",
    }

Two decomposition strategies
-----------------------------
1. LLM-based (preferred when an LLM client is available):
   Sends a structured prompt to the LLM asking it to break down the
   goal into numbered, concrete tasks with natural-language utterances
   and optional structured action objects.

2. Heuristic-based (fallback when LLM is unavailable or fails):
   Uses keyword/pattern matching to recognise common workflows
   and produces a fixed sequence of tasks for each pattern.
   For well-known patterns (app launch, browser search) the heuristic
   now generates structured actions in addition to the utterance.

Design
------
- The decomposer is stateless -- no side effects, pure input->output
- LLM path: gracefully falls back to heuristic on parse failure
- All returned TaskRecords start in TaskState.PENDING
- Each task has both a human-readable description AND an utterance
  (the utterance is the fallback when no structured action is available)

Usage
-----
    decomposer = TaskDecomposer()
    tasks = await decomposer.decompose(
        "Create a Flask project",
        goal_id=goal.goal_id,
        llm_client=llm_client,
    )
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from spidy.agent.types import TaskRecord, TaskState
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.llm.client import BaseLLMClient

log = get_logger(__name__)

# ─── LLM Decomposition Prompt ─────────────────────────────────────────────────

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a task planning assistant for SPIDY. Your job is to break down a high-level goal
into an ordered sequence of concrete, deterministic, executable steps with structured action contracts.

Rules:
- Return ONLY a valid JSON array, no prose, no markdown fences.
- Each item must be a JSON object with these fields:
    "description": short human-readable step name (max 10 words)
    "utterance": the natural-language command to execute this step
    "terminal": true if this step achieves the final objective and no further steps are needed after it succeeds
    "action": structured action object for direct skill routing:
        skill:            "browser" | "desktop" | "file" | "dialog"
        action:           e.g. "open_app" | "navigate" | "search" | "extract_text" | "focus" | "type_text" | "save_file" | "close_app" | "calculate" | "ask_user"
        target:           app name ("Chrome", "Edge", "Notepad", "Calculator") or site name ("YouTube", "Google")
        query:            verbatim search query string (for search actions) or calculation expression
        text:             text to type or template expression (e.g. "${prev}.text", "${step_2}.url")
        url:              explicit URL (for navigate actions)
        input_from:       optional reference to prior task ("step_N", "prev", or task_id)
        expected_outcome: what success looks like (plain English observation criterion)
- CRITICAL for ambiguous/underspecified goals (e.g. "search something"):
    Do NOT invent a query. Set skill="dialog", action="ask_user", target="user", and utterance="What would you like me to search for?".
- For compound tasks like "Search X and type into Notepad":
    Include an explicit read/extract step, focus step, and a type_text step referencing the extracted result via input_from.
"""

_DECOMPOSE_USER_TEMPLATE = 'Break down this goal into executable steps: "{goal}"'


# ─── Heuristic Patterns ───────────────────────────────────────────────────────

_HEURISTIC_PATTERNS: list[tuple[list[str], list[tuple[str, str]]]] = [
    # ── VS Code + file creation (compound) — BEFORE single-step VS Code ──────
    # Must appear FIRST so "Open VS Code and create main.py" matches this
    # 2-step pattern rather than the single-step "open vs code" below.
    (
        ["vs code and create", "vscode and create",
         "vs code and make", "vscode and make",
         "open vs code and", "launch vs code and"],
        [
            ("Open VS Code", "open vs code"),
            ("Create a new file", "create a new file in vs code"),
        ],
    ),

    # ── Single-action app launches ─────────────────────────────────────────

    # These are single-step goals — the utterance IS the task.
    # Keeping them explicit means the heuristic decomposer beats the
    # "last resort passthrough" AND generates a human-readable description.
    (
        ["open notepad", "launch notepad", "start notepad"],
        [("Open Notepad", "open notepad")],
    ),
    (
        ["open chrome", "launch chrome", "start chrome", "open google chrome"],
        [("Open Chrome", "open chrome")],
    ),
    (
        ["open edge", "launch edge", "open microsoft edge"],
        [("Open Edge", "open edge")],
    ),
    (
        ["open firefox", "launch firefox"],
        [("Open Firefox", "open firefox")],
    ),
    (
        [
            "open vs code", "launch vs code", "open vscode", "launch vscode",
            "start vs code", "open visual studio code",
        ],
        [("Open VS Code", "open vs code")],
    ),
    (
        ["open terminal", "open cmd", "open command prompt", "open powershell", "launch terminal"],
        [("Open Terminal", "open terminal")],
    ),
    (
        ["open discord", "launch discord"],
        [("Open Discord", "open discord")],
    ),
    (
        ["open spotify", "launch spotify"],
        [("Open Spotify", "open spotify")],
    ),
    (
        ["open calculator", "launch calculator"],
        [("Open Calculator", "open calculator")],
    ),
    (
        ["open paint", "launch paint"],
        [("Open Paint", "open paint")],
    ),
    (
        ["open task manager"],
        [("Open Task Manager", "open task manager")],
    ),
    (
        ["lock screen", "lock computer", "lock my pc"],
        [("Lock Screen", "lock screen")],
    ),
    (
        ["take a screenshot", "screenshot", "capture screen"],
        [("Take Screenshot", "take a screenshot")],
    ),

    # ── Window control — close / minimize / maximize / focus ───────────────
    # BUG 3 FIX: These commands previously fell through to LLM planning,
    # generating useless GUI-click plans. They are single-step desktop
    # operations that map directly to AppSkill actions.

    # close
    (
        ["close notepad", "quit notepad", "exit notepad"],
        [("Close Notepad", "close notepad")],
    ),
    (
        ["close edge", "quit edge", "close microsoft edge", "quit microsoft edge"],
        [("Close Edge", "close edge")],
    ),
    (
        ["close chrome", "quit chrome", "close google chrome", "quit google chrome"],
        [("Close Chrome", "close chrome")],
    ),
    (
        ["close firefox", "quit firefox"],
        [("Close Firefox", "close firefox")],
    ),
    (
        ["close discord", "quit discord"],
        [("Close Discord", "close discord")],
    ),
    (
        ["close spotify", "quit spotify"],
        [("Close Spotify", "close spotify")],
    ),
    (
        ["close vs code", "quit vs code", "close vscode", "quit vscode"],
        [("Close VS Code", "close vs code")],
    ),
    (
        ["close terminal", "close cmd", "close powershell", "quit terminal"],
        [("Close Terminal", "close terminal")],
    ),
    (
        ["close calculator", "quit calculator"],
        [("Close Calculator", "close calculator")],
    ),

    # minimize
    (
        ["minimize notepad", "minimise notepad"],
        [("Minimize Notepad", "minimize notepad")],
    ),
    (
        ["minimize edge", "minimise edge", "minimize microsoft edge"],
        [("Minimize Edge", "minimize edge")],
    ),
    (
        ["minimize chrome", "minimise chrome", "minimize google chrome"],
        [("Minimize Chrome", "minimize chrome")],
    ),
    (
        ["minimize firefox", "minimise firefox"],
        [("Minimize Firefox", "minimize firefox")],
    ),
    (
        ["minimize vs code", "minimise vs code", "minimize vscode", "minimise vscode"],
        [("Minimize VS Code", "minimize vs code")],
    ),
    (
        ["minimize discord", "minimise discord"],
        [("Minimize Discord", "minimize discord")],
    ),
    (
        ["minimize spotify", "minimise spotify"],
        [("Minimize Spotify", "minimize spotify")],
    ),
    (
        ["minimize calculator", "minimise calculator"],
        [("Minimize Calculator", "minimize calculator")],
    ),

    # maximize / restore
    (
        ["maximize notepad", "maximise notepad", "restore notepad"],
        [("Maximize Notepad", "maximize notepad")],
    ),
    (
        ["maximize edge", "maximise edge", "restore edge", "maximize microsoft edge"],
        [("Maximize Edge", "maximize edge")],
    ),
    (
        ["maximize chrome", "maximise chrome", "restore chrome"],
        [("Maximize Chrome", "maximize chrome")],
    ),
    (
        ["maximize firefox", "maximise firefox", "restore firefox"],
        [("Maximize Firefox", "maximize firefox")],
    ),
    (
        ["maximize vs code", "maximise vs code", "restore vs code",
         "maximize vscode", "maximise vscode"],
        [("Maximize VS Code", "maximize vs code")],
    ),
    (
        ["maximize calculator", "maximise calculator", "restore calculator"],
        [("Maximize Calculator", "maximize calculator")],
    ),

    # focus / bring to front / switch to
    (
        ["focus notepad", "bring notepad to front", "switch to notepad",
         "bring notepad to foreground"],
        [("Focus Notepad", "focus notepad")],
    ),
    (
        ["focus edge", "bring edge to front", "switch to edge",
         "bring edge to foreground", "focus microsoft edge"],
        [("Focus Edge", "focus edge")],
    ),
    (
        ["focus chrome", "bring chrome to front", "switch to chrome",
         "bring chrome to foreground", "focus google chrome"],
        [("Focus Chrome", "focus chrome")],
    ),
    (
        ["focus firefox", "bring firefox to front", "switch to firefox"],
        [("Focus Firefox", "focus firefox")],
    ),
    (
        ["focus vs code", "bring vs code to front", "switch to vs code",
         "focus vscode", "bring vscode to front"],
        [("Focus VS Code", "focus vs code")],
    ),
    (
        ["focus discord", "bring discord to front", "switch to discord"],
        [("Focus Discord", "focus discord")],
    ),
    (
        ["focus spotify", "bring spotify to front", "switch to spotify"],
        [("Focus Spotify", "focus spotify")],
    ),

    # ── Create folder ──────────────────────────────────────────────────────
    (
        ["create a folder", "create folder", "make a folder", "make folder",
         "new folder", "make a directory", "create a directory", "mkdir"],
        [("Create folder on Desktop", "create a folder on the desktop")],
    ),

    # ── File search ─────────────────────────────────────────────────────────
    (
        ["find all pdf", "find pdf files", "find pdfs", "search for pdf",
         "find pdf in downloads", "search downloads for pdf"],
        [
            ("Search for PDF files", "find all pdf files in the downloads folder"),
        ],
    ),
    (
        ["find all docx", "find word files", "find all word files"],
        [
            ("Search for Word files", "find all docx files in the documents folder"),
        ],
    ),
    (
        ["find all images", "find all photos", "find all pictures"],
        [
            ("Search for image files", "find all image files in the pictures folder"),
        ],
    ),

    # ── Webpage summarization ───────────────────────────────────────────────
    (
        ["summarize webpage", "summarize this page", "summarize the current page",
         "summarize current page", "read this webpage", "summarize current webpage",
         "read the current page", "what does this page say"],
        [
            ("Read current webpage", "read the current browser page"),
            ("Summarize content", "summarize the webpage content"),
        ],
    ),

    # ── Open project in VS Code ─────────────────────────────────────────────
    (
        ["open my portfolio", "open portfolio project", "open portfolio in vs code",
         "open my portfolio in vs code"],
        [
            ("Find Portfolio folder", "find the portfolio folder"),
            ("Open VS Code", "open vs code in the portfolio folder"),
        ],
    ),
    (
        ["open my project", "open project in vs code", "open project in vscode",
         "open my project in vs code", "open the project"],
        [
            ("Find project folder", "find the project folder"),
            ("Open VS Code", "open vs code in the project folder"),
        ],
    ),

    # ── Open Edge + search YouTube (compound browser workflow) ─────────────
    # Handles "Open Edge and search YouTube for X" / "Open Edge. Insert YouTube..."
    # These are caught by _split_compound_goal first when connectors are used;
    # the heuristic entry handles the "open edge" portion when called as a sub-goal.
    (
        ["open edge and search youtube", "open edge and navigate to youtube",
         "open edge and go to youtube", "open edge then search youtube"],
        [
            ("Open Edge", "open edge"),
            ("Search YouTube", "search youtube"),
        ],
    ),

    # ── YouTube search ──────────────────────────────────────────────────────
    (
        ["search youtube", "youtube", "search on youtube",
         "navigate to youtube", "go to youtube", "insert youtube",
         "open youtube", "launch youtube"],
        [
            ("Open browser", "open chrome"),
            ("Search YouTube", "search youtube"),
        ],
    ),

    # ── Google search ────────────────────────────────────────────────────────
    (
        ["search google", "google search", "search on google"],
        [
            ("Open browser", "open chrome"),
            ("Search Google", "search google"),
        ],
    ),

    # ── Flask project ────────────────────────────────────────────────────────
    (
        ["flask", "flask project", "flask app", "flask application"],
        [
            ("Create project folder", "create a folder named flask_project on the desktop"),
            ("Open terminal", "open terminal"),
            ("Create virtual environment", "create a python virtual environment named venv"),
            ("Install Flask", "install flask using pip"),
            ("Create app.py", "create a file named app.py with a basic flask hello world app"),
            ("Open VS Code", "open vs code"),
            ("Run Flask app", "run the flask application"),
        ],
    ),

    # ── React project ────────────────────────────────────────────────────────
    (
        ["react", "react project", "react app", "react application"],
        [
            ("Open terminal", "open terminal"),
            ("Create React app", "create a new react application using create-react-app"),
            ("Open VS Code", "open vs code in the project folder"),
            ("Install dependencies", "install node dependencies using npm install"),
            ("Start dev server", "start the react development server using npm start"),
        ],
    ),

    # ── Python project ───────────────────────────────────────────────────────
    (
        ["python project", "python app", "python application", "create a python project",
         "new python project"],
        [
            ("Create project folder", "create a folder named python_project on the desktop"),
            ("Open VS Code", "open vs code"),
            ("Create main.py", "create a file named main.py"),
            ("Open terminal", "open terminal"),
        ],
    ),

    # ── Browser research ─────────────────────────────────────────────────────
    (
        ["research", "find information about"],
        [
            ("Open browser", "open chrome"),
            ("Search for topic", "search google for the topic"),
            ("Read and summarize", "summarize what you found"),
        ],
    ),

    # ── Generic project creation ──────────────────────────────────────────────
    # Catch-all for "create a project" if the specific type isn't matched above.
    # The LLM decomposer handles the real work; this just ensures ≥1 task.
    (
        ["create a project", "new project", "start a project", "start new project"],
        [
            ("Create project folder", "create a folder for the project on the desktop"),
            ("Open VS Code", "open vs code"),
            ("Open terminal", "open terminal"),
        ],
    ),
]

# ─── Compound Goal Splitter ───────────────────────────────────────────────────
#
# BUG 2 FIX: Split compound voice commands like:
#   "Open Edge and search Python"      → ["Open Edge", "search Python"]
#   "Open Chrome and open YouTube"     → ["Open Chrome", "open YouTube"]
#   "Open VS Code then open project"   → ["Open VS Code", "open project"]
#
# The split only fires when:
#   1. A connector word (and/then/after/followed by) is present.
#   2. BOTH sides of the split start with an action verb (open/launch/close/
#      search/find/minimize/maximize/focus/bring/create/make/start/run).
#
# This prevents false splits on phrases like "search for cats and dogs"
# where "dogs" has no action verb.

_ACTION_VERBS = re.compile(
    r'\b(?:open|launch|close|quit|exit|minimize|minimise|maximize|maximise|restore|'
    r'focus|bring|switch|search|find|look\s+up|create|make|start|run|'
    r'navigate|go\s+to|show|insert|type|click|press|scroll|'
    r'download|upload|play|pause|stop|skip|next|'
    r'calculate|verify|check|tell|delete|remove|execute|save|write|read)\b',
    re.IGNORECASE,
)

# Connector words that may signal a compound command.
# Also matches sentence-boundary periods/semicolons between two clauses (with trailing space)
# so that "Open Edge. Search YouTube for Python." is split correctly without breaking filenames.
_COMPOUND_CONNECTORS_RE = re.compile(
    r'(?:\s*(?:,|and|then|after\s+that|followed\s+by|next)\s+|\s*,\s*|[;!]\s*|\.\s+)',
    re.IGNORECASE,
)

# Conversational fillers, greetings, and leading conjunctions that Whisper
# or user speech may prepend to actionable commands.
_LEADING_FILLERS = re.compile(
    r'^(?:and|then|after\s+that|followed\s+by|next|also|please|can\s+you|could\s+you|would\s+you|'
    r'i\s+want\s+to|help\s+me|spidy|jarvis|hey\s+spidy|hey\s+jarvis|that\'s\s+[\w]+)\s*',
    re.IGNORECASE,
)

_KNOWN_WEB_TARGETS: dict[str, str] = {
    "youtube":    "https://www.youtube.com",
    "google":     "https://www.google.com",
    "github":     "https://github.com",
    "gmail":      "https://mail.google.com",
    "reddit":     "https://www.reddit.com",
    "wikipedia":  "https://www.wikipedia.org",
    "twitter":    "https://www.twitter.com",
    "x":          "https://www.x.com",
    "facebook":   "https://www.facebook.com",
    "instagram":  "https://www.instagram.com",
    "linkedin":   "https://www.linkedin.com",
    "bing":       "https://www.bing.com",
    "duckduckgo": "https://duckduckgo.com",
}


_KNOWN_DESKTOP_APPS: dict[str, str] = {
    "edge": "Edge",
    "microsoft edge": "Edge",
    "chrome": "Chrome",
    "google chrome": "Chrome",
    "firefox": "Firefox",
    "notepad": "Notepad",
    "calculator": "Calculator",
    "calc": "Calculator",
    "vs code": "VS Code",
    "vscode": "VS Code",
    "visual studio code": "VS Code",
    "terminal": "Terminal",
    "cmd": "Command Prompt",
    "powershell": "PowerShell",
    "spotify": "Spotify",
    "discord": "Discord",
    "paint": "Paint",
    "task manager": "Task Manager",
    "settings": "Settings",
    "windows settings": "Windows Settings",
    "explorer": "File Explorer",
    "file explorer": "File Explorer",
}


def _extract_actionable_clauses(goal: str) -> list[str]:
    """
    Extract meaningful, actionable command clauses from a raw goal string.

    Tolerates conversational noise, preamble (e.g. "That's Shiva", "Hey Spidy"),
    and punctuation/conjunction separators.
    """
    parts = _COMPOUND_CONNECTORS_RE.split(goal.strip())
    clauses: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Strip leading filler repeatedly
        while True:
            m = _LEADING_FILLERS.match(part)
            if m:
                part = part[m.end():].strip()
            else:
                break
        if not part:
            continue

        if _ACTION_VERBS.match(part):
            clauses.append(part)
        else:
            m = _ACTION_VERBS.search(part)
            if m:
                clean_part = part[m.start():].strip()
                if clean_part:
                    clauses.append(clean_part)

    return clauses


def _split_compound_goal(goal: str) -> list[str] | None:
    """
    Split a compound goal string into ordered sub-goals.

    Returns a list of 2+ sub-goals if the utterance is compound, or None
    if it is a single goal (caller uses normal decomposition).
    """
    clauses = _extract_actionable_clauses(goal)
    if len(clauses) >= 2:
        return clauses
    return None


class TaskDecomposer:
    """
    Converts a high-level goal string into an ordered list of TaskRecords.

    Parameters
    ----------
    max_tasks:
        Maximum number of tasks to generate (safety cap).
    """

    def __init__(self, max_tasks: int = 12) -> None:
        self._max_tasks = max_tasks

    async def decompose(
        self,
        goal_description: str,
        goal_id: str = "",
        llm_client: "BaseLLMClient | None" = None,
        session_id: str = "",
    ) -> list[TaskRecord]:
        """
        Decompose a goal into an ordered list of TaskRecords.

        Tries LLM-based decomposition first; falls back to heuristic on
        failure or when no LLM is configured.

        Parameters
        ----------
        goal_description:
            The user's goal (e.g. "Create a Flask project").
        goal_id:
            Parent goal ID to attach to each TaskRecord.
        llm_client:
            Optional LLM client. If None, uses heuristic decomposition.
        session_id:
            For logging / tracing.

        Returns
        -------
        list[TaskRecord]
            Ordered tasks, all in PENDING state.
        """
        tasks: list[TaskRecord] | None = None

        # ── BUG 2 FIX: Compound goal splitting ────────────────────────────────
        # Detect compound commands like "Open Edge and search Python" BEFORE
        # the heuristic patterns run. Split into sub-goals, decompose each
        # independently, and concatenate the task lists.
        sub_goals = _split_compound_goal(goal_description)
        if sub_goals is not None:
            log.info(
                "TaskDecomposer: compound goal detected — {n} sub-goals: {parts}",
                n=len(sub_goals),
                parts=[s[:40] for s in sub_goals],
            )
            all_tasks: list[TaskRecord] = []
            last_target = ""
            active_browser = ""

            goal_lower = goal_description.lower()
            if any(b in goal_lower for b in ("edge", "microsoft edge", "msedge")):
                active_browser = "edge"
            elif any(b in goal_lower for b in ("chrome", "google chrome")):
                active_browser = "chrome"
            elif "firefox" in goal_lower:
                active_browser = "firefox"

            for i, sub in enumerate(sub_goals):
                is_last = (i == len(sub_goals) - 1)
                sub_lower = sub.strip().lower()
                sub_tasks: list[TaskRecord] | None = None

                # Detect if this sub-goal specifically names a browser
                if any(b in sub_lower for b in ("edge", "microsoft edge", "msedge")):
                    active_browser = "edge"
                elif any(b in sub_lower for b in ("chrome", "google chrome")):
                    active_browser = "chrome"
                elif "firefox" in sub_lower:
                    active_browser = "firefox"

                # 1. Search clause (e.g. "search for Python tutorials", "search youtube for ...", "search something")
                if re.match(r'^(?:search|look\s+up)\b', sub_lower, re.IGNORECASE):
                    s_target, s_query = self._extract_search_params(sub)
                    if not s_target and last_target:
                        s_target = last_target
                    if not active_browser:
                        active_browser = detect_default_browser()

                    if is_placeholder_or_ambiguous_query(s_query):
                        # Ambiguity handling: ask user for clarification
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description="Clarify search query",
                            utterance="What would you like me to search for?",
                            skill="dialog",
                            action="ask_user",
                            target="user",
                            query="",
                            expected_outcome="User provided search query",
                            terminal=is_last,
                            browser=active_browser,
                        )]
                    else:
                        s_target_disp = s_target.capitalize() if s_target else ""
                        eo = (
                            f"{s_target_disp} search results for {s_query} are visible"
                            if s_target_disp
                            else f"Search results for {s_query} are visible"
                        )
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description=f"Search {s_target_disp} for {s_query}" if s_target_disp else f"Search for {s_query}",
                            utterance=f"search {s_target} for {s_query}" if s_target else f"search for {s_query}",
                            skill="browser",
                            action="search",
                            target=s_target or active_browser or "google",
                            query=s_query,
                            expected_outcome=eo,
                            terminal=is_last,
                            browser=active_browser,
                        )]

                # 2. Browser extract / read result (e.g. "read the first useful result", "read the result")
                if sub_tasks is None:
                    m_extract = re.match(
                        r'^(?:read|extract|copy|get)\s+(?:the\s+)?(?:first\s+useful\s+)?(?:search\s+)?result',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_extract:
                        if not active_browser:
                            active_browser = detect_default_browser()
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description="Read search result",
                            utterance="read the first search result",
                            skill="browser",
                            action="extract_text",
                            target=last_target or active_browser or "browser",
                            expected_outcome="Search result text extracted",
                            terminal=is_last,
                            browser=active_browser,
                        )]

                # 3. Focus / switch window (e.g. "switch to notepad", "focus notepad", "switch to calculator")
                if sub_tasks is None:
                    m_focus = re.match(
                        r'^(?:switch\s+to|focus|bring\s+up|activate)\s+([\w\s]+)',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_focus:
                        focus_target = m_focus.group(1).strip()
                        if focus_target.lower() in _KNOWN_DESKTOP_APPS:
                            focus_target = _KNOWN_DESKTOP_APPS[focus_target.lower()]
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description=f"Focus {focus_target}",
                            utterance=f"focus {focus_target}",
                            skill="desktop",
                            action="focus",
                            target=focus_target,
                            expected_outcome=f"{focus_target} is in foreground",
                            terminal=is_last,
                        )]

                # 4. Browser website navigation (e.g. "open youtube", "go to google")
                if sub_tasks is None:
                    m_web = re.match(
                        r'^(?:open|go\s+to|navigate\s+to|launch)\s+([\w\.-]+|https?://[^\s]+)\b',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_web:
                        target_raw = m_web.group(1).lower().strip()
                        if target_raw in _KNOWN_WEB_TARGETS or target_raw.startswith("http") or target_raw.endswith(".com") or target_raw.endswith(".org"):
                            target_disp = target_raw.capitalize()
                            url = _KNOWN_WEB_TARGETS.get(target_raw, target_raw)
                            last_target = target_raw
                            if not active_browser:
                                active_browser = detect_default_browser()
                            sub_tasks = [self._make_structured_task(
                                goal_id=goal_id,
                                description=f"Navigate to {target_disp}",
                                utterance=f"navigate to {url}",
                                skill="browser",
                                action="navigate",
                                target=target_disp,
                                url=url,
                                expected_outcome=f"{target_disp} is loaded",
                                terminal=is_last,
                                browser=active_browser,
                            )]

                # 5. Desktop app launch (e.g. "open edge", "open notepad", "open calculator")
                if sub_tasks is None:
                    m_app = re.match(
                        r'^(?:open|launch|start)\s+([\w\s]+)',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_app:
                        app_key = m_app.group(1).lower().strip()
                        if app_key in _KNOWN_DESKTOP_APPS:
                            app_name = _KNOWN_DESKTOP_APPS[app_key]
                            is_browser_app = app_key in ("edge", "microsoft edge", "chrome", "google chrome", "firefox")
                            if is_browser_app:
                                active_browser = "edge" if "edge" in app_key else ("chrome" if "chrome" in app_key else "firefox")
                            sub_tasks = [self._make_structured_task(
                                goal_id=goal_id,
                                description=f"Open {app_name}",
                                utterance=f"open {app_key}",
                                skill="desktop",
                                action="open_app",
                                target=app_name,
                                expected_outcome=f"{app_name} is running",
                                terminal=is_last,
                                browser=active_browser if is_browser_app else "",
                            )]

                # 6. Desktop typing / entering text (e.g. "type Hello from Spidy", "type 'test'", "type the result")
                if sub_tasks is None:
                    m_type = re.match(
                        r'^(?:type|write|enter)\s+(?:["\'](.*?)["\']|([\w\s,\.\?!]+?))(?:\s+into\s+([\w\s]+))?$',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_type:
                        text_typed = (m_type.group(1) or m_type.group(2) or "").strip()
                        target_app = (m_type.group(3) or "").strip()
                        text_lower = text_typed.lower()
                        action_text = text_typed
                        input_from_ref = ""
                        if text_lower in ("the result", "the search result", "search result", "result"):
                            action_text = "${prev}.text"
                            input_from_ref = "<input_from:prev>"
                        elif text_lower in ("the query", "the search query", "search query", "query"):
                            action_text = "${prev}.query"
                            input_from_ref = "<input_from:prev>"

                        task_rec = self._make_structured_task(
                            goal_id=goal_id,
                            description=f"Type '{text_typed}'" + (f" into {target_app}" if target_app else ""),
                            utterance=f"type {text_typed}",
                            skill="desktop",
                            action="type_text",
                            target=target_app,
                            query=text_typed,
                            expected_outcome=f"Text '{text_typed}' entered",
                            terminal=is_last,
                        )
                        if task_rec.action:
                            task_rec.action["text"] = action_text
                            if input_from_ref:
                                task_rec.action["input_from"] = input_from_ref
                                task_rec.input_from = input_from_ref
                        sub_tasks = [task_rec]

                # 7. Desktop save file (e.g. "save it as spidy_test.txt", "save as test.txt")
                if sub_tasks is None:
                    m_save = re.match(
                        r'^(?:save(?:\s+it)?(?:\s+as)?)\s+([\w\.\-_/\\:]+)',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_save:
                        filename = m_save.group(1).strip()
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description=f"Save file as {filename}",
                            utterance=f"save as {filename}",
                            skill="desktop",
                            action="save_file",
                            target=filename,
                            expected_outcome=f"File {filename} saved",
                            terminal=is_last,
                        )]

                # 8. Desktop close app (e.g. "close notepad", "close calculator")
                if sub_tasks is None:
                    m_close = re.match(
                        r'^(?:close|quit|exit)\s+([\w\s]+)',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_close:
                        close_target = m_close.group(1).strip()
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description=f"Close {close_target}",
                            utterance=f"close {close_target}",
                            skill="desktop",
                            action="close_app",
                            target=close_target,
                            expected_outcome=f"{close_target} is closed",
                            terminal=is_last,
                        )]

                # 9. Desktop calculate (e.g. "calculate 25 multiplied by 18", "calculate 25 * 18")
                if sub_tasks is None:
                    m_calc = re.match(
                        r'^(?:calculate|compute)\s+(.*)',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_calc:
                        raw_expr = m_calc.group(1).strip()
                        norm_expr = (
                            raw_expr.replace("multiplied by", "*")
                            .replace("times", "*")
                            .replace("x", "*")
                            .replace("plus", "+")
                            .replace("minus", "-")
                            .replace("divided by", "/")
                        )
                        sub_tasks = [
                            self._make_structured_task(
                                goal_id=goal_id,
                                description=f"Calculate {raw_expr}",
                                utterance=f"calculate {norm_expr}",
                                skill="desktop",
                                action="calculate",
                                query=norm_expr,
                                expected_outcome=f"Calculated {raw_expr}",
                                terminal=is_last,
                            )
                        ]

                # 10. Desktop verify / check (e.g. "verify the file exists", "verify test.txt exists")
                if sub_tasks is None:
                    m_ver = re.match(
                        r'^(?:verify|check)\s+(?:that\s+|if\s+)?(.*)',
                        sub,
                        re.IGNORECASE,
                    )
                    if m_ver:
                        ver_target = m_ver.group(1).strip()
                        sub_tasks = [self._make_structured_task(
                            goal_id=goal_id,
                            description=f"Verify {ver_target}",
                            utterance=f"verify {ver_target}",
                            skill="desktop",
                            action="find_file" if "file" in ver_target.lower() else "verify",
                            target=ver_target,
                            expected_outcome=f"Verified {ver_target}",
                            terminal=is_last,
                        )]

                # 9. Standard heuristic patterns
                if sub_tasks is None:
                    sub_tasks = self._decompose_heuristic(sub, goal_id)

                # 5. LLM decomposition fallback
                if not sub_tasks and llm_client is not None:
                    sub_tasks = await self._decompose_with_llm(sub, goal_id, llm_client)

                # 6. Passthrough fallback to ensure NO clause is silently dropped
                if not sub_tasks:
                    log.warning(
                        "TaskDecomposer: sub-goal produced 0 tasks, using passthrough: '{s}'",
                        s=sub[:60],
                    )
                    sub_tasks = [self._make_task(
                        goal_id=goal_id,
                        description=sub,
                        utterance=sub,
                        terminal=is_last,
                    )]

                # Ensure only the last task of the last sub-goal is terminal
                if is_last and sub_tasks:
                    t = sub_tasks[-1]
                    sub_tasks[-1] = TaskRecord(
                        task_id=t.task_id,
                        goal_id=t.goal_id,
                        description=t.description,
                        utterance=t.utterance,
                        action=t.action,
                        input_from=t.input_from,
                        state=t.state,
                        extra=t.extra,
                        terminal=True,
                    )
                else:
                    for j, t in enumerate(sub_tasks):
                        if t.terminal:
                            sub_tasks[j] = TaskRecord(
                                task_id=t.task_id,
                                goal_id=t.goal_id,
                                description=t.description,
                                utterance=t.utterance,
                                action=t.action,
                                input_from=t.input_from,
                                state=t.state,
                                extra=t.extra,
                                terminal=False,
                            )
                all_tasks.extend(sub_tasks)

            tasks = all_tasks[:self._max_tasks]

            if len(tasks) < len(sub_goals):
                log.error(
                    "TaskDecomposer: compound goal has {n_clauses} clauses but only "
                    "{n_tasks} tasks produced — possible silent drop!",
                    n_clauses=len(sub_goals),
                    n_tasks=len(tasks),
                )

            log.info(
                "Decomposed compound goal '{desc}' → {n} tasks",
                desc=goal_description[:80],
                n=len(tasks),
            )
            return tasks

        # ── Heuristic-first strategy for single-clause goals ────────────────
        tasks = self._decompose_heuristic(goal_description, goal_id)

        if not tasks and llm_client is not None:
            tasks = await self._decompose_with_llm(goal_description, goal_id, llm_client)

        if not tasks:
            # Last resort: single-task passthrough (the goal IS the utterance)
            tasks = [self._make_task(
                goal_id=goal_id,
                description=goal_description,
                utterance=goal_description,
                terminal=True,
            )]

        # Safety cap
        tasks = tasks[:self._max_tasks]

        log.info(
            "Decomposed goal '{desc}' → {n} tasks",
            desc=goal_description[:80],
            n=len(tasks),
        )
        for i, t in enumerate(tasks):
            log.debug("  Task {i}: {desc}", i=i + 1, desc=t.description)

        return tasks

    # ── LLM decomposition ──────────────────────────────────────────────────

    async def _decompose_with_llm(
        self,
        goal_description: str,
        goal_id: str,
        llm_client: "BaseLLMClient",
    ) -> list[TaskRecord] | None:
        """
        Ask the LLM to decompose the goal and parse its JSON response.

        Returns None on any parse or LLM failure (heuristic will be tried next).
        """
        from spidy.llm.client import LLMMessage

        messages = [
            LLMMessage(role="system", content=_DECOMPOSE_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=_DECOMPOSE_USER_TEMPLATE.format(goal=goal_description),
            ),
        ]

        try:
            response = await llm_client.complete(messages)
            if not response.success or not response.text:
                log.debug("TaskDecomposer: LLM response unsuccessful, falling back")
                return None

            return self._parse_llm_response(response.text, goal_id)

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "TaskDecomposer: LLM decomposition failed (non-fatal): {exc}",
                exc=exc,
            )
            return None

    def _parse_llm_response(
        self,
        text: str,
        goal_id: str,
    ) -> list[TaskRecord] | None:
        """
        Parse the LLM's JSON array of {description, utterance} objects.

        Returns None if parsing fails.
        """
        # Strip any markdown fences the LLM might include
        cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        cleaned = cleaned.rstrip("```").strip()

        # Find the JSON array
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not array_match:
            log.debug("TaskDecomposer: no JSON array found in LLM response")
            return None

        try:
            data = json.loads(array_match.group(0))
            if not isinstance(data, list) or not data:
                return None

            tasks: list[TaskRecord] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                description = str(item.get("description", "")).strip()
                utterance = str(item.get("utterance", "")).strip()
                if not description or not utterance:
                    continue
                terminal_raw = item.get("terminal", False)
                terminal = bool(terminal_raw) if isinstance(terminal_raw, bool) else str(terminal_raw).lower() == "true"
                action = item.get("action")
                if action and not isinstance(action, dict):
                    action = None
                input_from = str(item.get("input_from", "")).strip()
                task_record = self._make_task(
                    goal_id=goal_id,
                    description=description,
                    utterance=utterance,
                    terminal=terminal,
                    action=action,
                )
                if input_from:
                    task_record.input_from = input_from
                tasks.append(task_record)

            return tasks if tasks else None


        except (json.JSONDecodeError, ValueError) as exc:
            log.debug("TaskDecomposer: JSON parse error: {exc}", exc=exc)
            return None

    # ── Heuristic decomposition ────────────────────────────────────────────

    def _decompose_heuristic(
        self,
        goal_description: str,
        goal_id: str,
    ) -> list[TaskRecord] | None:
        """
        Pattern-match the goal against known workflow templates.

        Single-step patterns (exactly one task) automatically receive
        ``terminal=True`` so the ExecutionLoop stops immediately after
        the decisive action succeeds.

        Returns None if no pattern matches (caller will use passthrough).
        """
        # Guard: if the goal contains multiple distinct actionable clauses,
        # never greedily match a single-step pattern and drop remaining clauses.
        clauses = _extract_actionable_clauses(goal_description)
        if len(clauses) >= 2:
            return None

        goal_lower = goal_description.lower().strip().rstrip(".?!")

        # Direct desktop app launch
        m_app = re.match(r'^(?:open|launch|start)\s+([\w\s]+)$', goal_lower)
        if m_app:
            app_key = m_app.group(1).lower().strip()
            if app_key in _KNOWN_DESKTOP_APPS:
                app_name = _KNOWN_DESKTOP_APPS[app_key]
                is_browser_app = app_key in ("edge", "microsoft edge", "chrome", "google chrome", "firefox")
                active_browser = "edge" if "edge" in app_key else ("chrome" if "chrome" in app_key else "firefox") if is_browser_app else ""
                return [
                    self._make_structured_task(
                        goal_id=goal_id,
                        description=f"Open {app_name}",
                        utterance=f"open {app_key}",
                        skill="desktop",
                        action="open_app",
                        target=app_name,
                        expected_outcome=f"{app_name} is running",
                        terminal=True,
                        browser=active_browser,
                    )
                ]

        for keywords, step_templates in _HEURISTIC_PATTERNS:
            if any(kw in goal_lower for kw in keywords):
                log.debug(
                    "TaskDecomposer: heuristic match for keywords: {kws}",
                    kws=keywords[:3],
                )
                tasks = [
                    self._make_task(
                        goal_id=goal_id,
                        description=desc,
                        utterance=utterance,
                    )
                    for desc, utterance in step_templates
                ]
                # Mark the sole task (or the last task in a multi-step
                # heuristic) as terminal so the loop exits after success.
                if tasks:
                    tasks[-1] = TaskRecord(
                        task_id=tasks[-1].task_id,
                        goal_id=tasks[-1].goal_id,
                        description=tasks[-1].description,
                        utterance=tasks[-1].utterance,
                        state=tasks[-1].state,
                        extra=tasks[-1].extra,
                        terminal=True,
                    )
                return tasks

        return None

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _make_task(
        goal_id: str,
        description: str,
        utterance: str,
        terminal: bool = False,
        action: "dict | None" = None,
    ) -> TaskRecord:
        """Construct a pending TaskRecord with optional structured action."""
        return TaskRecord(
            goal_id=goal_id,
            description=description,
            utterance=utterance,
            action=action,
            state=TaskState.PENDING,
            terminal=terminal,
        )

    @staticmethod
    def _make_structured_task(
        goal_id: str,
        description: str,
        utterance: str,
        skill: str,
        action: str,
        target: str = "",
        query: str = "",
        url: str = "",
        expected_outcome: str = "",
        terminal: bool = False,
        browser: str = "",
    ) -> TaskRecord:
        """
        Construct a TaskRecord with a fully populated structured action.

        This is the primary M18 factory for heuristic tasks where we know
        the exact parameters at decomposition time.
        """
        action_dict: dict = {
            "skill": skill,
            "action": action,
        }
        if target:
            action_dict["target"] = target
        if query:
            action_dict["query"] = _normalize_search_query(query)
        if url:
            action_dict["url"] = url
        if expected_outcome:
            action_dict["expected_outcome"] = expected_outcome
        if browser:
            action_dict["browser"] = browser
        return TaskRecord(
            goal_id=goal_id,
            description=description,
            utterance=utterance,
            action=action_dict,
            state=TaskState.PENDING,
            terminal=terminal,
        )

    @staticmethod
    def _extract_search_params(clause: str) -> tuple[str, str]:
        """
        Extract (target, query) from a search clause.

        Handles patterns like:
          "search YouTube for Python tutorials"
          "search for Python tutorials on YouTube"
          "search Google for best Python books"
          "search Python tutorials"
          "search something"

        Returns (target, query) where:
          target -- lowercased site name ("youtube", "google", ...) or ""
          query  -- normalized query string with trailing sentence punctuation stripped

        Returns ("", "") if extraction fails.
        """
        text = clause.strip()
        # Pattern 1: "search <site> for <query>"
        m = re.match(
            r'search\s+([\w]+)\s+for\s+(.+)',
            text, re.IGNORECASE,
        )
        if m:
            target = m.group(1).lower().strip()
            query  = _normalize_search_query(m.group(2))
            return target, query

        # Pattern 2: "search for <query> on <site>"
        m = re.match(
            r'search\s+(?:for\s+)?(.+)\s+on\s+([\w]+)',
            text, re.IGNORECASE,
        )
        if m:
            query  = _normalize_search_query(m.group(1))
            target = m.group(2).lower().strip()
            return target, query

        # Pattern 3: "search for <query>" (no site specified)
        m = re.match(r'search\s+for\s+(.+)', text, re.IGNORECASE)
        if m:
            return "", _normalize_search_query(m.group(1))

        # Pattern 4: "search <query>" (e.g. "search Python tutorials" or "search something")
        m = re.match(r'search\s+(.+)', text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            parts = raw.split(maxsplit=1)
            if parts and parts[0].lower() in _KNOWN_WEB_TARGETS and len(parts) > 1:
                return parts[0].lower(), _normalize_search_query(parts[1])
            return "", _normalize_search_query(raw)

        return "", ""


def detect_default_browser() -> str:
    """
    Detect the system default web browser on Windows, or return 'edge' / 'chrome'.
    """
    import sys
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
            ) as key:
                prog_id, _ = winreg.QueryValueEx(key, "ProgId")
                prog_id_lower = str(prog_id).lower()
                if "chrome" in prog_id_lower:
                    return "chrome"
                if "edge" in prog_id_lower:
                    return "edge"
        except Exception:
            pass
    return "edge"


def is_placeholder_or_ambiguous_query(query: str) -> bool:
    """
    Check if a search query is ambiguous or a placeholder word like 'something'.
    """
    q = query.strip().lower().rstrip(".?!")
    if not q:
        return True
    return q in (
        "something", "anything", "whatever", "stuff", "it", "this", "that",
        "something online", "for something", "something on the web",
        "for anything", "something on youtube", "something on google",
    )


def _normalize_search_query(raw_query: str) -> str:
    """
    Strip terminal sentence punctuation (. ! ? ,) from search queries
    while preserving meaningful internal punctuation (e.g. 'node.js', 'C++', 'Python 3.12').
    """
    q = raw_query.strip()
    while q and q[-1] in (".", "!", "?", ","):
        q = q[:-1].rstrip()
    return q

