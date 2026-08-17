"""
TaskDecomposer — Goal → Task List Conversion (Milestone 13)
============================================================
Converts a high-level goal description into an ordered list of
executable TaskRecords that the ExecutionLoop can run step by step.

Two decomposition strategies
-----------------------------
1. LLM-based (preferred when an LLM client is available):
   Sends a structured prompt to the LLM asking it to break down the
   goal into numbered, concrete tasks with natural-language utterances.
   The LLM output is parsed into TaskRecord objects.

2. Heuristic-based (fallback when LLM is unavailable or fails):
   Uses keyword/pattern matching to recognise common workflows
   (Flask project, React app, Python project, VS Code, browser search)
   and produces a fixed sequence of tasks for each pattern.

Design
------
- The decomposer is stateless — no side effects, pure input→output
- LLM path: gracefully falls back to heuristic on parse failure
- All returned TaskRecords start in TaskState.PENDING
- Each task has both a human-readable description AND an utterance
  (the utterance is what Brain.process() will actually receive)

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
You are a task planning assistant. Your job is to break down a high-level goal
into a small number of concrete, executable steps.

Rules:
- Return ONLY a valid JSON array, no prose, no markdown fences.
- Each item must be a JSON object with these fields:
    "description": short human-readable step name (max 10 words)
    "utterance": the natural-language command to execute this step (what a user would say to an AI assistant)
    "terminal": true if this step achieves the final objective and no further steps are needed after it succeeds
- Produce 3–10 steps total. Fewer is better when steps can be combined.
- Steps must be ordered (first to last).
- Each step should be independently executable.
- Do NOT include steps that require no action (e.g. "Wait", "Think").
- For simple single-action goals (e.g. "open notepad"), return exactly ONE step marked terminal=true.

Example for "Create a Flask project":
[
  {"description": "Create project folder", "utterance": "create a folder named flask_project on the desktop", "terminal": false},
  {"description": "Open terminal", "utterance": "open a terminal or command prompt", "terminal": false},
  {"description": "Create virtual environment", "utterance": "create a python virtual environment named venv in the terminal", "terminal": false},
  {"description": "Install Flask", "utterance": "install flask using pip in the terminal", "terminal": false},
  {"description": "Create app.py", "utterance": "create a file named app.py with a basic flask hello world application", "terminal": false},
  {"description": "Open VS Code", "utterance": "open vs code in the project folder", "terminal": false},
  {"description": "Run the Flask app", "utterance": "run the flask application using python app.py", "terminal": true}
]
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
    r'^(?:open|launch|close|quit|exit|minimize|minimise|maximize|maximise|restore|'
    r'focus|bring|switch|search|find|look\s+up|create|make|start|run|'
    r'navigate|go\s+to|show|insert|type|click|press|scroll|'
    r'download|upload|play|pause|stop|skip|next)\b',
    re.IGNORECASE,
)

# Connector words that may signal a compound command.
# Also matches sentence-boundary periods/semicolons between two clauses
# so that "Open Edge. Search YouTube for Python." is split correctly.
_COMPOUND_CONNECTORS_RE = re.compile(
    r'(?:\s+(?:and|then|after\s+that|followed\s+by|next)\s+|[.;!]\s+)',
    re.IGNORECASE,
)


def _split_compound_goal(goal: str) -> list[str] | None:
    """
    Split a compound goal string into ordered sub-goals.

    Returns a list of 2+ sub-goals if the utterance is compound, or None
    if it is a single goal (caller uses normal decomposition).

    Only splits when BOTH sides of the connector start with an action verb
    to prevent false positives on "search for cats and dogs".
    """
    parts = _COMPOUND_CONNECTORS_RE.split(goal.strip())
    if len(parts) < 2:
        return None

    result: list[str] = []
    pending = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        combined = pending + (" " + part if pending else part)
        if _ACTION_VERBS.match(combined.strip()):
            if pending:
                result.append(pending.strip())
            pending = part
        else:
            # No action verb — merge with previous part
            pending = combined

    if pending:
        result.append(pending.strip())

    # Must have ≥2 distinct parts where both start with action verbs
    if len(result) >= 2 and all(_ACTION_VERBS.match(p) for p in result):
        return result
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
        # the heuristic patterns run.  Split into sub-goals, decompose each
        # independently, and concatenate the task lists.
        # Single-command goals ("Open Notepad") are unaffected — the split
        # only fires when both sides start with an action verb.
        sub_goals = _split_compound_goal(goal_description)
        if sub_goals is not None:
            log.info(
                "TaskDecomposer: compound goal detected — {n} sub-goals: {parts}",
                n=len(sub_goals),
                parts=[s[:40] for s in sub_goals],
            )
            all_tasks: list[TaskRecord] = []
            for i, sub in enumerate(sub_goals):
                is_last = (i == len(sub_goals) - 1)
                sub_tasks = self._decompose_heuristic(sub, goal_id)
                if not sub_tasks and llm_client is not None:
                    sub_tasks = await self._decompose_with_llm(sub, goal_id, llm_client)
                if not sub_tasks:
                    # SAFETY: every sub-goal MUST produce at least one task.
                    # A passthrough task is created so no requested clause is
                    # silently dropped.
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
                # Only the last task of the last sub-goal is terminal
                if is_last and sub_tasks:
                    sub_tasks[-1] = TaskRecord(
                        task_id=sub_tasks[-1].task_id,
                        goal_id=sub_tasks[-1].goal_id,
                        description=sub_tasks[-1].description,
                        utterance=sub_tasks[-1].utterance,
                        state=sub_tasks[-1].state,
                        extra=sub_tasks[-1].extra,
                        terminal=True,
                    )
                else:
                    # Intermediate sub-goals must NOT be terminal
                    for j, t in enumerate(sub_tasks):
                        if t.terminal:
                            sub_tasks[j] = TaskRecord(
                                task_id=t.task_id,
                                goal_id=t.goal_id,
                                description=t.description,
                                utterance=t.utterance,
                                state=t.state,
                                extra=t.extra,
                                terminal=False,
                            )
                all_tasks.extend(sub_tasks)

            tasks = all_tasks[:self._max_tasks]

            # Invariant: compound goals must never produce fewer tasks than
            # there were distinct clauses.  Log a warning when this occurs
            # (it should not happen due to the passthrough above, but we
            # check explicitly so the problem is visible in logs).
            if len(tasks) < len(sub_goals):
                log.error(
                    "TaskDecomposer: compound goal has {n_clauses} clauses but only "
                    "{n_tasks} tasks produced — possible silent drop!",
                    n_clauses=len(sub_goals),
                    n_tasks=len(tasks),
                )

            log.info(
                "Decomposed compound goal '{desc}' \u2192 {n} tasks",
                desc=goal_description[:80],
                n=len(tasks),
            )
            return tasks

        # ── Heuristic-first strategy ────────────────────────────────────────
        # Always try the heuristic pattern table first.
        # If the goal matches a known single-action pattern (e.g. "open edge"),
        # the heuristic result is authoritative — the LLM is NOT called.
        # This prevents the LLM from inflating simple launches into multi-step
        # plans (e.g. "find shortcut → double-click → minimize other windows").
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
                tasks.append(self._make_task(
                    goal_id=goal_id,
                    description=description,
                    utterance=utterance,
                    terminal=terminal,
                ))

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
        goal_lower = goal_description.lower()

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

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_task(
        goal_id: str,
        description: str,
        utterance: str,
        terminal: bool = False,
    ) -> TaskRecord:
        """Construct a pending TaskRecord."""
        return TaskRecord(
            goal_id=goal_id,
            description=description,
            utterance=utterance,
            state=TaskState.PENDING,
            terminal=terminal,
        )
