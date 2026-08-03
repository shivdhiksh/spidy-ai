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
- Each item must be a JSON object with two fields:
    "description": short human-readable step name (max 10 words)
    "utterance": the natural-language command to execute this step (what a user would say to an AI assistant)
- Produce 3–10 steps total. Fewer is better when steps can be combined.
- Steps must be ordered (first to last).
- Each step should be independently executable.
- Do NOT include steps that require no action (e.g. "Wait", "Think").

Example for "Create a Flask project":
[
  {"description": "Create project folder", "utterance": "create a folder named flask_project on the desktop"},
  {"description": "Open terminal", "utterance": "open a terminal or command prompt"},
  {"description": "Create virtual environment", "utterance": "create a python virtual environment named venv in the terminal"},
  {"description": "Install Flask", "utterance": "install flask using pip in the terminal"},
  {"description": "Create app.py", "utterance": "create a file named app.py with a basic flask hello world application"},
  {"description": "Open VS Code", "utterance": "open vs code in the project folder"},
  {"description": "Run the Flask app", "utterance": "run the flask application using python app.py"}
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

    # ── YouTube search ──────────────────────────────────────────────────────
    (
        ["search youtube", "youtube", "search on youtube"],
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

        if llm_client is not None:
            tasks = await self._decompose_with_llm(goal_description, goal_id, llm_client)

        if not tasks:
            tasks = self._decompose_heuristic(goal_description, goal_id)

        if not tasks:
            # Last resort: single-task passthrough (the goal IS the utterance)
            tasks = [self._make_task(
                goal_id=goal_id,
                description=goal_description,
                utterance=goal_description,
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
                tasks.append(self._make_task(
                    goal_id=goal_id,
                    description=description,
                    utterance=utterance,
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

        Returns None if no pattern matches (caller will use passthrough).
        """
        goal_lower = goal_description.lower()

        for keywords, step_templates in _HEURISTIC_PATTERNS:
            if any(kw in goal_lower for kw in keywords):
                log.debug(
                    "TaskDecomposer: heuristic match for keywords: {kws}",
                    kws=keywords[:3],
                )
                return [
                    self._make_task(
                        goal_id=goal_id,
                        description=desc,
                        utterance=utterance,
                    )
                    for desc, utterance in step_templates
                ]

        return None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_task(
        goal_id: str,
        description: str,
        utterance: str,
    ) -> TaskRecord:
        """Construct a pending TaskRecord."""
        return TaskRecord(
            goal_id=goal_id,
            description=description,
            utterance=utterance,
            state=TaskState.PENDING,
        )
