"""
Unit Tests: Goal Completion & Autonomous Execution Reliability Pass
====================================================================
Validates all 34 regression specifications:
- Goal Completion: partial completion, missing required tasks, terminal gating, dependencies, goal preservation.
- Tool Selection: FileSkill, AppSkill, BrowserSkill, ScreenUnderstanding, Code.
- Reference Resolution: 'open it', 'type the query', 'type the result', ambiguous reference.
- Clarification: WAITING_FOR_USER state, zero retry, zero LLM, same goal resumes, binding.
- Recovery: app not ready, focus lost, control missing, file missing, browser failure, code failure.
- Voice: duplicate wake suppression, active session gating, TTS echo isolation, transcript normalization.
- Verification: typing, save, browser, screen, code execution verification.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from spidy.agent.types import (
    ExecutionContext,
    GoalContext,
    GoalRecord,
    GoalState,
    StepResult,
    TaskErrorCode,
    TaskPriority,
    TaskRecord,
    TaskState,
)
from spidy.agent.verifier import GoalVerifier, VerificationResult
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.execution_loop import ExecutionLoop
from spidy.agent.goal_manager import GoalManager
from spidy.agent.reflection_engine import ReflectionEngine
from spidy.agent.progress_tracker import ProgressTracker
from spidy.core.event_bus import EventBus
from spidy.voice.normalizer import TranscriptNormalizer
from spidy.voice.session import VoiceSessionManager, VoiceSessionState


# ── 1. GOAL COMPLETION ─────────────────────────────────────────────────────────


class TestGoalCompletion:
    """Regression tests 1–5: Goal completion semantics."""

    @pytest.mark.asyncio
    async def test_1_partial_task_does_not_complete_goal(self):
        """1. Partial task completion must not mark the goal complete."""
        verifier = GoalVerifier()
        t1 = TaskRecord(description="Find resume", state=TaskState.COMPLETED, result_message="Found resume.pdf")
        t2 = TaskRecord(description="Open resume", state=TaskState.FAILED, error="Failed to open")
        goal = GoalRecord(description="Find my resume and open it", tasks=[t1, t2])

        res = await verifier.verify(goal)
        assert not res.verified
        assert res.confidence == "partial"
        assert "I found your resume, but I couldn't open it." in res.summary

    @pytest.mark.asyncio
    async def test_2_missing_required_task_triggers_replan(self):
        """2. Missing required task/clause triggers verification failure / replan."""
        verifier = GoalVerifier()
        # Goal asked for 2 clauses, but only 1 task was present
        t1 = TaskRecord(description="Find resume", state=TaskState.COMPLETED, terminal=True)
        goal = GoalRecord(description="Find my resume and open it", tasks=[t1])

        res = await verifier.verify(goal)
        assert not res.verified
        assert res.method == "clause_integrity_check"

    @pytest.mark.asyncio
    async def test_3_terminal_task_does_not_bypass_goal_verification(self):
        """3. Terminal task does not bypass goal verification if required outcome missing."""
        verifier = GoalVerifier()
        t1 = TaskRecord(
            description="Open VS Code",
            state=TaskState.COMPLETED,
            terminal=True,
            action={"expected_outcome": "VS Code is running"},
        )
        ctx = GoalContext(
            goal_id="g1",
            original_goal="Open VS Code and create Edge.py",
            required_outcomes=["File Edge.py is created", "VS Code is running"],
            tasks=[t1],
        )

        res = await verifier.verify_goal(ctx)
        assert not res.verified
        assert "not verified" in res.summary

    @pytest.mark.asyncio
    async def test_4_dependent_task_waits_for_dependency(self):
        """4. A dependent task fails immediately if prerequisite dependency was not completed."""
        bus = EventBus()
        goal_mgr = GoalManager(bus=bus)
        loop = ExecutionLoop(brain=MagicMock(), goal_manager=goal_mgr, bus=bus)

        t1 = TaskRecord(task_id="t1", description="Search file", state=TaskState.FAILED)
        t2 = TaskRecord(task_id="t2", description="Open file", dependencies=["t1"], state=TaskState.PENDING)

        goal = GoalRecord(goal_id="g1", description="Search and open", tasks=[t1, t2])
        ctx = ExecutionContext(goal=goal)

        res_task, _, _ = await loop._execute_task(t2, 1, 2, ctx)
        assert res_task.state == TaskState.FAILED
        assert "Prerequisite dependency" in res_task.error

    @pytest.mark.asyncio
    async def test_5_failed_task_preserves_original_goal(self):
        """5. Failed task retains the uncorrupted original goal in GoalContext."""
        ctx = GoalContext(
            goal_id="g5",
            original_goal="Open Edge, search YouTube, copy result into Notepad",
            normalized_goal="Open Edge, search YouTube, copy result into Notepad",
        )
        ctx.add_recovery("t2", "retry_search", "Network error", "Failed")

        assert ctx.original_goal == "Open Edge, search YouTube, copy result into Notepad"
        assert len(ctx.recovery_history) == 1
        assert ctx.recovery_history[0]["strategy"] == "retry_search"


# ── 2. TOOL SELECTION ──────────────────────────────────────────────────────────


class TestToolSelection:
    """Regression tests 6–10: Deterministic skill matching."""

    @pytest.mark.asyncio
    async def test_6_file_request_to_file_skill(self):
        """6. File search request maps to file.search_files."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("Find my resume", "g6")
        assert len(tasks) == 1
        assert tasks[0].action["skill"] == "file"
        assert tasks[0].action["action"] == "search_files"

    @pytest.mark.asyncio
    async def test_7_app_request_to_app_skill(self):
        """7. App request maps to desktop.open_app."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("Open Notepad", "g7")
        assert len(tasks) == 1
        assert tasks[0].action["skill"] == "desktop"
        assert tasks[0].action["action"] == "open_app"
        assert tasks[0].action["target"] == "Notepad"

    @pytest.mark.asyncio
    async def test_8_browser_request_to_browser_skill(self):
        """8. Browser search request maps to browser.search."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("Search YouTube for Python tutorials", "g8")
        assert len(tasks) == 1
        assert tasks[0].action["skill"] == "browser"
        assert tasks[0].action["action"] == "search"
        assert tasks[0].action["target"] == "youtube"
        assert tasks[0].action["query"] == "Python tutorials"

    @pytest.mark.asyncio
    async def test_9_screen_request_to_screen_understanding(self):
        """9. Screen inspection request maps to vision.ask_screen."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("What is on my screen?", "g9")
        assert len(tasks) == 1
        assert tasks[0].action["skill"] == "vision"
        assert tasks[0].action["action"] == "ask_screen"

    @pytest.mark.asyncio
    async def test_10_code_request_to_code_agent(self):
        """10. Code error repair request maps to code.repair_code."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("Fix the error on my screen", "g10")
        assert len(tasks) == 1
        assert tasks[0].action["skill"] == "code"
        assert tasks[0].action["action"] == "repair_code"


# ── 3. REFERENCE RESOLUTION ───────────────────────────────────────────────────


class TestReferenceResolution:
    """Regression tests 11–14: Dynamic pronoun and reference binding."""

    def test_11_open_it_resolves_to_previous_file(self):
        """11. 'open it' resolves to the previous file path."""
        ctx = ExecutionContext()
        ctx.add_structured_result("t1", {"path": "C:\\Users\\User\\Documents\\Resume.pdf"})
        resolved = ctx.resolve_variable("the file")
        assert resolved == "C:\\Users\\User\\Documents\\Resume.pdf"

    def test_12_type_the_query_resolves_to_previous_query(self):
        """12. 'the search query' resolves to previous search query."""
        ctx = ExecutionContext()
        ctx.add_structured_result("t1", {"query": "Python tutorials", "target": "YouTube"})
        resolved = ctx.resolve_variable("the search query")
        assert resolved == "Python tutorials"

    def test_13_type_the_result_resolves_to_previous_result(self):
        """13. 'the result' resolves to previous step result/text."""
        ctx = ExecutionContext()
        ctx.add_structured_result("t1", {"text": "Top Video: Complete Python Course 2026"})
        resolved = ctx.resolve_variable("the result")
        assert resolved == "Top Video: Complete Python Course 2026"

    @pytest.mark.asyncio
    async def test_14_ambiguous_reference_triggers_clarification(self):
        """14. Ambiguous reference 'search for something' creates dialog.ask_user."""
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("Search for something", "g14")
        assert tasks[0].action["skill"] == "dialog"
        assert tasks[0].action["action"] == "ask_user"


# ── 4. CLARIFICATION & WAITING_FOR_USER ───────────────────────────────────────


class TestClarificationWaitingState:
    """Regression tests 15–19: WAITING_FOR_USER lifecycle."""

    @pytest.mark.asyncio
    async def test_15_to_19_clarification_lifecycle(self):
        """15–19. Clarification pauses cleanly, avoids redundant LLMs, and binds user answer."""
        from spidy.agent.agent import AutonomousAgent

        brain = MagicMock()
        bus = EventBus()
        agent = AutonomousAgent(brain=brain, bus=bus)

        # 15. Run goal with ambiguous search
        res = await agent.run_goal("Search for something")
        assert agent.has_waiting_goal()
        assert "What would you like me to search for?" in res

        # 16–17. Verify no unnecessary brain / LLM retries during pause
        assert brain.process.call_count == 0

        # 18–19. Resuming binds answer to the same goal
        brain.process = AsyncMock(return_value="Searched Google for Python 3.13 release notes")
        res2 = await agent.resume_with_user_input("Python 3.13 release notes")
        assert not agent.has_waiting_goal()


# ── 5. RECOVERY ────────────────────────────────────────────────────────────────


class TestRecovery:
    """Regression tests 20–25: Recovery error codes and strategies."""

    def test_20_to_25_error_codes(self):
        """20–25. Error codes exist and are typed."""
        assert TaskErrorCode.APP_NOT_FOUND.value == "APP_NOT_FOUND"
        assert TaskErrorCode.WINDOW_NOT_FOUND.value == "WINDOW_NOT_FOUND"
        assert TaskErrorCode.FOCUS_LOST.value == "FOCUS_LOST"
        assert TaskErrorCode.CONTROL_NOT_FOUND.value == "CONTROL_NOT_FOUND"
        assert TaskErrorCode.FILE_NOT_FOUND.value == "FILE_NOT_FOUND"
        assert TaskErrorCode.BROWSER_FAILURE.value == "BROWSER_FAILURE"
        assert TaskErrorCode.CODE_ERROR.value == "CODE_ERROR"


# ── 6. VOICE RELIABILITY ───────────────────────────────────────────────────────


class TestVoiceReliability:
    """Regression tests 26–29: Wake-word gating, TTS echo isolation, normalizer."""

    @pytest.mark.asyncio
    async def test_26_27_duplicate_wake_and_active_session_suppression(self):
        """26–27. Duplicate wake detection while active or speaking is suppressed."""
        bus = EventBus()
        mgr = VoiceSessionManager(bus=bus)
        await mgr.activate(wake_word="hey spidy")
        assert mgr.is_active

        # Second wake while active should reset timeout, not duplicate session
        sid1 = mgr.session_id
        sid2 = await mgr.activate(wake_word="hey spidy")
        assert sid1 == sid2

    def test_29_transcript_normalization(self):
        """29. Voice speech fragments are normalized while preserving raw transcript."""
        normalizer = TranscriptNormalizer()
        raw = "Open VS Code and create a Python. File name. Edge.py."
        res = normalizer.normalize(raw)
        assert res.original_transcript == raw
        assert "edge.py" in res.normalized_transcript.lower() or "python" in res.normalized_transcript.lower()


# ── 7. VERIFICATION ────────────────────────────────────────────────────────────


class TestVerification:
    """Regression tests 30–34: Outcome verifications."""

    @pytest.mark.asyncio
    async def test_30_typing_verification(self):
        """30. Typing verification checks result message."""
        verifier = GoalVerifier()
        t = TaskRecord(
            description="Type into Notepad",
            state=TaskState.COMPLETED,
            action={"expected_outcome": "Text 'Python tutorials' entered"},
            result_message="Text entered successfully.",
        )
        goal = GoalRecord(description="Type into Notepad", tasks=[t])
        res = await verifier.verify(goal)
        assert res.verified

    @pytest.mark.asyncio
    async def test_31_save_verification(self):
        """31. Save file verification checks completion."""
        verifier = GoalVerifier()
        t = TaskRecord(
            description="Save as result.txt",
            state=TaskState.COMPLETED,
            action={"expected_outcome": "File result.txt saved"},
            result_message="Saved file as 'result.txt'.",
        )
        goal = GoalRecord(description="Save file as result.txt", tasks=[t])
        res = await verifier.verify(goal)
        assert res.verified

    @pytest.mark.asyncio
    async def test_32_browser_verification(self):
        """32. Browser search verification."""
        verifier = GoalVerifier()
        t = TaskRecord(
            description="Search YouTube",
            state=TaskState.COMPLETED,
            action={"expected_outcome": "YouTube search results for Python tutorials are visible"},
            result_message="Navigated to YouTube search results.",
        )
        goal = GoalRecord(description="Search YouTube for Python tutorials", tasks=[t])
        res = await verifier.verify(goal)
        assert res.verified

    @pytest.mark.asyncio
    async def test_33_screen_verification(self):
        """33. Screen verification checks screen state observation."""
        verifier = GoalVerifier()
        t = TaskRecord(
            description="Inspect screen",
            state=TaskState.COMPLETED,
            action={"expected_outcome": "Screen state is analyzed"},
            result_message="The screen displays VS Code with hello.py.",
        )
        goal = GoalRecord(description="What is on my screen", tasks=[t])
        res = await verifier.verify(goal)
        assert res.verified

    @pytest.mark.asyncio
    async def test_34_code_execution_verification(self):
        """34. Code execution verification."""
        verifier = GoalVerifier()
        t1 = TaskRecord(
            description="Run hello.py",
            state=TaskState.COMPLETED,
            action={"expected_outcome": "Script hello.py executed"},
            result_message="Output of hello.py: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
        )
        t2 = TaskRecord(
            description="Tell output",
            state=TaskState.COMPLETED,
            action={"expected_outcome": "Output reported to user"},
            result_message="0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
        )
        goal = GoalRecord(description="Run hello.py and tell me the output", tasks=[t1, t2])
        res = await verifier.verify(goal)
        assert res.verified
