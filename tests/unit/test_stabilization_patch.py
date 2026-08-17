"""
tests/unit/test_stabilization_patch.py
=========================================
Regression tests for SPIDY Small Runtime Stabilization Patch.

Covers:
  1. Goal-completion UI notification — no TypeError
  2. Repeated wake-phrase transcript is dropped
  3. Repeated wake phrase does not reach Brain
  4. Repeated wake phrase does not reach LLM
  5. Normal wake + command still works
  6. Compound goal creates all required tasks
  7. Compound goal cannot silently drop a clause
  8. Punctuation variants of compound goals work
  9. Incomplete goal is not falsely marked verified
 10. Existing simple commands remain correct
"""
from __future__ import annotations

import asyncio
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Goal-completion UI notification does not throw TypeError
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalCompletionNotification:
    """UISignalBridge.request_notification(title, msg, level) — 3 args only."""

    def _make_bridge(self):
        pytest.importorskip("PySide6", reason="PySide6 not installed")
        from spidy.ui.overlay import UISignalBridge
        import sys
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication(sys.argv[:1])
        return UISignalBridge()

    def test_three_arg_call_does_not_raise(self):
        bridge = self._make_bridge()
        # Must not raise TypeError
        bridge.request_notification("Goal Complete", "Done.", "success")

    def test_two_arg_call_does_not_raise(self):
        bridge = self._make_bridge()
        bridge.request_notification("Info", "Something happened.")

    def test_four_arg_call_raises_type_error(self):
        """Ensure callers know the signature is 3-arg, not 4."""
        bridge = self._make_bridge()
        with pytest.raises(TypeError):
            bridge.request_notification("Title", "Body", "info", 5000)

    def test_goal_completed_handler_uses_correct_args(self):
        """
        The _handle_agent_goal_completed coroutine must call
        request_notification with exactly 3 positional args (no duration_ms).
        """
        import ast, pathlib
        src = pathlib.Path("spidy/ui/app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_agent_goal_completed":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        attr = func.attr if isinstance(func, ast.Attribute) else None
                        if attr == "request_notification":
                            total_args = len(child.args) + len(child.keywords)
                            assert total_args <= 3, (
                                f"_handle_agent_goal_completed passes {total_args} args "
                                "to request_notification; expected <= 3."
                            )
                return
        pytest.fail("_handle_agent_goal_completed not found in app.py")


# ─────────────────────────────────────────────────────────────────────────────
# 2 & 3. Repeated wake-phrase transcript is dropped
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeOnlyDetection:
    def _stripper(self):
        from spidy.voice.wake_word_stripper import WakeWordStripper
        return WakeWordStripper()

    # must be detected as wake-only
    @pytest.mark.parametrize("text", [
        "Hey Jarvis",
        "Hey Jarvis.",
        "Hey Jarvis. Hey Jarvis.",
        "Hey Jarvis, Hey Jarvis.",
        "Hey Jarvis. Hey Jarvis. Hey Jarvis.",
        "Okay Jarvis. Okay Jarvis.",
        "Ok Jarvis. Ok Jarvis.",
        "Hey Spidy. Hey Spidy.",
    ])
    def test_is_wake_only_true(self, text):
        s = self._stripper()
        assert s.is_wake_only(text), f"Expected wake-only for: {text!r}"

    # must NOT be detected as wake-only
    @pytest.mark.parametrize("text", [
        "Hey Jarvis, open Edge.",
        "Hey Jarvis, what is Python?",
        "Hey Jarvis open Edge",
        "Who is Jarvis?",
        "Okay Jarvis, search YouTube for Python tutorials.",
        "open Edge",
    ])
    def test_is_wake_only_false(self, text):
        s = self._stripper()
        assert not s.is_wake_only(text), f"Expected NOT wake-only for: {text!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Repeated wake phrase does not reach Brain (pipeline simulation)
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeOnlyPipelineDrop:
    """
    Simulate the continuous.py Step 1 logic directly.

    The guard must return (drop) before forwarding to Brain when the
    transcript is wake-only.
    """

    def _guard_step(self, text: str) -> bool:
        """Returns True if the text would be DROPPED by Step 1."""
        from spidy.voice.wake_word_stripper import WakeWordStripper
        stripper = WakeWordStripper()
        if stripper.is_wake_only(text):
            return True  # dropped
        stripped = stripper.strip_wake_prefix(text)
        if not stripped:
            return True  # also dropped
        return False     # would proceed to Brain

    @pytest.mark.parametrize("text", [
        "Hey Jarvis. Hey Jarvis.",
        "Hey Jarvis, Hey Jarvis.",
        "Okay Jarvis. Okay Jarvis.",
        "Hey Jarvis. Hey Jarvis. Hey Jarvis.",
    ])
    def test_repeated_wake_dropped_before_brain(self, text):
        assert self._guard_step(text), f"Repeated wake phrase must be dropped: {text!r}"

    @pytest.mark.parametrize("text", [
        "Hey Jarvis, open Edge.",
        "Hey Jarvis, what is Python?",
        "open Edge",
    ])
    def test_valid_command_not_dropped(self, text):
        assert not self._guard_step(text), f"Valid command must not be dropped: {text!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Normal wake + command still works after stripping
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalWakeStripWorks:
    def _strip(self, text: str) -> str:
        from spidy.voice.wake_word_stripper import WakeWordStripper
        s = WakeWordStripper()
        assert not s.is_wake_only(text), f"Should NOT be wake-only: {text!r}"
        return s.strip_wake_prefix(text)

    def test_strip_hey_jarvis_open_edge(self):
        result = self._strip("Hey Jarvis, open Edge.")
        assert "open" in result.lower()

    def test_strip_okay_jarvis_search_youtube(self):
        result = self._strip("Okay Jarvis, search YouTube for Python.")
        assert "search" in result.lower() or "youtube" in result.lower()

    def test_no_wake_prefix_unchanged(self):
        from spidy.voice.wake_word_stripper import WakeWordStripper
        s = WakeWordStripper()
        assert s.strip_wake_prefix("open Edge") == "open Edge"

    def test_who_is_jarvis_not_stripped(self):
        from spidy.voice.wake_word_stripper import WakeWordStripper
        s = WakeWordStripper()
        result = s.strip_wake_prefix("Who is Jarvis?")
        assert "jarvis" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 6 & 7. Compound goal creates all required tasks and never drops a clause
# ─────────────────────────────────────────────────────────────────────────────

class TestCompoundGoalDecomposition:
    def _decompose(self, goal: str) -> list:
        from spidy.agent.task_decomposer import TaskDecomposer
        decomposer = TaskDecomposer()
        return asyncio.run(
            decomposer.decompose(goal, goal_id="test-g1", llm_client=None)
        )

    def test_open_edge_period_search_youtube_produces_multiple_tasks(self):
        tasks = self._decompose("Open Edge. Search YouTube for Python tutorials.")
        assert len(tasks) >= 2, (
            f"Expected >= 2 tasks for compound goal, got {len(tasks)}: "
            f"{[t.description for t in tasks]}"
        )

    def test_open_edge_and_search_youtube_produces_multiple_tasks(self):
        tasks = self._decompose("Open Edge and search YouTube for Python tutorials.")
        assert len(tasks) >= 2, f"Got {len(tasks)}: {[t.description for t in tasks]}"

    def test_open_edge_then_search_youtube_produces_multiple_tasks(self):
        tasks = self._decompose("Open Edge, then search YouTube for Python tutorials.")
        assert len(tasks) >= 2, f"Got {len(tasks)}: {[t.description for t in tasks]}"

    def test_open_edge_insert_youtube_produces_multiple_tasks(self):
        """Original user-observed failure case."""
        tasks = self._decompose("Open Edge. Insert YouTube for Python Tutorials.")
        assert len(tasks) >= 2, (
            f"'Insert YouTube' clause was silently dropped — got {len(tasks)} tasks: "
            f"{[t.description for t in tasks]}"
        )

    def test_compound_goal_no_silent_clause_drop(self):
        """Number of tasks must be >= number of clauses in compound goal."""
        from spidy.agent.task_decomposer import _split_compound_goal
        goal = "Open Edge. Search YouTube for Python tutorials."
        sub_goals = _split_compound_goal(goal)
        assert sub_goals is not None and len(sub_goals) >= 2

        tasks = self._decompose(goal)
        assert len(tasks) >= len(sub_goals), (
            f"Task count ({len(tasks)}) < clause count ({len(sub_goals)}) — silent drop!"
        )

    def test_single_open_edge_not_split(self):
        """Single-action goals must not be spuriously split."""
        tasks = self._decompose("Open Edge")
        # Should be exactly 1 task for a single-step goal
        assert len(tasks) == 1, f"Single goal should produce 1 task, got {len(tasks)}"

    def test_open_edge_only_produced_one_task_before_fix(self):
        """Regression: the BUG scenario produced only 1 task for 2-clause goal.
        Now we require >= 2 tasks for compound input."""
        tasks = self._decompose("Open Edge. Search YouTube for Python tutorials.")
        descriptions = [t.description.lower() for t in tasks]
        # At least one task must reference browser/Edge
        assert any("edge" in d or "browser" in d or "open" in d for d in descriptions), (
            f"No browser task found: {descriptions}"
        )
        # At least one task must reference YouTube/search
        assert any("youtube" in d or "search" in d for d in descriptions), (
            f"No YouTube/search task found: {descriptions}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Punctuation variants of compound goals
# ─────────────────────────────────────────────────────────────────────────────

class TestCompoundGoalPunctuationVariants:
    def _decompose(self, goal: str) -> list:
        from spidy.agent.task_decomposer import TaskDecomposer
        decomposer = TaskDecomposer()
        return asyncio.run(
            decomposer.decompose(goal, goal_id="test-g2", llm_client=None)
        )

    @pytest.mark.parametrize("goal", [
        "Open Edge. Search YouTube for Python tutorials.",
        "Open Edge and search YouTube for Python tutorials.",
        "Open Edge, then search YouTube for Python tutorials.",
        "Open Edge. Insert YouTube for Python Tutorials.",
    ])
    def test_variant_produces_multiple_tasks(self, goal):
        tasks = self._decompose(goal)
        assert len(tasks) >= 2, (
            f"Expected >= 2 tasks for '{goal}', got {len(tasks)}: "
            f"{[t.description for t in tasks]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Incomplete goal is not falsely marked verified
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifierIncompleteGoal:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_failed_task_not_verified(self):
        from spidy.agent.verifier import GoalVerifier
        from spidy.agent.types import GoalRecord, TaskRecord, TaskState

        goal = GoalRecord(description="Open Edge and search YouTube", goal_id="g1")
        t1 = TaskRecord(goal_id="g1", description="Open Edge", utterance="open edge",
                        state=TaskState.COMPLETED, terminal=False)
        t2 = TaskRecord(goal_id="g1", description="Search YouTube",
                        utterance="search youtube", state=TaskState.FAILED, terminal=True)
        goal.tasks = [t1, t2]

        result = self._run(GoalVerifier().verify(goal))
        assert not result.verified, "Goal with a failed task must NOT be verified"

    def test_partially_run_goal_not_verified(self):
        from spidy.agent.verifier import GoalVerifier
        from spidy.agent.types import GoalRecord, TaskRecord, TaskState

        goal = GoalRecord(description="Open Edge and search YouTube", goal_id="g2")
        t1 = TaskRecord(goal_id="g2", description="Open Edge", utterance="open edge",
                        state=TaskState.COMPLETED, terminal=False)
        # Second task never ran (still PENDING)
        t2 = TaskRecord(goal_id="g2", description="Search YouTube",
                        utterance="search youtube", state=TaskState.PENDING, terminal=True)
        goal.tasks = [t1, t2]

        result = self._run(GoalVerifier().verify(goal))
        assert not result.verified, "Partial completion (pending task) must NOT be verified"

    def test_all_completed_is_verified(self):
        from spidy.agent.verifier import GoalVerifier
        from spidy.agent.types import GoalRecord, TaskRecord, TaskState

        goal = GoalRecord(description="Open Edge and search YouTube", goal_id="g3")
        t1 = TaskRecord(goal_id="g3", description="Open Edge", utterance="open edge",
                        state=TaskState.COMPLETED, terminal=False)
        t2 = TaskRecord(goal_id="g3", description="Search YouTube",
                        utterance="search youtube", state=TaskState.COMPLETED, terminal=True)
        goal.tasks = [t1, t2]

        result = self._run(GoalVerifier().verify(goal))
        assert result.verified, "All-completed goal must be verified"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Existing simple commands remain fast and correct
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleCommandsUnaffected:
    def _decompose(self, goal: str) -> list:
        from spidy.agent.task_decomposer import TaskDecomposer
        decomposer = TaskDecomposer()
        return asyncio.run(
            decomposer.decompose(goal, goal_id="test-s", llm_client=None)
        )

    @pytest.mark.parametrize("goal,expected_desc", [
        ("open notepad", "Open Notepad"),
        ("open edge", "Open Edge"),
        ("open chrome", "Open Chrome"),
        ("lock screen", "Lock Screen"),
        ("take a screenshot", "Take Screenshot"),
        ("open vs code", "Open VS Code"),
    ])
    def test_single_action_produces_one_task(self, goal, expected_desc):
        tasks = self._decompose(goal)
        assert len(tasks) == 1, f"'{goal}' should produce 1 task, got {len(tasks)}"
        assert tasks[0].terminal, f"Single task must be marked terminal"

    def test_search_google_not_affected(self):
        tasks = self._decompose("search google for cats")
        assert len(tasks) >= 1

    def test_flask_project_produces_multi_tasks(self):
        tasks = self._decompose("create a flask project")
        assert len(tasks) >= 3, f"Flask project should produce >= 3 tasks, got {len(tasks)}"

