"""
Unit Tests — GoalIntentClassifier (Milestone 13.1)
===================================================
Tests the classification routing logic in goal_intent_classifier.py.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from spidy.brain.goal_intent_classifier import (
    GoalIntentClassifier,
    GOAL_ACTIONS,
    CHAT_ACTIONS,
    is_goal_intent,
    get_route,
)
from spidy.brain.types import Intent


def _make_intent(action: str, confidence: float = 0.9) -> Intent:
    """Create a minimal Intent mock for testing."""
    m = MagicMock(spec=Intent)
    m.action = action
    m.confidence = confidence
    return m


class TestGoalIntentClassifier:
    def setup_method(self):
        self.clf = GoalIntentClassifier()

    def test_goal_actions_route_to_goal(self):
        for action in GOAL_ACTIONS:
            intent = _make_intent(action, confidence=0.9)
            assert self.clf.classify(intent) == "goal", f"'{action}' should be 'goal'"

    def test_chat_actions_route_to_chat(self):
        for action in CHAT_ACTIONS:
            intent = _make_intent(action, confidence=0.9)
            assert self.clf.classify(intent) == "chat", f"'{action}' should be 'chat'"

    def test_low_confidence_always_chat(self):
        # Even normally-executable actions route to chat when confidence is low
        for action in ["launch_app", "search_files", "create_folder", "search_youtube"]:
            intent = _make_intent(action, confidence=0.3)
            assert self.clf.classify(intent) == "chat", \
                f"Low-confidence '{action}' should be 'chat'"

    def test_just_above_threshold_routes_to_goal(self):
        intent = _make_intent("launch_app", confidence=0.65)
        assert self.clf.classify(intent) == "goal"

    def test_just_below_threshold_routes_to_chat(self):
        intent = _make_intent("launch_app", confidence=0.64)
        assert self.clf.classify(intent) == "chat"

    def test_unknown_action_routes_to_chat_conservatively(self):
        intent = _make_intent("totally_unknown_action", confidence=0.9)
        assert self.clf.classify(intent) == "chat"

    def test_is_executable_returns_bool(self):
        goal_intent = _make_intent("launch_app")
        chat_intent = _make_intent("greet")
        assert self.clf.is_executable(goal_intent) is True
        assert self.clf.is_executable(chat_intent) is False

    def test_cancel_goal_routes_to_goal(self):
        intent = _make_intent("cancel_goal")
        assert self.clf.classify(intent) == "goal"

    def test_compound_routes_to_goal(self):
        intent = _make_intent("compound")
        assert self.clf.classify(intent) == "goal"

    def test_create_folder_routes_to_goal(self):
        intent = _make_intent("create_folder")
        assert self.clf.classify(intent) == "goal"

    def test_open_project_routes_to_goal(self):
        intent = _make_intent("open_project")
        assert self.clf.classify(intent) == "goal"

    def test_read_webpage_routes_to_goal(self):
        intent = _make_intent("read_webpage")
        assert self.clf.classify(intent) == "goal"

    def test_search_files_routes_to_goal(self):
        intent = _make_intent("search_files")
        assert self.clf.classify(intent) == "goal"


class TestModuleLevelFunctions:
    def test_is_goal_intent_with_goal_action(self):
        intent = _make_intent("launch_app")
        assert is_goal_intent(intent) is True

    def test_is_goal_intent_with_chat_action(self):
        intent = _make_intent("greet")
        assert is_goal_intent(intent) is False

    def test_get_route_goal(self):
        intent = _make_intent("search_youtube")
        assert get_route(intent) == "goal"

    def test_get_route_chat(self):
        intent = _make_intent("calculate")
        assert get_route(intent) == "chat"


class TestGoalActionCoverage:
    """Verify critical M13.1 action set membership."""

    def test_all_desktop_actions_are_goals(self):
        desktop_actions = {
            "launch_app", "close_app", "bring_app_to_foreground",
            "detect_running_apps", "search_files", "open_file",
            "open_folder", "create_folder", "delete_folder",
            "list_recent_files", "create_project", "open_project",
            "open_terminal",
        }
        missing = desktop_actions - GOAL_ACTIONS
        assert not missing, f"Desktop actions missing from GOAL_ACTIONS: {missing}"

    def test_all_browser_actions_are_goals(self):
        # P1-2 fix: search_web has moved to CHAT_ACTIONS.
        # All other explicit browser-engine actions still require goal routing.
        browser_actions = {
            "search_youtube", "search_bing",
            "open_url", "open_browser_and_search", "read_webpage",
        }
        missing = browser_actions - GOAL_ACTIONS
        assert not missing, f"Browser actions missing from GOAL_ACTIONS: {missing}"

    def test_search_web_not_in_goal_actions(self):
        """P1-2 regression guard: search_web must NOT be in GOAL_ACTIONS."""
        assert "search_web" not in GOAL_ACTIONS, (
            "search_web must be in CHAT_ACTIONS, not GOAL_ACTIONS, after P1-2 fix."
        )

    def test_search_web_in_chat_actions(self):
        """P1-2: search_web must be in CHAT_ACTIONS for the fast LLM-direct path."""
        assert "search_web" in CHAT_ACTIONS, (
            "search_web must be in CHAT_ACTIONS after P1-2 fix."
        )

    def test_conversational_actions_not_in_goals(self):
        """Greet/farewell/chat must never be in GOAL_ACTIONS."""
        conversational = {"greet", "farewell", "chat", "introduce"}
        overlap = conversational & GOAL_ACTIONS
        assert not overlap, f"Conversational actions in GOAL_ACTIONS: {overlap}"

    def test_cancel_goal_in_goal_actions(self):
        assert "cancel_goal" in GOAL_ACTIONS

    def test_compound_in_goal_actions(self):
        assert "compound" in GOAL_ACTIONS
