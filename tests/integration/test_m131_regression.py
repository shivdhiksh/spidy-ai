"""
M13.1 Regression Tests — All 12 Manual Validation Commands
===========================================================
Verifies that each of the 12 validation commands:

  1. Is classified by IntentClassifier with the expected action (not "chat").
  2. Is routed by GoalIntentClassifier as "goal" (not "chat").
  3. Is decomposed by TaskDecomposer into at least one task.
  4. Does NOT produce a "I'm not sure" / "I don't understand" fallback response.

These tests use mocked Brain, so no real OS operations are performed.
They cover the full classification → routing → decomposition pipeline.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from spidy.brain.intent_classifier import IntentClassifier
from spidy.brain.goal_intent_classifier import GoalIntentClassifier, GOAL_ACTIONS
from spidy.agent.task_decomposer import TaskDecomposer

# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    return IntentClassifier()

@pytest.fixture
def goal_clf():
    return GoalIntentClassifier()

@pytest.fixture
def decomposer():
    return TaskDecomposer()


# ─── Helper ────────────────────────────────────────────────────────────────────

async def _classify(classifier: IntentClassifier, text: str):
    return await classifier.classify(text)


# ─── 1. "Create a folder on the Desktop" ──────────────────────────────────────

@pytest.mark.asyncio
async def test_create_folder_classification(classifier, goal_clf):
    intent = await classifier.classify("Create a folder on the Desktop")
    assert intent.action == "create_folder", f"Expected create_folder, got '{intent.action}'"
    assert goal_clf.is_executable(intent), "create_folder should be routed to goal"

@pytest.mark.asyncio
async def test_create_folder_decomposition(decomposer):
    tasks = await decomposer.decompose("Create a folder on the Desktop")
    assert len(tasks) >= 1
    assert any("folder" in t.description.lower() or "folder" in t.utterance.lower() for t in tasks)


# ─── 2. "Open VS Code" ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_vscode_classification(classifier, goal_clf):
    for utterance in ["Open VS Code", "open vs code", "launch vscode", "Start VS Code"]:
        intent = await classifier.classify(utterance)
        assert intent.action == "launch_app", f"Expected launch_app for '{utterance}', got '{intent.action}'"
        assert goal_clf.is_executable(intent), f"launch_app should be a goal for '{utterance}'"

@pytest.mark.asyncio
async def test_open_vscode_decomposition(decomposer):
    tasks = await decomposer.decompose("Open VS Code")
    assert len(tasks) >= 1
    assert any("vs code" in t.utterance.lower() or "vscode" in t.utterance.lower() for t in tasks)


# ─── 3. "Open Chrome" ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_chrome_classification(classifier, goal_clf):
    for utterance in ["Open Chrome", "launch Chrome", "start chrome"]:
        intent = await classifier.classify(utterance)
        assert intent.action == "launch_app", f"Expected launch_app for '{utterance}', got '{intent.action}'"
        assert goal_clf.is_executable(intent)

@pytest.mark.asyncio
async def test_open_chrome_decomposition(decomposer):
    tasks = await decomposer.decompose("Open Chrome")
    assert len(tasks) >= 1
    assert any("chrome" in t.utterance.lower() for t in tasks)


# ─── 4. "Open Notepad" ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_notepad_classification(classifier, goal_clf):
    intent = await classifier.classify("Open Notepad")
    assert intent.action == "launch_app", f"Expected launch_app, got '{intent.action}'"
    assert goal_clf.is_executable(intent)

@pytest.mark.asyncio
async def test_open_notepad_decomposition(decomposer):
    tasks = await decomposer.decompose("Open Notepad")
    assert len(tasks) >= 1
    assert any("notepad" in t.utterance.lower() for t in tasks)


# ─── 5. "Search YouTube for Python tutorials" ─────────────────────────────────

@pytest.mark.asyncio
async def test_search_youtube_classification(classifier, goal_clf):
    intent = await classifier.classify("Search YouTube for Python tutorials")
    assert intent.action == "search_youtube", f"Expected search_youtube, got '{intent.action}'"
    assert goal_clf.is_executable(intent)
    # Entity extraction: should have query
    query_entities = [e for e in intent.entities if e.name == "query"]
    assert len(query_entities) > 0, "Expected 'query' entity for YouTube search"
    assert "python" in query_entities[0].value.lower()

@pytest.mark.asyncio
async def test_search_youtube_decomposition(decomposer):
    tasks = await decomposer.decompose("Search YouTube for Python tutorials")
    assert len(tasks) >= 1
    assert any("youtube" in t.utterance.lower() for t in tasks)


# ─── 6. "Find all PDF files in Downloads" ────────────────────────────────────

@pytest.mark.asyncio
async def test_find_pdf_classification(classifier, goal_clf):
    for utterance in [
        "Find all PDF files in Downloads",
        "find all pdf files",
        "find pdfs",
        "search downloads for pdf",
    ]:
        intent = await classifier.classify(utterance)
        assert intent.action == "search_files", f"Expected search_files for '{utterance}', got '{intent.action}'"
        assert goal_clf.is_executable(intent)

@pytest.mark.asyncio
async def test_find_pdf_entity_extraction(classifier):
    intent = await classifier.classify("Find all PDF files in Downloads")
    ext_entities = [e for e in intent.entities if e.name == "extensions"]
    assert len(ext_entities) > 0, "Expected 'extensions' entity for PDF search"
    assert ext_entities[0].value == "pdf"

@pytest.mark.asyncio
async def test_find_pdf_decomposition(decomposer):
    tasks = await decomposer.decompose("Find all PDF files in Downloads")
    assert len(tasks) >= 1
    assert any("pdf" in t.utterance.lower() for t in tasks)


# ─── 7. "Summarize the current webpage" ──────────────────────────────────────

@pytest.mark.asyncio
async def test_summarize_webpage_classification(classifier, goal_clf):
    for utterance in [
        "Summarize the current webpage",
        "summarize this page",
        "summarize current page",
        "what does this page say",
        "read this webpage",
    ]:
        intent = await classifier.classify(utterance)
        assert intent.action == "read_webpage", f"Expected read_webpage for '{utterance}', got '{intent.action}'"
        assert goal_clf.is_executable(intent)

@pytest.mark.asyncio
async def test_summarize_webpage_decomposition(decomposer):
    tasks = await decomposer.decompose("Summarize the current webpage")
    assert len(tasks) >= 1
    assert any(
        "webpage" in t.utterance.lower() or "page" in t.utterance.lower() or "summarize" in t.utterance.lower()
        for t in tasks
    )


# ─── 8. "Create a React project called Portfolio" ────────────────────────────

@pytest.mark.asyncio
async def test_create_react_project_decomposition(decomposer):
    tasks = await decomposer.decompose("Create a React project called Portfolio")
    assert len(tasks) >= 2, "React project should decompose into multiple tasks"
    utterances = " ".join(t.utterance.lower() for t in tasks)
    assert "react" in utterances or "npm" in utterances or "terminal" in utterances

@pytest.mark.asyncio
async def test_create_react_project_classification(classifier, goal_clf):
    intent = await classifier.classify("Create a React project called Portfolio")
    assert intent.action == "create_project", (
        f"Expected create_project, got '{intent.action}'"
    )
    assert goal_clf.is_executable(intent), "create_project should be a goal"


# ─── 9. "Open my Portfolio project in VS Code" ───────────────────────────────

@pytest.mark.asyncio
async def test_open_portfolio_classification(classifier, goal_clf):
    intent = await classifier.classify("Open my Portfolio project in VS Code")
    assert intent.action == "open_project", f"Expected open_project, got '{intent.action}'"
    assert goal_clf.is_executable(intent)

@pytest.mark.asyncio
async def test_open_portfolio_entity_extraction(classifier):
    intent = await classifier.classify("Open my Portfolio project in VS Code")
    project_entities = [e for e in intent.entities if e.name == "project_name"]
    assert len(project_entities) > 0, "Expected 'project_name' entity"
    assert "portfolio" in project_entities[0].value.lower()

@pytest.mark.asyncio
async def test_open_portfolio_decomposition(decomposer):
    tasks = await decomposer.decompose("Open my Portfolio project in VS Code")
    assert len(tasks) >= 1
    assert any("vs code" in t.utterance.lower() or "portfolio" in t.utterance.lower() for t in tasks)


# ─── 10. "Create a Python project" ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_python_project_decomposition(decomposer):
    tasks = await decomposer.decompose("Create a Python project")
    assert len(tasks) >= 1
    utterances = " ".join(t.utterance.lower() for t in tasks)
    assert "python" in utterances or "folder" in utterances

@pytest.mark.asyncio
async def test_create_python_project_not_chat(classifier, goal_clf):
    intent = await classifier.classify("Create a Python project")
    assert intent.action == "create_project", (
        f"Expected create_project, got '{intent.action}'"
    )
    assert goal_clf.is_executable(intent), "create_project should be a goal"


# ─── 11. "Search Google for OpenAI" ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_google_classification(classifier, goal_clf):
    for utterance in [
        "Search Google for OpenAI",
        "google search for openai",
        "search on google for openai",
    ]:
        intent = await classifier.classify(utterance)
        # Should be search_web or search_google (both route to goal)
        assert intent.action in {"search_web", "search_google", "search_youtube", "open_url"} or \
               goal_clf.is_executable(intent), \
               f"Expected web search action for '{utterance}', got '{intent.action}'"

@pytest.mark.asyncio
async def test_search_google_decomposition(decomposer):
    tasks = await decomposer.decompose("Search Google for OpenAI")
    assert len(tasks) >= 1
    utterances = " ".join(t.utterance.lower() for t in tasks)
    assert "google" in utterances or "chrome" in utterances or "search" in utterances


# ─── 12. "Cancel the current goal" ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_goal_classification(classifier, goal_clf):
    for utterance in [
        "Cancel the current goal",
        "cancel",
        "stop that",
        "nevermind",
        "abort",
        "cancel goal",
    ]:
        intent = await classifier.classify(utterance)
        assert intent.action == "cancel_goal", f"Expected cancel_goal for '{utterance}', got '{intent.action}'"
        assert goal_clf.is_executable(intent), f"cancel_goal should be executable for '{utterance}'"


# ─── GoalIntentClassifier unit tests ──────────────────────────────────────────

def test_goal_intent_classifier_goal_actions(goal_clf, classifier):
    """All GOAL_ACTIONS should be routed as 'goal'."""
    from unittest.mock import MagicMock
    from spidy.brain.types import Intent

    for action in GOAL_ACTIONS:
        mock_intent = MagicMock(spec=Intent)
        mock_intent.action = action
        mock_intent.confidence = 0.9
        result = goal_clf.classify(mock_intent)
        assert result == "goal", f"Action '{action}' should be 'goal'"


def test_goal_intent_classifier_chat_actions(goal_clf):
    """Conversational actions should be routed as 'chat'."""
    from unittest.mock import MagicMock
    from spidy.brain.types import Intent
    from spidy.brain.goal_intent_classifier import CHAT_ACTIONS

    for action in CHAT_ACTIONS:
        mock_intent = MagicMock(spec=Intent)
        mock_intent.action = action
        mock_intent.confidence = 0.9
        result = goal_clf.classify(mock_intent)
        assert result == "chat", f"Action '{action}' should be 'chat'"


def test_goal_intent_classifier_low_confidence(goal_clf):
    """Low-confidence intents should always route to 'chat'."""
    from unittest.mock import MagicMock
    from spidy.brain.types import Intent

    mock_intent = MagicMock(spec=Intent)
    mock_intent.action = "launch_app"
    mock_intent.confidence = 0.4  # Below 0.65 threshold
    result = goal_clf.classify(mock_intent)
    assert result == "chat", "Low-confidence intent should be 'chat'"


# ─── Intent classifier correctness ─────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("utterance,expected_action", [
    ("Create a folder on the Desktop", "create_folder"),
    ("Make a folder", "create_folder"),
    ("Make a directory", "create_folder"),
    ("Open VS Code", "launch_app"),
    ("launch vscode", "launch_app"),
    ("Open Chrome", "launch_app"),
    ("Start Chrome", "launch_app"),
    ("Open Notepad", "launch_app"),
    ("Find all PDF files", "search_files"),
    ("find pdfs", "search_files"),
    ("Summarize the current webpage", "read_webpage"),
    ("summarize this page", "read_webpage"),
    ("Search YouTube for Python tutorials", "search_youtube"),
    ("Open my Portfolio project in VS Code", "open_project"),
    ("Cancel the current goal", "cancel_goal"),
    ("cancel", "cancel_goal"),
    ("nevermind", "cancel_goal"),
    ("Search Google for OpenAI", "search_web"),
    ("Create a React project called Portfolio", "create_project"),
    ("Create a Python project", "create_project"),
])
async def test_intent_classification_parametrized(classifier, utterance, expected_action):
    intent = await classifier.classify(utterance)
    assert intent.action == expected_action, (
        f"Utterance '{utterance}': expected '{expected_action}', got '{intent.action}'"
    )


# ─── No fallback to generic chat for validation commands ───────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("utterance", [
    "Create a folder on the Desktop",
    "Open VS Code",
    "Open Chrome",
    "Open Notepad",
    "Search YouTube for Python tutorials",
    "Find all PDF files in Downloads",
    "Summarize the current webpage",
    "Create a React project called Portfolio",
    "Open my Portfolio project in VS Code",
    "Create a Python project",
    "Search Google for OpenAI",
    "Cancel the current goal",
])
async def test_no_chat_fallback_for_validation_commands(classifier, goal_clf, utterance):
    """All 12 validation commands must NOT fall back to 'chat' action."""
    intent = await classifier.classify(utterance)
    # Either the action itself is not "chat", OR the GoalIntentClassifier routes it as goal
    is_not_chat = intent.action != "chat" or goal_clf.is_executable(intent)
    assert is_not_chat, (
        f"Command '{utterance}' fell back to chat! action='{intent.action}', "
        f"confidence={intent.confidence:.2f}. This is a milestone failure."
    )
