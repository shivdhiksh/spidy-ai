"""
Unit tests for Browser Targeting, Process Identity, and Session Reuse
======================================================================
Verifies:
1. Target Normalization (Edge, Chrome, Firefox, Chromium).
2. Playwright Channel selection ('msedge', 'chrome') & explicit failure without silent fallback.
3. Process Identity & User-Agent verification.
4. BrowserAgent and BrowserSkill session reuse across multi-step goals.
5. StructuredRouter browser routing & desktop browser app launch handling.
6. TaskDecomposer browser target propagation.
7. TaskObserver and GoalVerifier observation & verification checks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.agent.observer import Observation, TaskObserver
from spidy.agent.structured_router import StructuredRouter
from spidy.agent.task_decomposer import TaskDecomposer
from spidy.agent.types import GoalRecord, GoalState, TaskRecord, TaskState
from spidy.agent.verifier import GoalVerifier
from spidy.browser.agent import BrowserAgent
from spidy.browser.backends.playwright_backend import PlaywrightBackend
from spidy.browser.types import (
    BROWSER_PLAYWRIGHT_CHANNELS,
    BROWSER_PROCESS_MAP,
    PageInfo,
    normalize_browser_target,
)
from spidy.skills.base import SkillContext
from spidy.skills.browser.browser_skill import BrowserSkill


# ── 1. Browser Target Normalization ───────────────────────────────────────────


class TestBrowserTargetNormalization:
    def test_edge_aliases(self):
        assert normalize_browser_target("edge") == "edge"
        assert normalize_browser_target("msedge") == "edge"
        assert normalize_browser_target("Microsoft Edge") == "edge"
        assert normalize_browser_target("Edge browser") == "edge"
        assert normalize_browser_target("ms edge") == "edge"

    def test_chrome_aliases(self):
        assert normalize_browser_target("chrome") == "chrome"
        assert normalize_browser_target("Google Chrome") == "chrome"
        assert normalize_browser_target("chrome browser") == "chrome"

    def test_firefox_aliases(self):
        assert normalize_browser_target("firefox") == "firefox"
        assert normalize_browser_target("Mozilla Firefox") == "firefox"

    def test_chromium_and_fallback(self):
        assert normalize_browser_target("chromium") == "chromium"
        assert normalize_browser_target(None) == "chromium"
        assert normalize_browser_target("unknown_browser") == "chromium"
        assert normalize_browser_target("unknown_browser", default="edge") == "edge"

    def test_channel_and_process_maps(self):
        assert BROWSER_PLAYWRIGHT_CHANNELS["edge"] == "msedge"
        assert BROWSER_PLAYWRIGHT_CHANNELS["chrome"] == "chrome"
        assert BROWSER_PROCESS_MAP["edge"] == "msedge.exe"
        assert BROWSER_PROCESS_MAP["chrome"] == "chrome.exe"


# ── 2. Playwright Channel Selection & Explicit Failure ────────────────────────


class TestPlaywrightBackendTargeting:
    @pytest.mark.asyncio
    async def test_start_launches_edge_channel(self):
        backend = PlaywrightBackend()
        mock_pw = AsyncMock()
        mock_chromium = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_pw.chromium = mock_chromium
        mock_chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        with patch("spidy.browser.backends.playwright_backend.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_pw)
            await backend.start(browser_type="Microsoft Edge", connect_to_existing=False)

            assert backend.active_browser_type == "edge"
            assert backend.active_channel == "msedge"
            mock_chromium.launch.assert_called_once_with(channel="msedge", headless=False)

    @pytest.mark.asyncio
    async def test_start_launches_chrome_channel(self):
        backend = PlaywrightBackend()
        mock_pw = AsyncMock()
        mock_chromium = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_pw.chromium = mock_chromium
        mock_chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        with patch("spidy.browser.backends.playwright_backend.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_pw)
            await backend.start(browser_type="Google Chrome", connect_to_existing=False)

            assert backend.active_browser_type == "chrome"
            assert backend.active_channel == "chrome"
            mock_chromium.launch.assert_called_once_with(channel="chrome", headless=False)

    @pytest.mark.asyncio
    async def test_edge_launch_failure_raises_explicit_error_without_fallback(self):
        backend = PlaywrightBackend()
        mock_pw = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch.side_effect = Exception("msedge channel not found")
        mock_pw.chromium = mock_chromium

        with patch("spidy.browser.backends.playwright_backend.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_pw)
            with pytest.raises(RuntimeError, match="Microsoft Edge could not be launched or controlled"):
                await backend.start(browser_type="edge", connect_to_existing=False)


# ── 3. Process Identity Verification ──────────────────────────────────────────


class TestProcessIdentityVerification:
    @pytest.mark.asyncio
    async def test_verify_process_identity_edge_success(self):
        backend = PlaywrightBackend()
        backend._running = True
        backend._active_browser_type = "edge"
        backend._active_channel = "msedge"

        mock_page = AsyncMock()
        mock_page.evaluate.return_value = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edg/131.0.0.0"
        backend._pages = [mock_page]
        backend._active_page_idx = 0

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="msedge.exe 12344 Console 1 120,000 K")
            ok, msg = await backend.verify_process_identity()
            assert ok is True
            assert "Verified genuine edge session" in msg

    @pytest.mark.asyncio
    async def test_verify_process_identity_edge_mismatch_fails(self):
        backend = PlaywrightBackend()
        backend._running = True
        backend._active_browser_type = "edge"
        backend._active_channel = "msedge"

        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36")
        backend._pages = [mock_page]
        backend._active_page_idx = 0

        ok, msg = await backend.verify_process_identity()
        assert ok is False
        assert "Browser identity mismatch" in msg


# ── 4. Session Reuse across Multi-step Goals ──────────────────────────────────


class TestBrowserSessionReuse:
    @pytest.mark.asyncio
    async def test_agent_reuses_running_session_for_same_browser(self):
        mock_backend = AsyncMock()
        mock_backend.is_running = True
        mock_backend.active_browser_type = "edge"
        mock_backend.open_url.return_value = PageInfo(title="YouTube", url="https://www.youtube.com")

        agent = BrowserAgent(backend=mock_backend, browser_type="edge")

        # First operation: open URL
        p1 = await agent.open_url("https://www.youtube.com", browser_type="edge")
        assert p1.url == "https://www.youtube.com"

        # Second operation: search YouTube
        mock_backend.open_url.return_value = PageInfo(title="Python tutorials - YouTube", url="https://www.youtube.com/results?search_query=Python+tutorials")
        p2 = await agent.search_youtube("Python tutorials", browser_type="edge")
        assert "Python" in p2.url

        # Backend start was NOT called a second time
        mock_backend.start.assert_not_called()
        assert mock_backend.open_url.call_count == 2

    @pytest.mark.asyncio
    async def test_skill_open_browser_and_navigate_reuses_agent(self):
        skill = BrowserSkill(browser_type="edge")
        mock_agent = AsyncMock()
        mock_agent.set_browser_type = MagicMock()
        mock_agent.is_running = False
        mock_agent.active_browser_type = "edge"
        mock_agent.active_channel = "msedge"
        mock_agent.verify_process_identity.return_value = (True, "OK")
        mock_agent.open_url.return_value = PageInfo(title="YouTube", url="https://www.youtube.com")

        skill._agent = mock_agent

        # Step 1: Open Edge
        res1 = await skill.execute("open_browser", SkillContext(action="open_browser", params={"browser_type": "edge"}))
        assert res1.success
        mock_agent.start.assert_called_once()

        mock_agent.is_running = True

        # Step 2: Navigate YouTube
        res2 = await skill.execute("open_url", SkillContext(action="open_url", params={"url": "https://www.youtube.com", "browser_type": "edge"}))
        assert res2.success
        assert "YouTube" in res2.message
        # start was not called again
        assert mock_agent.start.call_count == 1


# ── 5. Structured Router & Decomposer Browser Propagation ─────────────────────


class TestStructuredRouterAndDecomposer:
    @pytest.mark.asyncio
    async def test_decomposer_propagates_browser_target(self):
        decomposer = TaskDecomposer()
        goal = "Open Edge. Open YouTube. And search for Python tutorials."
        tasks = await decomposer.decompose(goal, goal_id="g1")

        assert len(tasks) == 3
        # Task 1: Open Edge
        assert tasks[0].action["skill"] == "desktop"
        assert tasks[0].action["action"] == "open_app"
        assert tasks[0].action["browser"] == "edge"

        # Task 2: Navigate YouTube
        assert tasks[1].action["action"] == "navigate"
        assert tasks[1].action["url"] == "https://www.youtube.com"
        assert tasks[1].action["browser"] == "edge"

        # Task 3: Search YouTube
        assert tasks[2].action["action"] == "search"
        assert tasks[2].action["query"] == "Python tutorials"
        assert tasks[2].action["browser"] == "edge"

    @pytest.mark.asyncio
    async def test_structured_router_routes_open_edge_to_browser_skill(self):
        mock_brain = MagicMock()
        mock_registry = MagicMock()
        mock_brain._registry = mock_registry

        mock_browser_skill = AsyncMock()
        mock_browser_skill.name = "BrowserSkill"
        from spidy.skills.base import SkillResult
        mock_browser_skill.execute.return_value = SkillResult.ok("Browser launched (edge).", data={"browser_type": "edge"})

        mock_registry.find_skill_for_action.side_effect = lambda a: mock_browser_skill if a in ("open_browser", "open_url", "search_youtube") else None

        router = StructuredRouter(mock_brain)

        # Task 1: desktop/open_app with target="Edge"
        task1 = TaskRecord(
            task_id="t1", goal_id="g1", description="Open Edge", utterance="open edge",
            action={"skill": "desktop", "action": "open_app", "target": "Edge"},
        )
        msg1, ok1 = await router.execute(task1.action, task1, session_id="s1")
        assert ok1 is True
        mock_browser_skill.execute.assert_called_once()
        call_action, call_ctx = mock_browser_skill.execute.call_args[0]
        assert call_action == "open_browser"
        assert call_ctx.get("browser_type") == "edge"


# ── 6. Observation & Verification ─────────────────────────────────────────────


class TestObserverAndVerifierChecks:
    @pytest.mark.asyncio
    async def test_observer_extracts_url_and_checks_edge_process(self):
        observer = TaskObserver()
        task = TaskRecord(
            task_id="t1", goal_id="g1", description="Open Edge", utterance="open edge",
            action={"skill": "browser", "action": "open_browser", "target": "Edge", "browser": "edge"},
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="msedge.exe 12344")
            obs = await observer.observe(task, "Opened \"YouTube\" in tab. URL: https://www.youtube.com", skip_for_terminal=False)

            assert obs.app_running is True
            assert obs.browser_url == "https://www.youtube.com"
            assert obs.success_signal is True

    @pytest.mark.asyncio
    async def test_goal_verifier_passes_on_completed_multi_step_edge_goal(self):
        verifier = GoalVerifier()
        t1 = TaskRecord(task_id="t1", goal_id="g1", description="Open Edge", utterance="open edge", state=TaskState.COMPLETED)
        t2 = TaskRecord(task_id="t2", goal_id="g1", description="Navigate to YouTube", utterance="navigate to https://www.youtube.com", state=TaskState.COMPLETED)
        t3 = TaskRecord(task_id="t3", goal_id="g1", description="Search for Python tutorials", utterance="search for Python tutorials", state=TaskState.COMPLETED)

        goal = GoalRecord(goal_id="g1", description="Open Edge. Open YouTube. And search for Python tutorials.", tasks=[t1, t2, t3], state=GoalState.COMPLETED)
        obs = [
            Observation(method="app_state", success_signal=True, app_running=True, summary="App 'edge' is running"),
            Observation(method="browser_url", success_signal=True, browser_url="https://www.youtube.com", summary="URL: https://www.youtube.com"),
            Observation(method="browser_url", success_signal=True, browser_url="https://www.youtube.com/results?search_query=Python+tutorials", summary="Search complete"),
        ]

        result = await verifier.verify(goal, observations=obs)
        assert result.verified is True
        assert result.confidence == "high"

    @pytest.mark.asyncio
    async def test_goal_verifier_fails_on_negative_observation_signal(self):
        verifier = GoalVerifier()
        t1 = TaskRecord(task_id="t1", goal_id="g1", description="Open Edge", utterance="open edge", state=TaskState.COMPLETED)
        goal = GoalRecord(goal_id="g1", description="Open Edge", tasks=[t1], state=GoalState.COMPLETED)

        # Negative observation signal (e.g. process not running or mismatch)
        obs = [
            Observation(method="app_state", success_signal=False, app_running=False, summary="App 'edge' is NOT running"),
        ]

        result = await verifier.verify(goal, observations=obs)
        assert result.verified is False
        assert "Verification failed" in result.summary
