"""
Unit tests for YouTube video discovery, playback, and verification.
Tests both BrowserBackend/PlaywrightBackend and BrowserSkill layers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from spidy.browser.backends.base import BrowserBackend
from spidy.skills.browser.browser_skill import BrowserSkill
from spidy.skills.base import SkillContext


@pytest.fixture
def mock_backend():
    backend = AsyncMock(spec=BrowserBackend)
    backend.name = "playwright"
    backend.is_running = True
    backend.get_page_info = AsyncMock(return_value={
        "url": "https://www.youtube.com/results?search_query=python",
        "title": "python - YouTube",
    })
    backend.get_youtube_video_results = AsyncMock(return_value=[
        {
            "index": 0,
            "title": "Python Full Course for Beginners [2026]",
            "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
            "video_id": "kqtD5dpn9C8",
        },
        {
            "index": 1,
            "title": "Python in 100 Seconds",
            "url": "https://www.youtube.com/watch?v=x7X9w_GIm1s",
            "video_id": "x7X9w_GIm1s",
        },
    ])
    backend.play_youtube_video = AsyncMock(return_value={
        "status": "playing",
        "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
        "title": "Python Full Course for Beginners [2026]",
        "video_id": "kqtD5dpn9C8",
        "player_state": 1,
        "is_playing": True,
        "is_paused": False,
        "current_time": 2.5,
        "duration": 3600.0,
        "volume": 100,
        "playback_verified": True,
    })
    return backend


@pytest.fixture
def browser_skill(mock_backend):
    agent = MagicMock()
    agent.backend = mock_backend
    agent.is_running = True
    agent.get_page_info = mock_backend.get_page_info
    agent.get_youtube_video_results = mock_backend.get_youtube_video_results
    agent.play_youtube_video = mock_backend.play_youtube_video

    skill = BrowserSkill(browser_agent=agent)
    return skill


class TestYouTubePlaybackUnit:
    @pytest.mark.asyncio
    async def test_video_results_discovery(self, browser_skill, mock_backend):
        """Verify video cards discovery extracts valid video titles and URLs."""
        videos = await browser_skill._agent.get_youtube_video_results()
        assert len(videos) == 2
        assert "watch?v=" in videos[0]["url"]
        assert videos[0]["video_id"] == "kqtD5dpn9C8"
        assert videos[1]["title"] == "Python in 100 Seconds"

    @pytest.mark.asyncio
    async def test_play_youtube_video_success(self, browser_skill, mock_backend):
        """Verify play_youtube_video succeeds and returns rich structured evidence."""
        ctx = SkillContext(
            action="play_youtube_video",
            params={"random_pick": True},
            session_id="test-session",
        )
        result = await browser_skill.execute("play_youtube_video", ctx)

        assert result.success is True
        assert "Python Full Course for Beginners [2026]" in result.message
        assert result.data["playback_verified"] is True
        assert result.data["is_playing"] is True
        assert result.data["current_time"] > 0
        assert "watch?v=" in result.data["url"]

    @pytest.mark.asyncio
    async def test_play_youtube_video_specific_query(self, browser_skill, mock_backend):
        """Verify play_youtube_video accepts query parameter."""
        ctx = SkillContext(
            action="play_youtube_video",
            params={"query": "python tutorial", "index": 0},
            session_id="test-session",
        )
        result = await browser_skill.execute("play_youtube_video", ctx)

        assert result.success is True
        mock_backend.play_youtube_video.assert_awaited_once_with(
            query="python tutorial",
            video_index=0,
            select_mode="random",
            max_retries=3,
            browser_type=None,
        )

    @pytest.mark.asyncio
    async def test_play_youtube_video_failure(self, browser_skill, mock_backend):
        """Verify play_youtube_video gracefully reports failure when playback cannot be confirmed."""
        mock_backend.play_youtube_video.return_value = {
            "status": "failed",
            "error": "Playback verification failed: video currentTime did not advance.",
            "playback_verified": False,
            "is_playing": False,
        }

        ctx = SkillContext(
            action="play_youtube_video",
            params={},
            session_id="test-session",
        )
        result = await browser_skill.execute("play_youtube_video", ctx)

        assert result.success is False
        assert "Playback verification failed" in result.message
        assert result.data["playback_verified"] is False


class TestPlaywrightBackendYouTubeLogic:
    @pytest.mark.asyncio
    async def test_inspect_youtube_player_helper(self):
        """Test _inspect_youtube_player returns accurate player state from page DOM."""
        from spidy.browser.backends.playwright_backend import PlaywrightBackend

        backend = PlaywrightBackend()
        page = AsyncMock()
        page.url = "https://www.youtube.com/watch?v=kqtD5dpn9C8"
        page.title = AsyncMock(return_value="Python Course - YouTube")
        page.evaluate = AsyncMock(return_value={
            "has_video": True,
            "paused": False,
            "currentTime": 4.2,
            "duration": 1800.0,
            "playerState": 1,
            "is_unavailable": False,
        })

        info = await backend._inspect_youtube_player(page)
        assert info["has_video"] is True
        assert info["paused"] is False
        assert info["currentTime"] == 4.2
        assert info["playerState"] == 1

    @pytest.mark.asyncio
    async def test_trigger_youtube_play_helper(self):
        """Test _trigger_youtube_play attempts play button click or HTML5 play() call."""
        from spidy.browser.backends.playwright_backend import PlaywrightBackend

        backend = PlaywrightBackend()
        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="clicked_play_button")

        triggered = await backend._trigger_youtube_play(page)
        assert triggered == "clicked_play_button"

    @pytest.mark.asyncio
    async def test_play_youtube_video_advancement_check(self):
        """Test PlaywrightBackend.play_youtube_video verifies currentTime advancement."""
        from spidy.browser.backends.playwright_backend import PlaywrightBackend

        backend = PlaywrightBackend()
        backend._is_running = True
        backend._page = AsyncMock()
        backend._page.url = "https://www.youtube.com/results?search_query=python"

        # Mock results list
        backend.get_youtube_video_results = AsyncMock(return_value=[
            {"index": 0, "title": "Test Video", "url": "https://www.youtube.com/watch?v=123", "video_id": "123"}
        ])

        # Step 1: inspect shows video playing and time advancing
        time_samples = [1.0, 2.5, 3.2]
        sample_idx = 0

        async def mock_inspect(page):
            nonlocal sample_idx
            curr = time_samples[min(sample_idx, len(time_samples) - 1)]
            sample_idx += 1
            return {
                "has_video": True,
                "paused": False,
                "currentTime": curr,
                "duration": 500.0,
                "playerState": 1,
                "is_unavailable": False,
            }

        backend._inspect_youtube_player = AsyncMock(side_effect=mock_inspect)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await backend.play_youtube_video(video_index=0)

        assert result["status"] == "SUCCEEDED"
        assert result["playback_verified"] is True
        assert result["current_time"] >= 2.5
