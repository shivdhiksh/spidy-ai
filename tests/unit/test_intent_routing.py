"""
tests/unit/test_intent_routing.py — Intent Classifier v2 Regression Suite
===========================================================================
Full coverage of the new IntentClassifier, verifying:
  - URL auto-detection (bare domains, http/https, www)
  - New intent patterns: shutdown_system, restart_system, calculate,
    take_screenshot, search_bing, search_duckduckgo, search_wikipedia,
    search_maps, search_youtube (shorthand), lock_workstation
  - Entity extraction accuracy
  - Priority ordering (URL > specific engine > generic search > app > chat)
  - Regression: no utterance that clearly maps to an action should fall to chat
"""

from __future__ import annotations

import asyncio
import pytest

from spidy.brain.intent_classifier import IntentClassifier, _detect_url


# ─── URL Detection Unit Tests ──────────────────────────────────────────────────

class TestUrlDetection:
    """Unit-test _detect_url() directly."""

    def test_bare_domain_com(self):
        assert _detect_url("github.com") == "https://github.com"

    def test_bare_domain_io(self):
        assert _detect_url("python.io") == "https://python.io"

    def test_bare_domain_org(self):
        assert _detect_url("wikipedia.org") == "https://wikipedia.org"

    def test_bare_domain_with_path(self):
        result = _detect_url("youtube.com/watch?v=abc")
        assert result is not None
        assert "youtube.com" in result

    def test_http_url(self):
        assert _detect_url("http://example.com") == "http://example.com"

    def test_https_url(self):
        assert _detect_url("https://github.com") == "https://github.com"

    def test_www_url(self):
        assert _detect_url("www.google.com") == "https://www.google.com"

    def test_prefixed_open(self):
        result = _detect_url("open github.com")
        assert result == "https://github.com"

    def test_prefixed_go_to(self):
        result = _detect_url("go to stackoverflow.com")
        assert result == "https://stackoverflow.com"

    def test_prefixed_visit(self):
        result = _detect_url("visit youtube.com")
        assert result == "https://youtube.com"

    def test_prefixed_navigate_to(self):
        result = _detect_url("navigate to google.com")
        assert result == "https://google.com"

    def test_unknown_tld_not_detected(self):
        """'hello.xyz123' should not be detected as a URL (unknown TLD)."""
        result = _detect_url("hello.xyz123")
        assert result is None

    def test_plain_word_not_detected(self):
        assert _detect_url("hello world") is None

    def test_number_not_detected(self):
        assert _detect_url("15 * 32") is None

    def test_empty_not_detected(self):
        assert _detect_url("") is None


# ─── Intent Classifier Tests ───────────────────────────────────────────────────


@pytest.fixture
def classifier():
    return IntentClassifier()


class TestUrlIntents:
    """URL utterances should always produce open_url intent."""

    @pytest.mark.asyncio
    async def test_bare_domain_intent(self, classifier):
        intent = await classifier.classify("github.com")
        assert intent.action == "open_url"
        assert intent.source == "heuristic_url"
        url = intent.get_entity("url")
        assert "github.com" in url

    @pytest.mark.asyncio
    async def test_open_domain_intent(self, classifier):
        intent = await classifier.classify("open github.com")
        assert intent.action == "open_url"
        url = intent.get_entity("url")
        assert "github.com" in url

    @pytest.mark.asyncio
    async def test_go_to_domain_intent(self, classifier):
        intent = await classifier.classify("go to stackoverflow.com")
        assert intent.action == "open_url"

    @pytest.mark.asyncio
    async def test_full_https_url_intent(self, classifier):
        intent = await classifier.classify("open https://docs.python.org/3/")
        assert intent.action == "open_url"
        url = intent.get_entity("url")
        assert "docs.python.org" in url

    @pytest.mark.asyncio
    async def test_url_beats_app_launch(self, classifier):
        """'open github.com' should be open_url, not launch_app."""
        intent = await classifier.classify("open github.com")
        assert intent.action == "open_url", (
            f"Expected open_url, got {intent.action!r} — URL detection must run first"
        )

    @pytest.mark.asyncio
    async def test_url_confidence_is_high(self, classifier):
        intent = await classifier.classify("github.com")
        assert intent.confidence >= 0.9


class TestCalculateIntents:
    """Calculate utterances should produce calculate intent."""

    @pytest.mark.asyncio
    async def test_calculate_keyword(self, classifier):
        intent = await classifier.classify("calculate 15 * 32")
        assert intent.action == "calculate"

    @pytest.mark.asyncio
    async def test_what_is_math(self, classifier):
        intent = await classifier.classify("what is 5 plus 3")
        assert intent.action == "calculate"

    @pytest.mark.asyncio
    async def test_compute_keyword(self, classifier):
        intent = await classifier.classify("compute 100 divided by 4")
        assert intent.action == "calculate"

    @pytest.mark.asyncio
    async def test_expression_entity_extracted(self, classifier):
        intent = await classifier.classify("calculate 15 * 32")
        expr = intent.get_entity("expression")
        assert expr != ""  # something was extracted

    @pytest.mark.asyncio
    async def test_how_much_is(self, classifier):
        intent = await classifier.classify("how much is 200 minus 75")
        assert intent.action == "calculate"

    @pytest.mark.asyncio
    async def test_work_out(self, classifier):
        intent = await classifier.classify("work out 7 times 8")
        assert intent.action == "calculate"


class TestScreenshotIntents:
    """Screenshot utterances should produce take_screenshot intent."""

    @pytest.mark.asyncio
    async def test_take_screenshot(self, classifier):
        intent = await classifier.classify("take a screenshot")
        assert intent.action == "take_screenshot"

    @pytest.mark.asyncio
    async def test_screenshot_bare(self, classifier):
        intent = await classifier.classify("screenshot")
        assert intent.action == "take_screenshot"

    @pytest.mark.asyncio
    async def test_capture_screen(self, classifier):
        intent = await classifier.classify("capture the screen")
        assert intent.action == "take_screenshot"

    @pytest.mark.asyncio
    async def test_screen_capture(self, classifier):
        intent = await classifier.classify("screen capture")
        assert intent.action == "take_screenshot"

    @pytest.mark.asyncio
    async def test_grab_screenshot(self, classifier):
        intent = await classifier.classify("grab a screenshot")
        assert intent.action == "take_screenshot"


class TestSystemShutdownIntents:
    """Shutdown/restart utterances must route to real SystemControlSkill actions."""

    @pytest.mark.asyncio
    async def test_shutdown_computer(self, classifier):
        intent = await classifier.classify("shutdown computer")
        assert intent.action == "shutdown_system"

    @pytest.mark.asyncio
    async def test_shut_down(self, classifier):
        intent = await classifier.classify("shut down")
        assert intent.action == "shutdown_system"

    @pytest.mark.asyncio
    async def test_power_off(self, classifier):
        intent = await classifier.classify("power off my computer")
        assert intent.action == "shutdown_system"

    @pytest.mark.asyncio
    async def test_turn_off_computer(self, classifier):
        intent = await classifier.classify("turn off computer")
        assert intent.action == "shutdown_system"

    @pytest.mark.asyncio
    async def test_restart_laptop(self, classifier):
        intent = await classifier.classify("restart laptop")
        assert intent.action == "restart_system"

    @pytest.mark.asyncio
    async def test_reboot(self, classifier):
        intent = await classifier.classify("reboot")
        assert intent.action == "restart_system"

    @pytest.mark.asyncio
    async def test_restart_the_computer(self, classifier):
        intent = await classifier.classify("restart the computer")
        assert intent.action == "restart_system"

    @pytest.mark.asyncio
    async def test_shutdown_confidence_is_high(self, classifier):
        intent = await classifier.classify("shutdown computer")
        assert intent.confidence >= 0.9


class TestLockScreenIntent:
    """lock_screen utterances must map to lock_workstation (the real action)."""

    @pytest.mark.asyncio
    async def test_lock_screen(self, classifier):
        intent = await classifier.classify("lock screen")
        assert intent.action == "lock_workstation"

    @pytest.mark.asyncio
    async def test_lock_my_computer(self, classifier):
        intent = await classifier.classify("lock my computer")
        assert intent.action == "lock_workstation"

    @pytest.mark.asyncio
    async def test_lock_workstation(self, classifier):
        intent = await classifier.classify("lock workstation")
        assert intent.action == "lock_workstation"

    @pytest.mark.asyncio
    async def test_lock_my_laptop(self, classifier):
        intent = await classifier.classify("lock my laptop")
        assert intent.action == "lock_workstation"


class TestYouTubeIntents:
    """YouTube shorthand like 'youtube cats' should produce search_youtube."""

    @pytest.mark.asyncio
    async def test_youtube_bare_query(self, classifier):
        intent = await classifier.classify("youtube cats")
        assert intent.action == "search_youtube"
        query = intent.get_entity("query")
        assert "cats" in query

    @pytest.mark.asyncio
    async def test_search_youtube_for(self, classifier):
        intent = await classifier.classify("search youtube for python tutorial")
        assert intent.action == "search_youtube"
        query = intent.get_entity("query")
        assert "python tutorial" in query

    @pytest.mark.asyncio
    async def test_youtube_music(self, classifier):
        intent = await classifier.classify("youtube lofi music")
        assert intent.action == "search_youtube"

    @pytest.mark.asyncio
    async def test_youtube_beats_generic_search(self, classifier):
        """'youtube python' should be search_youtube, not search_web."""
        intent = await classifier.classify("youtube python")
        assert intent.action == "search_youtube", (
            f"Expected search_youtube, got {intent.action!r}"
        )


class TestAlternateSearchEngineIntents:
    """Bing, DuckDuckGo, Wikipedia, Maps intents."""

    @pytest.mark.asyncio
    async def test_search_bing(self, classifier):
        intent = await classifier.classify("search bing for weather")
        assert intent.action == "search_bing"
        query = intent.get_entity("query")
        assert "weather" in query

    @pytest.mark.asyncio
    async def test_bing_bare(self, classifier):
        intent = await classifier.classify("bing best laptops 2024")
        assert intent.action == "search_bing"

    @pytest.mark.asyncio
    async def test_duckduckgo(self, classifier):
        intent = await classifier.classify("duckduckgo privacy tips")
        assert intent.action == "search_duckduckgo"

    @pytest.mark.asyncio
    async def test_search_duckduckgo_for(self, classifier):
        intent = await classifier.classify("search duckduckgo for vpn comparison")
        assert intent.action == "search_duckduckgo"

    @pytest.mark.asyncio
    async def test_wikipedia(self, classifier):
        intent = await classifier.classify("wikipedia black holes")
        assert intent.action == "search_wikipedia"

    @pytest.mark.asyncio
    async def test_search_wikipedia_for(self, classifier):
        intent = await classifier.classify("search wikipedia for quantum computing")
        assert intent.action == "search_wikipedia"
        query = intent.get_entity("query")
        assert "quantum computing" in query

    @pytest.mark.asyncio
    async def test_wiki(self, classifier):
        intent = await classifier.classify("wiki python programming")
        assert intent.action == "search_wikipedia"

    @pytest.mark.asyncio
    async def test_maps_directions(self, classifier):
        intent = await classifier.classify("directions to Times Square")
        assert intent.action == "search_maps"
        query = intent.get_entity("query")
        assert "times square" in query.lower()

    @pytest.mark.asyncio
    async def test_maps_show(self, classifier):
        intent = await classifier.classify("show map of Paris")
        assert intent.action == "search_maps"

    @pytest.mark.asyncio
    async def test_google_maps(self, classifier):
        intent = await classifier.classify("google maps nearest coffee shop")
        assert intent.action == "search_maps"


class TestWebSearchIntents:
    """Generic web search should produce search_web (aliases to search_google)."""

    @pytest.mark.asyncio
    async def test_google_query(self, classifier):
        intent = await classifier.classify("google python tutorials")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_search_for(self, classifier):
        intent = await classifier.classify("search for best IDE for Python")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_query_extracted(self, classifier):
        intent = await classifier.classify("google Python tutorials")
        query = intent.get_entity("query")
        assert "python tutorials" in query.lower()

    @pytest.mark.asyncio
    async def test_look_up(self, classifier):
        intent = await classifier.classify("look up machine learning basics")
        assert intent.action == "search_web"

    @pytest.mark.asyncio
    async def test_what_is(self, classifier):
        intent = await classifier.classify("what is recursion in programming")
        assert intent.action == "search_web"


class TestGreetingIntents:
    """Greetings should always be classified correctly, not as 'search_web'."""

    @pytest.mark.asyncio
    async def test_hello(self, classifier):
        intent = await classifier.classify("hello")
        assert intent.action == "greet"

    @pytest.mark.asyncio
    async def test_hi(self, classifier):
        intent = await classifier.classify("hi")
        assert intent.action == "greet"

    @pytest.mark.asyncio
    async def test_good_morning(self, classifier):
        intent = await classifier.classify("good morning")
        assert intent.action == "greet"

    @pytest.mark.asyncio
    async def test_goodbye(self, classifier):
        intent = await classifier.classify("goodbye")
        assert intent.action == "farewell"

    @pytest.mark.asyncio
    async def test_who_are_you(self, classifier):
        intent = await classifier.classify("who are you")
        assert intent.action == "introduce"


class TestAppIntents:
    """App launch/close should still work correctly after classifier rewrite."""

    @pytest.mark.asyncio
    async def test_open_notepad(self, classifier):
        intent = await classifier.classify("open notepad")
        assert intent.action == "launch_app"
        name = intent.get_entity("name")
        assert "notepad" in name.lower()

    @pytest.mark.asyncio
    async def test_launch_vscode(self, classifier):
        intent = await classifier.classify("launch vscode")
        assert intent.action == "launch_app"

    @pytest.mark.asyncio
    async def test_open_chrome(self, classifier):
        intent = await classifier.classify("open chrome")
        # chrome without a domain suffix should be launch_app
        assert intent.action == "launch_app"

    @pytest.mark.asyncio
    async def test_close_chrome(self, classifier):
        intent = await classifier.classify("close chrome")
        assert intent.action == "close_app"

    @pytest.mark.asyncio
    async def test_open_spotify(self, classifier):
        intent = await classifier.classify("open spotify")
        assert intent.action == "launch_app"


class TestSystemInfoIntents:
    @pytest.mark.asyncio
    async def test_system_info(self, classifier):
        intent = await classifier.classify("system info")
        assert intent.action == "get_system_info"

    @pytest.mark.asyncio
    async def test_cpu_usage(self, classifier):
        intent = await classifier.classify("cpu usage")
        assert intent.action == "get_system_info"

    @pytest.mark.asyncio
    async def test_battery_level(self, classifier):
        intent = await classifier.classify("battery level")
        assert intent.action == "get_system_info"


class TestTimerIntents:
    @pytest.mark.asyncio
    async def test_set_timer(self, classifier):
        intent = await classifier.classify("set a timer for 5 minutes")
        assert intent.action == "set_timer"

    @pytest.mark.asyncio
    async def test_timer_entities(self, classifier):
        intent = await classifier.classify("set a timer for 10 minutes")
        amount = intent.get_entity("amount")
        unit = intent.get_entity("unit")
        assert amount == "10"
        assert "minute" in unit

    @pytest.mark.asyncio
    async def test_remind_me_in(self, classifier):
        intent = await classifier.classify("remind me in 30 seconds")
        assert intent.action == "set_timer"


class TestVolumeIntents:
    @pytest.mark.asyncio
    async def test_volume_up(self, classifier):
        intent = await classifier.classify("volume up")
        assert intent.action == "set_volume"

    @pytest.mark.asyncio
    async def test_mute(self, classifier):
        intent = await classifier.classify("mute")
        assert intent.action == "set_volume"

    @pytest.mark.asyncio
    async def test_set_volume_to_50(self, classifier):
        intent = await classifier.classify("set volume to 50")
        assert intent.action == "set_volume"
        level = intent.get_entity("level")
        assert level == "50"


class TestNeverReturnChatForClearRequests:
    """
    Regression: clear/specific utterances must NEVER fall through to 'chat'.
    This is the core production quality requirement.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("utterance,expected_action", [
        # Browser & URL
        ("open github.com", "open_url"),
        ("go to youtube.com", "open_url"),
        ("visit stackoverflow.com", "open_url"),
        # System control
        ("shutdown computer", "shutdown_system"),
        ("restart laptop", "restart_system"),
        ("lock screen", "lock_workstation"),
        ("lock my computer", "lock_workstation"),
        ("mute", "set_volume"),
        ("volume up", "set_volume"),
        # Search
        ("youtube cats", "search_youtube"),
        ("google python", "search_web"),
        ("search bing for weather", "search_bing"),
        ("duckduckgo privacy", "search_duckduckgo"),
        ("wikipedia black holes", "search_wikipedia"),
        ("directions to Paris", "search_maps"),
        # Compute
        ("calculate 15 * 32", "calculate"),
        ("what is 100 divided by 4", "calculate"),
        # Screenshot
        ("take a screenshot", "take_screenshot"),
        ("screenshot", "take_screenshot"),
        # Apps
        ("open notepad", "launch_app"),
        ("launch vscode", "launch_app"),
        ("close chrome", "close_app"),
        # Social
        ("hello", "greet"),
        ("goodbye", "farewell"),
        ("who are you", "introduce"),
        # System info
        ("system info", "get_system_info"),
        ("battery level", "get_system_info"),
        # Timers
        ("set a timer for 5 minutes", "set_timer"),
        # Notes
        ("take a note: buy milk", "take_note"),
        # Help
        ("help", "help"),
    ])
    async def test_no_chat_fallback_for_clear_request(
        self, classifier, utterance, expected_action
    ):
        intent = await classifier.classify(utterance)
        assert intent.action == expected_action, (
            f"'{utterance}' → expected '{expected_action}', got '{intent.action}'"
        )
        assert intent.action != "chat", (
            f"'{utterance}' fell through to 'chat' — this is a production routing failure"
        )


class TestFallbackToChat:
    """Truly ambiguous inputs should gracefully return chat."""

    @pytest.mark.asyncio
    async def test_empty_utterance(self, classifier):
        intent = await classifier.classify("")
        assert intent.action == "chat"

    @pytest.mark.asyncio
    async def test_whitespace_only(self, classifier):
        intent = await classifier.classify("   ")
        assert intent.action == "chat"

    @pytest.mark.asyncio
    async def test_gibberish(self, classifier):
        intent = await classifier.classify("xyzzy florp blargh")
        assert intent.action == "chat"

    @pytest.mark.asyncio
    async def test_open_ended_question(self, classifier):
        # Open-ended questions should go to LLM (chat), not fail
        intent = await classifier.classify("tell me a joke")
        # It's fine if this is chat — the LLM will handle it
        assert intent.action in ("chat", "search_web")  # either is acceptable


class TestPriorityOrdering:
    """Verify that higher-priority rules win over lower-priority ones."""

    @pytest.mark.asyncio
    async def test_url_beats_everything(self, classifier):
        """A bare domain should always be open_url regardless of other keywords."""
        intent = await classifier.classify("youtube.com")
        # youtube.com is a URL, not a YouTube search
        assert intent.action == "open_url"

    @pytest.mark.asyncio
    async def test_youtube_beats_generic_search(self, classifier):
        intent = await classifier.classify("youtube python")
        assert intent.action == "search_youtube"
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_wikipedia_beats_generic_search(self, classifier):
        intent = await classifier.classify("wikipedia history of the internet")
        assert intent.action == "search_wikipedia"
        assert intent.action != "search_web"

    @pytest.mark.asyncio
    async def test_screenshot_beats_generic(self, classifier):
        intent = await classifier.classify("take a screenshot")
        assert intent.action == "take_screenshot"

    @pytest.mark.asyncio
    async def test_shutdown_not_chat(self, classifier):
        intent = await classifier.classify("shutdown")
        assert intent.action == "shutdown_system"
        assert intent.action != "chat"
