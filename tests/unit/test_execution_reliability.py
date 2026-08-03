"""
tests/unit/test_execution_reliability.py
=========================================
Milestone 13.2 — Execution Reliability Regression Tests

Covers every fix made in M13.2:

1. OS alias mapping (Finder, Trash, Terminal, Applications)
2. Intent classification fixes:
   - "open finder" → launch_app
   - "go to the desktop" → show_desktop (NOT bring_app_to_foreground)
   - "right click on the desktop" → desktop_context_menu
   - "create a new folder" → create_folder with default "New Folder" name
3. Planner decisions:
   - show_desktop routes correctly
   - open_terminal → launch_app
4. FileSkill folder CRUD with post-execution verification
5. False-success prevention (create_folder, delete_folder)
6. Error message sanitization
7. show_desktop skill action registration
8. open_folder path resolution for spoken names
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest


# ─── Context helper ───────────────────────────────────────────────────────────


def make_ctx(action: str, **params):
    from spidy.skills.base import SkillContext
    return SkillContext(action=action, params=params, session_id="test", user_name="Test")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — OS Alias Mapping in AppSkill
# ═══════════════════════════════════════════════════════════════════════════════


class TestOSAliasMapping:
    """Verify that macOS / cross-platform aliases resolve to Windows equivalents."""

    def test_finder_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["finder"] == "explorer.exe"

    def test_mac_finder_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["mac finder"] == "explorer.exe"

    def test_applications_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["applications"] == "explorer.exe"

    def test_file_manager_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["file manager"] == "explorer.exe"

    def test_file_browser_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["file browser"] == "explorer.exe"

    def test_files_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["files"] == "explorer.exe"

    def test_trash_maps_to_explorer(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["trash"] == "explorer.exe"

    def test_iterm_maps_to_wt(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["iterm"] == "wt.exe"
        assert _APP_ALIASES["iterm2"] == "wt.exe"

    def test_bash_maps_to_wt(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["bash"] == "wt.exe"

    def test_console_maps_to_wt(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["console"] == "wt.exe"

    def test_existing_aliases_unchanged(self):
        from spidy.skills.desktop.app_skill import _APP_ALIASES
        assert _APP_ALIASES["chrome"] == "chrome.exe"
        assert _APP_ALIASES["terminal"] == "wt.exe"
        assert _APP_ALIASES["vs code"] == "code.exe"
        assert _APP_ALIASES["notepad"] == "notepad.exe"


class TestTerminalFallback:
    """_resolve_exe_path falls back to cmd.exe when wt.exe is absent."""

    def test_wt_fallback_to_cmd_when_not_found(self):
        from spidy.skills.desktop.app_skill import AppSkill
        import shutil
        with patch.object(shutil, "which", side_effect=lambda x: "cmd.exe" if x == "cmd.exe" else None):
            result = AppSkill._resolve_exe_path("wt.exe")
        assert result == "cmd.exe"

    def test_wt_not_fallen_back_when_found(self):
        from spidy.skills.desktop.app_skill import AppSkill
        import shutil
        with patch.object(shutil, "which", return_value="C:\\Windows\\wt.exe"):
            result = AppSkill._resolve_exe_path("wt.exe")
        assert result == "wt.exe"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Intent Classifier Fixes
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntentClassifierFixes:

    @pytest.mark.asyncio
    async def test_open_finder_classifies_as_launch_app(self):
        """Bug 1 fix: 'Open Finder' must not fall through to 'chat'."""
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("Open Finder")
        assert intent.action == "launch_app", (
            f"Expected launch_app, got {intent.action!r}. Finder should map to File Explorer."
        )

    @pytest.mark.asyncio
    async def test_go_to_desktop_classifies_as_show_desktop(self):
        """Bug 2 fix: 'Go to the Desktop' must route to show_desktop."""
        from spidy.brain.intent_classifier import IntentClassifier
        classifier = IntentClassifier()
        for utterance in [
            "Go to the Desktop",
            "go to desktop",
            "show the desktop",
            "show desktop",
            "minimize all windows",
        ]:
            intent = await classifier.classify(utterance)
            assert intent.action == "show_desktop", (
                f"{utterance!r} → {intent.action!r}, expected show_desktop."
            )

    @pytest.mark.asyncio
    async def test_go_to_desktop_not_bring_to_foreground(self):
        """Regression: 'go to desktop' must NOT trigger bring_app_to_foreground."""
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("go to desktop")
        assert intent.action != "bring_app_to_foreground"

    @pytest.mark.asyncio
    async def test_right_click_desktop_classifies_as_context_menu(self):
        """Bug 3 fix: 'Right click on the Desktop' must not fall to chat."""
        from spidy.brain.intent_classifier import IntentClassifier
        classifier = IntentClassifier()
        for utterance in [
            "Right click on the Desktop",
            "right click the desktop",
            "right-click on the desktop",
        ]:
            intent = await classifier.classify(utterance)
            assert intent.action == "desktop_context_menu", (
                f"{utterance!r} → {intent.action!r}, expected desktop_context_menu."
            )
            assert intent.action != "chat"

    @pytest.mark.asyncio
    async def test_create_new_folder_provides_default_name(self):
        """Bug 4 fix: 'create a new folder' must provide a folder_name entity."""
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("create a new folder")
        assert intent.action == "create_folder"
        entities = {e.name: e.value for e in intent.entities}
        assert "folder_name" in entities, f"No folder_name entity! Got: {entities}"
        assert entities["folder_name"]

    @pytest.mark.asyncio
    async def test_create_folder_named_test_extracts_name(self):
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("Create folder Test")
        assert intent.action == "create_folder"
        entities = {e.name: e.value for e in intent.entities}
        assert entities.get("folder_name", "").lower() == "test"

    @pytest.mark.asyncio
    async def test_open_downloads_classifies_as_open_folder(self):
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("open downloads")
        assert intent.action == "open_folder"
        entities = {e.name: e.value for e in intent.entities}
        assert entities.get("path", "").lower() == "downloads"

    @pytest.mark.asyncio
    async def test_open_documents_classifies_as_open_folder(self):
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("open documents")
        assert intent.action == "open_folder"
        entities = {e.name: e.value for e in intent.entities}
        assert entities.get("path", "").lower() == "documents"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Planner Decision Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlannerM132:

    def _make_intent(self, action: str, **entities):
        from spidy.brain.types import Entity, Intent
        ents = tuple(Entity(name=k, value=v) for k, v in entities.items())
        return Intent(action=action, entities=ents, confidence=0.9,
                      raw_utterance=f"test {action}", source="heuristic")

    def _make_decision(self, action: str, **entities):
        from spidy.brain.types import Decision, DecisionMode
        return Decision(mode=DecisionMode.SKILL, intent=self._make_intent(action, **entities),
                        skill_name=action, rationale="test")

    @pytest.mark.asyncio
    async def test_show_desktop_routed_correctly(self):
        from spidy.brain.planner import Planner
        plan = await Planner().plan(self._make_decision("show_desktop"), session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "show_desktop"

    @pytest.mark.asyncio
    async def test_desktop_context_menu_routed_correctly(self):
        from spidy.brain.planner import Planner
        plan = await Planner().plan(self._make_decision("desktop_context_menu"), session_id="test")
        assert not plan.is_empty
        assert plan.first.action == "desktop_context_menu"

    def test_open_terminal_aliased_to_launch_app(self):
        from spidy.brain.planner import _ACTION_ALIASES
        assert _ACTION_ALIASES.get("open_terminal") == "launch_app"

    def test_show_desktop_in_aliases(self):
        from spidy.brain.planner import _ACTION_ALIASES
        assert _ACTION_ALIASES.get("show_desktop") == "show_desktop"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — FileSkill Folder Operations with Verification
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def file_skill():
    from spidy.skills.desktop.file_skill import FileSkill
    return FileSkill(bus=None)


class TestCreateFolderWithVerification:

    @pytest.mark.asyncio
    async def test_creates_folder_successfully(self, tmp_path, file_skill):
        ctx = make_ctx("create_folder", folder_name="TestFolder", path=str(tmp_path))
        result = await file_skill.execute("create_folder", ctx)
        assert result.success, f"Expected success: {result.message}"
        assert (tmp_path / "TestFolder").exists()
        assert result.data.get("verified") is True

    @pytest.mark.asyncio
    async def test_no_name_defaults_to_new_folder(self, tmp_path, file_skill):
        ctx = make_ctx("create_folder", path=str(tmp_path))
        result = await file_skill.execute("create_folder", ctx)
        assert result.success, f"Expected success: {result.message}"
        assert (tmp_path / "New Folder").exists()

    @pytest.mark.asyncio
    async def test_fails_if_folder_already_exists(self, tmp_path, file_skill):
        (tmp_path / "AlreadyExists").mkdir()
        ctx = make_ctx("create_folder", folder_name="AlreadyExists", path=str(tmp_path))
        result = await file_skill.execute("create_folder", ctx)
        assert not result.success
        assert "already exists" in result.message.lower()

    @pytest.mark.asyncio
    async def test_never_reports_success_without_verification(self, tmp_path, file_skill):
        """Critical: if mkdir is a no-op, verification must catch it."""
        ctx = make_ctx("create_folder", folder_name="Ghost", path=str(tmp_path))
        with patch("pathlib.Path.mkdir"):  # no-op — folder not actually created
            result = await file_skill.execute("create_folder", ctx)
        assert not result.success, "MUST NOT report success when verification fails!"

    @pytest.mark.asyncio
    async def test_verified_flag_present_on_success(self, tmp_path, file_skill):
        ctx = make_ctx("create_folder", folder_name="Verified", path=str(tmp_path))
        result = await file_skill.execute("create_folder", ctx)
        assert result.success
        assert result.data.get("verified") is True


class TestRenameFolderWithVerification:

    @pytest.mark.asyncio
    async def test_renames_folder_successfully(self, tmp_path, file_skill):
        (tmp_path / "OldName").mkdir()
        ctx = make_ctx("rename_folder", old_name="OldName", new_name="NewName",
                       path=str(tmp_path))
        result = await file_skill.execute("rename_folder", ctx)
        assert result.success, f"Expected success: {result.message}"
        assert (tmp_path / "NewName").exists()
        assert not (tmp_path / "OldName").exists()
        assert result.data.get("verified") is True

    @pytest.mark.asyncio
    async def test_fails_when_old_folder_not_found(self, tmp_path, file_skill):
        ctx = make_ctx("rename_folder", old_name="NoSuchFolder", new_name="NewName",
                       path=str(tmp_path))
        result = await file_skill.execute("rename_folder", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_fails_when_missing_old_name(self, tmp_path, file_skill):
        ctx = make_ctx("rename_folder", new_name="NewName", path=str(tmp_path))
        result = await file_skill.execute("rename_folder", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_fails_when_missing_new_name(self, tmp_path, file_skill):
        (tmp_path / "OldFolder").mkdir()
        ctx = make_ctx("rename_folder", old_name="OldFolder", path=str(tmp_path))
        result = await file_skill.execute("rename_folder", ctx)
        assert not result.success


class TestDeleteFolderWithVerification:

    @pytest.mark.asyncio
    async def test_deletes_folder_successfully(self, tmp_path, file_skill):
        (tmp_path / "ToDelete").mkdir()
        ctx = make_ctx("delete_folder", folder_name="ToDelete", path=str(tmp_path), force="true")
        result = await file_skill.execute("delete_folder", ctx)
        assert result.success, f"Expected success: {result.message}"
        assert not (tmp_path / "ToDelete").exists()
        assert result.data.get("verified") is True

    @pytest.mark.asyncio
    async def test_fails_when_folder_not_found(self, tmp_path, file_skill):
        ctx = make_ctx("delete_folder", folder_name="NoSuchFolder", path=str(tmp_path))
        result = await file_skill.execute("delete_folder", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_force_delete_is_permanent(self, tmp_path, file_skill):
        target = tmp_path / "ForceDelete"
        target.mkdir()
        (target / "file.txt").write_text("data")
        ctx = make_ctx("delete_folder", folder_name="ForceDelete", path=str(tmp_path), force="true")
        result = await file_skill.execute("delete_folder", ctx)
        assert result.success
        assert not target.exists()
        assert result.data.get("permanent") is True

    @pytest.mark.asyncio
    async def test_never_reports_success_when_folder_still_exists(self, tmp_path, file_skill):
        """Critical: if deletion fails silently, we must report failure."""
        (tmp_path / "Stubborn").mkdir()
        import shutil
        with patch.object(shutil, "rmtree"):  # no-op — folder stays
            ctx = make_ctx("delete_folder", folder_name="Stubborn", path=str(tmp_path), force="true")
            result = await file_skill.execute("delete_folder", ctx)
        assert not result.success, "MUST NOT report success when folder still exists!"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — show_desktop Skill
# ═══════════════════════════════════════════════════════════════════════════════


class TestShowDesktopSkill:

    def test_show_desktop_in_capabilities(self):
        from spidy.skills.desktop.system_control_skill import SystemControlSkill
        actions = [cap.action for cap in SystemControlSkill().capabilities()]
        assert "show_desktop" in actions

    @pytest.mark.asyncio
    async def test_show_desktop_calls_implementation(self):
        from spidy.skills.desktop.system_control_skill import SystemControlSkill
        skill = SystemControlSkill()
        ctx = make_ctx("show_desktop")
        with patch.object(SystemControlSkill, "_do_show_desktop", return_value=True):
            result = await skill.execute("show_desktop", ctx)
        assert result.success
        assert result.action_taken == "show_desktop"

    @pytest.mark.asyncio
    async def test_show_desktop_fails_gracefully_on_non_windows(self):
        from spidy.skills.desktop.system_control_skill import SystemControlSkill
        skill = SystemControlSkill()
        ctx = make_ctx("show_desktop")
        with patch.object(SystemControlSkill, "_do_show_desktop", return_value=False):
            result = await skill.execute("show_desktop", ctx)
        assert not result.success
        assert "traceback" not in result.message.lower()
        assert "exception" not in result.message.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — Error Message Sanitization
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorSanitization:

    def test_sanitize_appskill_bring_to_foreground_error(self):
        from spidy.brain.response_composer import _sanitize_message
        raw = "AppSkill: 'name' parameter is required for bring_app_to_foreground."
        result = _sanitize_message(raw)
        assert "AppSkill:" not in result

    def test_sanitize_fileskill_prefix(self):
        from spidy.brain.response_composer import _sanitize_message
        raw = "FileSkill: 'path' parameter is required for open_folder."
        result = _sanitize_message(raw)
        assert "FileSkill:" not in result

    def test_sanitize_systemcontrolskill_prefix(self):
        from spidy.brain.response_composer import _sanitize_message
        raw = "SystemControlSkill: failed to set brightness"
        result = _sanitize_message(raw)
        assert "SystemControlSkill:" not in result

    def test_clean_message_unchanged(self):
        from spidy.brain.response_composer import _sanitize_message
        clean = "I've opened Chrome for you."
        assert _sanitize_message(clean) == clean

    def test_is_raw_detects_appskill_prefix(self):
        from spidy.brain.response_composer import _is_raw_message
        assert _is_raw_message("AppSkill: 'name' parameter is required")
        assert _is_raw_message("FileSkill: folder not found")
        assert _is_raw_message("SystemControlSkill: failed")

    def test_is_raw_clean_messages(self):
        from spidy.brain.response_composer import _is_raw_message
        assert not _is_raw_message("I've opened Chrome for you.")
        assert not _is_raw_message("Desktop is now visible.")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 — open_folder Path Resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenFolderPathResolution:

    def test_no_path_defaults_to_desktop(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path(None) == Path.home() / "Desktop"

    def test_empty_string_defaults_to_desktop(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("") == Path.home() / "Desktop"

    def test_downloads_resolves(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("downloads") == Path.home() / "Downloads"

    def test_documents_resolves(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("documents") == Path.home() / "Documents"

    def test_desktop_resolves(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("desktop") == Path.home() / "Desktop"

    def test_pictures_resolves(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("pictures") == Path.home() / "Pictures"

    def test_music_resolves(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("music") == Path.home() / "Music"

    def test_videos_resolves(self):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path("videos") == Path.home() / "Videos"

    def test_absolute_path_returned_unchanged(self, tmp_path):
        from spidy.skills.desktop.file_skill import FileSkill
        assert FileSkill()._resolve_parent_path(str(tmp_path)) == tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8 — No-Skill Honest Messages
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoSkillMessages:

    def test_desktop_context_menu_message(self):
        from spidy.brain.tool_router import _build_no_skill_message
        msg = _build_no_skill_message("desktop_context_menu")
        assert "right" in msg.lower() or "context menu" in msg.lower()
        assert "desktop_context_menu" not in msg  # raw action name not exposed
        assert "AppSkill:" not in msg

    def test_unknown_action_is_helpful(self):
        from spidy.brain.tool_router import _build_no_skill_message
        msg = _build_no_skill_message("some_unknown_action")
        assert "help" in msg.lower()

    def test_right_click_message_is_honest(self):
        from spidy.brain.tool_router import _build_no_skill_message
        msg = _build_no_skill_message("right_click")
        assert len(msg) > 20


# ═══════════════════════════════════════════════════════════════════════════════
# PART 9 — Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:

    @pytest.mark.asyncio
    async def test_launch_app_still_works(self):
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("open chrome")
        assert intent.action == "launch_app"

    @pytest.mark.asyncio
    async def test_switch_to_chrome_still_foreground(self):
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("switch to Chrome")
        assert intent.action == "bring_app_to_foreground"

    @pytest.mark.asyncio
    async def test_lock_screen_still_works(self):
        from spidy.brain.intent_classifier import IntentClassifier
        intent = await IntentClassifier().classify("lock the screen")
        assert intent.action == "lock_workstation"

    def test_planner_existing_aliases_preserved(self):
        from spidy.brain.planner import _ACTION_ALIASES
        assert _ACTION_ALIASES["lock_screen"] == "lock_workstation"
        assert _ACTION_ALIASES["search_web"] == "search_google"
        assert _ACTION_ALIASES["math"] == "calculate"
        assert _ACTION_ALIASES["shutdown"] == "shutdown_system"
        assert _ACTION_ALIASES["restart"] == "restart_system"

    def test_file_skill_old_capabilities_still_registered(self):
        from spidy.skills.desktop.file_skill import FileSkill
        actions = [cap.action for cap in FileSkill().capabilities()]
        assert "search_files" in actions
        assert "open_folder" in actions
        assert "list_recent_files" in actions

    def test_new_file_skill_capabilities_registered(self):
        from spidy.skills.desktop.file_skill import FileSkill
        actions = [cap.action for cap in FileSkill().capabilities()]
        assert "create_folder" in actions
        assert "rename_folder" in actions
        assert "delete_folder" in actions
