"""
Tests for Milestone 6 — FileSkill
===================================
Unit tests for all FileSkill actions:
- search_files
- search_folders
- open_file
- open_folder
- reveal_in_explorer
- list_recent_files
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spidy.skills.base import SkillContext, SkillResult
from spidy.skills.desktop.file_skill import FileSkill


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(action: str = "", params: dict | None = None, session_id: str = "test") -> SkillContext:
    return SkillContext(action=action, params=params or {}, session_id=session_id)


def _make_skill(bus=None, tmp_root: Path | None = None, max_results: int = 50) -> FileSkill:
    return FileSkill(
        bus=bus,
        search_root=str(tmp_root) if tmp_root else str(Path.home()),
        max_results=max_results,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Capability declarations
# ──────────────────────────────────────────────────────────────────────────────


class TestFileSkillCapabilities:
    def test_all_actions_declared(self):
        skill = _make_skill()
        actions = skill.capability_names()
        expected = {
            # Existing actions
            "search_files", "search_folders", "open_file",
            "open_folder", "reveal_in_explorer", "list_recent_files",
            # M13.2 — folder mutation actions
            "create_folder", "rename_folder", "delete_folder",
        }
        assert expected == set(actions)

    def test_search_actions_are_t0(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            if cap.action in ("search_files", "search_folders", "list_recent_files"):
                assert cap.permission_tier == "T0", f"{cap.action} should be T0"

    def test_open_actions_are_t1(self):
        skill = _make_skill()
        for cap in skill.capabilities():
            if cap.action in ("open_file", "open_folder", "reveal_in_explorer"):
                assert cap.permission_tier == "T1", f"{cap.action} should be T1"

    def test_skill_name_and_version(self):
        assert FileSkill.name == "file_skill"
        assert FileSkill.version == "1.0.0"

    def test_supports_all_actions(self):
        skill = _make_skill()
        for action in ("search_files", "search_folders", "open_file",
                       "open_folder", "reveal_in_explorer", "list_recent_files"):
            assert skill.supports(action)

    def test_unknown_action_not_supported(self):
        skill = _make_skill()
        assert not skill.supports("does_not_exist")


# ──────────────────────────────────────────────────────────────────────────────
# 2. search_files
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchFiles:
    @pytest.fixture
    def tmp_tree(self, tmp_path):
        """Create a temporary file tree for testing."""
        (tmp_path / "report.pdf").write_text("pdf content")
        (tmp_path / "notes.txt").write_text("note content")
        (tmp_path / "invoice_2024.pdf").write_text("invoice")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep_report.pdf").write_text("deep pdf")
        (sub / "image.png").write_text("png")
        return tmp_path

    @pytest.mark.asyncio
    async def test_search_finds_matching_files(self, tmp_tree):
        skill = _make_skill(tmp_root=tmp_tree)
        ctx = _ctx("search_files", {"query": "report", "path": str(tmp_tree)})
        result = await skill.execute("search_files", ctx)
        assert result.success
        assert result.data["total_found"] >= 1
        # Should find report.pdf and deep_report.pdf
        found_names = [Path(p).name for p in result.data["results"]]
        assert any("report" in n.lower() for n in found_names)

    @pytest.mark.asyncio
    async def test_search_with_extension_filter(self, tmp_tree):
        skill = _make_skill(tmp_root=tmp_tree)
        ctx = _ctx("search_files", {"query": "*", "path": str(tmp_tree), "extensions": "pdf"})
        result = await skill.execute("search_files", ctx)
        assert result.success
        for path in result.data["results"]:
            assert Path(path).suffix.lower() == ".pdf"

    @pytest.mark.asyncio
    async def test_search_no_results(self, tmp_tree):
        skill = _make_skill(tmp_root=tmp_tree)
        ctx = _ctx("search_files", {"query": "nonexistent_xyz_file", "path": str(tmp_tree)})
        result = await skill.execute("search_files", ctx)
        assert result.success  # No results is not a failure
        assert result.data["results"] == []

    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        skill = _make_skill()
        ctx = _ctx("search_files", {})
        result = await skill.execute("search_files", ctx)
        assert not result.success
        assert "query" in result.message.lower()

    @pytest.mark.asyncio
    async def test_search_invalid_path(self):
        skill = _make_skill()
        ctx = _ctx("search_files", {"query": "test", "path": "/nonexistent/path/xyz"})
        result = await skill.execute("search_files", ctx)
        assert not result.success
        assert "not exist" in result.message.lower()

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, tmp_tree):
        # Create many files
        for i in range(10):
            (tmp_tree / f"file_{i:03d}.txt").write_text("content")

        skill = _make_skill(tmp_root=tmp_tree, max_results=3)
        ctx = _ctx("search_files", {"query": "file_", "path": str(tmp_tree), "max_results": "10"})
        result = await skill.execute("search_files", ctx)
        assert result.success
        # max_results on skill is 3, so even if we ask for 10, we get at most 3
        assert len(result.data["results"]) <= 3

    @pytest.mark.asyncio
    async def test_search_publishes_event(self, tmp_tree):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus, tmp_root=tmp_tree)
        ctx = _ctx("search_files", {"query": "report", "path": str(tmp_tree)})
        await skill.execute("search_files", ctx)
        bus.publish.assert_called_once()
        event = bus.publish.call_args[0][0]
        from spidy.skills.desktop.events import FileSearchResultEvent
        assert isinstance(event, FileSearchResultEvent)
        assert event.query == "report"


# ──────────────────────────────────────────────────────────────────────────────
# 3. search_folders
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchFolders:
    @pytest.fixture
    def tmp_tree(self, tmp_path):
        (tmp_path / "project_alpha").mkdir()
        (tmp_path / "project_beta").mkdir()
        (tmp_path / "documents").mkdir()
        sub = tmp_path / "work"
        sub.mkdir()
        (sub / "project_gamma").mkdir()
        return tmp_path

    @pytest.mark.asyncio
    async def test_search_finds_folders(self, tmp_tree):
        skill = _make_skill(tmp_root=tmp_tree)
        ctx = _ctx("search_folders", {"query": "project", "path": str(tmp_tree)})
        result = await skill.execute("search_folders", ctx)
        assert result.success
        # Should find project_alpha, project_beta, project_gamma
        assert result.data["total_found"] >= 2

    @pytest.mark.asyncio
    async def test_search_folders_not_files(self, tmp_tree):
        (tmp_tree / "project_file.txt").write_text("not a folder")
        skill = _make_skill(tmp_root=tmp_tree)
        ctx = _ctx("search_folders", {"query": "project_file", "path": str(tmp_tree)})
        result = await skill.execute("search_folders", ctx)
        assert result.success
        # Should NOT find the txt file
        assert result.data["total_found"] == 0

    @pytest.mark.asyncio
    async def test_search_folders_requires_query(self):
        skill = _make_skill()
        result = await skill.execute("search_folders", _ctx("search_folders", {}))
        assert not result.success


# ──────────────────────────────────────────────────────────────────────────────
# 4. open_file
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenFile:
    @pytest.mark.asyncio
    async def test_open_file_success(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        with patch("os.startfile") as mock_startfile:
            skill = _make_skill()
            ctx = _ctx("open_file", {"path": str(test_file)})
            result = await skill.execute("open_file", ctx)
        assert result.success
        mock_startfile.assert_called_once_with(str(test_file))

    @pytest.mark.asyncio
    async def test_open_file_not_found(self):
        skill = _make_skill()
        ctx = _ctx("open_file", {"path": "/nonexistent/file.txt"})
        result = await skill.execute("open_file", ctx)
        assert not result.success
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_open_file_requires_path(self):
        skill = _make_skill()
        result = await skill.execute("open_file", _ctx("open_file", {}))
        assert not result.success
        assert "path" in result.message.lower()

    @pytest.mark.asyncio
    async def test_open_file_is_not_a_file(self, tmp_path):
        # Pass a directory path
        skill = _make_skill()
        ctx = _ctx("open_file", {"path": str(tmp_path)})
        result = await skill.execute("open_file", ctx)
        assert not result.success
        assert "not a file" in result.message.lower()

    @pytest.mark.asyncio
    async def test_open_file_publishes_event(self, tmp_path):
        test_file = tmp_path / "doc.pdf"
        test_file.write_text("pdf")

        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        ctx = _ctx("open_file", {"path": str(test_file)})

        with patch("os.startfile"):
            await skill.execute("open_file", ctx)

        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import FileOpenedEvent
        assert isinstance(bus.publish.call_args[0][0], FileOpenedEvent)

    @pytest.mark.asyncio
    async def test_open_file_not_supported_non_windows(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        skill = _make_skill()
        ctx = _ctx("open_file", {"path": str(test_file)})
        with patch("os.startfile", side_effect=AttributeError("not on windows")):
            result = await skill.execute("open_file", ctx)
        assert not result.success
        assert "windows" in result.message.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 5. open_folder
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenFolder:
    @pytest.mark.asyncio
    async def test_open_folder_success(self, tmp_path):
        with patch("subprocess.Popen") as mock_popen:
            skill = _make_skill()
            ctx = _ctx("open_folder", {"path": str(tmp_path)})
            result = await skill.execute("open_folder", ctx)
        assert result.success
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "explorer" in call_args

    @pytest.mark.asyncio
    async def test_open_folder_not_found(self):
        skill = _make_skill()
        ctx = _ctx("open_folder", {"path": "/nonexistent/folder/xyz"})
        result = await skill.execute("open_folder", ctx)
        assert not result.success
        # M13.2: error message was humanized from "not found" to "couldn't find"
        assert (
            "not found" in result.message.lower()
            or "couldn't find" in result.message.lower()
            or "could not" in result.message.lower()
        )

    @pytest.mark.asyncio
    async def test_open_folder_no_path_defaults_to_desktop(self):
        # M13.2: open_folder with no path now defaults to the Desktop
        # (a valid directory), so it should succeed or open Desktop.
        skill = _make_skill()
        desktop = __import__('pathlib').Path.home() / "Desktop"
        if not desktop.exists():
            pytest.skip("Desktop directory does not exist on this system")
        with __import__('unittest.mock', fromlist=['patch']).patch("subprocess.Popen"):
            result = await skill.execute("open_folder", _ctx("open_folder", {}))
        assert result.success  # Desktop exists, should succeed

    @pytest.mark.asyncio
    async def test_open_folder_rejects_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        skill = _make_skill()
        ctx = _ctx("open_folder", {"path": str(f)})
        result = await skill.execute("open_folder", ctx)
        assert not result.success
        assert "not a folder" in result.message.lower()

    @pytest.mark.asyncio
    async def test_open_folder_publishes_event(self, tmp_path):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        ctx = _ctx("open_folder", {"path": str(tmp_path)})
        with patch("subprocess.Popen"):
            await skill.execute("open_folder", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import FolderOpenedEvent
        event = bus.publish.call_args[0][0]
        assert isinstance(event, FolderOpenedEvent)
        assert event.reveal_mode is False


# ──────────────────────────────────────────────────────────────────────────────
# 6. reveal_in_explorer
# ──────────────────────────────────────────────────────────────────────────────


class TestRevealInExplorer:
    @pytest.mark.asyncio
    async def test_reveal_success(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content")
        with patch("subprocess.Popen") as mock_popen:
            skill = _make_skill()
            ctx = _ctx("reveal_in_explorer", {"path": str(f)})
            result = await skill.execute("reveal_in_explorer", ctx)
        assert result.success
        call_args = mock_popen.call_args[0][0]
        assert "/select," in call_args

    @pytest.mark.asyncio
    async def test_reveal_not_found(self):
        skill = _make_skill()
        ctx = _ctx("reveal_in_explorer", {"path": "/nonexistent/file.txt"})
        result = await skill.execute("reveal_in_explorer", ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_reveal_publishes_event(self, tmp_path):
        f = tmp_path / "file.pdf"
        f.write_text("pdf")
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        ctx = _ctx("reveal_in_explorer", {"path": str(f)})
        with patch("subprocess.Popen"):
            await skill.execute("reveal_in_explorer", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import FolderOpenedEvent
        event = bus.publish.call_args[0][0]
        assert isinstance(event, FolderOpenedEvent)
        assert event.reveal_mode is True


# ──────────────────────────────────────────────────────────────────────────────
# 7. list_recent_files
# ──────────────────────────────────────────────────────────────────────────────


class TestListRecentFiles:
    @pytest.mark.asyncio
    async def test_list_recent_returns_result(self):
        skill = _make_skill()
        ctx = _ctx("list_recent_files", {"limit": "5"})
        with patch.object(FileSkill, "_get_recent_files", return_value=["file1.txt", "file2.pdf"]):
            result = await skill.execute("list_recent_files", ctx)
        assert result.success
        assert "file1.txt" in result.message or "file1.txt" in str(result.data)

    @pytest.mark.asyncio
    async def test_list_recent_empty(self):
        skill = _make_skill()
        ctx = _ctx("list_recent_files", {})
        with patch.object(FileSkill, "_get_recent_files", return_value=[]):
            result = await skill.execute("list_recent_files", ctx)
        assert result.success
        assert "no recent" in result.message.lower()

    @pytest.mark.asyncio
    async def test_list_recent_publishes_event(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        skill = _make_skill(bus=bus)
        ctx = _ctx("list_recent_files", {})
        with patch.object(FileSkill, "_get_recent_files", return_value=["a.txt"]):
            await skill.execute("list_recent_files", ctx)
        bus.publish.assert_called_once()
        from spidy.skills.desktop.events import RecentFilesResultEvent
        assert isinstance(bus.publish.call_args[0][0], RecentFilesResultEvent)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Unknown action
# ──────────────────────────────────────────────────────────────────────────────


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action_fails_gracefully(self):
        skill = _make_skill()
        result = await skill.execute("does_not_exist", _ctx("does_not_exist"))
        assert not result.success
        assert "unknown action" in result.message.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 9. Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestFileSkillInternals:
    def test_do_file_search_finds_files(self, tmp_path):
        (tmp_path / "needle.txt").write_text("content")
        (tmp_path / "haystack.txt").write_text("other")
        results = FileSkill._do_file_search(
            tmp_path, "needle", None, 10, files_only=True
        )
        assert len(results) == 1
        assert "needle.txt" in results[0]

    def test_do_file_search_glob_pattern(self, tmp_path):
        for i in range(5):
            (tmp_path / f"file{i}.pdf").write_text("pdf")
        (tmp_path / "other.txt").write_text("txt")
        results = FileSkill._do_file_search(
            tmp_path, "*.pdf", None, 10, files_only=True
        )
        assert len(results) == 5

    def test_do_file_search_extension_filter(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("pdf")
        (tmp_path / "doc.txt").write_text("txt")
        results = FileSkill._do_file_search(
            tmp_path, "doc", [".pdf"], 10, files_only=True
        )
        assert len(results) == 1
        assert results[0].endswith(".pdf")

    def test_do_file_search_respects_max_results(self, tmp_path):
        for i in range(20):
            (tmp_path / f"file{i}.txt").write_text("")
        results = FileSkill._do_file_search(
            tmp_path, "file", None, 5, files_only=True
        )
        assert len(results) <= 5

    def test_do_file_search_folders_only(self, tmp_path):
        (tmp_path / "my_folder").mkdir()
        (tmp_path / "my_file.txt").write_text("txt")
        results = FileSkill._do_file_search(
            tmp_path, "my_", None, 10, files_only=False
        )
        assert all(Path(r).is_dir() for r in results)
        assert len(results) == 1
