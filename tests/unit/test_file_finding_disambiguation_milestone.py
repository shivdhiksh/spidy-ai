"""
Tests for Smart File Finding and Disambiguation (Phase E)
"""

import pytest
import os
import tempfile
from pathlib import Path
from spidy.skills.desktop.file_skill import FileSkill
from spidy.skills.base import SkillContext


@pytest.mark.asyncio
async def test_file_skill_smart_search_single_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "spidy_unique_report.txt"
        test_file.write_text("Hello from spidy", encoding="utf-8")

        skill = FileSkill(search_root=tmpdir)
        ctx = SkillContext(
            action="search_files",
            params={"query": "spidy_unique_report", "path": tmpdir},
            session_id="test",
        )
        res = await skill.execute("search_files", ctx)
        assert res.success is True
        assert "spidy_unique_report.txt" in res.message
        assert res.data.get("file_path") == str(test_file)
        assert res.data.get("total_found") == 1


@pytest.mark.asyncio
async def test_file_skill_smart_search_multiple_matches_disambiguation():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "invoice_january.pdf").write_text("1", encoding="utf-8")
        (Path(tmpdir) / "invoice_february.pdf").write_text("2", encoding="utf-8")
        (Path(tmpdir) / "invoice_march.pdf").write_text("3", encoding="utf-8")

        skill = FileSkill(search_root=tmpdir)
        ctx = SkillContext(
            action="search_files",
            params={"query": "invoice", "path": tmpdir},
            session_id="test",
        )
        res = await skill.execute("search_files", ctx)
        assert res.success is True
        assert "matching file(s)" in res.message
        assert len(res.data.get("results", [])) == 3
        assert len(res.data.get("candidates", [])) == 3
