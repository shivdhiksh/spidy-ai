"""
FileSkill — Windows File & Folder Agent
=======================================
Provides safe file system interaction capabilities.

Permission Tiers
----------------
T0 — Read-only actions (search, list recent)
T1 — Actions that open/reveal files or folders

Actions
-------
search_files        — Recursive file search by name pattern        (T0)
search_folders      — Recursive folder search by name              (T0)
open_file           — Open a file with its default application     (T1)
open_folder         — Open a folder in Windows Explorer            (T1)
reveal_in_explorer  — Reveal and select a file in Explorer         (T1)
list_recent_files   — List recently accessed files                 (T0)

Architecture
------------
- No Windows API required for search (pure pathlib).
- open_file uses os.startfile() (Windows-native, safe).
- open_folder / reveal_in_explorer use subprocess + explorer.exe.
- All results are published to the EventBus after execution.
- The Brain never calls this module directly — only through the Executor.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger
from spidy.skills.base import (
    BaseSkill,
    ParamSchema,
    SkillCapability,
    SkillContext,
    SkillResult,
)

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus

log = get_logger(__name__)

# Maximum search results to prevent runaway searches
_DEFAULT_MAX_RESULTS = 50
# Default search root (expanded at runtime)
_DEFAULT_SEARCH_ROOT = str(Path.home())


class FileSkill(BaseSkill):
    """
    File and folder interaction agent.

    Parameters
    ----------
    bus:
        EventBus for publishing desktop events. May be None.
    search_root:
        Default directory to search when no path is specified.
    max_results:
        Hard cap on search results to prevent performance issues.
    """

    name = "file_skill"
    version = "1.0.0"

    def __init__(
        self,
        bus: "EventBus | None" = None,
        search_root: str | None = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> None:
        self._bus = bus
        self._search_root = Path(search_root or _DEFAULT_SEARCH_ROOT).expanduser()
        self._max_results = max_results

    # ── Capabilities ──────────────────────────────────────────────────────

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="search_files",
                description=(
                    "Search for files by name or pattern. "
                    "Returns a list of matching file paths."
                ),
                permission_tier="T0",
                params=[
                    ParamSchema("query", "string", required=True,
                                description="File name or glob pattern (e.g. '*.pdf', 'report')."),
                    ParamSchema("path", "path", required=False,
                                description="Directory to search. Defaults to home directory."),
                    ParamSchema("extensions", "string", required=False,
                                description="Comma-separated extensions filter (e.g. 'pdf,docx')."),
                    ParamSchema("max_results", "int", required=False,
                                default=_DEFAULT_MAX_RESULTS,
                                description="Maximum number of results to return."),
                ],
                examples=[
                    "find resume.pdf", "search for invoice files", "look for notes.txt",
                    "find all PDF files", "search files named report",
                ],
            ),
            SkillCapability(
                action="search_folders",
                description="Search for folders by name.",
                permission_tier="T0",
                params=[
                    ParamSchema("query", "string", required=True,
                                description="Folder name or pattern."),
                    ParamSchema("path", "path", required=False,
                                description="Directory to search. Defaults to home directory."),
                    ParamSchema("max_results", "int", required=False,
                                default=_DEFAULT_MAX_RESULTS,
                                description="Maximum number of results to return."),
                ],
                examples=[
                    "find downloads folder", "search for project folder",
                    "where is the documents folder",
                ],
            ),
            SkillCapability(
                action="open_file",
                description="Open a file with its default application.",
                permission_tier="T1",
                params=[
                    ParamSchema("path", "path", required=True,
                                description="Full path to the file to open."),
                ],
                examples=[
                    "open resume.pdf", "open the file at C:/notes.txt",
                    "launch this document",
                ],
            ),
            SkillCapability(
                action="open_folder",
                description="Open a folder in Windows Explorer.",
                permission_tier="T1",
                params=[
                    ParamSchema("path", "path", required=True,
                                description="Full path to the folder to open."),
                ],
                examples=[
                    "open the downloads folder", "open C:/Users/me/Documents",
                    "show this folder in explorer",
                ],
            ),
            SkillCapability(
                action="reveal_in_explorer",
                description="Reveal and select a specific file in Windows Explorer.",
                permission_tier="T1",
                params=[
                    ParamSchema("path", "path", required=True,
                                description="Full path to the file to reveal."),
                ],
                examples=[
                    "reveal in explorer", "show file location", "locate this file",
                    "open containing folder",
                ],
            ),
            SkillCapability(
                action="list_recent_files",
                description="List recently accessed files from the Windows Recent folder.",
                permission_tier="T0",
                params=[
                    ParamSchema("limit", "int", required=False, default=10,
                                description="Maximum number of recent files to return."),
                ],
                examples=[
                    "show recent files", "what did I open recently",
                    "list recent documents", "recently used files",
                ],
            ),
        ]

    # ── Dispatch ──────────────────────────────────────────────────────────

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "search_files":
            return await self._search_files(context)
        if action == "search_folders":
            return await self._search_folders(context)
        if action == "open_file":
            return await self._open_file(context)
        if action == "open_folder":
            return await self._open_folder(context)
        if action == "reveal_in_explorer":
            return await self._reveal_in_explorer(context)
        if action == "list_recent_files":
            return await self._list_recent_files(context)
        return SkillResult.fail(f"FileSkill: unknown action '{action}'.")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _search_files(self, context: SkillContext) -> SkillResult:
        query: str = context.get("query", "")
        if not query:
            return SkillResult.fail("FileSkill: 'query' parameter is required for search_files.")

        raw_path = context.get("path")
        search_root = Path(raw_path).expanduser() if raw_path else self._search_root
        if not search_root.exists():
            return SkillResult.fail(
                f"FileSkill: search path does not exist: '{search_root}'"
            )

        extensions_raw: str = context.get("extensions", "")
        extensions: list[str] | None = None
        if extensions_raw:
            extensions = [
                f".{e.strip().lstrip('.')}" for e in extensions_raw.split(",") if e.strip()
            ]

        max_r = int(context.get("max_results") or self._max_results)
        max_r = min(max_r, self._max_results)

        # Run blocking search in thread pool
        results = await asyncio.to_thread(
            self._do_file_search, search_root, query, extensions, max_r, files_only=True
        )

        await self._emit_search_result(query, results, str(search_root), context.session_id)

        if not results:
            return SkillResult.ok(
                f"No files matching '{query}' found in {search_root}.",
                data={"results": [], "query": query, "search_root": str(search_root),
                      "total_found": 0},
                action_taken="search_files",
            )

        paths_str = "\n".join(results[:10])
        suffix = f"\n… and {len(results) - 10} more." if len(results) > 10 else ""
        return SkillResult.ok(
            f"Found {len(results)} file(s) matching '{query}':\n{paths_str}{suffix}",
            data={"results": results, "query": query, "search_root": str(search_root),
                  "total_found": len(results)},
            action_taken="search_files",
        )

    async def _search_folders(self, context: SkillContext) -> SkillResult:
        query: str = context.get("query", "")
        if not query:
            return SkillResult.fail("FileSkill: 'query' parameter is required for search_folders.")

        raw_path = context.get("path")
        search_root = Path(raw_path).expanduser() if raw_path else self._search_root
        if not search_root.exists():
            return SkillResult.fail(
                f"FileSkill: search path does not exist: '{search_root}'"
            )

        max_r = int(context.get("max_results") or self._max_results)
        max_r = min(max_r, self._max_results)

        results = await asyncio.to_thread(
            self._do_file_search, search_root, query, None, max_r, files_only=False
        )

        await self._emit_search_result(query, results, str(search_root), context.session_id)

        if not results:
            return SkillResult.ok(
                f"No folders matching '{query}' found in {search_root}.",
                data={"results": [], "query": query, "total_found": 0},
                action_taken="search_folders",
            )

        paths_str = "\n".join(results[:10])
        suffix = f"\n… and {len(results) - 10} more." if len(results) > 10 else ""
        return SkillResult.ok(
            f"Found {len(results)} folder(s) matching '{query}':\n{paths_str}{suffix}",
            data={"results": results, "query": query, "total_found": len(results)},
            action_taken="search_folders",
        )

    async def _open_file(self, context: SkillContext) -> SkillResult:
        raw_path = context.get("path", "")
        if not raw_path:
            return SkillResult.fail("FileSkill: 'path' parameter is required for open_file.")

        file_path = Path(raw_path).expanduser().resolve()
        if not file_path.exists():
            return SkillResult.fail(f"FileSkill: file not found: '{file_path}'")
        if not file_path.is_file():
            return SkillResult.fail(f"FileSkill: path is not a file: '{file_path}'")

        try:
            await asyncio.to_thread(os.startfile, str(file_path))  # type: ignore[attr-defined]
        except AttributeError:
            # os.startfile is Windows-only
            return SkillResult.fail(
                "FileSkill: open_file is only supported on Windows."
            )
        except OSError as exc:
            return SkillResult.fail(f"FileSkill: could not open file: {exc}", error=exc)

        await self._emit_file_opened(str(file_path), context.session_id)
        return SkillResult.ok(
            f"Opened '{file_path.name}'.",
            data={"path": str(file_path)},
            action_taken="open_file",
        )

    async def _open_folder(self, context: SkillContext) -> SkillResult:
        raw_path = context.get("path", "")
        if not raw_path:
            return SkillResult.fail("FileSkill: 'path' parameter is required for open_folder.")

        folder_path = Path(raw_path).expanduser().resolve()
        if not folder_path.exists():
            return SkillResult.fail(f"FileSkill: folder not found: '{folder_path}'")
        if not folder_path.is_dir():
            return SkillResult.fail(f"FileSkill: path is not a folder: '{folder_path}'")

        try:
            await asyncio.to_thread(
                self._run_explorer, str(folder_path), select_mode=False
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(f"FileSkill: could not open folder: {exc}", error=exc)

        await self._emit_folder_opened(str(folder_path), reveal_mode=False,
                                       session_id=context.session_id)
        return SkillResult.ok(
            f"Opened folder '{folder_path}'.",
            data={"path": str(folder_path)},
            action_taken="open_folder",
        )

    async def _reveal_in_explorer(self, context: SkillContext) -> SkillResult:
        raw_path = context.get("path", "")
        if not raw_path:
            return SkillResult.fail(
                "FileSkill: 'path' parameter is required for reveal_in_explorer."
            )

        file_path = Path(raw_path).expanduser().resolve()
        if not file_path.exists():
            return SkillResult.fail(f"FileSkill: path not found: '{file_path}'")

        try:
            await asyncio.to_thread(
                self._run_explorer, str(file_path), select_mode=True
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"FileSkill: could not reveal in Explorer: {exc}", error=exc
            )

        await self._emit_folder_opened(str(file_path), reveal_mode=True,
                                       session_id=context.session_id)
        return SkillResult.ok(
            f"Revealed '{file_path.name}' in Explorer.",
            data={"path": str(file_path)},
            action_taken="reveal_in_explorer",
        )

    async def _list_recent_files(self, context: SkillContext) -> SkillResult:
        limit = int(context.get("limit") or 10)
        limit = max(1, min(limit, 50))

        recent_files = await asyncio.to_thread(self._get_recent_files, limit)

        await self._emit_recent_result(recent_files, context.session_id)

        if not recent_files:
            return SkillResult.ok(
                "No recent files found.",
                data={"files": []},
                action_taken="list_recent_files",
            )

        files_str = "\n".join(recent_files[:limit])
        return SkillResult.ok(
            f"Recent files ({len(recent_files)}):\n{files_str}",
            data={"files": recent_files},
            action_taken="list_recent_files",
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _do_file_search(
        root: Path,
        query: str,
        extensions: list[str] | None,
        max_results: int,
        files_only: bool,
    ) -> list[str]:
        """
        Synchronous recursive search. Runs in a thread pool.

        Matches by case-insensitive glob against the file/folder name.
        If query has no wildcard, wraps with '*query*' for substring match.
        """
        pattern = query if any(c in query for c in ("*", "?", "[")) else f"*{query}*"
        results: list[str] = []

        try:
            for item in root.rglob("*"):
                if len(results) >= max_results:
                    break
                try:
                    is_match = fnmatch.fnmatch(item.name.lower(), pattern.lower())
                    if not is_match:
                        continue
                    if files_only and not item.is_file():
                        continue
                    if not files_only and not item.is_dir():
                        continue
                    if extensions and item.is_file():
                        if item.suffix.lower() not in extensions:
                            continue
                    results.append(str(item))
                except (PermissionError, OSError):
                    continue  # Skip inaccessible paths silently
        except (PermissionError, OSError):
            pass

        return results

    @staticmethod
    def _run_explorer(path: str, select_mode: bool) -> None:
        """Open Explorer. select_mode=True selects the file in its parent folder."""
        import subprocess
        if select_mode:
            subprocess.Popen(["explorer", "/select,", path], shell=False)
        else:
            subprocess.Popen(["explorer", path], shell=False)

    @staticmethod
    def _get_recent_files(limit: int) -> list[str]:
        """
        Read recently accessed files from the Windows Recent folder.
        Returns resolved target paths (not .lnk shortcut paths).
        """
        recent_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Recent"
        if not recent_dir.exists():
            return []

        lnk_files: list[Any] = sorted(
            recent_dir.glob("*.lnk"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        results: list[str] = []
        for lnk in lnk_files[:limit * 3]:  # over-fetch to account for resolution failures
            if len(results) >= limit:
                break
            try:
                # Try to resolve .lnk targets via winshell
                import winshell  # type: ignore[import-untyped]
                target = winshell.shortcut(str(lnk)).path
                if target and Path(target).exists():
                    results.append(target)
            except Exception:  # noqa: BLE001
                # Fallback: just return the .lnk path name without extension
                try:
                    results.append(lnk.stem)
                except Exception:  # noqa: BLE001
                    pass

        return results

    # ── EventBus publishers ───────────────────────────────────────────────

    async def _emit_search_result(
        self, query: str, results: list[str], search_root: str, session_id: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import FileSearchResultEvent
        await self._bus.publish(FileSearchResultEvent(
            query=query,
            results=results,
            total_found=len(results),
            search_root=search_root,
            session_id=session_id,
        ))

    async def _emit_file_opened(self, path: str, session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import FileOpenedEvent
        await self._bus.publish(FileOpenedEvent(path=path, session_id=session_id))

    async def _emit_folder_opened(
        self, path: str, reveal_mode: bool, session_id: str
    ) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import FolderOpenedEvent
        await self._bus.publish(FolderOpenedEvent(
            path=path, reveal_mode=reveal_mode, session_id=session_id
        ))

    async def _emit_recent_result(self, files: list[str], session_id: str) -> None:
        if self._bus is None:
            return
        from spidy.skills.desktop.events import RecentFilesResultEvent
        await self._bus.publish(RecentFilesResultEvent(files=files, session_id=session_id))
