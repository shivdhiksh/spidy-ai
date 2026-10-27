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
        default_root: str | None = None,
    ) -> None:
        self._bus = bus
        root = search_root or default_root or _DEFAULT_SEARCH_ROOT
        self._search_root = Path(root).expanduser()
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
            SkillCapability(
                action="create_file",
                description="Create or write content to a file.",
                permission_tier="T1",
                params=[
                    ParamSchema("filename", "string", required=False, description="File name."),
                    ParamSchema("path", "path", required=False, description="Target file path."),
                    ParamSchema("content", "string", required=False, default="", description="File contents."),
                ],
                examples=["create hello.py", "create file Edge.py", "save notes.txt"],
            ),
            SkillCapability(
                action="read_file",
                description="Read contents of a file.",
                permission_tier="T0",
                params=[
                    ParamSchema("path", "path", required=True, description="File path to read."),
                ],
                examples=["read calc.py", "view report.txt"],
            ),
            SkillCapability(
                action="create_folder",
                description="Create a new folder at the specified path (defaults to Desktop).",
                permission_tier="T1",
                params=[
                    ParamSchema("folder_name", "string", required=True,
                                description="Name of the folder to create."),
                    ParamSchema("path", "path", required=False,
                                description="Parent directory. Defaults to the Desktop."),
                ],
                examples=[
                    "create a folder", "make a new folder", "new folder",
                    "create folder Test", "make a folder called Projects",
                    "create a folder on the desktop",
                ],
            ),
            SkillCapability(
                action="rename_folder",
                description="Rename an existing folder.",
                permission_tier="T1",
                params=[
                    ParamSchema("old_name", "string", required=True,
                                description="Current folder name."),
                    ParamSchema("new_name", "string", required=True,
                                description="New folder name."),
                    ParamSchema("path", "path", required=False,
                                description="Parent directory. Defaults to the Desktop."),
                ],
                examples=[
                    "rename folder Test to MyProject",
                    "rename folder Projects to Archive",
                ],
            ),
            SkillCapability(
                action="delete_folder",
                description="Delete a folder (moves to Recycle Bin by default; use force=true to permanently delete).",
                permission_tier="T1",
                params=[
                    ParamSchema("folder_name", "string", required=True,
                                description="Name of the folder to delete."),
                    ParamSchema("path", "path", required=False,
                                description="Parent directory. Defaults to the Desktop."),
                    ParamSchema("force", "bool", required=False, default=False,
                                description="If true, permanently delete instead of sending to Recycle Bin."),
                ],
                examples=[
                    "delete folder Test", "remove folder Projects",
                    "delete the test folder",
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
        if action in ("create_file", "write_file", "save_file"):
            return await self._create_file(context)
        if action == "read_file":
            return await self._read_file(context)
        if action == "reveal_in_explorer":
            return await self._reveal_in_explorer(context)
        if action == "list_recent_files":
            return await self._list_recent_files(context)
        if action == "create_folder":
            return await self._create_folder(context)
        if action == "rename_folder":
            return await self._rename_folder(context)
        if action == "delete_folder":
            return await self._delete_folder(context)
        return SkillResult.fail(f"FileSkill: unknown action '{action}'.")

    @staticmethod
    def _clean_path_input(raw: str) -> str:
        """Clean path strings extracted from natural speech / transcripts."""
        s = str(raw).strip()
        s = s.strip("'\"`")
        s = s.rstrip(".?!,;:")
        return s

    async def _create_file(self, context: SkillContext) -> SkillResult:
        raw_path = self._clean_path_input(context.get("path") or context.get("filename") or context.get("name") or "")
        content = context.get("content", "")

        if not raw_path:
            return SkillResult.fail("FileSkill: 'path' or 'filename' is required to create a file.")

        p = Path(raw_path)
        if not p.is_absolute():
            p = self._search_root / p

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            # Never report a successful desktop write based only on the
            # absence of an exception.  Resolve the exact path and verify the
            # object and byte count that Windows made visible on disk.
            p = p.resolve()
            st_size = p.stat().st_size if p.exists() else -1
            expected_size = len(content.encode("utf-8"))
            if not p.is_file() or st_size != expected_size:
                return SkillResult.fail(
                    f"FileSkill: write verification failed for '{p}'.",
                )
            log.info(
                "FileSkill: created file '{path}' ({size} bytes)",
                path=str(p),
                size=st_size,
            )
            return SkillResult.ok(
                f"Created and verified file '{p}'.",
                data={
                    "path": str(p),
                    "file_path": str(p),
                    "filename": p.name,
                    "size": p.stat().st_size,
                    "content": content,
                    "exists": True,
                    "verified": True,
                },
                action_taken="create_file",
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(f"FileSkill: failed to create file '{p}': {exc}", error=exc)

    async def _read_file(self, context: SkillContext) -> SkillResult:
        raw_path = self._clean_path_input(context.get("path") or context.get("filename") or "")
        if not raw_path:
            return SkillResult.fail("FileSkill: 'path' is required to read a file.")

        p = Path(raw_path)
        if not p.is_absolute():
            p = self._search_root / p

        if not p.exists():
            return SkillResult.fail(f"FileSkill: file not found '{p}'.")

        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            return SkillResult.ok(
                f"Read file '{p.name}'.",
                data={"path": str(p), "filename": p.name, "content": data},
                action_taken="read_file",
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(f"FileSkill: failed to read file '{p}': {exc}", error=exc)

    # ── Actions ───────────────────────────────────────────────────────────

    async def _search_files(self, context: SkillContext) -> SkillResult:
        query: str = context.get("query", "")
        if not query:
            return SkillResult.fail("FileSkill: 'query' parameter is required for search_files.")

        raw_path = context.get("path")
        if raw_path:
            p = Path(raw_path).expanduser()
            if not p.exists():
                return SkillResult.fail(f"FileSkill: search path does not exist: '{raw_path}'")

        extensions_raw: str = context.get("extensions", "")
        max_r = int(context.get("max_results") or self._max_results)
        max_r = min(max_r, self._max_results)

        # Run smart search
        file_items = await asyncio.to_thread(
            self._smart_file_search, query, raw_path, extensions_raw, max_r, str(self._search_root)
        )

        results = [item["path"] for item in file_items]
        search_root = raw_path or str(self._search_root)
        await self._emit_search_result(query, results, search_root, context.session_id)

        if not file_items:
            return SkillResult.ok(
                f"No files matching '{query}' found.",
                data={"results": [], "query": query, "total_found": 0},
                action_taken="search_files",
            )

        # Single confident match
        if len(file_items) == 1:
            best = file_items[0]
            return SkillResult.ok(
                f"Found '{best['name']}' at {best['path']} ({best['size_str']}).",
                data={
                    "results": results,
                    "file_path": best["path"],
                    "path": best["path"],
                    "name": best["name"],
                    "filename": best["name"],
                    "size": best["size"],
                    "mtime": best["mtime"],
                    "total_found": 1,
                },
                action_taken="search_files",
            )

        # Multiple matches — provide ranked/disambiguated list
        lines = [f"{i+1}. {item['name']} ({item['size_str']}) — {item['path']}" for i, item in enumerate(file_items[:5])]
        msg = f"I found {len(file_items)} matching file(s):\n" + "\n".join(lines)
        if len(file_items) > 5:
            msg += f"\n... and {len(file_items) - 5} more."

        best = file_items[0]
        return SkillResult.ok(
            msg,
            data={
                "results": results,
                "file_path": best["path"],
                "path": best["path"],
                "name": best["name"],
                "filename": best["name"],
                "candidates": file_items[:10],
                "total_found": len(file_items),
            },
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
        raw_path = self._clean_path_input(context.get("path", ""))
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

        # Resolve spoken names ("downloads", "desktop", etc.) to real paths.
        # If a full path was provided use it directly; otherwise treat the
        # raw_path value as a spoken shorthand.
        folder_path = self._resolve_parent_path(raw_path or None)

        if not folder_path.exists():
            return SkillResult.fail(
                f"I couldn't find the folder '{folder_path.name}'. "
                "Make sure it exists and try again."
            )
        if not folder_path.is_dir():
            return SkillResult.fail(
                f"'{folder_path.name}' is not a folder."
            )

        try:
            await asyncio.to_thread(
                self._run_explorer, str(folder_path), select_mode=False
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"I couldn't open the folder: {exc}", error=exc
            )

        await self._emit_folder_opened(str(folder_path), reveal_mode=False,
                                       session_id=context.session_id)
        return SkillResult.ok(
            f"Opened folder '{folder_path.name}'.",
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

    # ── Folder mutation actions (with post-execution verification) ──────────────────

    async def _create_folder(self, context: SkillContext) -> SkillResult:
        """
        Create a new folder, then VERIFY it exists before reporting success.

        The default parent is the user's Desktop so that natural commands like
        "create a new folder" place the folder somewhere visible.
        """
        folder_name: str = (context.get("folder_name") or context.get("name") or "").strip()
        if not folder_name:
            folder_name = "New Folder"  # default name when not specified

        raw_path = context.get("path")
        parent = self._resolve_parent_path(raw_path)

        folder_path = parent / folder_name

        # Check for existing folder
        if folder_path.exists():
            return SkillResult.fail(
                f"A folder named '{folder_name}' already exists at {parent}. "
                "Please choose a different name."
            )

        # Execute
        try:
            await asyncio.to_thread(folder_path.mkdir, True, False)
        except FileExistsError:
            return SkillResult.fail(
                f"A folder named '{folder_name}' already exists at {parent}."
            )
        except PermissionError:
            return SkillResult.fail(
                f"I don't have permission to create a folder in '{parent}'. "
                "Try a different location."
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"I couldn't create the folder '{folder_name}': {exc}", error=exc
            )

        # ── Post-execution verification ───────────────────────────────────
        if not (folder_path.exists() and folder_path.is_dir()):
            return SkillResult.fail(
                f"I attempted to create '{folder_name}' but verification failed — "
                "the folder was not found on disk after creation. Please try again."
            )

        log.info(
            "FileSkill: created folder '{name}' at '{path}'",
            name=folder_name, path=str(folder_path),
        )
        return SkillResult.ok(
            f"Created folder '{folder_name}' at {parent}.",
            data={"path": str(folder_path), "folder_name": folder_name,
                  "verified": True},
            action_taken="create_folder",
        )

    async def _rename_folder(self, context: SkillContext) -> SkillResult:
        """
        Rename a folder, then VERIFY the new name exists and the old name is gone.
        """
        old_name: str = (context.get("old_name") or "").strip()
        new_name: str = (context.get("new_name") or "").strip()

        if not old_name:
            return SkillResult.fail(
                "I need the current folder name to rename it. "
                "Try: 'rename folder <old name> to <new name>'."
            )
        if not new_name:
            return SkillResult.fail(
                "I need the new folder name. "
                "Try: 'rename folder <old name> to <new name>'."
            )

        raw_path = context.get("path")
        parent = self._resolve_parent_path(raw_path)

        old_path = parent / old_name
        new_path = parent / new_name

        if not old_path.exists():
            return SkillResult.fail(
                f"I couldn't find a folder named '{old_name}' at {parent}."
            )
        if not old_path.is_dir():
            return SkillResult.fail(
                f"'{old_name}' is not a folder."
            )
        if new_path.exists():
            return SkillResult.fail(
                f"A folder named '{new_name}' already exists at {parent}."
            )

        try:
            await asyncio.to_thread(old_path.rename, new_path)
        except PermissionError:
            return SkillResult.fail(
                f"I don't have permission to rename '{old_name}'. "
                "Close any applications using it and try again."
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"I couldn't rename '{old_name}' to '{new_name}': {exc}", error=exc
            )

        # ── Post-execution verification ───────────────────────────────────
        if not (new_path.exists() and new_path.is_dir()):
            return SkillResult.fail(
                f"I attempted to rename '{old_name}' to '{new_name}' but the new folder "
                "was not found on disk after the operation. Please try again."
            )
        if old_path.exists():
            return SkillResult.fail(
                f"The rename appeared to run but '{old_name}' still exists. "
                "The folder may be locked by another process."
            )

        log.info(
            "FileSkill: renamed '{old}' → '{new}' at '{path}'",
            old=old_name, new=new_name, path=str(parent),
        )
        return SkillResult.ok(
            f"Renamed folder '{old_name}' to '{new_name}'.",
            data={"old_path": str(old_path), "new_path": str(new_path),
                  "verified": True},
            action_taken="rename_folder",
        )

    async def _delete_folder(self, context: SkillContext) -> SkillResult:
        """
        Delete a folder, then VERIFY it no longer exists.

        By default sends to the Recycle Bin using winshell (safe/reversible).
        With force=True, permanently deletes using shutil.rmtree.
        """
        folder_name: str = (context.get("folder_name") or "").strip()
        if not folder_name:
            return SkillResult.fail(
                "I need the folder name to delete it. "
                "Try: 'delete folder <name>'."
            )

        force_raw = context.get("force", False)
        if isinstance(force_raw, str):
            force = force_raw.lower() not in ("false", "0", "no", "")
        else:
            force = bool(force_raw)

        raw_path = context.get("path")
        parent = self._resolve_parent_path(raw_path)
        folder_path = parent / folder_name

        if not folder_path.exists():
            return SkillResult.fail(
                f"I couldn't find a folder named '{folder_name}' at {parent}."
            )
        if not folder_path.is_dir():
            return SkillResult.fail(
                f"'{folder_name}' is not a folder."
            )

        try:
            if force:
                import shutil
                await asyncio.to_thread(shutil.rmtree, str(folder_path))
            else:
                await asyncio.to_thread(
                    self._send_to_recycle_bin, str(folder_path)
                )
        except PermissionError:
            return SkillResult.fail(
                f"I don't have permission to delete '{folder_name}'. "
                "Close any applications using it and try again."
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"I couldn't delete '{folder_name}': {exc}", error=exc
            )

        # ── Post-execution verification ───────────────────────────────────
        if folder_path.exists():
            return SkillResult.fail(
                f"I attempted to delete '{folder_name}' but it still exists on disk. "
                "The folder may be in use by another application."
            )

        action_word = "Permanently deleted" if force else "Moved to Recycle Bin"
        log.info(
            "FileSkill: deleted folder '{name}' at '{path}' (force={f})",
            name=folder_name, path=str(parent), f=force,
        )
        return SkillResult.ok(
            f"{action_word} folder '{folder_name}'.",
            data={"path": str(folder_path), "folder_name": folder_name,
                  "verified": True, "permanent": force},
            action_taken="delete_folder",
        )


    # ── Internal helpers ──────────────────────────────────────────────────

    def _resolve_parent_path(self, raw_path: str | None) -> Path:
        """
        Resolve the parent directory for folder operations.

        When raw_path is None or a spoken shorthand ("desktop", "downloads"),
        map it to the corresponding real path.  Default is the Desktop.

        Mapping
        -------
        None / "" / "desktop"   → ~/Desktop
        "downloads"             → ~/Downloads
        "documents"             → ~/Documents
        "pictures"              → ~/Pictures
        "music"                 → ~/Music
        "videos"                → ~/Videos
        Any other value         → Path(raw_path).expanduser()
        """
        if not raw_path:
            return Path.home() / "Desktop"

        normalized = raw_path.strip().lower()
        _SPOKEN_PATHS: dict[str, Path] = {
            "desktop":   Path.home() / "Desktop",
            "downloads": Path.home() / "Downloads",
            "documents": Path.home() / "Documents",
            "pictures":  Path.home() / "Pictures",
            "music":     Path.home() / "Music",
            "videos":    Path.home() / "Videos",
            "home":      Path.home(),
        }
        if normalized in _SPOKEN_PATHS:
            return _SPOKEN_PATHS[normalized]

        return Path(raw_path).expanduser()

    @staticmethod
    def _send_to_recycle_bin(path: str) -> None:
        """
        Send a path to the Windows Recycle Bin.

        Tries winshell first; falls back to SHFileOperationW via ctypes.
        """
        try:
            import winshell  # type: ignore[import-untyped]
            winshell.delete_file(path, no_confirm=True, allow_undo=True)
            return
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

        # Fallback: SHFileOperationW (FOF_ALLOWUNDO = 0x40, FOF_NOCONFIRMATION = 0x10)
        try:
            import ctypes
            from ctypes.wintypes import HWND, UINT, LPCWSTR, DWORD, BOOL
            class SHFILEOPSTRUCTW(ctypes.Structure):
                _fields_ = [
                    ("hwnd", HWND),
                    ("wFunc", UINT),
                    ("pFrom", LPCWSTR),
                    ("pTo", LPCWSTR),
                    ("fFlags", DWORD),
                    ("fAnyOperationsAborted", BOOL),
                    ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", LPCWSTR),
                ]
            FO_DELETE = 3
            FOF_ALLOWUNDO = 0x40
            FOF_NOCONFIRMATION = 0x10
            FOF_SILENT = 0x4
            op = SHFILEOPSTRUCTW()
            op.wFunc = FO_DELETE
            op.pFrom = path + "\0\0"
            op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
            ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        except Exception:  # noqa: BLE001
            # Ultimate fallback: permanent delete
            import shutil
            shutil.rmtree(path, ignore_errors=True)

    @classmethod
    def _smart_file_search(
        cls,
        query: str,
        raw_path: str | None,
        extensions_raw: str,
        max_results: int,
        default_root: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Intelligent multi-location file search with metadata extraction,
        size ranking, time filtering, and prioritized folders.
        """
        import os
        import re

        q_lower = query.lower().strip()

        # Parse target roots
        search_roots: list[Path] = []
        if raw_path:
            p = Path(raw_path).expanduser()
            if p.exists():
                search_roots.append(p)
        else:
            # Check if query targets a specific directory
            if "download" in q_lower:
                dl = Path.home() / "Downloads"
                if dl.exists():
                    search_roots.append(dl)
            elif "desktop" in q_lower:
                dt = Path.home() / "Desktop"
                if dt.exists():
                    search_roots.append(dt)
                od_dt = Path.home() / "OneDrive" / "Desktop"
                if od_dt.exists() and od_dt not in search_roots:
                    search_roots.append(od_dt)
            elif "document" in q_lower:
                dc = Path.home() / "Documents"
                if dc.exists():
                    search_roots.append(dc)
                od_dc = Path.home() / "OneDrive" / "Documents"
                if od_dc.exists() and od_dc not in search_roots:
                    search_roots.append(od_dc)
            else:
                # Prioritized default search locations (fast & local first)
                candidates = [
                    Path(os.getcwd()),
                    Path.home() / "Desktop",
                    Path.home() / "OneDrive" / "Desktop",
                    Path.home() / "Downloads",
                    Path.home() / "Documents",
                    Path.home() / "OneDrive" / "Documents",
                    Path.home() / "OneDrive",
                ]
                for c in candidates:
                    if c.exists() and c not in search_roots:
                        search_roots.append(c)

        # Include custom configured search root if provided
        if default_root:
            dr = Path(default_root).expanduser()
            if dr.exists():
                if dr not in search_roots:
                    search_roots.insert(0, dr)
                for sub_dir in ("Downloads", "Desktop", "Documents"):
                    sdr = dr / sub_dir
                    if sdr.exists() and sdr not in search_roots:
                        search_roots.insert(0, sdr)

        # Parse extension filter
        extensions: list[str] = []
        if extensions_raw:
            extensions = [f".{e.strip().lstrip('.')}".lower() for e in extensions_raw.split(",") if e.strip()]
        else:
            # Infer extension from query
            if any(k in q_lower for k in ("resume", "cv", "curriculum vitae", "biodata")):
                extensions.extend([".pdf", ".docx", ".doc", ".txt", ".md"])
            elif "pdf" in q_lower:
                extensions.append(".pdf")
            elif "python" in q_lower or ".py" in q_lower:
                extensions.append(".py")
            elif "csv" in q_lower:
                extensions.append(".csv")
            elif "text" in q_lower or "txt" in q_lower:
                extensions.append(".txt")
            elif "screenshot" in q_lower or "png" in q_lower or "image" in q_lower:
                extensions.extend([".png", ".jpg", ".jpeg", ".bmp"])

        # Extract core search keyword (stripping search noise)
        clean_q = q_lower
        for noise in (
            "find my", "find the", "find all", "find files", "find file", "find",
            "search for", "search all", "search", "look for", "get my",
            "in downloads", "in desktop", "in documents", "in my downloads",
            "largest file", "latest file", "newest file", "largest", "latest", "newest",
            "pdfs", "pdf files", "python files", "python file",
        ):
            clean_q = clean_q.replace(noise, " ")
        clean_q = clean_q.strip()
        if not clean_q and extensions:
            clean_q = "*"

        # Perform search across candidate roots
        items: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for root in search_roots:
            try:
                # Search shallow depth first for fast response
                for p in root.rglob("*"):
                    if len(items) >= max_results * 2:
                        break
                    try:
                        if not p.is_file():
                            continue
                        str_path = str(p)
                        if str_path in seen_paths:
                            continue

                        # Check extension
                        if extensions and p.suffix.lower() not in extensions:
                            continue

                        # Check name match
                        if clean_q and clean_q != "*":
                            if clean_q in ("resume", "cv", "my resume", "my cv", "curriculum vitae", "biodata"):
                                if not any(k in p.name.lower() for k in ("resume", "cv", "curriculum", "vitae", "biodata")):
                                    continue
                            else:
                                pattern = f"*{clean_q}*" if not any(c in clean_q for c in ("*", "?")) else clean_q
                                if not fnmatch.fnmatch(p.name.lower(), pattern.lower()):
                                    continue

                        stat = p.stat()
                        size = stat.st_size
                        mtime = stat.st_mtime

                        seen_paths.add(str_path)
                        items.append({
                            "path": str_path,
                            "name": p.name,
                            "size": size,
                            "size_str": cls._format_size(size),
                            "mtime": mtime,
                        })
                    except (PermissionError, OSError):
                        continue
            except (PermissionError, OSError):
                continue

        # Sort based on criteria
        if "largest" in q_lower or "biggest" in q_lower:
            items.sort(key=lambda x: x["size"], reverse=True)
        elif "latest" in q_lower or "newest" in q_lower or "yesterday" in q_lower or "today" in q_lower or "screenshot" in q_lower:
            items.sort(key=lambda x: x["mtime"], reverse=True)
        else:
            # Default sort: exact match first, then most recently modified
            def _rank(item: dict[str, Any]) -> tuple[int, float]:
                exact = 0 if clean_q and clean_q.lower() == item["name"].lower() else 1
                return exact, -item["mtime"]
            items.sort(key=_rank)

        return items[:max_results]

    @staticmethod
    def _format_size(bytes_val: int) -> str:
        """Format byte size to human readable string."""
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f} KB"
        elif bytes_val < 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.1f} MB"
        return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"

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
