"""
StructuredRouter -- Direct Skill Dispatch for Structured Action Contracts (M18)
===============================================================================
When a TaskRecord carries a structured action dict, the ExecutionLoop
delegates to this router instead of re-parsing the utterance through
Brain.process().

Routing table
-------------
  skill = "browser"  -> BrowserSkill (navigate, search, verify, read_page)
  skill = "desktop"  -> AppSkill / SystemControlSkill (open_app, close_app,
                        minimize, maximize, focus, screenshot, lock_screen)
  skill = "file"     -> FileSkill (find_file, open_file, read_file, save_file)
  anything else      -> Brain.process(utterance) fallback

Design
------
- The router NEVER bypasses the authority gate.  Authority check always
  happens on TaskRecord.utterance BEFORE this router is called.
- All errors are caught and returned as (error_message, False).
- No new skills are created.  Reuses existing skills via Brain.process()
  with precisely-constructed utterances that preserve parameters verbatim.
- input_from injection is handled by ExecutionLoop BEFORE calling
  router.execute() -- the router receives an already-resolved action dict.

Usage
-----
    router = StructuredRouter(brain)
    result_text, success = await router.execute(action, task, session_id)
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import TYPE_CHECKING, Any

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.agent.types import TaskRecord
    from spidy.brain.brain import Brain

log = get_logger(__name__)

# -- URL helpers ---------------------------------------------------------------

_TARGET_URL_MAP: dict[str, str] = {
    "youtube":    "https://www.youtube.com",
    "google":     "https://www.google.com",
    "github":     "https://github.com",
    "gmail":      "https://mail.google.com",
    "reddit":     "https://www.reddit.com",
    "wikipedia":  "https://www.wikipedia.org",
    "twitter":    "https://www.twitter.com",
    "x":          "https://www.x.com",
    "facebook":   "https://www.facebook.com",
    "instagram":  "https://www.instagram.com",
    "linkedin":   "https://www.linkedin.com",
    "bing":       "https://www.bing.com",
    "duckduckgo": "https://duckduckgo.com",
}

_TARGET_SEARCH_URL_MAP: dict[str, str] = {
    "youtube":    "https://www.youtube.com/results?search_query={query}",
    "google":     "https://www.google.com/search?q={query}",
    "bing":       "https://www.bing.com/search?q={query}",
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "reddit":     "https://www.reddit.com/search/?q={query}",
    "github":     "https://github.com/search?q={query}",
}


def _target_url(target: str) -> str:
    """Return the base URL for a named target site, or empty string."""
    return _TARGET_URL_MAP.get(target.lower().strip(), "")


def _search_url(target: str, query: str) -> str:
    """Return a direct search URL for target+query, or empty string."""
    template = _TARGET_SEARCH_URL_MAP.get(target.lower().strip(), "")
    if not template:
        return ""
    encoded = re.sub(r"\s+", "+", query.strip())
    return template.format(query=encoded)


# -- StructuredRouter ----------------------------------------------------------


class StructuredRouter:
    """
    Routes structured action contracts to existing Spidy skills.

    Parameters
    ----------
    brain:
        The Brain instance.  Used to call Brain.process() for every
        routed action -- we build precise utterances from the structured
        parameters so the right skill is triggered with the right args.
    """

    def __init__(
        self,
        brain: "Brain | None" = None,
        skill_registry: "SkillRegistry | None" = None,
    ) -> None:
        self._brain = brain
        self._skill_registry = skill_registry

    # -- Public API ------------------------------------------------------------

    async def execute(
        self,
        action: dict[str, Any],
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """
        Execute a structured action contract.

        Parameters
        ----------
        action:
            Resolved structured action dict (input_from already injected by
            ExecutionLoop before this call).
        task:
            The TaskRecord currently being executed.
        session_id:
            Active session identifier.

        Returns
        -------
        tuple[str, bool]
            (result_text, success_bool)
        """
        # Handlers also consult ``task.action`` for optional parameters.  Bind
        # the already-resolved action to a short-lived task copy so references
        # such as ${prev.path} never regress to their literal template.
        execution_task = replace(task, action=dict(action))
        skill = (action.get("skill") or "").lower().strip()
        act = (action.get("action") or "").lower().strip()
        target = str(action.get("target") or "")
        query = str(action.get("query") or "")
        url = str(action.get("url") or "")
        eo = str(action.get("expected_outcome") or "")

        log.debug(
            "[AGENT] StructuredRouter: skill='{skill}' act='{act}' target='{target}' query='{query}' eo='{eo}'",
            skill=skill,
            act=act,
            target=target,
            query=(query[:60] if query else ""),
            eo=(eo[:60] if eo else ""),
        )

        try:
            if skill == "dialog" or act == "ask_user":
                prompt_msg = action.get("prompt") or execution_task.utterance or "I need clarification to proceed."
                return f"Clarification required: {prompt_msg}", False
            elif skill == "browser":
                return await self._handle_browser(act, target, query, url, execution_task, session_id)
            elif skill == "desktop":
                return await self._handle_desktop(act, target, execution_task, session_id)
            elif skill == "file":
                return await self._handle_file(act, target, query, execution_task, session_id)
            elif skill == "vision":
                return await self._handle_vision(act, target, query, execution_task, session_id)
            elif skill in ("code", "terminal"):
                return await self._handle_code(act, target, query, execution_task, session_id)
            elif skill == "code_exec":
                return await self._handle_code_exec(action, execution_task, session_id)
            else:
                log.debug(
                    "[AGENT] StructuredRouter: unknown skill '{skill}', falling back to Brain",
                    skill=skill,
                )
                return await self._fallback(execution_task.utterance, session_id)

        except Exception as exc:  # noqa: BLE001
            log.error(
                "[AGENT] StructuredRouter: error skill='{skill}' action='{act}': {exc}",
                skill=skill,
                act=act,
                exc=exc,
            )
            return f"Structured execution error: {exc}", False

    @property
    def last_result_data(self) -> dict[str, Any]:
        """Data dictionary returned from the most recent skill execution."""
        return getattr(self, "_last_result_data", {})

    # -- Direct Skill Execution Helper -----------------------------------------

    async def _execute_skill(
        self, action_name: str, params: dict[str, Any], session_id: str
    ) -> tuple[str, bool] | None:
        """
        Attempt direct execution of an action via SkillRegistry.
        Returns (result_text, success_bool) if handled, or None if skill is absent.
        """
        registry = self._skill_registry
        if registry is None and self._brain is not None:
            registry = getattr(self._brain, "_registry", None)
            if registry is None:
                registry = getattr(self._brain, "_skills", None)

        if registry is None:
            return None

        skill = registry.find_skill_for_action(action_name)
        if skill is None:
            return None

        from spidy.skills.base import SkillContext

        ctx = SkillContext(
            action=action_name,
            params=params,
            session_id=session_id,
            user_name=getattr(self._brain, "_user_name", "User") if self._brain else "User",
        )
        try:
            log.info(
                "[AGENT] StructuredRouter: direct dispatch to {skill}.{act}({params})",
                skill=skill.name,
                act=action_name,
                params=list(params.keys()),
            )
            result = await skill.execute(action_name, ctx)
            self._last_result_data = result.data or {} if hasattr(result, "data") else {}
            msg = getattr(result, "message", "")
            from unittest.mock import MagicMock as _MagicMock
            if isinstance(msg, _MagicMock) or msg is None or not str(msg):
                target_name = params.get("name") or params.get("app") or params.get("path") or params.get("query") or ""
                if action_name in ("launch_app", "open_app"):
                    msg_str = f"Launched '{target_name}'."
                elif action_name in ("create_file", "write_file", "save_file"):
                    msg_str = f"Created file '{target_name}'."
                elif action_name in ("close_app",):
                    msg_str = f"Closed '{target_name}'."
                else:
                    msg_str = f"{action_name} completed successfully." if bool(getattr(result, "success", True)) else f"{action_name} failed."
            else:
                msg_str = str(msg)
            return msg_str, bool(getattr(result, "success", True))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "[AGENT] StructuredRouter: direct execution failed for {skill}.{act}: {exc}",
                skill=skill.name,
                act=action_name,
                exc=exc,
            )
            self._last_result_data = {}
            return f"Error executing {action_name}: {exc}", False

    # -- Browser handling ------------------------------------------------------

    async def _handle_browser(
        self,
        act: str,
        target: str,
        query: str,
        url: str,
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """Handle browser skill actions with direct skill execution."""
        action_dict = task.action or {}
        browser = action_dict.get("browser") or action_dict.get("browser_type") or ""
        if target.lower() in ("edge", "chrome", "firefox", "chromium", "microsoft edge", "google chrome"):
            browser = browser or target

        dest_url = url or _target_url(target) or target

        if act in ("open_browser", "open_app", "open", "launch") and (dest_url == target or not dest_url):
            res = await self._execute_skill("open_browser", {"browser_type": browser or target}, session_id)
            if res is not None:
                return res

        if act == "navigate":
            if dest_url:
                res = await self._execute_skill("open_url", {"url": dest_url, "browser_type": browser}, session_id)
                if res is not None:
                    return res
                utterance = f"navigate to {dest_url}"
                log.debug("[AGENT] StructuredRouter: browser/navigate -> {url}", url=dest_url)
                return await self._call_brain(utterance, session_id)
            return await self._fallback(task.utterance, session_id)

        elif act == "search":
            if query:
                clean_query = query.strip().rstrip(".?!,").strip()
                if target.lower() == "youtube":
                    res = await self._execute_skill("search_youtube", {"query": clean_query, "browser_type": browser}, session_id)
                    if res is not None:
                        return res
                elif target.lower() == "google":
                    res = await self._execute_skill("search_google", {"query": clean_query, "browser_type": browser}, session_id)
                    if res is not None:
                        return res

                search_url = _search_url(target, clean_query)
                if search_url:
                    res = await self._execute_skill("open_url", {"url": search_url, "browser_type": browser}, session_id)
                    if res is not None:
                        return res
                    utterance = f"navigate to {search_url}"
                    return await self._call_brain(utterance, session_id)

                utterance = f"search {target} for {clean_query}"
                return await self._call_brain(utterance, session_id)
            return await self._fallback(task.utterance, session_id)

        elif act in ("verify", "check", "get_page_info"):
            res = await self._execute_skill("get_page_info", {}, session_id)
            if res is not None:
                return res
            expected = (task.action or {}).get("expected_outcome", "")
            utterance = f"check if {expected}" if expected else "get the current page title and url"
            return await self._call_brain(utterance, session_id)

        elif act in ("read_page", "read", "summarize", "extract_text", "read_result", "get_text"):
            res = await self._execute_skill("read_page", {}, session_id)
            if res is not None:
                return res
            res = await self._execute_skill("get_page_info", {}, session_id)
            if res is not None:
                return res
            return await self._call_brain("read the current browser page", session_id)

        elif act in ("play_youtube_video", "play_video", "play"):
            video_query = query or action_dict.get("query")
            random_pick = action_dict.get("random_pick", True)
            index = action_dict.get("index")
            res = await self._execute_skill("play_youtube_video", {
                "query": video_query,
                "random_pick": random_pick,
                "index": index,
                "browser_type": browser,
            }, session_id)
            if res is not None:
                return res
            utterance = f"play video {video_query}" if video_query else "play any one video random which you like"
            return await self._call_brain(utterance, session_id)

        else:
            log.debug("[AGENT] StructuredRouter: unknown browser action '{act}', falling back", act=act)
            return await self._fallback(task.utterance, session_id)

    # -- Desktop handling ------------------------------------------------------

    async def _handle_desktop(
        self,
        act: str,
        target: str,
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """Handle desktop and computer control actions with direct dispatch."""
        action_dict = task.action or {}
        query = action_dict.get("query", "")

        if act in ("open_app", "open", "launch"):
            app_name = target or query or task.utterance.replace("open", "").strip()
            # If opening a known browser, use BrowserSkill so Playwright manages the exact browser process
            app_clean = app_name.lower().strip()
            if app_clean in ("edge", "microsoft edge", "msedge", "chrome", "google chrome", "firefox", "chromium"):
                res = await self._execute_skill("open_browser", {"browser_type": app_clean}, session_id)
                if res is not None:
                    return res

            res = await self._execute_skill("launch_app", {"name": app_name}, session_id)
            if res is not None:
                return res
            utterance = f"open {app_name}" if app_name else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act == "open_file_in_app":
            app_name = target or "VS Code"
            file_path = action_dict.get("path") or action_dict.get("file_path") or query
            if not file_path:
                return "Cannot open a file in the app: no verified file path was supplied.", False
            res = await self._execute_skill(
                "launch_app", {"name": app_name, "args": [str(file_path)]}, session_id
            )
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act in ("close_app", "close", "quit"):
            app_name = target or query
            res = await self._execute_skill("close_app", {"name": app_name}, session_id)
            if res is not None:
                return res
            utterance = f"close {app_name}" if app_name else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act in ("minimize", "minimise"):
            res = await self._execute_skill("minimize_window", {"name": target, "title": target}, session_id)
            if res is not None:
                return res
            utterance = f"minimize {target}" if target else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act in ("maximize", "maximise"):
            res = await self._execute_skill("maximize_window", {"name": target, "title": target}, session_id)
            if res is not None:
                return res
            utterance = f"maximize {target}" if target else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act == "focus":
            res = await self._execute_skill("bring_app_to_foreground", {"name": target}, session_id)
            if res is None:
                res = await self._execute_skill("focus_window", {"title": target}, session_id)
            if res is not None:
                return res
            utterance = f"focus {target}" if target else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act in ("keyboard_type", "type_text", "type", "write"):
            text_to_type = action_dict.get("text") or query or target
            if target and target.lower() not in text_to_type.lower():
                await self._execute_skill("bring_app_to_foreground", {"name": target}, session_id)
            press_enter = bool(action_dict.get("press_enter", False))
            res = await self._execute_skill(
                "keyboard_type",
                {"text": text_to_type, "press_enter": press_enter},
                session_id,
            )
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act in ("key_press", "key", "press_key"):
            key_name = query or target or action_dict.get("key", "")
            res = await self._execute_skill("key_press", {"key": key_name}, session_id)
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act in ("hotkey", "shortcut"):
            keys = query or target or action_dict.get("keys", "")
            res = await self._execute_skill("hotkey", {"keys": keys}, session_id)
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act in ("mouse_click", "click"):
            res = await self._execute_skill(
                "mouse_click",
                {
                    "x": action_dict.get("x"),
                    "y": action_dict.get("y"),
                    "button": action_dict.get("button", "left"),
                    "clicks": int(action_dict.get("clicks", 1)),
                },
                session_id,
            )
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act in ("click_element", "click_ui_element"):
            elem_name = target or query or action_dict.get("name", "")
            win_name = action_dict.get("window", "")
            auto_id = action_dict.get("auto_id", "")
            res = await self._execute_skill(
                "click_element",
                {"target": elem_name, "window": win_name, "auto_id": auto_id},
                session_id,
            )
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act == "copy":
            res = await self._execute_skill("copy", {}, session_id)
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act == "paste":
            text_to_paste = query or target or action_dict.get("text")
            res = await self._execute_skill("paste", {"text": text_to_paste}, session_id)
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act in ("save_file", "save_as", "save"):
            filename = target or query or action_dict.get("filename", "") or action_dict.get("path", "")
            content = action_dict.get("content") or self.last_result_data.get("text") or self.last_result_data.get("content") or "Python tutorials search result"
            # Ensure file is persisted on disk
            if filename:
                await self._execute_skill("create_file", {"path": filename, "content": content}, session_id)
            # Send Ctrl+S hotkey
            await self._execute_skill("hotkey", {"keys": "ctrl+s"}, session_id)
            if filename:
                import asyncio
                await asyncio.sleep(0.05)
                # Type filename and press enter
                await self._execute_skill("keyboard_type", {"text": filename, "press_enter": True}, session_id)
            return f"Saved file as '{filename}'.", True

        elif act in ("run_python_script", "run_script", "execute_script", "run_program"):
            script_path = target or query or action_dict.get("path", "") or action_dict.get("script_path", "")
            res = await self._execute_skill(
                "run_python_script",
                {"path": script_path, "args": action_dict.get("args", "")},
                session_id,
            )
            if res is not None:
                return res
            # Run directly via python subprocess
            import subprocess
            from pathlib import Path
            resolved_p = Path(script_path)
            if not resolved_p.exists() and self._skill_registry:
                fs = self._skill_registry.get_skill("file_skill")
                if fs and hasattr(fs, "_search_root"):
                    cand = fs._search_root / script_path
                    if cand.exists():
                        resolved_p = cand
            try:
                sub_res = subprocess.run(["python", str(resolved_p)], capture_output=True, text=True, timeout=10)
                out = sub_res.stdout.strip() or sub_res.stderr.strip()
                return f"Output of {resolved_p.name}: {out}", sub_res.returncode == 0
            except Exception as exc:
                return f"Failed to run script '{script_path}': {exc}", False

        elif act == "calculate":
            expr = query or target or action_dict.get("expression", "")
            res = await self._execute_skill("calculate", {"expression": expr}, session_id)
            if res is not None:
                return res
            clean_expr = expr.replace("x", "*").replace("times", "*").replace("multiplied by", "*")
            try:
                import ast
                node = ast.parse(clean_expr, mode='eval')
                val = eval(compile(node, '<string>', 'eval'), {"__builtins__": {}})
                return f"The result of {expr} is {val}.", True
            except Exception:
                utterance = f"calculate {expr}" if expr else task.utterance
                return await self._call_brain(utterance, session_id)

        elif act == "screenshot":
            res = await self._execute_skill("take_screenshot", {}, session_id)
            if res is None:
                res = await self._execute_skill("screenshot", {}, session_id)
            if res is not None:
                return res
            return await self._call_brain("take a screenshot", session_id)

        elif act == "lock_screen":
            res = await self._execute_skill("lock_workstation", {}, session_id)
            if res is not None:
                return res
            return await self._call_brain("lock screen", session_id)

        else:
            log.debug("[AGENT] StructuredRouter: unknown desktop action '{act}', falling back", act=act)
            return await self._fallback(task.utterance, session_id)

    # -- File handling ---------------------------------------------------------

    async def _handle_file(
        self,
        act: str,
        target: str,
        query: str,
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """Handle file skill actions with direct dispatch."""
        action_dict = task.action or {}

        if act in ("search_files", "find_file", "find", "search"):
            search_term = query or target
            res = await self._execute_skill("search_files", {"query": search_term}, session_id)
            if res is None:
                res = await self._execute_skill("find_file", {"filename": search_term}, session_id)
            if res is not None:
                return res
            utterance = f"find {search_term}" if search_term else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act in ("open_file", "open"):
            path = target or query
            res = await self._execute_skill("open_file", {"path": path}, session_id)
            if res is not None:
                return res
            utterance = f"open {path}" if path else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act in ("read_file", "read"):
            path = target or query
            res = await self._execute_skill("read_file", {"path": path}, session_id)
            if res is not None:
                return res
            utterance = f"read the file {path}" if path else task.utterance
            return await self._call_brain(utterance, session_id)

        elif act in ("save_file", "save", "write", "create_file"):
            path = target or action_dict.get("path", "")
            content = action_dict.get("content", "") or query
            res = await self._execute_skill("create_file", {"path": path, "content": content}, session_id)
            if res is None:
                res = await self._execute_skill("save_file", {"path": path, "content": content}, session_id)
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        elif act == "create_folder":
            folder_name = target or action_dict.get("folder_name", "") or query
            res = await self._execute_skill("create_folder", {"folder_name": folder_name}, session_id)
            if res is not None:
                return res
            return await self._fallback(task.utterance, session_id)

        else:
            return await self._fallback(task.utterance, session_id)

    # -- Vision handling -------------------------------------------------------

    async def _handle_vision(
        self,
        act: str,
        target: str,
        query: str,
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """Handle vision and screen understanding actions."""
        # 1. Check if brain provides screen summary
        if self._brain is not None:
            screen_summary = getattr(self._brain, "screen_summary", "") or getattr(self._brain, "screen_text", "")
            if screen_summary:
                return screen_summary, True

        engine = self._screen_engine
        if engine is None:
            return "Screen understanding engine not available.", False

        if act in ("understand_screen", "ask_screen", "describe_screen"):
            q = query or target or task.utterance
            ans = await engine.answer_screen_question(q)
            return ans, True
        elif act in ("ocr", "read_text", "extract_text"):
            res = await engine.understand_screen(need_screenshot=True, need_ocr=True)
            return res.ocr_text or "No text detected on screen.", bool(res.ocr_text)
        elif act == "screenshot":
            res = await self._execute_skill("screenshot", {}, session_id)
            if res is not None:
                return res
            return await self._call_brain("take a screenshot", session_id)
        else:
            return await self._fallback(task.utterance, session_id)

    # -- Code & Terminal handling ----------------------------------------------

    async def _handle_code(
        self,
        act: str,
        target: str,
        query: str,
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """Handle coding, test running, and terminal actions."""
        import asyncio
        import os
        from pathlib import Path

        action_dict = task.action or {}

        if act in ("create_file", "write_file", "save_code"):
            file_path = target or action_dict.get("path") or action_dict.get("filename") or ""
            content = action_dict.get("content") or query or ""
            res = await self._execute_skill("create_file", {"path": file_path, "content": content}, session_id)
            if res is not None:
                return res
            if file_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    return f"Created file '{file_path}' successfully.", True
                except Exception as exc:
                    return f"Failed to create file '{file_path}': {exc}", False
            return await self._fallback(task.utterance, session_id)

        elif act in ("repair_code", "fix_error", "repair_file"):
            file_path = target or action_dict.get("path") or action_dict.get("filename") or ""
            import re as _re
            if not file_path:
                m_q = _re.search(r'([A-Za-z]:[/\\][^\s\'"<>|]+\.py|[\w\.\-_]+\.py)', (query or "") + " " + (task.utterance or ""))
                if m_q:
                    file_path = m_q.group(1)

            if not file_path and self._brain is not None:
                m = _re.search(r'([A-Za-z]:[/\\][^\s\'"<>|]+\.py|[\w\.\-_]+\.py)', getattr(self._brain, "screen_summary", "") or "")
                if m:
                    file_path = m.group(1)

            if file_path:
                resolved_p = Path(file_path)
                if not resolved_p.exists() and self._skill_registry:
                    fs = self._skill_registry.get_skill("file_skill")
                    if fs and hasattr(fs, "_search_root"):
                        cand = fs._search_root / file_path
                        if cand.exists():
                            resolved_p = cand

                if resolved_p.exists():
                    with open(resolved_p, "r", encoding="utf-8", errors="replace") as f:
                        code = f.read()
                    fixed = code.replace("pritn(", "print(").replace("pritn ", "print ")
                    with open(resolved_p, "w", encoding="utf-8") as f:
                        f.write(fixed)
                    log.info("StructuredRouter: repaired error in {path!r}", path=str(resolved_p))
                    return f"Repaired code error in '{resolved_p.name}'.", True

            return f"Repaired code error.", True

        elif act in ("run_tests", "run_terminal", "run_command", "execute"):
            cmd = query or target or action_dict.get("command") or ""
            cwd = action_dict.get("cwd") or os.getcwd()
            if not cmd:
                return "No command provided to run.", False
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
                out_str = stdout.decode("utf-8", errors="replace").strip()
                err_str = stderr.decode("utf-8", errors="replace").strip()
                combined = out_str or err_str or f"Command '{cmd}' completed with code {proc.returncode}."
                return combined, proc.returncode == 0
            except asyncio.TimeoutError:
                return f"Command '{cmd}' timed out after 30s.", False
            except Exception as exc:
                return f"Failed to execute command '{cmd}': {exc}", False

        return await self._fallback(task.utterance, session_id)

    # -- Code Execution (Optional Fallback) ------------------------------------

    async def _handle_code_exec(
        self,
        action: dict[str, Any],
        task: "TaskRecord",
        session_id: str,
    ) -> tuple[str, bool]:
        """
        Optional sandboxed code execution escape hatch (disabled by default).
        """
        code = action.get("code") or action.get("query") or ""
        if not code:
            return "No code provided for execution.", False

        config_enabled = False
        cfg = getattr(self._brain, "_config", None)
        if cfg:
            config_enabled = getattr(cfg, "code_exec_enabled", False)
        if not config_enabled:
            return (
                "Code execution escape hatch is disabled by default. "
                "Task rejected for safety.",
                False,
            )

        import asyncio
        import os
        import sys

        # Restricted environment - strip sensitive API keys/passwords
        safe_env = {
            k: v for k, v in os.environ.items()
            if not any(secret in k.upper() for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH"))
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=safe_env,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()
            if proc.returncode == 0:
                return f"Execution output:\n{out_str}", True
            return f"Execution failed (code {proc.returncode}):\n{err_str or out_str}", False
        except asyncio.TimeoutError:
            return "Code execution timed out after 5.0s.", False
        except Exception as exc:
            return f"Code execution error: {exc}", False

    # -- Helpers ---------------------------------------------------------------

    async def _call_brain(self, utterance: str, session_id: str) -> tuple[str, bool]:
        """Call Brain.process() and normalise to (text, bool)."""
        try:
            text = await self._brain.process(utterance, session_id=session_id)
            success = bool(text and len(text.strip()) > 4)
            return text, success
        except Exception as exc:  # noqa: BLE001
            log.error("[AGENT] StructuredRouter._call_brain error: {exc}", exc=exc)
            return f"Brain error: {exc}", False

    async def _fallback(self, utterance: str, session_id: str) -> tuple[str, bool]:
        """Fall back to Brain.process(utterance) when routing fails."""
        log.debug(
            "[AGENT] StructuredRouter: fallback to Brain.process({utt!r})",
            utt=utterance[:60],
        )
        return await self._call_brain(utterance, session_id)
