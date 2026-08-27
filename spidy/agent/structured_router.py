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

    def __init__(self, brain: "Brain") -> None:
        self._brain = brain

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
            The original TaskRecord (utterance used as fallback).
        session_id:
            Brain session ID.

        Returns
        -------
        tuple[str, bool]
            (result_text, success_bool) -- identical interface to Brain.process().
        """
        skill  = (action.get("skill")  or "").lower().strip()
        act    = (action.get("action") or "").lower().strip()
        target = (action.get("target") or "").lower().strip()
        query  = action.get("query", "")
        url    = action.get("url", "")
        eo     = action.get("expected_outcome", "")

        log.info(
            "[AGENT] StructuredRouter: skill=%r action=%r target=%r query=%r eo=%r",
            skill, act, target, (query[:60] if query else ""), (eo[:60] if eo else ""),
        )

        try:
            if skill == "dialog" or act == "ask_user":
                prompt_msg = action.get("prompt") or task.utterance or "I need clarification to proceed."
                return f"Clarification required: {prompt_msg}", False
            elif skill == "browser":
                return await self._handle_browser(act, target, query, url, task, session_id)
            elif skill == "desktop":
                return await self._handle_desktop(act, target, task, session_id)
            elif skill == "file":
                return await self._handle_file(act, target, query, task, session_id)
            elif skill == "code_exec":
                return await self._handle_code_exec(action, task, session_id)
            else:
                log.debug(
                    "[AGENT] StructuredRouter: unknown skill %r, falling back to Brain",
                    skill,
                )
                return await self._fallback(task.utterance, session_id)

        except Exception as exc:  # noqa: BLE001
            log.error(
                "[AGENT] StructuredRouter: error skill=%r action=%r: %s",
                skill, act, exc,
            )
            return f"Structured execution error: {exc}", False

    # -- Direct Skill Execution Helper -----------------------------------------

    async def _execute_skill(
        self, action_name: str, params: dict[str, Any], session_id: str
    ) -> tuple[str, bool] | None:
        """
        Attempt direct execution of an action via SkillRegistry.
        Returns (result_text, success_bool) if handled, or None if skill is absent.
        """
        registry = getattr(self._brain, "_registry", None)
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
            user_name=getattr(self._brain, "_user_name", "User"),
        )
        try:
            log.info(
                "[AGENT] StructuredRouter: direct dispatch to %s.%s(%s)",
                skill.name,
                action_name,
                list(params.keys()),
            )
            result = await skill.execute(action_name, ctx)
            return result.message, result.success
        except Exception as exc:  # noqa: BLE001
            log.error(
                "[AGENT] StructuredRouter: direct execution failed for %s.%s: %s",
                skill.name,
                action_name,
                exc,
            )
            return None

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
                log.debug("[AGENT] StructuredRouter: browser/navigate -> %r", dest_url)
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

        else:
            log.debug("[AGENT] StructuredRouter: unknown browser action %r, falling back", act)
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
            # Send Ctrl+S hotkey
            await self._execute_skill("hotkey", {"keys": "ctrl+s"}, session_id)
            if filename:
                import asyncio
                await asyncio.sleep(0.3)
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
            return await self._fallback(task.utterance, session_id)

        elif act == "calculate":
            expr = query or target or action_dict.get("expression", "")
            res = await self._execute_skill("calculate", {"expression": expr}, session_id)
            if res is not None:
                return res
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
            log.debug("[AGENT] StructuredRouter: unknown desktop action %r, falling back", act)
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

        if act in ("find_file", "find", "search"):
            search_term = query or target
            res = await self._execute_skill("find_file", {"filename": search_term}, session_id)
            if res is None:
                res = await self._execute_skill("search_files", {"query": search_term}, session_id)
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
            if path and content:
                res = await self._execute_skill("save_file", {"path": path, "content": content}, session_id)
                if res is not None:
                    return res
            # Complex action -- defer to Brain with original utterance
            return await self._fallback(task.utterance, session_id)

        else:
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
            log.error("[AGENT] StructuredRouter._call_brain error: %s", exc)
            return f"Brain error: {exc}", False

    async def _fallback(self, utterance: str, session_id: str) -> tuple[str, bool]:
        """Fall back to Brain.process(utterance) when routing fails."""
        log.debug(
            "[AGENT] StructuredRouter: fallback to Brain.process(%r)",
            utterance[:60],
        )
        return await self._call_brain(utterance, session_id)

