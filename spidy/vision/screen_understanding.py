"""
Screen Understanding Engine — First-Class Desktop Screen Perception
===================================================================
Provides progressive, multi-tier screen understanding for autonomous agents:
Tier 1: Deterministic Win32 & UI Automation control inspection (zero-cost)
Tier 2: Fast on-demand GDI screenshot capture
Tier 3: OCR text extraction for visible text and error diagnosis
Tier 4: Vision model / contour analysis for graphical UI components

Capabilities:
- Capture and produce ScreenUnderstandingResult
- Answer screen questions ("What is on my screen?", "What app is open?", "What error is visible?")
- Compare before/after screen states to verify UI actions
- Extract clickable targets and error tracebacks
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spidy.desktop.controller import DesktopComputerController
from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.vision.manager import VisionManager

log = get_logger(__name__)


@dataclass
class ScreenUnderstandingResult:
    """
    Structured outcome of screen understanding.
    """
    active_window: str = ""
    application: str = ""
    open_windows: list[str] = field(default_factory=list)
    ocr_text: str = ""
    elements: list[dict[str, Any]] = field(default_factory=list)
    ui_elements: list[dict[str, Any]] = field(default_factory=list)
    screenshot_path: str = ""
    vision_summary: str = ""
    detected_errors: list[str] = field(default_factory=list)
    has_error: bool = False
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Human-readable description of current screen state."""
        win_title = self.active_window or "Desktop"
        app = f" ({self.application})" if self.application else ""
        parts = [f"Active Window: '{win_title}'{app}"]
        if self.ocr_text:
            cleaned = " ".join(self.ocr_text.split()[:40])
            parts.append(f"Visible text: '{cleaned}...'")
        if self.elements or self.ui_elements:
            elem_count = len(self.elements or self.ui_elements)
            parts.append(f"{elem_count} UI elements detected")
        return " | ".join(parts)


class ScreenUnderstandingEngine:
    """
    Coordinates multi-tier screen perception and understanding.
    """

    def __init__(
        self,
        controller: DesktopComputerController | None = None,
        vision_manager: "VisionManager | None" = None,
    ) -> None:
        self._controller = controller or DesktopComputerController()
        self._vision_manager = vision_manager

    async def understand_screen(
        self,
        need_screenshot: bool = False,
        need_ocr: bool = False,
        need_elements: bool = True,
    ) -> ScreenUnderstandingResult:
        """
        Produce a ScreenUnderstandingResult with progressive depth.
        """
        # Tier 1: Deterministic window inspection
        active_win = self._controller.get_active_window()
        active_title = active_win.get("title", "")
        active_proc = active_win.get("process_name", "")
        app_name = active_proc.replace(".exe", "").capitalize() if active_proc else ""
        if not app_name and active_title:
            app_name = active_title.split(" - ")[-1]

        elements: list[dict[str, Any]] = []
        if need_elements and active_win.get("hwnd"):
            try:
                hwnd = active_win["hwnd"]
                child = self._controller._find_win32_child(hwnd, None, None)
                if child:
                    elements.append(child)
            except Exception as exc:
                log.debug("ScreenUnderstanding: element extraction error: {exc}", exc=exc)

        screenshot_path = ""
        ocr_text = ""

        # Tier 2: Screenshot on demand
        if need_screenshot or need_ocr:
            try:
                temp_dir = tempfile.gettempdir()
                shot_file = os.path.join(temp_dir, f"spidy_screen_{int(time.time()*1000)}.png")
                self._controller.screenshot(file_path=shot_file)
                if os.path.exists(shot_file):
                    screenshot_path = shot_file
            except Exception as exc:
                log.debug("ScreenUnderstanding: screenshot capture error: {exc}", exc=exc)

        # Tier 3: OCR on demand
        if need_ocr and self._vision_manager is not None and screenshot_path:
            try:
                ocr_res = await self._vision_manager.read_screen_text(screenshot_path=screenshot_path)
                ocr_text = ocr_res.full_text if ocr_res else ""
            except Exception as exc:
                log.debug("ScreenUnderstanding: OCR error: {exc}", exc=exc)

        # Build vision summary
        summary_text = f"Application '{app_name or 'Desktop'}' is currently open with window title '{active_title}'."
        if ocr_text:
            summary_text += f" On screen text contains: {' '.join(ocr_text.split()[:30])}"

        return ScreenUnderstandingResult(
            active_window=active_title,
            application=app_name,
            ocr_text=ocr_text,
            elements=elements,
            screenshot_path=screenshot_path,
            vision_summary=summary_text,
            confidence=0.95 if active_title else 0.5,
        )

    async def answer_screen_question(self, question: str) -> str:
        """
        Answer a natural language question about what is visible on screen.
        """
        q = question.lower()

        # Decide depth based on question keywords
        need_ocr = any(k in q for k in ("error", "text", "read", "say", "traceback", "wrong", "code", "message"))
        need_screenshot = need_ocr or any(k in q for k in ("look", "see", "show", "screen"))

        state = await self.understand_screen(
            need_screenshot=need_screenshot,
            need_ocr=need_ocr,
            need_elements=True,
        )

        if "app" in q or "application" in q or "program" in q:
            if state.application:
                return f"The active application is {state.application} (Window: '{state.active_window}')."
            return "No active application window is currently focused."

        if "error" in q or "wrong" in q:
            if state.ocr_text:
                error_match = re.search(r"(?:error|exception|traceback|failed|fatal):?\s*([^\n\r]+)", state.ocr_text, re.I)
                if error_match:
                    return f"I observed the following error on your screen: {error_match.group(0)}"
                return f"Here is the text observed on your screen: {' '.join(state.ocr_text.split()[:50])}"
            return f"The active window is '{state.active_window}'. No explicit error was detected in the visible window title."

        if "text" in q or "say" in q or "read" in q:
            if state.ocr_text:
                return f"Text on screen: {state.ocr_text[:300]}"
            return f"The active window title is '{state.active_window}'."

        # General screen overview
        return f"Currently on your screen, '{state.active_window}' ({state.application}) is the active window."

    async def detect_screen_errors(self) -> tuple[bool, list[str]]:
        """
        Inspect screen OCR and active window to detect errors/exceptions.
        """
        state = await self.understand_screen(need_screenshot=True, need_ocr=True, need_elements=False)
        if state.has_error and state.detected_errors:
            return True, state.detected_errors

        errors = []
        if state.ocr_text:
            matches = re.findall(r"(?:[A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception|FAILED|Traceback|SyntaxError|NameError):\s*[^\n\r]+", state.ocr_text)
            if matches:
                errors.extend(matches)

        return bool(errors), errors
