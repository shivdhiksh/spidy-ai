"""
ScreenshotSkill — Screen Capture
==================================
Captures the current screen and saves to a file.

Permission Tier: T1 (non-destructive, writes a file to disk)

Actions
-------
take_screenshot    — Capture the full screen                    (T1)

Architecture
------------
Primary backend:   PIL/Pillow ImageGrab (Windows/macOS)
Fallback backend:  mss (cross-platform)
Ultimate fallback: subprocess + Windows Snipping Tool / Snip & Sketch

Output
------
Saved to: ~/Pictures/Screenshots/spidy_YYYYMMDD_HHMMSS.png
Returns the full path of the saved file in SkillResult.data.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from spidy.skills.base import BaseSkill, ParamSchema, SkillCapability, SkillContext, SkillResult
from spidy.logging.logger import get_logger

log = get_logger(__name__)

_DEFAULT_SAVE_DIR = Path.home() / "Pictures" / "Screenshots"


class ScreenshotSkill(BaseSkill):
    """
    Screen capture skill.

    Captures the entire screen and saves it as a PNG.
    Tries PIL.ImageGrab first, then mss, then fails gracefully.
    """

    name = "screenshot_skill"
    version = "1.0.0"

    def capabilities(self) -> list[SkillCapability]:
        return [
            SkillCapability(
                action="take_screenshot",
                description=(
                    "Capture the current screen and save it as an image file."
                ),
                permission_tier="T1",
                params=[
                    ParamSchema(
                        "save_dir",
                        "path",
                        required=False,
                        description=(
                            "Directory to save the screenshot. "
                            "Defaults to ~/Pictures/Screenshots."
                        ),
                    ),
                    ParamSchema(
                        "filename",
                        "string",
                        required=False,
                        description=(
                            "Custom filename (without extension). "
                            "Defaults to spidy_YYYYMMDD_HHMMSS."
                        ),
                    ),
                ],
                examples=[
                    "take a screenshot",
                    "screenshot",
                    "capture the screen",
                    "take a screen capture",
                    "grab a screenshot",
                    "screen capture",
                    "print screen",
                ],
            ),
        ]

    async def execute(self, action: str, context: SkillContext) -> SkillResult:
        if action == "take_screenshot":
            return await self._take_screenshot(context)
        return SkillResult.fail(f"ScreenshotSkill: unknown action '{action}'.")

    async def _take_screenshot(self, context: SkillContext) -> SkillResult:
        save_dir_raw = context.get("save_dir")
        save_dir = Path(save_dir_raw).expanduser() if save_dir_raw else _DEFAULT_SAVE_DIR

        custom_name = context.get("filename", "") or ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (custom_name.strip() or f"spidy_{timestamp}") + ".png"

        # Ensure directory exists
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return SkillResult.fail(
                f"ScreenshotSkill: could not create save directory '{save_dir}': {exc}",
                error=exc,
            )

        output_path = save_dir / filename

        try:
            saved_path = await asyncio.to_thread(
                self._capture_screen, output_path
            )
        except RuntimeError as exc:
            return SkillResult.fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return SkillResult.fail(
                f"ScreenshotSkill: screenshot failed: {exc}", error=exc
            )

        return SkillResult.ok(
            f"Screenshot saved to: {saved_path}",
            data={
                "path": str(saved_path),
                "filename": saved_path.name,
                "directory": str(saved_path.parent),
            },
            action_taken="take_screenshot",
        )

    @staticmethod
    def _capture_screen(output_path: Path) -> Path:
        """
        Capture the screen using the best available backend.

        Tries:
        1. PIL.ImageGrab (Pillow — most reliable on Windows)
        2. mss (fast, cross-platform)
        3. Raises RuntimeError with helpful install instructions
        """
        # ── Backend 1: PIL/Pillow ────────────────────────────────────────
        try:
            from PIL import ImageGrab  # type: ignore[import-untyped]
            img = ImageGrab.grab()
            img.save(str(output_path), "PNG")
            log.debug("Screenshot captured via PIL to {p}", p=output_path)
            return output_path
        except ImportError:
            pass  # PIL not available — try next
        except Exception as exc:
            log.warning("PIL screenshot failed: {exc}", exc=exc)

        # ── Backend 2: mss ───────────────────────────────────────────────
        try:
            import mss  # type: ignore[import-untyped]
            import mss.tools
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # Primary + all monitors combined
                screenshot = sct.grab(monitor)
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(output_path))
            log.debug("Screenshot captured via mss to {p}", p=output_path)
            return output_path
        except ImportError:
            pass  # mss not available
        except Exception as exc:
            log.warning("mss screenshot failed: {exc}", exc=exc)

        raise RuntimeError(
            "ScreenshotSkill: no screenshot backend available. "
            "Install Pillow: pip install Pillow  OR  mss: pip install mss"
        )
