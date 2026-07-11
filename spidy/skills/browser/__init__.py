"""
Browser Skills Loader
=====================
Registers all browser-automation skills with the SkillRegistry.

Called from ``spidy/core/app.py`` during the Brain startup sequence,
following the same pattern as ``register_desktop_skills()``.

Usage
-----
    from spidy.skills.browser import register_browser_skills

    registered = register_browser_skills(
        registry,
        config=settings.skills,
        bus=bus,
    )
    # returns: ["browser_skill"]  (or [] if disabled)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)


def register_browser_skills(
    registry: "SkillRegistry",
    config: object = None,
    bus: "EventBus | None" = None,
) -> list[str]:
    """
    Register browser automation skills with the SkillRegistry.

    Parameters
    ----------
    registry:
        The application's SkillRegistry.
    config:
        SkillsConfig object. Reads: ``browser_skill_enabled``,
        ``browser_type``, ``browser_headless``, ``browser_download_dir``,
        ``browser_connect_to_existing``, ``browser_cdp_endpoint``,
        ``browser_page_load_timeout_ms``, ``browser_navigation_timeout_ms``,
        ``browser_read_page_max_chars``, ``browser_history_limit``.
    bus:
        EventBus for publishing browser events.

    Returns
    -------
    list[str]
        Names of skills that were successfully registered.
    """
    registered: list[str] = []

    # ── BrowserSkill ──────────────────────────────────────────────────────
    browser_skill_enabled = getattr(config, "browser_skill_enabled", True)
    if browser_skill_enabled:
        try:
            from spidy.skills.browser.browser_skill import BrowserSkill

            skill = BrowserSkill(
                bus=bus,
                browser_type=getattr(config, "browser_type", "chromium"),
                headless=getattr(config, "browser_headless", False),
                download_dir=getattr(config, "browser_download_dir", "~"),
                connect_to_existing=getattr(config, "browser_connect_to_existing", True),
                cdp_endpoint=getattr(config, "browser_cdp_endpoint", "http://localhost:9222"),
                page_load_timeout_ms=getattr(config, "browser_page_load_timeout_ms", 30_000),
                navigation_timeout_ms=getattr(config, "browser_navigation_timeout_ms", 10_000),
                read_page_max_chars=getattr(config, "browser_read_page_max_chars", 5000),
                history_limit=getattr(config, "browser_history_limit", 20),
            )
            registry.register(skill)
            registered.append("browser_skill")
            log.info("BrowserSkill registered ({n} actions).", n=len(skill.capability_names()))
        except Exception as exc:  # noqa: BLE001
            log.warning("BrowserSkill registration failed (non-fatal): {exc}", exc=exc)
    else:
        log.debug("BrowserSkill is disabled in config.")

    return registered
