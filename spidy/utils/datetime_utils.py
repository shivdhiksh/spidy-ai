"""
spidy.utils.datetime_utils
==========================
Centralized datetime and timezone utility functions.
Supports explicit timezone configuration, automatic Windows/OS local timezone detection,
and consistent formatting for assistant skills and LLM system prompts.
"""

from __future__ import annotations

import os
import zoneinfo
from datetime import datetime, tzinfo


def get_local_timezone(tz_override: str | None = None) -> tzinfo:
    """
    Determine the authoritative local timezone.

    Priority:
    1. Explicit `tz_override` argument (if provided and valid).
    2. `SPIDY_TIMEZONE` or `TZ` environment variables.
    3. `settings.app.timezone` configuration (if configured and not 'auto').
    4. OS / Windows system local timezone via `datetime.now().astimezone().tzinfo`.
    """
    candidates = []
    if tz_override and str(tz_override).strip():
        candidates.append(str(tz_override).strip())

    env_tz = os.environ.get("SPIDY_TIMEZONE") or os.environ.get("TZ")
    if env_tz and env_tz.strip():
        candidates.append(env_tz.strip())

    try:
        from spidy.config import settings
        cfg_tz = getattr(getattr(settings, "app", None), "timezone", None)
        if cfg_tz and str(cfg_tz).strip():
            candidates.append(str(cfg_tz).strip())
    except Exception:
        pass

    for tz_name in candidates:
        if tz_name.lower() in ("auto", "system", "local", "none", ""):
            continue
        try:
            return zoneinfo.ZoneInfo(tz_name)
        except Exception:
            continue

    # Default to the OS local timezone (e.g. Asia/Kolkata on Windows in India)
    system_tz = datetime.now().astimezone().tzinfo
    return system_tz or datetime.now().astimezone().tzinfo


def get_local_datetime(tz_override: str | None = None) -> datetime:
    """
    Return the current datetime in the authoritative local timezone.
    Always returns a timezone-aware datetime.
    """
    tz = get_local_timezone(tz_override)
    return datetime.now(tz)
