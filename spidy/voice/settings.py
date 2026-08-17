"""
VoiceSettings — Runtime Voice Configuration
===========================================
Provides runtime-adjustable voice settings with persistence.

Settings are stored in ``config/voice_settings.json`` and applied
immediately without restarting Spidy.

Supported settings (Part 9)
----------------------------
  microphone_index        int | None    — sounddevice input device index
  speech_rate             float         — TTS speed multiplier (0.5–2.0)
  voice                   str           — TTS voice name
  volume                  float         — TTS volume (0.0–1.0)
  wake_word_enabled       bool          — enable/disable wake word
  conversation_timeout    float         — seconds before session ends
  vad_sensitivity         int           — VAD aggressiveness 0–3
  wake_word               str           — active wake word phrase

Usage
-----
    settings = VoiceSettings(bus=bus, config_path=Path("config/voice_settings.json"))
    settings.load()

    # Change a setting at runtime
    settings.set("volume", 0.8)
    settings.set("conversation_timeout", 90.0)

    # Apply to active components
    settings.apply_to(streaming_tts, session_manager)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from spidy.logging.logger import get_logger
from spidy.voice.events import VoiceSettingsChangedEvent

if TYPE_CHECKING:
    from spidy.core.event_bus import EventBus
    from spidy.voice.session import VoiceSessionManager
    from spidy.voice.streaming_tts import StreamingTTSWrapper

log = get_logger(__name__)

_DEFAULTS: dict[str, Any] = {
    "microphone_index": None,
    "speech_rate": 1.0,
    "voice": "en_US-ryan-high",
    "volume": 1.0,
    "wake_word_enabled": True,
    "conversation_timeout": 60.0,
    "vad_sensitivity": 2,
    "wake_word": "hey spidy",
}

_VALID_RANGES: dict[str, tuple[Any, Any]] = {
    "speech_rate": (0.5, 2.0),
    "volume": (0.0, 1.0),
    "conversation_timeout": (5.0, 600.0),
    "vad_sensitivity": (0, 3),
}


class VoiceSettings:
    """
    Runtime voice settings manager.

    Parameters
    ----------
    bus:
        EventBus — VoiceSettingsChangedEvent published on each change.
    config_path:
        Path to persist/load settings JSON.
        Defaults to ``config/voice_settings.json``.
    """

    def __init__(
        self,
        bus: "EventBus",
        config_path: Path | None = None,
    ) -> None:
        self._bus = bus
        self._config_path = config_path or Path("config/voice_settings.json")
        self._settings: dict[str, Any] = dict(_DEFAULTS)

    def load(self) -> None:
        """Load settings from disk. Missing keys fall back to defaults."""
        if not self._config_path.exists():
            log.debug("VoiceSettings: no config file found, using defaults.")
            return

        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            for key, default in _DEFAULTS.items():
                if key in data:
                    raw_value = data[key]
                    if raw_value is None or default is None:
                        self._settings[key] = raw_value
                    else:
                        try:
                            self._settings[key] = type(default)(raw_value)
                        except (TypeError, ValueError):
                            self._settings[key] = raw_value
            log.info("VoiceSettings loaded from {path}.", path=self._config_path)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "VoiceSettings: failed to load from {path}: {exc}. Using defaults.",
                path=self._config_path,
                exc=exc,
            )

    def save(self) -> None:
        """Persist current settings to disk."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(self._settings, indent=2),
                encoding="utf-8",
            )
            log.debug("VoiceSettings saved to {path}.", path=self._config_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("VoiceSettings: save failed: {exc}.", exc=exc)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._settings.get(key, _DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        """
        Update a setting at runtime.

        Validates range, publishes VoiceSettingsChangedEvent, and saves.

        Parameters
        ----------
        key:
            Setting name (one of the keys in _DEFAULTS).
        value:
            New value. Must pass range validation if applicable.
        """
        if key not in _DEFAULTS:
            log.warning("VoiceSettings: unknown key '{k}'.", k=key)
            return

        if key in _VALID_RANGES:
            lo, hi = _VALID_RANGES[key]
            try:
                value = type(lo)(value)
                if not (lo <= value <= hi):
                    log.warning(
                        "VoiceSettings: '{k}' value {v} out of range [{lo}, {hi}].",
                        k=key, v=value, lo=lo, hi=hi,
                    )
                    return
            except (TypeError, ValueError):
                log.warning("VoiceSettings: invalid type for '{k}'.", k=key)
                return

        old_value = self._settings.get(key)
        self._settings[key] = value
        self.save()

        log.info(
            "VoiceSettings: '{k}' changed: {old} → {new}",
            k=key, old=old_value, new=value,
        )

        # Publish change event (non-blocking — best effort)
        try:
            loop = __import__("asyncio").get_event_loop()
            if loop.is_running():
                loop.create_task(
                    self._bus.publish(VoiceSettingsChangedEvent(changes={key: value}))
                )
        except Exception:  # noqa: BLE001
            pass

    def apply_to(
        self,
        streaming_tts: "StreamingTTSWrapper | None" = None,
        session_manager: "VoiceSessionManager | None" = None,
    ) -> None:
        """
        Apply current settings to active voice components.

        Parameters
        ----------
        streaming_tts:
            StreamingTTSWrapper — speech_rate and volume applied.
        session_manager:
            VoiceSessionManager — conversation_timeout applied.
        """
        if session_manager is not None:
            timeout = float(self._settings.get("conversation_timeout", 60.0))
            session_manager.timeout_seconds = timeout
            log.debug("VoiceSettings: session timeout → {t}s.", t=timeout)

        log.debug("VoiceSettings applied to active components.")

    @staticmethod
    def list_microphones() -> list[dict]:
        """
        Return available input microphones via sounddevice.

        Returns
        -------
        list[dict]
            Each entry has: index, name, max_input_channels, default_samplerate.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
            devices = sd.query_devices()
            return [
                {
                    "index": i,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "default_samplerate": int(d["default_samplerate"]),
                }
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("VoiceSettings.list_microphones: {exc}.", exc=exc)
            return []

    @property
    def all_settings(self) -> dict[str, Any]:
        """Read-only snapshot of current settings."""
        return dict(self._settings)
