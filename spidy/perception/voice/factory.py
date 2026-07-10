"""
VoiceEngine Factory
=====================
Constructs the full VoiceEngine from configuration.

Design
------
This is the only place in the codebase that knows which concrete
implementations to use. Changing the voice backend (e.g. Piper → Kokoro)
requires changing the config, not the factory.

The factory pattern keeps SpidyCore.py clean — it just calls
VoiceEngineFactory.build(config, bus) and gets back a fully wired engine.
"""

from __future__ import annotations

from spidy.config.manager import SpidyConfig
from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.voice.engine import VoiceEngine
from spidy.perception.voice.wake_word.openwakeword import OpenWakeWordModel
from spidy.perception.voice.stt.whisper import FasterWhisperRecognizer
from spidy.perception.voice.tts.piper import PiperTTSEngine

log = get_logger(__name__)


class VoiceEngineFactory:
    """
    Builds and loads a fully configured VoiceEngine.

    Parameters
    ----------
    config:
        The loaded SpidyConfig object.
    bus:
        The application EventBus.

    Returns
    -------
    VoiceEngine
        A ready-to-start VoiceEngine with all models loaded.
    """

    @staticmethod
    def build(config: SpidyConfig, bus: EventBus) -> VoiceEngine:
        voice_cfg = config.voice

        # ── Wake Word ─────────────────────────────────────────────────────
        log.info("Building wake word engine: model='{m}'", m=voice_cfg.wake_word.model)
        wake_model = OpenWakeWordModel(
            model_name=voice_cfg.wake_word.model,
            threshold=voice_cfg.wake_word.threshold,
        )
        wake_model.load()

        # ── STT ───────────────────────────────────────────────────────────
        log.info("Building STT engine: model='{m}' device='{d}'",
                 m=voice_cfg.stt.model, d=voice_cfg.stt.device)
        recognizer = FasterWhisperRecognizer(
            model_size=voice_cfg.stt.model,
            device=voice_cfg.stt.device,
            compute_type=voice_cfg.stt.compute_type,
            language=voice_cfg.stt.language,
            vad_filter=voice_cfg.stt.vad_filter,
            vad_threshold=voice_cfg.stt.vad_threshold,
        )
        recognizer.load()

        # ── TTS ───────────────────────────────────────────────────────────
        log.info("Building TTS engine: voice='{v}'", v=voice_cfg.tts.voice)

        # Select TTS engine from config
        tts_engine_name = voice_cfg.tts.engine.lower()
        if tts_engine_name == "piper":
            tts = PiperTTSEngine(
                voice=voice_cfg.tts.voice,
                speed=voice_cfg.tts.speed,
                volume=voice_cfg.tts.volume,
            )
            tts.load()
        else:
            log.warning(
                "Unknown TTS engine '{name}'. Falling back to Piper.",
                name=tts_engine_name,
            )
            tts = PiperTTSEngine(voice=voice_cfg.tts.voice)
            tts.load()

        # ── Assemble VoiceEngine ──────────────────────────────────────────
        engine = VoiceEngine(
            bus=bus,
            wake_word_model=wake_model,
            recognizer=recognizer,
            tts=tts,
            sample_rate=voice_cfg.audio.sample_rate,
            wake_threshold=voice_cfg.wake_word.threshold,
        )

        log.info("VoiceEngine assembled successfully.")
        return engine
