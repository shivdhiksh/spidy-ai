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

Milestone 14 additions
-----------------------
- Supports ``model: "null"`` for testing (NullWakeWordModel, no file load)
- Builds ContinuousVoiceController and wires it to Brain when brain is provided
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spidy.config.manager import SpidyConfig
from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.voice.engine import VoiceEngine
from spidy.perception.voice.wake_word.openwakeword import (
    NullWakeWordModel,
    OpenWakeWordModel,
)
from spidy.perception.voice.stt.whisper import FasterWhisperRecognizer
from spidy.perception.voice.tts.piper import PiperTTSEngine
from spidy.perception.voice.tts.base import TTSEngine

if TYPE_CHECKING:
    from spidy.brain.brain import Brain
    from spidy.voice.continuous import ContinuousVoiceController

log = get_logger(__name__)


class _NullTTS(TTSEngine):
    """
    Silent no-op TTS used when the real TTS engine fails to load.

    Allows wake-word detection and STT to function even when Piper voice
    model files are missing.  Logs a warning on every speak() call so the
    operator knows audio responses are being dropped.
    """

    async def synthesize(self, text: str):  # type: ignore[override]
        from spidy.perception.voice.tts.base import AudioBuffer
        import numpy as np
        log.warning("[NullTTS] synthesize() called but TTS is unavailable: '{t}'", t=text[:60])
        return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=22050, duration_seconds=0.0)

    async def speak(self, text: str) -> None:
        log.warning("[NullTTS] speak() called but TTS is unavailable: '{t}'", t=text[:60])

    def stop(self) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def voice_name(self) -> str:
        return "null"

    @property
    def is_speaking(self) -> bool:
        return False


class VoiceEngineFactory:
    """
    Builds and loads a fully configured VoiceEngine.

    Parameters
    ----------
    config:
        The loaded SpidyConfig object.
    bus:
        The application EventBus.
    brain:
        Optional Brain instance. When provided, also builds and returns
        a ContinuousVoiceController (M14).

    Returns
    -------
    VoiceEngine
        A ready-to-start VoiceEngine with all models loaded.
    """

    @staticmethod
    def build(config: SpidyConfig, bus: EventBus) -> VoiceEngine:
        voice_cfg = config.voice

        # ── Wake Word ─────────────────────────────────────────────────────
        wake_model_name = voice_cfg.wake_word.model.lower()
        custom_path = voice_cfg.wake_word.custom_model_path or None
        log.info("Building wake word engine: model='{m}'{cp}",
                 m=wake_model_name,
                 cp=f" | custom_path='{custom_path}'" if custom_path else "")

        if wake_model_name == "null":
            # NullWakeWordModel for testing / CI environments
            wake_model = NullWakeWordModel()
            log.info("Using NullWakeWordModel (testing mode).")
        else:
            wake_model = OpenWakeWordModel(
                model_name=wake_model_name,
                threshold=voice_cfg.wake_word.threshold,
            )
        wake_model.load(model_path=custom_path)
        # [DIAG-6] Wake-word model finished loading.
        log.info(
            "[VOICE DIAG-6] Wake-word model loaded: name='{n}' | "
            "is_loaded={ok} | chunk_size={cs}",
            n=wake_model.model_name,
            ok=wake_model.is_loaded,
            cs=wake_model.chunk_size,
        )


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

        # Select TTS engine from config — failure is non-fatal;
        # a silent NullTTS is used so wake-word and STT still work.
        tts_engine_name = voice_cfg.tts.engine.lower()
        try:
            if tts_engine_name == "piper":
                tts: TTSEngine = PiperTTSEngine(
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
            log.info("TTS engine loaded successfully.")
        except Exception as tts_exc:  # noqa: BLE001
            log.error(
                "[VOICE DIAG] TTS failed to load (voice pipeline will run without TTS): {exc}",
                exc=tts_exc,
            )
            # Use a silent no-op TTS so wake-word detection and STT still function.
            tts = _NullTTS()

        # ── Assemble VoiceEngine ──────────────────────────────────────────
        # [DIAG-5] Log which microphone device will be used.
        _input_device = getattr(voice_cfg.audio, "input_device", None)
        log.info(
            "[VOICE DIAG-5] Microphone device selected: index={idx} "
            "(None = system default)",
            idx=_input_device,
        )

        engine = VoiceEngine(
            bus=bus,
            wake_word_model=wake_model,
            recognizer=recognizer,
            tts=tts,
            sample_rate=voice_cfg.audio.sample_rate,
            wake_threshold=voice_cfg.wake_word.threshold,
            mic_gain=getattr(voice_cfg.audio, "mic_gain", 1.0),
            # M14: VAD settings
            silence_timeout_seconds=voice_cfg.vad.silence_timeout_seconds,
        )

        log.info("VoiceEngine assembled successfully.")
        return engine

    @staticmethod
    def build_controller(
        voice_engine: VoiceEngine,
        brain: "Brain",
        bus: EventBus,
        config: SpidyConfig,
    ) -> "ContinuousVoiceController":
        """
        Build a ContinuousVoiceController for M14 continuous conversation.

        Called after VoiceEngine is assembled when a Brain instance is available.

        Parameters
        ----------
        voice_engine:
            The assembled VoiceEngine.
        brain:
            The running Brain instance.
        bus:
            Application EventBus.
        config:
            Full SpidyConfig.

        Returns
        -------
        ContinuousVoiceController
        """
        from spidy.voice.continuous import ContinuousVoiceController
        from spidy.voice.session import VoiceSessionManager
        from spidy.voice.streaming_tts import StreamingTTSWrapper
        from spidy.voice.interruption import InterruptionHandler
        from spidy.voice.wake_ack import WakeAcknowledger
        from spidy.voice.barge_in import BargeInDetector

        session_cfg = config.voice.session

        # Session manager
        session_mgr = VoiceSessionManager(
            bus=bus,
            timeout_seconds=session_cfg.conversation_timeout_seconds,
            max_turns=session_cfg.max_turns_per_session,
            continuous_mode=session_cfg.continuous_mode,
        )

        # Streaming TTS wrapper
        streaming_tts = StreamingTTSWrapper(
            engine=voice_engine._tts,  # noqa: SLF001 — factory has access
            stop_mid_sentence=session_cfg.stop_mid_sentence,
        )

        # Interruption handler
        interruption_handler = InterruptionHandler(
            bus=bus,
            brain=brain,
            tts=voice_engine._tts,  # noqa: SLF001
            session=session_mgr,
        )

        # ── Barge-in detector ─────────────────────────────────────────────
        # Loads tiny.en (separate from main base.en — no contention, ~20 MB extra).
        # Config: voice.barge_in section (all optional; defaults shown below).
        barge_in_cfg = config.voice.barge_in

        barge_in: BargeInDetector | None = None
        if barge_in_cfg.enabled:
            barge_in = BargeInDetector(
                interruption_handler=interruption_handler,
                bus=bus,
                model_size=barge_in_cfg.model,
                window_seconds=barge_in_cfg.window_seconds,
                min_rms_threshold=barge_in_cfg.min_rms_threshold,
                enabled=True,
            )
            # Load blocking — runs at startup (acceptable; same as wake-word / STT load)
            barge_in.load_model()

            # Register barge-in audio callback on the capture engine
            voice_engine._capture_engine._on_barge_in_chunk = barge_in.feed_chunk  # noqa: SLF001
            log.info(
                "BargeInDetector wired: model='{m}' | window={w}s | "
                "min_rms={r} | loaded={ok}",
                m=barge_in_cfg.model,
                w=barge_in_cfg.window_seconds,
                r=barge_in_cfg.min_rms_threshold,
                ok=barge_in.is_loaded,
            )
        else:
            log.info("BargeInDetector: disabled in config (voice.barge_in.enabled=false).")


        # Wake acknowledger — speaks "Yes Shiva." / "Hmm?" immediately on wake.
        # Uses the same Piper TTS engine (via streaming_tts); no second TTS path.
        # Disabled when voice.wake_ack.enabled=false in config.
        wake_ack_cfg = config.voice.wake_ack
        wake_acknowledger = WakeAcknowledger(wake_ack_cfg)
        log.info(
            "WakeAcknowledger: enabled={en}, phrases={ph}",
            en=wake_ack_cfg.enabled,
            ph=wake_ack_cfg.phrases if wake_ack_cfg.enabled else "(disabled)",
        )

        # ── TTS text filter ───────────────────────────────────────────────
        from spidy.voice.tts_text_filter import TTSTextFilter
        tts_filter_cfg = config.voice.tts_filter
        tts_filter = TTSTextFilter(
            code_block_replacement=tts_filter_cfg.code_block_replacement,
            url_replacement=tts_filter_cfg.url_replacement,
            enabled=tts_filter_cfg.enabled,
        )
        log.info(
            "TTSTextFilter: enabled={en} | code_replacement='{cr}'",
            en=tts_filter_cfg.enabled,
            cr=tts_filter_cfg.code_block_replacement[:40],
        )

        # Enable conversation mode on the engine
        if session_cfg.continuous_mode:
            voice_engine.enable_conversation_mode(
                timeout_seconds=session_cfg.conversation_timeout_seconds,
            )

        controller = ContinuousVoiceController(
            voice_engine=voice_engine,
            brain=brain,
            bus=bus,
            session_manager=session_mgr,
            streaming_tts=streaming_tts,
            interruption_handler=interruption_handler,
            barge_in=barge_in,
            tts_filter=tts_filter,
            continuous_mode=session_cfg.continuous_mode,
            measure_latency=session_cfg.measure_latency,
            wake_acknowledger=wake_acknowledger,
        )

        # Wire reliability config values
        controller._dedup_window = session_cfg.transcript_dedup_window_seconds
        controller._post_ack_flush_ms = session_cfg.post_ack_flush_ms
        log.info(
            "CVC reliability config: dedup_window={dw}s post_ack_flush={pf}ms",
            dw=session_cfg.transcript_dedup_window_seconds,
            pf=session_cfg.post_ack_flush_ms,
        )

        log.info("ContinuousVoiceController assembled successfully.")
        return controller

