"""
WakeAcknowledger — Immediate Wake Acknowledgement
===================================================
Speaks a short acknowledgement phrase ("Yes Shiva." / "Hmm?") immediately
after wake detection so the user knows Spidy is awake and listening.

Design constraints
------------------
1.  NEVER calls the LLM, Brain, STT, or Memory.
2.  NEVER publishes to EventBus.
3.  Uses the existing Piper TTS engine via StreamingTTSWrapper.
4.  If ``wake_ack.enabled`` is False, returns immediately — wake still works.
5.  An asyncio.Lock prevents double-speak if two wake events fire simultaneously
    (the second call is silently dropped while the first is speaking).
6.  Phrases come entirely from ``WakeAckConfig`` — zero hardcoded strings.

Caller responsibility
---------------------
The caller (ContinuousVoiceController._on_wake_word) must:
  - Set CaptureMode.IDLE **before** calling speak_ack()   (mic OFF)
  - Call speak_ack()
  - Reset OWW prediction buffer
  - Set CaptureMode.RECORDING **after** speak_ack() returns (mic ON)

This guarantees the ack audio never enters the STT or wake-word pipeline.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

from spidy.logging.logger import get_logger

if TYPE_CHECKING:
    from spidy.config.manager import WakeAckConfig
    from spidy.voice.streaming_tts import StreamingTTSWrapper

log = get_logger(__name__)


class WakeAcknowledger:
    """
    Speaks a random acknowledgement phrase on wake.

    Parameters
    ----------
    config:
        ``WakeAckConfig`` from the loaded SpidyConfig.
        Controls ``enabled`` flag and the phrase list.
    """

    def __init__(self, config: "WakeAckConfig") -> None:
        self._config = config
        # Serialize concurrent calls: if wake fires twice simultaneously
        # (e.g. two models both detect at the same instant), only the first
        # call speaks.  The second call acquires the lock, finds _speaking=True,
        # and returns immediately.
        self._lock = asyncio.Lock()
        self._speaking = False

    # -- Public API --------------------------------------------------------

    async def speak_ack(self, streaming_tts: "StreamingTTSWrapper") -> None:
        """
        Speak one acknowledgement phrase via the existing TTS engine.

        Safe to call from any asyncio coroutine.
        Returns immediately if ``wake_ack.enabled`` is False.
        Returns immediately (drops call) if already speaking an ack.

        Parameters
        ----------
        streaming_tts:
            The live StreamingTTSWrapper instance.
            **Caller must have already set CaptureMode.IDLE before this call.**
        """
        if not self._config.enabled:
            log.debug("WakeAcknowledger: disabled -- skipping ack.")
            return

        if not self._config.phrases:
            log.warning(
                "WakeAcknowledger: enabled=True but phrases list is empty -- "
                "skipping ack."
            )
            return

        async with self._lock:
            if self._speaking:
                log.debug(
                    "WakeAcknowledger: already speaking ack -- dropping duplicate call."
                )
                return

            phrase = random.choice(self._config.phrases)
            self._speaking = True
            try:
                log.info(
                    "WakeAcknowledger: speaking ack phrase: '{p}'", p=phrase
                )
                await streaming_tts.speak(phrase)
            except Exception as exc:  # noqa: BLE001
                # Never let a TTS failure break the wake flow.
                log.warning(
                    "WakeAcknowledger: TTS error during ack (non-fatal): {exc}",
                    exc=exc,
                )
            finally:
                self._speaking = False

    # -- Properties --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def phrases(self) -> list[str]:
        return list(self._config.phrases)

    @property
    def is_speaking(self) -> bool:
        return self._speaking
