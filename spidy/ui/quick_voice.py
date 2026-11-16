"""
QuickGreeter: the pet's greetings, spoken instantly
===================================================
The desktop pet greets you seconds after login ("Hi Boss!! Welcome Back"), but Spidy's own voice
pipeline only comes up ~20 s later. The greetings are fixed sentences, so they are synthesised once
with Spidy's Piper voice, cached as audio on disk, and played straight from the cache: no model
load at boot (after the first time) and no wait for the rest of Spidy.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np

from spidy.logging.logger import get_logger

log = get_logger(__name__)


def _cache_dir() -> Path:
    d = Path(os.environ.get("LOCALAPPDATA", ".")) / "Spidy" / "greetings"
    d.mkdir(parents=True, exist_ok=True)
    return d


class QuickGreeter:
    def __init__(self, voice: str = "en_US-lessac-high", speed: float = 1.0, volume: float = 1.0) -> None:
        self.voice, self.speed, self.volume = voice, speed, volume
        self._audio: dict[str, tuple[np.ndarray, int]] = {}
        self._lock = threading.Lock()

    def _path(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.voice}|{self.speed}|{self.volume}|{text}".encode()).hexdigest()[:16]
        return _cache_dir() / f"{key}.npz"

    def prepare(self, texts: list[str]) -> None:
        """Load cached audio now; synthesise any missing ones in the background."""
        missing = []
        for text in texts:
            p = self._path(text)
            try:
                if p.exists():
                    z = np.load(p)
                    with self._lock:
                        self._audio[text] = (z["samples"], int(z["sr"]))
                    continue
            except Exception:  # noqa: BLE001
                pass
            missing.append(text)
        if missing:
            threading.Thread(target=self._synthesise, args=(missing,), name="QuickGreeter", daemon=True).start()

    def _synthesise(self, texts: list[str]) -> None:
        try:
            from spidy.perception.voice.tts.piper import PiperTTSEngine
            try:
                from spidy.voice import emotion
                style = getattr(emotion, "HAPPY", None)
            except Exception:  # noqa: BLE001
                style = None
            eng = PiperTTSEngine(voice=self.voice, speed=self.speed, volume=self.volume)
            eng.load()
            try:
                for text in texts:
                    buf = eng._synthesize_sync(text, style)
                    samples = np.asarray(buf.samples, dtype=np.float32)
                    np.savez(self._path(text), samples=samples, sr=buf.sample_rate)
                    with self._lock:
                        self._audio[text] = (samples, buf.sample_rate)
            finally:
                try:
                    eng.unload()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            log.warning("[PET] greeting voice not prepared: {exc}", exc=exc)

    def has(self, text: str) -> bool:
        with self._lock:
            return text in self._audio

    def say(self, text: str) -> bool:
        """Play the greeting now. False if it isn't ready (the caller falls back to Spidy's voice)."""
        with self._lock:
            item = self._audio.get(text)
        if item is None:
            return False
        def play() -> None:                      # opening the audio device can take a moment: off the UI thread
            try:
                import sounddevice as sd
                sd.play(item[0], item[1])
                sd.wait()
            except Exception as exc:  # noqa: BLE001
                log.debug("[PET] quick greeting playback failed: {exc}", exc=exc)
        threading.Thread(target=play, name="PetGreeting", daemon=True).start()
        return True
