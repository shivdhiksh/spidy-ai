"""
OpenWakeWordModel — Concrete Wake Word Detector
================================================
Uses the openWakeWord library with a pre-trained TFLite model.

Milestone 1 status
------------------
We use ``alexa`` as the temporary wake trigger (a well-trained,
freely available openWakeWord model) while we validate the pipeline.

The final product uses "hey_spidy" — a custom-trained model.
To swap: train model → place .tflite in assets/models/wake_word/ →
change config.voice.wake_word.model to "hey_spidy" — no code changes.

openWakeWord internals
----------------------
- Runs a melspectrogram feature extractor on each audio chunk
- Feeds features to a TFLite model that outputs per-model scores
- Scores are averaged over a short window for stability
- We use a single model at a time (the one from config)

Thread safety
-------------
process_chunk() is called from the audio capture thread.
The openWakeWord Model object is not thread-safe by itself, but since
we call it from a single dedicated audio thread, this is fine.
"""

from __future__ import annotations

import numpy as np

from spidy.logging.logger import get_logger
from spidy.perception.voice.wake_word.base import WakeWordModel

log = get_logger(__name__)

# Maps config model names to openWakeWord's internal identifiers
# and expected score key names
_MODEL_MAP: dict[str, dict[str, str]] = {
    "alexa":        {"oww_name": "alexa",         "score_key": "alexa"},
    "hey_jarvis":   {"oww_name": "hey_jarvis",    "score_key": "hey jarvis"},
    "hey_mycroft":  {"oww_name": "hey_mycroft",   "score_key": "hey mycroft"},
    "hey_spidy":    {"oww_name": "hey_spidy",     "score_key": "hey spidy"},   # future custom
}

_CHUNK_SIZE = 1280      # 80ms at 16kHz — openWakeWord's required input size
_SAMPLE_RATE = 16000


class OpenWakeWordModel(WakeWordModel):
    """
    openWakeWord-based wake word detector.

    Configuration
    -------------
    Reads model name from SpidyConfig.voice.wake_word.model.
    Valid values: "alexa", "hey_jarvis", "hey_mycroft", "hey_spidy" (future).

    Parameters
    ----------
    model_name:
        Which pre-trained model to use. Defaults to "alexa".
    threshold:
        Detection confidence threshold (0.0–1.0). Detects when score > threshold.
    """

    def __init__(self, model_name: str = "alexa", threshold: float = 0.5) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._oww_model = None          # openwakeword.Model instance
        self._score_key: str = ""       # which key in scores dict to read
        self._loaded = False

    # ── WakeWordModel interface ───────────────────────────────────────────

    def load(self, model_path: str | None = None) -> None:
        """
        Load the openWakeWord model.

        Parameters
        ----------
        model_path:
            Optional path to a custom .tflite model file.
            If None, uses the library's built-in pre-trained models.
        """
        try:
            from openwakeword.model import Model as OWWModel
        except ImportError as exc:
            raise ImportError(
                "openWakeWord is not installed. "
                "Install it with: pip install openwakeword"
            ) from exc

        mapping = _MODEL_MAP.get(self._model_name, _MODEL_MAP["alexa"])
        oww_name = mapping["oww_name"]
        self._score_key = mapping["score_key"]

        log.info(
            "Loading wake word model: '{model}' (score_key='{key}')",
            model=oww_name,
            key=self._score_key,
        )

        try:
            if model_path:
                # Custom model (e.g. hey_spidy trained model)
                self._oww_model = OWWModel(
                    wakeword_models=[model_path],
                    inference_framework="tflite",
                )
                log.info("Custom wake word model loaded from: {path}", path=model_path)
            else:
                # Built-in pre-trained model
                self._oww_model = OWWModel(
                    wakeword_models=[oww_name],
                    inference_framework="tflite",
                )
                log.info("Pre-trained wake word model loaded: {name}", name=oww_name)

            self._loaded = True

        except Exception as exc:
            log.error("Failed to load wake word model: {exc}", exc=exc)
            raise

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Run inference on one audio chunk.

        Returns
        -------
        float
            Confidence score for the wake word. Range [0.0, 1.0].
            Returns 0.0 if model is not loaded.
        """
        if not self._loaded or self._oww_model is None:
            return 0.0

        # openWakeWord expects int16 PCM — convert from float32
        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        try:
            # predict() returns a dict: {model_name: score}
            scores = self._oww_model.predict(audio_int16)

            # Get score for our model (try direct key, then first available)
            score = scores.get(self._score_key, 0.0)
            if score == 0.0 and scores:
                score = list(scores.values())[0]

            return float(score)

        except Exception as exc:  # noqa: BLE001
            log.debug("Wake word inference error (non-fatal): {exc}", exc=exc)
            return 0.0

    def unload(self) -> None:
        """Release model resources."""
        if self._oww_model is not None:
            del self._oww_model
            self._oww_model = None
        self._loaded = False
        log.debug("Wake word model unloaded.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def chunk_size(self) -> int:
        return _CHUNK_SIZE

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_loaded(self) -> bool:
        return self._loaded
