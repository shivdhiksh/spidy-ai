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
# and expected score key names.
# IMPORTANT: score_key must exactly match the dict key returned by OWWModel.predict().
# Verified empirically: OWW returns 'hey_jarvis' (underscore), NOT 'hey jarvis' (space).
#
# Model quality notes (tested with Realtek Array mic, mic_gain=8.0):
#   alexa_v0.1      — large training corpus; broadest voice coverage; ONNX available
#   hey_jarvis_v0.1 — limited training data (~5 voices); max score 0.05 for many users;
#                     ONNX available but performs poorly outside training distribution
#   hey_mycroft_v0.1 — similar limited training; ONNX available
_MODEL_MAP: dict[str, dict[str, str]] = {
    "alexa":        {"oww_name": "alexa",       "score_key": "alexa"},
    "hey_jarvis":   {"oww_name": "hey_jarvis",  "score_key": "hey_jarvis"},
    "hey_mycroft":  {"oww_name": "hey_mycroft", "score_key": "hey_mycroft"},
    # M14: "hey_spidy" aliases to "alexa" (best coverage) until a custom model
    # is trained. The user says "Alexa" to wake Spidy in this interim period.
    # Replace oww_name + score_key with "hey_spidy" once custom model is trained.
    "hey_spidy":    {"oww_name": "alexa",       "score_key": "alexa"},
}


_CHUNK_SIZE = 1280      # 80ms at 16kHz — openWakeWord's required input size
_SAMPLE_RATE = 16000


def _detect_inference_framework() -> str:
    """
    Return the best available inference framework for openWakeWord.

    Preference order:
    1. 'onnx'   — onnxruntime (pip install onnxruntime) — most portable
    2. 'tflite' — tflite_runtime (pip install tflite-runtime) — lighter

    Hardcoding 'tflite' in the original code caused a silent ImportError on
    systems where only onnxruntime is installed (Windows is a common case).
    """
    try:
        import onnxruntime  # noqa: F401
        return "onnx"
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter  # noqa: F401
        return "tflite"
    except ImportError:
        pass
    # Last resort: let OWW choose and surface its own error
    return "onnx"


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
        self._first_chunk_logged = False  # one-shot diagnostics flag
        # Fix 1: software debounce — track last detection time to prevent
        # double-fire without needing to call OWWModel.reset() between detections.
        self._last_detection_time: float = 0.0
        self._debounce_seconds: float = 2.0   # ignore detections within 2s of last one

    # ── WakeWordModel interface ───────────────────────────────────────────

    def load(self, model_path: str | None = None) -> None:
        """
        Load the openWakeWord model.

        Parameters
        ----------
        model_path:
            Optional path to a custom .onnx model file.
            If provided, bypasses _MODEL_MAP entirely.
            score_key is derived automatically from the filename stem
            (e.g. ``assets/models/wake_word/hey_spidy.onnx`` → score_key ``hey_spidy``).
            If None, uses the library's built-in pre-trained model via _MODEL_MAP.
        """
        try:
            from openwakeword.model import Model as OWWModel
        except ImportError as exc:
            raise ImportError(
                "openWakeWord is not installed. "
                "Install it with: pip install openwakeword"
            ) from exc

        if model_path:
            # Custom model path (e.g. hey_spidy.onnx trained model).
            # OWW uses the filename stem as the score dict key.
            from pathlib import Path
            self._score_key = Path(model_path).stem
            oww_name = model_path
            log.info(
                "Loading custom wake word model: '{path}' (score_key='{key}')",
                path=model_path,
                key=self._score_key,
            )
        else:
            # Built-in model: look up in _MODEL_MAP
            mapping = _MODEL_MAP.get(self._model_name, _MODEL_MAP["hey_jarvis"])
            oww_name = mapping["oww_name"]
            self._score_key = mapping["score_key"]
            log.info(
                "Loading wake word model: '{model}' (score_key='{key}')",
                model=oww_name,
                key=self._score_key,
            )

        try:
            # Auto-detect inference backend: prefer onnxruntime (widely available),
            # fall back to tflite_runtime.  Hardcoding 'tflite' fails when
            # tflite_runtime is not installed (common on Windows).
            _backend = _detect_inference_framework()
            log.info(
                "[VOICE DIAG] Using inference_framework='{fw}' for wake word model.",
                fw=_backend,
            )

            self._oww_model = OWWModel(
                wakeword_models=[oww_name],
                inference_framework=_backend,
            )
            if model_path:
                log.info("Custom wake word model loaded from: {path}", path=model_path)
            else:
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

            # [DIAG-8] On the very first call, log what keys OWW returned.
            if not self._first_chunk_logged:
                self._first_chunk_logged = True
                log.info(
                    "[VOICE DIAG-8] First OWW predict() call: "
                    "score_keys={keys} | looking_for='{key}'",
                    keys=list(scores.keys()),
                    key=self._score_key,
                )

            # Get score for our model (try direct key, then first available)
            score = scores.get(self._score_key, 0.0)
            if score == 0.0 and scores:
                # Key not found — fall back to first value.
                # This means _score_key doesn't match what OWW actually uses.
                fallback_key = list(scores.keys())[0]
                fallback_score = list(scores.values())[0]
                if not self._first_chunk_logged or fallback_score > 0.05:
                    log.warning(
                        "[VOICE DIAG-8] score_key='{k}' NOT in OWW scores dict; "
                        "falling back to first key='{fk}' score={fs:.4f}. "
                        "Consider updating _MODEL_MAP score_key.",
                        k=self._score_key,
                        fk=fallback_key,
                        fs=fallback_score,
                    )
                score = fallback_score

            # Fix 1: software debounce — suppress score above threshold if we
            # detected too recently.  OWWModel.reset() is NOT called because
            # reset() reinitialises the feature_buffer with random-noise embeddings,
            # requiring ~6 seconds of audio to flush — causing 7-10 failed attempts.
            import time as _time
            if score >= self._threshold:
                now = _time.monotonic()
                if now - self._last_detection_time < self._debounce_seconds:
                    log.debug(
                        "Wake debounce: score={s:.4f} suppressed ({elapsed:.2f}s < {db}s cooldown)",
                        s=score, elapsed=now - self._last_detection_time,
                        db=self._debounce_seconds,
                    )
                    return 0.0  # suppressed — not a new detection
                self._last_detection_time = now

            return float(score)

        except Exception as exc:  # noqa: BLE001
            log.debug("Wake word inference error (non-fatal): {exc}", exc=exc)
            return 0.0

    def reset_buffer(self) -> None:
        """
        Clear the OWW prediction score window.

        IMPORTANT: Only the prediction_buffer (rolling 30-score deque) is
        cleared — NOT the full OWWModel state.  Calling OWWModel.reset()
        reinitialises the 120-frame feature_buffer with random-noise embeddings
        via AudioFeatures.reset(), which takes ~6.3 seconds of audio to flush.
        That was the root cause of the "7-10 attempts needed" reliability failure.

        The feature_buffer (melspectrogram embeddings from real audio) is healthy
        state that helps the next detection — it should NOT be cleared.
        The software debounce in process_chunk() handles double-fire prevention.
        """
        if self._oww_model is not None:
            try:
                pb = self._oww_model.prediction_buffer
                if hasattr(pb, 'clear'):
                    pb.clear()          # clears the defaultdict of score deques
                    log.debug("OWW prediction_buffer cleared (feature_buffer preserved).")
            except Exception as exc:  # noqa: BLE001
                log.debug("OWW prediction_buffer clear failed (non-fatal): {exc}", exc=exc)

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


# ── NullWakeWordModel — Testing / Development ──────────────────────────────────


class NullWakeWordModel(WakeWordModel):
    """
    A no-op wake word model for testing and development.

    Triggers after ``trigger_after_chunks`` calls to process_chunk().
    Returns a score of 0.0 otherwise.  Can also be forced to trigger
    by calling ``force_trigger()``.

    Use this in tests to avoid loading any real model files, and when
    running Spidy in CI environments without microphone hardware.

    Configuration: set ``voice.wake_word.model: "null"`` to activate.
    """

    def __init__(
        self,
        trigger_after_chunks: int = 0,
        always_trigger: bool = False,
    ) -> None:
        self._trigger_after = trigger_after_chunks
        self._always_trigger = always_trigger
        self._chunk_count = 0
        self._forced = False
        self._loaded = False

    def load(self, model_path: str | None = None) -> None:
        self._loaded = True
        log.debug("NullWakeWordModel loaded (testing mode).")

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        if not self._loaded:
            return 0.0
        self._chunk_count += 1

        if self._forced:
            self._forced = False
            return 1.0

        if self._always_trigger:
            return 1.0

        if self._trigger_after > 0 and self._chunk_count >= self._trigger_after:
            self._chunk_count = 0
            return 1.0

        return 0.0

    def unload(self) -> None:
        self._loaded = False
        log.debug("NullWakeWordModel unloaded.")

    def force_trigger(self) -> None:
        """Force the next process_chunk() call to return 1.0."""
        self._forced = True

    @property
    def model_name(self) -> str:
        return "null_wake_word"

    @property
    def chunk_size(self) -> int:
        return _CHUNK_SIZE

    @property
    def is_loaded(self) -> bool:
        return self._loaded

