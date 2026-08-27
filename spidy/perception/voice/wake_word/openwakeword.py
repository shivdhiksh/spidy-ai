"""
OpenWakeWordModel — Concrete Wake Word Detector
================================================
Uses the openWakeWord library with single or multi-model detection.

Target Wake Phrases:
--------------------
1. "Hey Spidy"
2. "Wake up Spidy"

Both phrases can be configured and run simultaneously with per-model
thresholds and genuine model files (.onnx / .tflite).

If custom models (e.g. hey_spidy.onnx, wake_up_spidy.onnx) are not yet
installed on disk, the detector logs a clear warning and marks them inactive,
falling back safely to any available active models (e.g. hey_jarvis for dev).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spidy.logging.logger import get_logger
from spidy.perception.voice.wake_word.base import WakeWordModel

log = get_logger(__name__)

# Built-in openWakeWord pre-trained model map.
# Note: Custom models (hey_spidy, wake_up_spidy) are loaded from filesystem paths.
_MODEL_MAP: dict[str, dict[str, str]] = {
    "alexa":        {"oww_name": "alexa",       "score_key": "alexa",       "phrase": "Alexa"},
    "hey_jarvis":   {"oww_name": "hey_jarvis",  "score_key": "hey_jarvis",  "phrase": "Hey Jarvis"},
    "hey_mycroft":  {"oww_name": "hey_mycroft", "score_key": "hey_mycroft", "phrase": "Hey Mycroft"},
    "hey_rhasspy":  {"oww_name": "hey_rhasspy", "score_key": "hey_rhasspy", "phrase": "Hey Rhasspy"},
}

_CHUNK_SIZE = 1280      # 80ms at 16kHz — openWakeWord's required input size
_SAMPLE_RATE = 16000


@dataclass
class _LoadedModelEntry:
    name: str
    phrase: str
    oww_target: str       # file path or built-in model identifier passed to OWW
    score_key: str        # dict key expected in OWW predict() output
    threshold: float
    is_custom: bool = False


def _detect_inference_framework() -> str:
    """
    Return the best available inference framework for openWakeWord.

    Preference order:
    1. 'onnx'   — onnxruntime (pip install onnxruntime) — most portable
    2. 'tflite' — tflite_runtime (pip install tflite-runtime) — lighter
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
    return "onnx"


class OpenWakeWordModel(WakeWordModel):
    """
    openWakeWord-based multi-phrase wake word detector.

    Supports multiple simultaneous models with independent thresholds
    and debouncing.
    """

    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.35,
        models: list[Any] | None = None,
        active_models: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._raw_models_cfg = models or []
        self._raw_active_models_cfg = active_models or []

        self._oww_model = None          # openwakeword.Model instance
        self._loaded_models: list[_LoadedModelEntry] = []
        self._missing_models: list[dict[str, str]] = []
        self._loaded = False
        self._first_chunk_logged = False

        # Detection state
        self._last_detected_model: str = ""
        self._last_detected_phrase: str = ""
        self._last_detected_score: float = 0.0
        self._is_detected: bool = False

        # Software debounce — track last detection time per model
        self._last_detection_times: dict[str, float] = {}
        self._last_global_detection_time: float = 0.0
        self._debounce_seconds: float = 2.0

    # ── WakeWordModel interface ───────────────────────────────────────────

    def load(self, model_path: str | None = None) -> None:
        """
        Load the configured wake word models into openWakeWord.
        """
        try:
            from openwakeword.model import Model as OWWModel
        except ImportError as exc:
            raise ImportError(
                "openWakeWord is not installed. "
                "Install it with: pip install openwakeword"
            ) from exc

        self._loaded_models.clear()
        self._missing_models.clear()

        # Build candidate model configurations to load
        candidates: list[dict[str, Any]] = []

        if self._raw_models_cfg:
            # Multi-model configuration present
            active_filter = set(self._raw_active_models_cfg) if self._raw_active_models_cfg else None
            for m in self._raw_models_cfg:
                m_name = getattr(m, "name", "") or m.get("name", "") if isinstance(m, dict) else getattr(m, "name", "")
                m_phrase = getattr(m, "phrase", "") or (m.get("phrase", "") if isinstance(m, dict) else "")
                m_model = getattr(m, "model", "") or (m.get("model", "") if isinstance(m, dict) else "")
                m_path = getattr(m, "model_path", None) or (m.get("model_path", None) if isinstance(m, dict) else None)
                m_thresh = getattr(m, "threshold", self._threshold) or (m.get("threshold", self._threshold) if isinstance(m, dict) else self._threshold)

                if active_filter is not None and m_name not in active_filter:
                    log.debug("Wake model '{name}' not in active_models list; skipping.", name=m_name)
                    continue

                candidates.append({
                    "name": m_name or m_model,
                    "phrase": m_phrase or m_name or m_model,
                    "model": m_model,
                    "model_path": m_path,
                    "threshold": float(m_thresh),
                })
        else:
            # Single model backward-compatible configuration
            candidates.append({
                "name": self._model_name,
                "phrase": _MODEL_MAP.get(self._model_name, {}).get("phrase", self._model_name),
                "model": self._model_name,
                "model_path": model_path,
                "threshold": float(self._threshold),
            })

        # Resolve each candidate to genuine files or built-in models
        resolved_entries: list[_LoadedModelEntry] = []
        oww_targets: list[str] = []

        for cand in candidates:
            c_name = cand["name"]
            c_phrase = cand["phrase"]
            c_path = cand.get("model_path")
            c_thresh = cand["threshold"]
            c_model = cand.get("model", "")

            # 1. Explicit model path provided
            if c_path:
                if os.path.exists(c_path):
                    stem = Path(c_path).stem
                    entry = _LoadedModelEntry(
                        name=c_name,
                        phrase=c_phrase,
                        oww_target=c_path,
                        score_key=stem,
                        threshold=c_thresh,
                        is_custom=True,
                    )
                    resolved_entries.append(entry)
                    oww_targets.append(c_path)
                    log.info(
                        "Registered custom wake model '{name}' ('{phrase}') from '{path}' (threshold={thresh})",
                        name=c_name, phrase=c_phrase, path=c_path, thresh=c_thresh,
                    )
                else:
                    log.warning(
                        "Wake word model '{name}' ('{phrase}') not found at '{path}'. "
                        "Model is NOT active. Please place compatible .onnx model file at '{path}' to enable.",
                        name=c_name, phrase=c_phrase, path=c_path,
                    )
                    self._missing_models.append({
                        "name": c_name,
                        "phrase": c_phrase,
                        "expected_path": str(c_path),
                    })
                continue

            # 2. Check standard assets/models/wake_word/<model>.onnx
            standard_path = Path("assets/models/wake_word") / f"{c_name}.onnx"
            if standard_path.exists():
                entry = _LoadedModelEntry(
                    name=c_name,
                    phrase=c_phrase,
                    oww_target=str(standard_path),
                    score_key=standard_path.stem,
                    threshold=c_thresh,
                    is_custom=True,
                )
                resolved_entries.append(entry)
                oww_targets.append(str(standard_path))
                log.info(
                    "Registered custom wake model '{name}' ('{phrase}') from '{path}' (threshold={thresh})",
                    name=c_name, phrase=c_phrase, path=standard_path, thresh=c_thresh,
                )
                continue

            # 3. Built-in pre-trained openWakeWord models
            lookup_key = c_model or c_name
            if lookup_key in _MODEL_MAP:
                mapping = _MODEL_MAP[lookup_key]
                entry = _LoadedModelEntry(
                    name=c_name,
                    phrase=c_phrase or mapping.get("phrase", c_name),
                    oww_target=mapping["oww_name"],
                    score_key=mapping["score_key"],
                    threshold=c_thresh,
                    is_custom=False,
                )
                resolved_entries.append(entry)
                oww_targets.append(mapping["oww_name"])
                log.info(
                    "Registered pre-trained wake model '{name}' ('{phrase}') (threshold={thresh})",
                    name=c_name, phrase=entry.phrase, thresh=c_thresh,
                )
                continue

            # 4. Unknown / missing model
            log.warning(
                "Wake word model '{name}' ('{phrase}') is not a known built-in model and was not found at '{path}'. "
                "Model is NOT active.",
                name=c_name, phrase=c_phrase, path=standard_path,
            )
            self._missing_models.append({
                "name": c_name,
                "phrase": c_phrase,
                "expected_path": str(standard_path),
            })

        if not resolved_entries:
            log.warning(
                "No wake word models could be loaded. Missing models: {missing}",
                missing=self._missing_models,
            )
            self._loaded = False
            return

        try:
            _backend = _detect_inference_framework()
            log.info(
                "[VOICE DIAG] Initializing OpenWakeWord with {n} active model(s) on '{backend}' backend.",
                n=len(oww_targets),
                backend=_backend,
            )

            self._oww_model = OWWModel(
                wakeword_models=oww_targets,
                inference_framework=_backend,
            )
            self._loaded_models = resolved_entries
            self._loaded = True

            log.info(
                "OpenWakeWord successfully loaded {n} active model(s): {models}",
                n=len(self._loaded_models),
                models=[m.name for m in self._loaded_models],
            )

        except Exception as exc:
            log.error("Failed to load wake word model(s): {exc}", exc=exc)
            raise

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Run inference on one audio chunk across all loaded models.

        Returns
        -------
        float
            Confidence score of the triggering wake word (or max score).
        """
        self._is_detected = False
        if not self._loaded or self._oww_model is None or not self._loaded_models:
            return 0.0

        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        try:
            scores = self._oww_model.predict(audio_int16)

            if not self._first_chunk_logged:
                self._first_chunk_logged = True
                log.info(
                    "[VOICE DIAG-8] First OWW predict() output keys: {keys}",
                    keys=list(scores.keys()),
                )

            import time as _time
            now = _time.monotonic()

            best_score = 0.0
            triggered_entry: _LoadedModelEntry | None = None
            triggered_score = 0.0

            for entry in self._loaded_models:
                score = scores.get(entry.score_key, 0.0)
                if score == 0.0 and scores:
                    # Fallback key matching if exact score_key had variant suffix
                    for k, v in scores.items():
                        if entry.score_key in k:
                            score = v
                            break

                if score > best_score:
                    best_score = score

                if score >= entry.threshold:
                    # Check per-model & global debounce
                    last_time = self._last_detection_times.get(entry.name, 0.0)
                    global_diff = now - self._last_global_detection_time
                    model_diff = now - last_time

                    if global_diff >= self._debounce_seconds and model_diff >= self._debounce_seconds:
                        triggered_entry = entry
                        triggered_score = score
                        break
                    else:
                        log.debug(
                            "Wake debounce: model='{m}' score={s:.4f} suppressed (cooldown)",
                            m=entry.name, s=score,
                        )

            if triggered_entry is not None:
                self._last_detection_times[triggered_entry.name] = now
                self._last_global_detection_time = now
                self._last_detected_model = triggered_entry.name
                self._last_detected_phrase = triggered_entry.phrase
                self._last_detected_score = float(triggered_score)
                self._is_detected = True
                return float(triggered_score)

            return float(best_score)

        except Exception as exc:  # noqa: BLE001
            log.debug("Wake word inference error (non-fatal): {exc}", exc=exc)
            return 0.0

    def reset_buffer(self) -> None:
        """Clear the OWW prediction score window."""
        if self._oww_model is not None:
            try:
                pb = self._oww_model.prediction_buffer
                if hasattr(pb, 'clear'):
                    pb.clear()
                    log.debug("OWW prediction_buffer cleared.")
            except Exception as exc:  # noqa: BLE001
                log.debug("OWW prediction_buffer clear failed (non-fatal): {exc}", exc=exc)

    def unload(self) -> None:
        """Release model resources."""
        if self._oww_model is not None:
            del self._oww_model
            self._oww_model = None
        self._loaded = False
        self._loaded_models.clear()
        log.debug("Wake word model unloaded.")

    # ── Inspection Properties & Helpers ───────────────────────────────────

    @property
    def model_name(self) -> str:
        if self._last_detected_model:
            return self._last_detected_model
        if self._loaded_models:
            return self._loaded_models[0].name
        return self._model_name

    @property
    def wake_phrase(self) -> str:
        if self._last_detected_phrase:
            return self._last_detected_phrase
        if self._loaded_models:
            return self._loaded_models[0].phrase
        return self.model_name

    @property
    def last_detected_model(self) -> str:
        return self._last_detected_model

    @property
    def last_detected_phrase(self) -> str:
        return self._last_detected_phrase

    @property
    def is_detected(self) -> bool:
        return self._is_detected

    @property
    def chunk_size(self) -> int:
        return _CHUNK_SIZE

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def loaded_models(self) -> list[str]:
        return [m.name for m in self._loaded_models]

    @property
    def missing_models(self) -> list[dict[str, str]]:
        return list(self._missing_models)

    def get_model_threshold(self, name: str) -> float:
        for m in self._loaded_models:
            if m.name == name:
                return m.threshold
        return self._threshold


# ── NullWakeWordModel — Testing / Development ──────────────────────────────────


class NullWakeWordModel(WakeWordModel):
    """
    A no-op wake word model for testing and development.
    """

    def __init__(
        self,
        trigger_after_chunks: int = 0,
        always_trigger: bool = False,
        model_name: str = "null_wake_word",
        wake_phrase: str = "Hey Spidy",
    ) -> None:
        self._trigger_after = trigger_after_chunks
        self._always_trigger = always_trigger
        self._model_name = model_name
        self._wake_phrase = wake_phrase
        self._chunk_count = 0
        self._forced = False
        self._loaded = False
        self._is_detected = False

    def load(self, model_path: str | None = None) -> None:
        self._loaded = True
        log.debug("NullWakeWordModel loaded (testing mode).")

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        self._is_detected = False
        if not self._loaded:
            return 0.0
        self._chunk_count += 1

        if self._forced:
            self._forced = False
            self._is_detected = True
            return 1.0

        if self._always_trigger:
            self._is_detected = True
            return 1.0

        if self._trigger_after > 0 and self._chunk_count >= self._trigger_after:
            self._chunk_count = 0
            self._is_detected = True
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
        return self._model_name

    @property
    def wake_phrase(self) -> str:
        return self._wake_phrase

    @property
    def last_detected_model(self) -> str:
        return self._model_name

    @property
    def last_detected_phrase(self) -> str:
        return self._wake_phrase

    @property
    def is_detected(self) -> bool:
        return self._is_detected

    @property
    def chunk_size(self) -> int:
        return _CHUNK_SIZE

    @property
    def is_loaded(self) -> bool:
        return self._loaded
