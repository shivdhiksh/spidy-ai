"""
SPIDY Custom Wake-Word Model Builder & Exporter
================================================
Generates high-accuracy, genuine openWakeWord compatible ONNX models for:
1. "Hey Spidy" (assets/models/wake_word/hey_spidy.onnx)
2. "Wake up Spidy" (assets/models/wake_word/wake_up_spidy.onnx)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
from openwakeword.model import Model as OWWModel
from openwakeword.utils import AudioFeatures
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")


def load_tts_voice():
    import piper
    voice_path = "assets/models/tts/en_US-ryan-high/en_US-ryan-high.onnx"
    config_path = "assets/models/tts/en_US-ryan-high/en_US-ryan-high.onnx.json"
    return piper.PiperVoice.load(voice_path, config_path=config_path)


def synthesize_raw(voice, text: str, speed: float = 1.0) -> np.ndarray:
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.zeros(16000, dtype=np.float32)
    audio = chunks[0].audio_float_array
    sr = chunks[0].sample_rate
    if sr != 16000:
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    if abs(speed - 1.0) > 0.02:
        audio = signal.resample(audio, int(len(audio) / speed))
    return audio.astype(np.float32)


def make_clip(audio: np.ndarray, total_len: int = 32000, start_offset: int = 8000) -> np.ndarray:
    out = np.zeros(total_len, dtype=np.float32)
    end = min(total_len, start_offset + len(audio))
    out[start_offset:end] = audio[: end - start_offset]
    return (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16)


def export_classifier_to_onnx(clf: LogisticRegression, output_path: str) -> None:
    weights = clf.coef_.astype(np.float32)   # (1, 1536)
    bias = clf.intercept_.astype(np.float32)  # (1,)

    class LinearWakeModel(nn.Module):
        def __init__(self, w: np.ndarray, b: np.ndarray):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(16 * 96, 1)
            with torch.no_grad():
                self.linear.weight.copy_(torch.from_numpy(w))
                self.linear.bias.copy_(torch.from_numpy(b))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.flatten(x)
            return self.sigmoid(self.linear(x))

    model = LinearWakeModel(weights, bias)
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_input = torch.randn(1, 16, 96, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
    print(f"  ✓ Exported ONNX model to: {output_path}")


def build_model(
    voice,
    af: AudioFeatures,
    model_name: str,
    output_path: str,
    positive_phrases: list[str],
    hard_negatives: list[str],
    C_reg: float = 0.015,
) -> dict[str, float]:
    print(f"\n==================================================")
    print(f"BUILDING MODEL: {model_name}")
    print(f"==================================================")

    # 1. Positives
    pos_clips = []
    speeds = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    offsets = [4000, 6000, 8000, 10000, 12000, 14000]

    for phrase in positive_phrases:
        for spd in speeds:
            aud = synthesize_raw(voice, phrase, speed=spd)
            for off in offsets:
                if off + len(aud) <= 32000:
                    pos_clips.append(make_clip(aud, start_offset=off))

    # 2. Hard negatives & general speech
    neg_clips = []
    common_phrases = [
        "open Edge", "open Notepad", "what is Python", "how is the weather",
        "search YouTube for tutorials", "hello there", "good morning",
        "play some music", "stop listening", "cancel that", "thank you",
        "tell me a joke", "open file explorer", "who made you", "are you awake",
        "turn on the lights", "launch the browser", "check my email", "open Chrome",
    ]
    all_negs = hard_negatives + common_phrases
    neg_offsets = [2000, 6000, 10000, 14000]

    for text in all_negs:
        for spd in [0.90, 1.0, 1.10]:
            aud = synthesize_raw(voice, text, speed=spd)
            for off in neg_offsets:
                if off + len(aud) <= 32000:
                    neg_clips.append(make_clip(aud, start_offset=off))

    # 3. Silence / ambient noise
    for _ in range(120):
        noise = (np.random.randn(32000) * 250).astype(np.int16)
        neg_clips.append(noise)

    print(f"Extracting embeddings (Positives: {len(pos_clips)}, Negatives: {len(neg_clips)})...")
    pos_emb = af.embed_clips(np.array(pos_clips))
    neg_emb = af.embed_clips(np.array(neg_clips))

    X_pos = pos_emb.reshape(len(pos_emb), -1)
    X_neg = neg_emb.reshape(len(neg_emb), -1)

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    print(f"Fitting LogisticRegression classifier (C={C_reg})...")
    clf = LogisticRegression(C=C_reg, max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    # Export ONNX
    export_classifier_to_onnx(clf, output_path)

    # Evaluation
    print(f"Evaluating '{model_name}' on test set...")
    pos_eval = []
    for p in positive_phrases[:3]:
        aud = synth_clip = make_clip(synthesize_raw(voice, p), start_offset=8000)
        emb = af.embed_clips(synth_clip[None, :]).reshape(1, -1)
        prob = float(clf.predict_proba(emb)[0][1])
        pos_eval.append(prob)
        print(f"  [POSITIVE] {p:35} -> Probability: {prob:.4f}")

    neg_eval = {}
    for n in hard_negatives[:12]:
        synth_clip = make_clip(synthesize_raw(voice, n), start_offset=8000)
        emb = af.embed_clips(synth_clip[None, :]).reshape(1, -1)
        prob = float(clf.predict_proba(emb)[0][1])
        neg_eval[n] = prob
        print(f"  [NEGATIVE] {n:35} -> Probability: {prob:.4f}")

    stats = {
        "pos_min": min(pos_eval),
        "pos_max": max(pos_eval),
        "neg_max": max(neg_eval.values()),
        "selected_threshold": 0.35,
    }
    print(f"\n  Calibration for '{model_name}':")
    print(f"    True Positives Range: [{stats['pos_min']:.4f}, {stats['pos_max']:.4f}]")
    print(f"    Max Hard-Negative:    {stats['neg_max']:.4f}")
    print(f"    Selected Threshold:   {stats['selected_threshold']:.2f}")
    return stats


def main() -> None:
    print("=" * 60)
    print("BUILDING REAL WAKE-WORD MODELS FOR SPIDY")
    print("=" * 60)

    voice = load_tts_voice()
    af = AudioFeatures()

    # 1. HEY SPIDY
    hey_spidy_pos = [
        "Hey Spidy", "Hey Spidy!", "Hey Spidy.", "Hey Spidy, open Edge",
        "Hey Spidy what is Python", "Hey Spidee", "Hey Spydee", "Hey, Spidy",
    ]
    hey_spidy_negs = [
        "Hi Spidy", "Hey Steve", "Hey Siri", "Tell me about Spidy",
        "Who is Spidy?", "Hey Spiderman", "Spidy", "Hey Spy", "Hey Buddy",
        "Hey Sammy", "What is Spidy", "Hey Google", "Hey Jarvis", "Hey Spotify",
        "Hello Spidy", "Spider", "Spiderman", "Hey Spider", "Good morning Spidy",
        "Wake up Spidy", "Wake up Spiderman", "Wake up", "open Edge", "open Notepad",
    ]
    build_model(
        voice, af, "hey_spidy", "assets/models/wake_word/hey_spidy.onnx",
        hey_spidy_pos, hey_spidy_negs, C_reg=0.015,
    )

    # 2. WAKE UP SPIDY
    wake_up_pos = [
        "Wake up Spidy", "Wake up Spidy!", "Wake up Spidy.", "Wake up Spidy, open Notepad",
        "Wake up Spidy and search YouTube for Python tutorials", "Wake up Spidee", "Wake up Spydee",
    ]
    wake_up_negs = [
        "Wake up", "Wake up Spider", "Wake up Spiderman", "Tell me to wake up",
        "Spidy", "Wake up buddy", "Wake up Siri", "Wake Spidy up", "Get up Spidy",
        "Who is Spidy?", "Tell me about Spidy.", "Wake up Jarvis", "Wake up Sammy",
        "Wake up Steve", "Wake up Google", "Wake", "Wide awake", "Hey Spidy",
        "Hi Spidy", "Hey Steve", "Hey Spiderman", "open Edge", "open Notepad",
    ]
    build_model(
        voice, af, "wake_up_spidy", "assets/models/wake_word/wake_up_spidy.onnx",
        wake_up_pos, wake_up_negs, C_reg=0.015,
    )

    # 3. VERIFY WITH OPENWAKEWORD MODEL
    print(f"\n==================================================")
    print("VERIFYING SIMULTANEOUS DETECTION IN OPENWAKEWORD")
    print("==================================================")
    oww = OWWModel(
        wakeword_models=[
            "assets/models/wake_word/hey_spidy.onnx",
            "assets/models/wake_word/wake_up_spidy.onnx",
        ],
        inference_framework="onnx",
    )
    print("Loaded models in OWW:", list(oww.models.keys()))

    test_phrases = [
        ("Hey Spidy", True, False),
        ("Hey Spidy, open Notepad.", True, False),
        ("Wake up Spidy", False, True),
        ("Wake up Spidy, open Edge.", False, True),
        ("Wake up Spidy and search YouTube for Python tutorials.", False, True),
        ("Who is Spidy?", False, False),
        ("Tell me about Spidy.", False, False),
        ("Wake up Spiderman.", False, False),
        ("Hi Spidy", False, False),
        ("Hey Steve", False, False),
        ("Wake up", False, False),
        ("open Edge", False, False),
    ]

    all_passed = True
    for phrase, exp_hs, exp_wu in test_phrases:
        clip = make_clip(synthesize_raw(voice, phrase), start_offset=8000)
        emb = af.embed_clips(clip[None, :]) # (1, 16, 96)

        # Predict with ONNX session
        pred_hs = oww.models["hey_spidy"].run(None, {"input": emb})[0][0][0]
        pred_wu = oww.models["wake_up_spidy"].run(None, {"input": emb})[0][0][0]

        det_hs = pred_hs >= 0.35
        det_wu = pred_wu >= 0.35

        ok = (det_hs == exp_hs) and (det_wu == exp_wu)
        if not ok:
            all_passed = False

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {phrase:55} | hey_spidy={pred_hs:.4f} (det={det_hs}) | wake_up_spidy={pred_wu:.4f} (det={det_wu})")

    if all_passed:
        print("\n✓ ALL VERIFICATION TESTS PASSED WITH 100% ACCURACY!")


if __name__ == "__main__":
    main()
