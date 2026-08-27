"""
Streaming OpenWakeWord Model Generator & Trainer for SPIDY
==========================================================
Produces genuine openWakeWord compatible ONNX classification models for:
1. "Hey Spidy"
2. "Wake up Spidy"

Extracts streaming feature trajectories using openWakeWord's AudioFeatures
streaming engine to ensure exact alignment between training data and real-time
inference buffers.
"""

from __future__ import annotations

import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
from openwakeword.utils import AudioFeatures

sys.stdout.reconfigure(encoding="utf-8")


class WakeWordClassifier(nn.Module):
    """OpenWakeWord classification head for (1, 16, 96) streaming embeddings."""

    def __init__(self, input_dim: int = 16 * 96, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_tts_voice():
    import piper
    voice_path = "assets/models/tts/en_US-ryan-high/en_US-ryan-high.onnx"
    config_path = "assets/models/tts/en_US-ryan-high/en_US-ryan-high.onnx.json"
    return piper.PiperVoice.load(voice_path, config_path=config_path)


def synthesize_raw(voice, text: str, speed: float = 1.0, pitch_shift: float = 1.0) -> np.ndarray:
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


def stream_audio_through_af(af: AudioFeatures, audio: np.ndarray, lead_sec: float = 0.8, trail_sec: float = 0.8) -> tuple[list[np.ndarray], int]:
    """Pass an audio clip through AudioFeatures in 1280-sample streaming chunks."""
    af.reset()
    lead = np.zeros(int(16000 * lead_sec), dtype=np.float32)
    trail = np.zeros(int(16000 * trail_sec), dtype=np.float32)
    full = np.concatenate([lead, audio, trail])
    full_int16 = (np.clip(full, -1.0, 1.0) * 32767).astype(np.int16)

    frames: list[np.ndarray] = []
    for i in range(0, len(full_int16) - 1280, 1280):
        af(full_int16[i : i + 1280])
        # Only start collecting after the 5 initial warm-up frames
        if len(frames) >= 0:
            frames.append(af.get_features(16))

    speech_end_sample = len(lead) + len(audio)
    peak_frame_idx = speech_end_sample // 1280
    return frames, peak_frame_idx


def build_streaming_dataset(
    voice,
    af: AudioFeatures,
    target_phrase: str,
    phonetic_variants: list[str],
    hard_negatives: list[str],
    num_positive_variations: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Build training dataset of (16, 96) feature frames with labels."""
    print(f"\n[BUILD] Generating streaming dataset for '{target_phrase}'...")
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    # 1. Positives
    print(f"  Synthesizing {num_positive_variations} augmented streaming positive runs...")
    for _ in range(num_positive_variations):
        text = random.choice(phonetic_variants)
        speed = random.uniform(0.82, 1.22)
        gain = random.uniform(0.4, 1.3)
        noise_level = random.uniform(0.001, 0.03)
        lead_s = random.uniform(0.6, 1.2)
        trail_s = random.uniform(0.6, 1.2)

        audio = synthesize_raw(voice, text, speed=speed) * gain
        if noise_level > 0:
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level

        frames, peak_idx = stream_audio_through_af(af, audio, lead_sec=lead_s, trail_sec=trail_s)

        # In positive run:
        # Peak frame and adjacent ±1 frame are positive (y = 1.0)
        for idx, feat in enumerate(frames):
            if idx < 5:  # skip initial buffer warm-up
                continue
            if abs(idx - peak_idx) <= 1:
                X_list.append(feat[0])
                y_list.append(1.0)
            elif abs(idx - peak_idx) >= 6:  # lead-in and tail are negative
                if random.random() < 0.3:   # sub-sample lead/tail
                    X_list.append(feat[0])
                    y_list.append(0.0)

    # 2. Hard Negatives & General speech
    common_phrases = [
        "open Edge", "open Notepad", "what is Python", "how is the weather",
        "search YouTube for tutorials", "hello there", "good morning",
        "play some music", "stop listening", "cancel that", "thank you",
        "tell me a joke", "open file explorer", "who made you", "are you awake",
        "turn on the lights", "launch the browser", "check my email",
    ]
    all_negatives = hard_negatives * 3 + common_phrases * 2

    print(f"  Synthesizing {len(all_negatives)} streaming negative runs...")
    for text in all_negatives:
        speed = random.uniform(0.85, 1.20)
        gain = random.uniform(0.4, 1.2)
        noise_level = random.uniform(0.001, 0.03)

        audio = synthesize_raw(voice, text, speed=speed) * gain
        if noise_level > 0:
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level

        frames, _ = stream_audio_through_af(af, audio, lead_sec=0.6, trail_sec=0.6)
        for idx, feat in enumerate(frames):
            if idx >= 5:
                X_list.append(feat[0])
                y_list.append(0.0)

    # 3. Pure noise & room ambience frames
    af.reset()
    noise_audio = (np.random.randn(16000 * 5) * 300).astype(np.int16)
    for i in range(0, len(noise_audio) - 1280, 1280):
        af(noise_audio[i : i + 1280])
        X_list.append(af.get_features(16)[0])
        y_list.append(0.0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Balance classes if negatives outnumber positives heavily
    pos_indices = np.where(y == 1.0)[0]
    neg_indices = np.where(y == 0.0)[0]
    print(f"  Extracted frames: Positives={len(pos_indices)} | Negatives={len(neg_indices)}")

    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def train_streaming_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    output_onnx_path: str,
    epochs: int = 35,
    lr: float = 0.002,
) -> tuple[float, dict[str, float]]:
    print(f"\n--- Training streaming classifier for {model_name} ---")

    # Split train/val
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Compute class weight for BCE
    num_pos = max(1, int(np.sum(y_train == 1.0)))
    num_neg = max(1, int(np.sum(y_train == 0.0)))
    pos_weight = torch.tensor([min(10.0, num_neg / num_pos)], dtype=torch.float32)

    model = WakeWordClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Note: WakeWordClassifier has Sigmoid at the end. For BCEWithLogitsLoss, we create a linear wrapper
    # or use BCELoss directly with weighted sampling.
    criterion = nn.BCELoss()

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)

    batch_size = 64
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).numpy().flatten()
                val_pos = val_preds[y_val == 1.0]
                val_neg = val_preds[y_val == 0.0]
                avg_pos = float(np.mean(val_pos)) if len(val_pos) > 0 else 0.0
                max_neg = float(np.max(val_neg)) if len(val_neg) > 0 else 0.0
                p95_neg = float(np.percentile(val_neg, 95)) if len(val_neg) > 0 else 0.0
                print(
                    f"  Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss/batches:.4f} | "
                    f"Val Pos Avg: {avg_pos:.4f} | Val Neg Max: {max_neg:.4f} (p95: {p95_neg:.4f})"
                )

    model.eval()
    with torch.no_grad():
        all_preds = model(torch.from_numpy(X).float()).numpy().flatten()
        pos = all_preds[y == 1.0]
        neg = all_preds[y == 0.0]

    min_pos = float(np.percentile(pos, 5))
    max_neg = float(np.percentile(neg, 99))
    chosen_threshold = float(np.clip(round((min_pos + max_neg) / 2.0, 2), 0.30, 0.45))

    stats = {
        "pos_min": float(np.min(pos)),
        "pos_mean": float(np.mean(pos)),
        "pos_p5": min_pos,
        "neg_max": float(np.max(neg)),
        "neg_mean": float(np.mean(neg)),
        "neg_p99": max_neg,
        "chosen_threshold": chosen_threshold,
    }

    # Export ONNX
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    dummy_input = torch.randn(1, 16, 96, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
    print(f"✓ Exported ONNX model to: {output_onnx_path}")
    return chosen_threshold, stats


def evaluate_streaming_audio(oww_model_path: str, model_key: str, voice, test_phrases: list[tuple[str, bool]]) -> None:
    from openwakeword.model import Model as OWWModel

    oww = OWWModel(wakeword_models=[oww_model_path], inference_framework="onnx")
    print(f"\n--- Streaming Evaluation for '{model_key}' ---")

    for phrase, is_pos in test_phrases:
        audio = synthesize_raw(voice, phrase)
        lead = np.zeros(16000, dtype=np.float32)
        trail = np.zeros(16000, dtype=np.float32)
        full = np.concatenate([lead, audio, trail])
        full_int16 = (np.clip(full, -1.0, 1.0) * 32767).astype(np.int16)

        oww.prediction_buffer[model_key].clear()
        scores = []
        for i in range(0, len(full_int16) - 1280, 1280):
            chunk = full_int16[i : i + 1280]
            pred = oww.predict(chunk)
            scores.append(pred.get(model_key, 0.0))

        max_score = max(scores) if scores else 0.0
        status = "PASS" if (is_pos and max_score >= 0.35) or (not is_pos and max_score < 0.35) else "FAIL"
        print(f"  [{status}] {phrase:35} (Expected: {'POS' if is_pos else 'NEG'}) -> Max Score: {max_score:.4f}")


def main() -> None:
    print("=" * 60)
    print("SPIDY STREAMING WAKE-WORD MODEL TRAINER")
    print("=" * 60)

    voice = load_tts_voice()
    af = AudioFeatures()

    # 1. HEY SPIDY
    hs_variants = [
        "Hey Spidy", "Hey Spidy!", "Hey Spidy.", "Hey Spidee",
        "Hey Spydee", "Hey Spydi", "Hey, Spidy",
    ]
    hs_hard_negs = [
        "Hi Spidy", "Hey Steve", "Hey Siri", "Tell me about Spidy",
        "Hey Spiderman", "Spidy", "Hey Spy", "Hey Buddy", "Hey Sammy",
        "What is Spidy", "Who is Spidy", "Hey Google", "Hey Jarvis",
        "Spider", "Spiderman", "Hey Spider", "Good morning Spidy",
    ]
    X_hs, y_hs = build_streaming_dataset(voice, af, "Hey Spidy", hs_variants, hs_hard_negs, num_positive_variations=180)
    hs_thresh, hs_stats = train_streaming_model(
        X_hs, y_hs, "hey_spidy", "assets/models/wake_word/hey_spidy.onnx", epochs=35, lr=0.002
    )

    # 2. WAKE UP SPIDY
    wu_variants = [
        "Wake up Spidy", "Wake up Spidy!", "Wake up Spidy.", "Wake up Spidee",
        "Wake up Spydee", "Wake up Spydi", "Wake up, Spidy",
    ]
    wu_hard_negs = [
        "Wake up", "Wake up Spider", "Wake up Spiderman", "Tell me to wake up",
        "Spidy", "Wake up buddy", "Wake up Siri", "Wake Spidy up", "Get up Spidy",
        "Who is Spidy", "Tell me about Spidy", "Wake up Jarvis", "Wake up Sammy",
        "Wake up Steve", "Wake up Google", "Wake", "Wide awake", "Hey Spidy",
    ]
    X_wu, y_wu = build_streaming_dataset(voice, af, "Wake up Spidy", wu_variants, wu_hard_negs, num_positive_variations=180)
    wu_thresh, wu_stats = train_streaming_model(
        X_wu, y_wu, "wake_up_spidy", "assets/models/wake_word/wake_up_spidy.onnx", epochs=35, lr=0.002
    )

    # 3. STREAMING VALIDATION
    evaluate_streaming_audio(
        "assets/models/wake_word/hey_spidy.onnx", "hey_spidy", voice,
        [
            ("Hey Spidy", True),
            ("Hey Spidy open Notepad", True),
            ("Hi Spidy", False),
            ("Tell me about Spidy", False),
            ("Who is Spidy?", False),
            ("Hey Steve", False),
            ("Hey Spiderman", False),
            ("Wake up Spidy", False),
            ("open Edge", False),
        ]
    )

    evaluate_streaming_audio(
        "assets/models/wake_word/wake_up_spidy.onnx", "wake_up_spidy", voice,
        [
            ("Wake up Spidy", True),
            ("Wake up Spidy, open Notepad", True),
            ("Wake up", False),
            ("Wake up Spiderman", False),
            ("Tell me about Spidy", False),
            ("Who is Spidy?", False),
            ("Wake up Steve", False),
            ("Hey Spidy", False),
            ("open Edge", False),
        ]
    )


if __name__ == "__main__":
    main()
