"""
Custom OpenWakeWord Model Generator & Trainer for SPIDY
======================================================
Trains genuine openWakeWord compatible ONNX classification models for:
1. "Hey Spidy"
2. "Wake up Spidy"

Pipeline:
1. Synthesizes positive phrases and hard negatives using Piper TTS.
2. Applies acoustic augmentations (speed/tempo, noise injection, gain, time shifts).
3. Extracts openWakeWord 16-frame 96-dim embeddings via AudioFeatures.
4. Trains PyTorch neural classification heads with BCE + regularization.
5. Calibrates per-model detection thresholds on positives and hard negatives.
6. Exports compatible ONNX models to assets/models/wake_word/.
7. Validates end-to-end inference using openwakeword.model.Model.
"""

from __future__ import annotations

import io
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
from openwakeword.utils import AudioFeatures

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")


class WakeWordClassifier(nn.Module):
    """OpenWakeWord classification head for (1, 16, 96) embeddings."""

    def __init__(self, input_dim: int = 16 * 96, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_tts_voice() -> Any:
    import piper
    voice_path = "assets/models/tts/en_US-ryan-high/en_US-ryan-high.onnx"
    config_path = "assets/models/tts/en_US-ryan-high/en_US-ryan-high.onnx.json"
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"TTS voice model not found at {voice_path}")
    return piper.PiperVoice.load(voice_path, config_path=config_path)


def synthesize_phrase(voice: Any, text: str) -> np.ndarray:
    """Synthesize raw float32 audio at 16kHz for a given text."""
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.zeros(16000, dtype=np.float32)
    audio = chunks[0].audio_float_array
    sr = chunks[0].sample_rate
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
    return audio.astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    target_length: int = 32000, # 2.0 seconds at 16kHz
    speed_factor: float = 1.0,
    noise_level: float = 0.0,
    gain: float = 1.0,
    shift_pos: float = 0.5, # 0.0 = left, 0.5 = center, 1.0 = right
) -> np.ndarray:
    """Apply speed adjustment, time alignment, gain, and noise augmentation."""
    # Speed adjustment
    if abs(speed_factor - 1.0) > 0.01 and len(audio) > 100:
        new_len = int(len(audio) / speed_factor)
        audio_mod = signal.resample(audio, new_len)
    else:
        audio_mod = audio.copy()

    # Gain scaling
    audio_mod = audio_mod * gain

    # Fit into target 2.0s buffer
    out = np.zeros(target_length, dtype=np.float32)
    if len(audio_mod) >= target_length:
        out = audio_mod[:target_length]
    else:
        max_start = target_length - len(audio_mod)
        start_idx = int(max_start * np.clip(shift_pos, 0.0, 1.0))
        out[start_idx : start_idx + len(audio_mod)] = audio_mod

    # Noise injection
    if noise_level > 0:
        noise = np.random.randn(target_length).astype(np.float32)
        out = out + noise * noise_level

    # Normalization & clipping
    out = np.clip(out, -1.0, 1.0)
    return (out * 32767).astype(np.int16)


def generate_dataset_for_phrase(
    voice: Any,
    af: AudioFeatures,
    target_phrase: str,
    phonetic_variants: list[str],
    hard_negatives: list[str],
    num_positives: int = 600,
    num_negatives: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate audio embeddings and binary labels for wake phrase."""
    print(f"\n--- Generating dataset for '{target_phrase}' ---")

    # 1. Base syntheses for positives
    pos_audios = [synthesize_phrase(voice, v) for v in phonetic_variants]

    # 2. Base syntheses for negatives
    common_phrases = [
        "open Edge", "open Notepad", "what is Python", "how is the weather",
        "search YouTube for tutorials", "hello there", "good morning",
        "play some music", "stop listening", "cancel that", "thank you",
        "tell me a joke", "open file explorer", "who made you", "are you awake",
    ]
    all_neg_texts = hard_negatives + common_phrases
    neg_audios = [synthesize_phrase(voice, n) for n in all_neg_texts]

    # Generate Positives
    pos_clips = []
    for _ in range(num_positives):
        base = random.choice(pos_audios)
        speed = random.uniform(0.80, 1.25)
        gain = random.uniform(0.3, 1.4)
        noise = random.uniform(0.001, 0.04)
        shift = random.uniform(0.1, 0.9)
        clip = augment_audio(base, speed_factor=speed, gain=gain, noise_level=noise, shift_pos=shift)
        pos_clips.append(clip)

    # Generate Negatives
    neg_clips = []
    # Pure noise / silence clips
    for _ in range(100):
        noise = (np.random.randn(32000) * random.uniform(50, 800)).astype(np.int16)
        neg_clips.append(noise)

    for _ in range(num_negatives - 100):
        base = random.choice(neg_audios)
        speed = random.uniform(0.80, 1.25)
        gain = random.uniform(0.3, 1.4)
        noise = random.uniform(0.001, 0.04)
        shift = random.uniform(0.1, 0.9)
        clip = augment_audio(base, speed_factor=speed, gain=gain, noise_level=noise, shift_pos=shift)
        neg_clips.append(clip)

    pos_array = np.array(pos_clips, dtype=np.int16)
    neg_array = np.array(neg_clips, dtype=np.int16)

    print(f"Extracting openWakeWord embeddings for {len(pos_clips)} positives...")
    pos_emb = af.embed_clips(pos_array) # (num_pos, 16, 96)

    print(f"Extracting openWakeWord embeddings for {len(neg_clips)} negatives...")
    neg_emb = af.embed_clips(neg_array) # (num_neg, 16, 96)

    X = np.vstack([pos_emb, neg_emb])
    y = np.concatenate([np.ones(len(pos_emb), dtype=np.float32), np.zeros(len(neg_emb), dtype=np.float32)])

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def train_and_export_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    output_onnx_path: str,
    epochs: int = 40,
    lr: float = 0.001,
) -> tuple[float, dict[str, float]]:
    """Train classification model, evaluate, and export to ONNX."""
    print(f"\n--- Training {model_name} on {len(X)} samples ---")

    # Train / Val split (80 / 20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = WakeWordClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)

    batch_size = 64
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).numpy().flatten()
                val_pos_scores = val_preds[y_val == 1.0]
                val_neg_scores = val_preds[y_val == 0.0]
                avg_pos = float(np.mean(val_pos_scores)) if len(val_pos_scores) > 0 else 0.0
                max_neg = float(np.max(val_neg_scores)) if len(val_neg_scores) > 0 else 0.0
                print(
                    f"Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss/max(1,num_batches):.4f} | "
                    f"Val Pos Avg: {avg_pos:.4f} | Val Neg Max: {max_neg:.4f}"
                )

    # Threshold calibration
    model.eval()
    with torch.no_grad():
        all_preds = model(torch.from_numpy(X).float()).numpy().flatten()
        pos_scores = all_preds[y == 1.0]
        neg_scores = all_preds[y == 0.0]

    min_pos = float(np.percentile(pos_scores, 5))
    max_neg = float(np.percentile(neg_scores, 99))
    chosen_threshold = float(np.clip(round((min_pos + max_neg) / 2.0, 2), 0.30, 0.45))

    stats = {
        "pos_min": float(np.min(pos_scores)),
        "pos_mean": float(np.mean(pos_scores)),
        "pos_p5": min_pos,
        "neg_max": float(np.max(neg_scores)),
        "neg_mean": float(np.mean(neg_scores)),
        "neg_p99": max_neg,
        "chosen_threshold": chosen_threshold,
    }
    print(f"Calibration for {model_name}:")
    print(f"  True Positives: min={stats['pos_min']:.4f}, mean={stats['pos_mean']:.4f}, p5={stats['pos_p5']:.4f}")
    print(f"  Negatives:      max={stats['neg_max']:.4f}, mean={stats['neg_mean']:.4f}, p99={stats['neg_p99']:.4f}")
    print(f"  Selected Threshold: {chosen_threshold}")

    # Export to ONNX
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


def main() -> None:
    print("=" * 60)
    print("SPIDY WAKE-WORD MODEL GENERATOR & TRAINER")
    print("=" * 60)

    voice = load_tts_voice()
    af = AudioFeatures()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HEY SPIDY
    # ─────────────────────────────────────────────────────────────────────────
    hey_spidy_variants = [
        "Hey Spidy", "Hey Spidy!", "Hey Spidy.", "Hey Spidee",
        "Hey Spydee", "Hey Spydi", "Hey, Spidy", "Hey Spidee!",
    ]
    hey_spidy_hard_negs = [
        "Hi Spidy", "Hey Steve", "Hey Siri", "Tell me about Spidy",
        "Hey Spiderman", "Spidy", "Hey Spy", "Hey Buddy", "Hey Sammy",
        "What is Spidy", "Who is Spidy", "Hey Google", "Hey Jarvis",
        "Hey Spotify", "Hello Spidy", "Spider", "Spiderman", "Hey Spider",
        "Can you hear me", "Good morning Spidy", "Spidy are you there",
    ]
    X_hs, y_hs = generate_dataset_for_phrase(
        voice, af, "Hey Spidy", hey_spidy_variants, hey_spidy_hard_negs,
        num_positives=700, num_negatives=900,
    )
    hs_thresh, hs_stats = train_and_export_model(
        X_hs, y_hs, "hey_spidy", "assets/models/wake_word/hey_spidy.onnx",
        epochs=40, lr=0.0015,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. WAKE UP SPIDY
    # ─────────────────────────────────────────────────────────────────────────
    wake_up_variants = [
        "Wake up Spidy", "Wake up Spidy!", "Wake up Spidy.", "Wake up Spidee",
        "Wake up Spydee", "Wake up Spydi", "Wake up, Spidy",
    ]
    wake_up_hard_negs = [
        "Wake up", "Wake up Spider", "Wake up Spiderman", "Tell me to wake up",
        "Spidy", "Wake up buddy", "Wake up Siri", "Wake Spidy up", "Get up Spidy",
        "Who is Spidy", "Tell me about Spidy", "Wake up Jarvis", "Wake up Sammy",
        "Wake up Steve", "Wake up Google", "Wake", "Wide awake", "Hey Spidy",
    ]
    X_wu, y_wu = generate_dataset_for_phrase(
        voice, af, "Wake up Spidy", wake_up_variants, wake_up_hard_negs,
        num_positives=700, num_negatives=900,
    )
    wu_thresh, wu_stats = train_and_export_model(
        X_wu, y_wu, "wake_up_spidy", "assets/models/wake_word/wake_up_spidy.onnx",
        epochs=40, lr=0.0015,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. VERIFY BOTH MODELS WITH OPENWAKEWORD
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Verifying Models with openwakeword.model.Model ---")
    from openwakeword.model import Model as OWWModel

    oww = OWWModel(
        wakeword_models=[
            "assets/models/wake_word/hey_spidy.onnx",
            "assets/models/wake_word/wake_up_spidy.onnx",
        ],
        inference_framework="onnx",
    )
    print("Loaded models in OWW:", list(oww.models.keys()))

    # Test streaming synthetic phrases
    for phrase in [
        "Hey Spidy",
        "Wake up Spidy",
        "Tell me about Spidy",
        "Wake up Spiderman",
        "Hi Spidy",
        "Hey Steve",
        "open Notepad",
    ]:
        audio = synthesize_phrase(voice, phrase)
        audio_int16 = (audio * 32767).astype(np.int16)
        oww.prediction_buffer.clear()

        scores_hs = []
        scores_wu = []
        for i in range(0, len(audio_int16) - 1280, 1280):
            chunk = audio_int16[i : i + 1280]
            sc = oww.predict(chunk)
            scores_hs.append(sc.get("hey_spidy", 0.0))
            scores_wu.append(sc.get("wake_up_spidy", 0.0))

        max_hs = max(scores_hs) if scores_hs else 0.0
        max_wu = max(scores_wu) if scores_wu else 0.0
        print(f"Streaming '{phrase:25}' -> hey_spidy: {max_hs:.4f} | wake_up_spidy: {max_wu:.4f}")

    print("\n✓ Training and verification complete!")


if __name__ == "__main__":
    main()
