"""
scripts/setup_models.py
========================
Downloads all AI model files required by Spidy.

Usage
-----
    # Download everything (recommended for first setup):
    python scripts/setup_models.py

    # Download specific components only:
    python scripts/setup_models.py --tts
    python scripts/setup_models.py --stt
    python scripts/setup_models.py --wake-word

    # Show what would be downloaded without downloading:
    python scripts/setup_models.py --dry-run

Models downloaded
-----------------
  Wake Word:  openWakeWord pre-trained models (downloaded by the library)
  STT:        faster-whisper base.en model (~145 MB, auto-downloaded by library)
  TTS:        Piper voice en_US-ryan-high (~65 MB, downloaded from HuggingFace)

Note: faster-whisper and openWakeWord download their models automatically
on first use. This script ensures Piper TTS voices are in the correct location.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────

PIPER_VOICES = {
    "en_US-ryan-high": {
        "model_url": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_US/ryan/high/en_US-ryan-high.onnx"
        ),
        "config_url": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_US/ryan/high/en_US-ryan-high.onnx.json"
        ),
    },
    "en_US-lessac-high": {
        "model_url": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_US/lessac/high/en_US-lessac-high.onnx"
        ),
        "config_url": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_US/lessac/high/en_US-lessac-high.onnx.json"
        ),
    },
}

ASSETS_DIR = Path("assets") / "models"
TTS_DIR = ASSETS_DIR / "tts"
STT_DIR = ASSETS_DIR / "stt"
WAKE_DIR = ASSETS_DIR / "wake_word"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def download_file(url: str, dest: Path, dry_run: bool = False) -> None:
    """Download a file from URL to dest, with progress reporting."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return

    if dry_run:
        print(f"  [DRY RUN] Would download: {url}")
        print(f"            → {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading: {dest.name}")
    print(f"    From: {url}")

    try:
        def progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r    {pct:3d}% — {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()  # newline after progress
        print(f"  ✓ Saved to: {dest}")
    except Exception as exc:
        print(f"  ✗ Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        raise


# ─── Download functions ───────────────────────────────────────────────────────

def download_piper_voice(voice: str, dry_run: bool = False) -> None:
    """Download a Piper TTS voice model and config."""
    if voice not in PIPER_VOICES:
        print(f"  ✗ Unknown voice: {voice}")
        print(f"    Available: {', '.join(PIPER_VOICES.keys())}")
        return

    voice_dir = TTS_DIR / voice
    info = PIPER_VOICES[voice]

    download_file(info["model_url"], voice_dir / f"{voice}.onnx", dry_run)
    download_file(info["config_url"], voice_dir / f"{voice}.onnx.json", dry_run)


def download_faster_whisper(model: str = "base.en", dry_run: bool = False) -> None:
    """
    faster-whisper downloads models automatically on first transcribe() call.
    This function pre-downloads them for offline use.
    """
    print(f"  faster-whisper '{model}' model will be auto-downloaded on first use.")
    print(f"  Default cache: ~/.cache/huggingface/hub/")
    print(f"  (Run Spidy once with an internet connection to cache it)")

    if dry_run:
        return

    try:
        from faster_whisper import WhisperModel
        print(f"  ↓ Pre-downloading faster-whisper '{model}'...")
        # This triggers the download
        _ = WhisperModel(model, device="cpu", compute_type="int8")
        print(f"  ✓ faster-whisper '{model}' cached successfully.")
        del _
    except ImportError:
        print("  ✗ faster-whisper not installed. Run: pip install faster-whisper")
    except Exception as exc:
        print(f"  ⚠ Download attempt: {exc}")


def download_openwakeword(dry_run: bool = False) -> None:
    """
    openWakeWord downloads its pre-trained models automatically.
    This function triggers the download and shows where models are cached.
    """
    print("  openWakeWord pre-trained models will be auto-downloaded on first use.")

    if dry_run:
        return

    try:
        from openwakeword.model import Model
        print("  ↓ Pre-downloading openWakeWord 'alexa' model...")
        _ = Model(wakeword_models=["alexa"], inference_framework="tflite")
        print("  ✓ openWakeWord model cached.")
        del _
    except ImportError:
        print("  ✗ openWakeWord not installed. Run: pip install openwakeword")
    except Exception as exc:
        print(f"  ⚠ {exc}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spidy Model Setup — Downloads all required AI models"
    )
    parser.add_argument("--tts", action="store_true", help="Download TTS voice only")
    parser.add_argument("--stt", action="store_true", help="Download STT model only")
    parser.add_argument("--wake-word", action="store_true", help="Download wake word model only")
    parser.add_argument("--voice", default="en_US-ryan-high", help="Piper voice to download")
    parser.add_argument("--whisper-model", default="base.en", help="Whisper model size")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    # If no specific flags, download everything
    download_all = not (args.tts or args.stt or args.wake_word)

    print("=" * 60)
    print("  Spidy Model Setup")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY RUN MODE] Nothing will actually be downloaded.")

    if download_all or args.tts:
        print_section("Piper TTS Voice")
        download_piper_voice(args.voice, dry_run=args.dry_run)

    if download_all or args.stt:
        print_section("faster-whisper STT")
        download_faster_whisper(args.whisper_model, dry_run=args.dry_run)

    if download_all or args.wake_word:
        print_section("openWakeWord")
        download_openwakeword(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Run Spidy with: python -m spidy.main")
    print("=" * 60)


if __name__ == "__main__":
    main()
