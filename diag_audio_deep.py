"""
Deep Audio Preprocessing Diagnostic
=====================================
This script captures real microphone audio, runs every preprocessing step
manually, and prints diagnostics at each stage. It also saves frames to WAV
and feeds them into OpenWakeWord to verify detection.

Run: py -3 diag_audio_deep.py
Speak "Hey Jarvis" repeatedly during the test.
"""
import sys
import time
import wave
import struct
import threading
import numpy as np

sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000         # what OpenWakeWord requires
CHANNELS    = 1             # mono
CHUNK_SIZE  = 1280          # 80ms at 16kHz (OWW requirement)
DTYPE       = "float32"
CAPTURE_SECONDS = 45        # total diagnostic duration
DIAG_INTERVAL   = 2.0       # print stats every N seconds
WAV_SAVE_PATH   = "diag_captured_audio.wav"  # saved WAV for manual listening
FRAMES_TO_SAVE  = 125       # ~10 seconds of audio (125 * 1280 / 16000 = 10s)
SPEAK_RMS_THRESHOLD = 0.002  # print enhanced diag when RMS above this

print("=" * 70)
print("DEEP AUDIO PREPROCESSING DIAGNOSTIC")
print("=" * 70)
print(f"  Sample rate  : {SAMPLE_RATE} Hz")
print(f"  Channels     : {CHANNELS}")
print(f"  Chunk size   : {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE*1000:.0f} ms)")
print(f"  Duration     : {CAPTURE_SECONDS}s")
print(f"  WAV output   : {WAV_SAVE_PATH}")
print()
print(">>> SPEAK 'HEY JARVIS' CLEARLY AND REPEATEDLY DURING THIS TEST <<<")
print()

# ── Step 1: Verify sounddevice opens at 16000 Hz ─────────────────────────────
print("─" * 50)
print("STAGE 1: Opening microphone at 16000 Hz...")
try:
    import sounddevice as sd
    device_info = sd.query_devices(kind="input")
    print(f"  Default input device : {device_info['name']}")
    print(f"  Device default rate  : {device_info['default_samplerate']} Hz")
    print(f"  Max input channels   : {device_info['max_input_channels']}")
    print(f"  Requesting rate      : {SAMPLE_RATE} Hz  (16000)")
    print(f"  Requesting channels  : {CHANNELS} (mono)")
    print(f"  Requesting dtype     : {DTYPE}")
    print()
except Exception as e:
    print(f"  ERROR querying device: {e}")
    sys.exit(1)

# ── Step 2: Load OpenWakeWord ─────────────────────────────────────────────────
print("─" * 50)
print("STAGE 2: Loading OpenWakeWord model...")
try:
    from openwakeword.model import Model as OWWModel
    oww = OWWModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    print("  OpenWakeWord loaded: hey_jarvis (onnx)")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# ── Step 3: Capture + full pipeline diagnostic ────────────────────────────────
print()
print("─" * 50)
print("STAGE 3: Capturing audio. SPEAK NOW!")
print()

saved_frames = []       # raw float32 frames for WAV
chunk_count = 0
last_diag_time = time.monotonic()
max_score_seen = 0.0
max_rms_seen   = 0.0
any_speech_detected = False

all_scores = []

def run_capture():
    global chunk_count, last_diag_time, max_score_seen, max_rms_seen, any_speech_detected
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_SIZE,
        ) as stream:
            print(f"  [OK] Stream opened: samplerate={stream.samplerate}, "
                  f"channels={stream.channels}, dtype={stream.dtype}")
            print()

            deadline = time.monotonic() + CAPTURE_SECONDS

            while time.monotonic() < deadline:
                # ── Raw capture ───────────────────────────────────────────────
                raw_chunk, overflowed = stream.read(CHUNK_SIZE)
                chunk_count += 1

                # ── Stage A: Raw shape/dtype ──────────────────────────────────
                # sounddevice returns shape (CHUNK_SIZE, CHANNELS)
                assert raw_chunk.shape == (CHUNK_SIZE, CHANNELS), \
                    f"Unexpected shape: {raw_chunk.shape}"
                assert raw_chunk.dtype == np.float32, \
                    f"Unexpected dtype: {raw_chunk.dtype}"

                # ── Stage B: Flatten (stereo→mono already done by channels=1) ─
                audio_flat = raw_chunk.flatten()  # shape: (CHUNK_SIZE,)
                assert audio_flat.shape == (CHUNK_SIZE,)

                # ── Stage C: Amplitude metrics ────────────────────────────────
                rms    = float(np.sqrt(np.mean(audio_flat ** 2)))
                peak   = float(np.max(np.abs(audio_flat)))
                mean_a = float(np.mean(np.abs(audio_flat)))
                vmin   = float(audio_flat.min())
                vmax   = float(audio_flat.max())

                max_rms_seen = max(max_rms_seen, rms)

                speaking = rms > SPEAK_RMS_THRESHOLD
                if speaking:
                    any_speech_detected = True

                # Save frames for WAV output
                if len(saved_frames) < FRAMES_TO_SAVE:
                    saved_frames.append(audio_flat.copy())

                # ── Stage D: int16 conversion ─────────────────────────────────
                # OWW's internal preprocessor needs int16 PCM values (-32768..32767)
                # from raw float32 data in [-1.0 .. 1.0]
                audio_int16 = (audio_flat * 32767.0).astype(np.int16)

                # Verify conversion fidelity
                audio_roundtrip = audio_int16.astype(np.float32) / 32767.0
                max_conversion_error = float(np.max(np.abs(audio_flat - audio_roundtrip)))

                # ── Stage E: Feed into OWW ────────────────────────────────────
                try:
                    scores = oww.predict(audio_int16)
                    score = float(scores.get("hey_jarvis", 0.0))
                except Exception as e:
                    print(f"  [ERROR] OWW predict failed: {e}")
                    score = 0.0

                all_scores.append(score)
                max_score_seen = max(max_score_seen, score)

                # ── Periodic diagnostic output ────────────────────────────────
                now = time.monotonic()
                elapsed = now - last_diag_time
                should_print_periodic = elapsed >= DIAG_INTERVAL
                should_print_speech   = speaking and (chunk_count % 5 == 0)
                should_print_score    = score > 0.05

                if should_print_periodic or should_print_speech or should_print_score:
                    last_diag_time = now
                    tag = "SPEAKING" if speaking else "silent  "
                    print(f"  [{tag}] chunk={chunk_count:5d} | "
                          f"rms={rms:.6f} | peak={peak:.6f} | mean={mean_a:.6f} | "
                          f"min={vmin:.4f} max={vmax:.4f} | "
                          f"score={score:.4f} | "
                          f"int16_range=[{audio_int16.min()},{audio_int16.max()}] | "
                          f"conv_err={max_conversion_error:.2e}")

                if score > 0.1:
                    print(f"  *** SCORE > 0.1: {score:.4f} at chunk {chunk_count} ***")
                if score >= 0.5:
                    print(f"  *** WAKE WORD DETECTED! score={score:.4f} ***")

    except Exception as e:
        print(f"  [FATAL] Stream error: {e}")
        import traceback; traceback.print_exc()

# Run capture in main thread
run_capture()

# ── Step 4: Save WAV ──────────────────────────────────────────────────────────
print()
print("─" * 50)
print("STAGE 4: Saving captured audio to WAV...")
if saved_frames:
    audio_concat = np.concatenate(saved_frames)
    audio_int16_wav = (audio_concat * 32767.0).astype(np.int16)
    with wave.open(WAV_SAVE_PATH, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16_wav.tobytes())
    duration_s = len(audio_concat) / SAMPLE_RATE
    print(f"  Saved {len(saved_frames)} chunks ({duration_s:.1f}s) → {WAV_SAVE_PATH}")
    print(f"  Play it back to verify audio quality: start {WAV_SAVE_PATH}")
else:
    print("  No frames captured!")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
print()
print("─" * 50)
print("STAGE 5: SUMMARY")
print(f"  Total chunks processed  : {chunk_count}")
print(f"  Max RMS seen            : {max_rms_seen:.6f}")
print(f"  Max OWW score seen      : {max_score_seen:.4f}")
print(f"  Any speech detected     : {any_speech_detected} (RMS > {SPEAK_RMS_THRESHOLD})")
print()

# ── Diagnosis ─────────────────────────────────────────────────────────────────
print("─" * 50)
print("DIAGNOSIS:")

if max_rms_seen < 0.001:
    print("  [PROBLEM] Microphone is capturing near-silence!")
    print("  → RMS never exceeded 0.001 over the entire test.")
    print("  → This means the mic is not picking up your voice at all.")
    print("  → Check Windows Sound settings: mic volume, privacy, exclusive mode.")
    print("  → The device may be muted or using wrong input device.")
elif max_rms_seen < 0.01:
    print("  [WARNING] Microphone signal is very weak (max RMS < 0.01).")
    print("  → Increase microphone volume in Windows Sound settings.")
    print("  → Current signal may be too quiet for OWW to detect the wake word.")
else:
    print(f"  [OK] Microphone signal is present (max RMS = {max_rms_seen:.6f})")

if max_score_seen < 0.05:
    print("  [PROBLEM] OWW score never exceeded 0.05.")
    if max_rms_seen < 0.001:
        print("  → Root cause: no audio reaching OWW (mic issue above).")
    else:
        print("  → Root cause: audio reaches OWW but model does not recognize phrase.")
        print("  → Verify the WAV file sounds like clear speech at 16kHz.")
elif max_score_seen < 0.5:
    print(f"  [PARTIAL] OWW scored up to {max_score_seen:.4f} but never hit threshold 0.5.")
    print("  → Speak louder and more clearly, or reduce the threshold.")
else:
    print(f"  [SUCCESS] Wake word detected! Max score = {max_score_seen:.4f}")

print()
print(f"  → Open {WAV_SAVE_PATH} in Audacity or Windows Media Player.")
print("     If it sounds clear, the audio pipeline is healthy.")
print("     If it sounds silent/corrupted, there is a preprocessing bug.")
print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
