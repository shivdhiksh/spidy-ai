"""
Live model comparison test — no diagnostic, no gain test.
Tests all available ONNX models simultaneously with live microphone input.
Say the model's wake word for each one. Watch which model scores highest.

Models being tested:
  - alexa      → say "Alexa"
  - hey_jarvis → say "Hey Jarvis"
  - hey_mycroft → say "Hey Mycroft"

Run: py -3 test_models_live.py
"""
import sys
import time
import numpy as np

print("=" * 65)
print("LIVE MODEL COMPARISON — SAY EACH WAKE PHRASE CLEARLY")
print("  'Alexa', 'Hey Jarvis', 'Hey Mycroft'")
print("=" * 65)

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed")
    sys.exit(1)

try:
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("ERROR: openwakeword not installed")
    sys.exit(1)

RATE = 16000
CHUNK = 1280
GAIN = 8.0
SECONDS = 60

print("Loading models...")
models = {}
for name in ["alexa", "hey_jarvis", "hey_mycroft"]:
    try:
        m = OWWModel(wakeword_models=[name], inference_framework="onnx")
        models[name] = m
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

if not models:
    print("No models loaded.")
    sys.exit(1)

print()
print(f"Listening for {SECONDS}s — speak each phrase 2-3 times.")
print("Scores printed when any model exceeds 0.05.")
print()

maxscores = {name: 0.0 for name in models}
chunk_count = 0

with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=CHUNK) as stream:
    deadline = time.monotonic() + SECONDS
    while time.monotonic() < deadline:
        raw, _ = stream.read(CHUNK)
        flat = raw.flatten()
        gained = np.clip(flat * GAIN, -1.0, 1.0)
        i16 = (gained * 32767).astype(np.int16)
        rms = float(np.sqrt(np.mean(gained ** 2)))
        chunk_count += 1

        # Run all models
        scores = {}
        for name, m in models.items():
            try:
                result = m.predict(i16)
                sc = float(result.get(name, 0.0))
            except Exception:
                sc = 0.0
            scores[name] = sc
            if sc > maxscores[name]:
                maxscores[name] = sc

        best_score = max(scores.values())
        best_name = max(scores, key=scores.get)

        # Print when any model is scoring meaningfully OR every 200 chunks
        if best_score > 0.05 or chunk_count % 200 == 0:
            remaining = int(deadline - time.monotonic())
            parts = [f"{n}={v:.4f}" for n, v in scores.items()]
            rms_tag = f"rms={rms:.3f}"
            print(f"  ch={chunk_count:4d} {rms_tag} | {' | '.join(parts)} | t={remaining}s")

        if best_score >= 0.3:
            print(f"  >>> DETECTION: {best_name} = {best_score:.4f} <<<")

print()
print("=" * 65)
print("RESULTS — Max scores seen over entire session:")
for name, sc in maxscores.items():
    bar = "#" * int(sc * 40)
    status = "DETECTS" if sc >= 0.3 else ("PARTIAL" if sc >= 0.1 else "FAIL")
    print(f"  {name:14s}: {sc:.4f}  [{bar:<40s}]  {status}")
print()
best = max(maxscores, key=maxscores.get)
print(f"Best model for this voice: {best} (max={maxscores[best]:.4f})")
if maxscores[best] >= 0.3:
    print(f"  --> Use: model: \"{best}\" in spidy_config.yaml")
else:
    print("  --> No model reached 0.3. Consider custom verifier training.")
print("=" * 65)
