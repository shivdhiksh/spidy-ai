"""
Quick live test: community jarvis_v1 + jarvis_v2 vs built-in hey_jarvis_v0.1.
Run: py -3 scratch_test_jarvis_community.py
Speak "Hey Jarvis" or just "Jarvis" clearly at normal volume.
"""
import sys, time, numpy as np, pathlib, openwakeword
from openwakeword.model import Model as OWWModel
import sounddevice as sd

RATE  = 16000
CHUNK = 1280
GAIN  = 8.0
SECS  = 45

models_dir = pathlib.Path(openwakeword.__file__).parent / "resources" / "models"

print("Loading models...")
models = {}
configs = [
    ("hey_jarvis_v0.1 [builtin]",  str(models_dir / "hey_jarvis_v0.1.onnx"), "hey_jarvis"),
    ("jarvis_v1 [community]",      str(models_dir / "jarvis_v1.onnx"),        "jarvis_v1"),
    ("jarvis_v2 [community]",      str(models_dir / "jarvis_v2.onnx"),        "jarvis_v2"),
]
for label, path, score_key in configs:
    try:
        m = OWWModel(wakeword_models=[path], inference_framework="onnx")
        models[label] = (m, score_key)
        print("  [OK] " + label)
    except Exception as e:
        print("  [FAIL] " + label + ": " + str(e))

print()
print("Listening for " + str(SECS) + "s. Say 'Hey Jarvis' or 'Jarvis' at NORMAL volume.")
print("(Scores printed every 3 chunks during speech, or whenever > 0.05)")
print()

maxscores = {k: 0.0 for k in models}
chunk_n = 0

with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=CHUNK) as stream:
    deadline = time.monotonic() + SECS
    while time.monotonic() < deadline:
        raw, _ = stream.read(CHUNK)
        flat   = raw.flatten()
        gained = np.clip(flat * GAIN, -1.0, 1.0)
        i16    = (gained * 32767).astype(np.int16)
        rms    = float(np.sqrt(np.mean(gained ** 2)))
        chunk_n += 1

        scores = {}
        for label, (m, key) in models.items():
            try:
                r = m.predict(i16)
                sc = float(r.get(key, 0.0))
            except Exception:
                sc = 0.0
            scores[label] = sc
            if sc > maxscores[label]:
                maxscores[label] = sc

        best = max(scores.values())
        if best > 0.05 or (rms > 0.02 and chunk_n % 3 == 0):
            t = int(deadline - time.monotonic())
            parts = " | ".join(k.split("[")[0].strip() + "=" + str(round(v, 4))
                               for k, v in scores.items())
            print("  ch=" + str(chunk_n).rjust(4) + " rms=" + str(round(rms, 3))
                  + " | " + parts + " | t=" + str(t) + "s")
        if best >= 0.4:
            print("  >>> DETECTION: " + max(scores, key=scores.get)
                  + " = " + str(round(best, 4)) + " <<<")

print()
print("=" * 65)
print("MAX SCORES:")
for label, sc in sorted(maxscores.items(), key=lambda x: -x[1]):
    bar    = "#" * int(sc * 40)
    status = "DETECTS" if sc >= 0.4 else ("CLOSE" if sc >= 0.2 else ("WEAK" if sc >= 0.05 else "FAIL"))
    print("  " + label.ljust(30) + ": " + str(round(sc, 4))
          + "  [" + bar.ljust(40) + "]  " + status)
print()
best_label = max(maxscores, key=maxscores.get)
print("Best: " + best_label + " (max=" + str(round(maxscores[best_label], 4)) + ")")
print("=" * 65)
