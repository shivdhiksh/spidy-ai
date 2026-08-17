
# Test OWW with 8x software gain applied in real-time
import sys, time, wave, numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

MIC_GAIN = 8.0
RATE = 16000
CHUNK = 1280
SECONDS = 60

print('Loading OWW (hey_jarvis)...')
oww = OWWModel(wakeword_models=['hey_jarvis'], inference_framework='onnx')
print('OWW loaded.')
print()
print('MIC GAIN: ' + str(MIC_GAIN) + 'x')
print('>>> SPEAK HEY JARVIS clearly and repeatedly <<<')
print()

max_score = 0.0
chunk_count = 0
last_print = time.monotonic()
ring = []
RING_LEN = 62

with sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=CHUNK) as stream:
    print('Stream open.')
    deadline = time.monotonic() + SECONDS
    while time.monotonic() < deadline:
        raw, _ = stream.read(CHUNK)
        chunk = raw.flatten()

        # Apply software gain
        chunk_gained = np.clip(chunk * MIC_GAIN, -1.0, 1.0)

        rms_raw    = float(np.sqrt(np.mean(chunk ** 2)))
        rms_gained = float(np.sqrt(np.mean(chunk_gained ** 2)))
        peak_i16_raw    = int(np.abs(chunk).max() * 32767)
        peak_i16_gained = int(np.abs(chunk_gained).max() * 32767)

        chunk_int16 = (chunk_gained * 32767).astype(np.int16)
        scores = oww.predict(chunk_int16)
        score = float(scores.get('hey_jarvis', 0.0))
        chunk_count += 1

        ring.append(chunk.copy())
        if len(ring) > RING_LEN:
            ring.pop(0)

        if score > max_score:
            max_score = score

        now = time.monotonic()
        if rms_raw > 0.003 and chunk_count % 3 == 0:
            print('  ch=' + str(chunk_count) + ' rms_raw=' + str(round(rms_raw,5)) + ' rms_8x=' + str(round(rms_gained,5)) + ' peak_raw_i16=' + str(peak_i16_raw) + ' peak_8x_i16=' + str(peak_i16_gained) + ' score=' + str(round(score,4)))
        elif now - last_print > 4.0:
            last_print = now
            print('  [quiet] ch=' + str(chunk_count) + ' max_score=' + str(round(max_score,4)))

        if score > 0.1:
            print()
            print('  *** SCORE > 0.1: ' + str(round(score,4)) + ' ***')
            print()
        if score >= 0.5:
            print()
            print('  *** WAKE WORD DETECTED! score=' + str(round(score,4)) + ' ***')
            print()

print()
print('=== SUMMARY ===')
print('Total chunks: ' + str(chunk_count))
print('Max score seen: ' + str(round(max_score,4)))
if max_score >= 0.5:
    print('SUCCESS: Wake word detected!')
elif max_score >= 0.1:
    print('PARTIAL: Detected but below threshold. Try lowering threshold or speaking louder.')
else:
    print('FAIL: Not detected. May need higher mic_gain or Windows mic volume increase.')
