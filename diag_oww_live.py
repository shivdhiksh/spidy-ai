
# Live OWW detection test - saves WAV when score peaks, runs 60 seconds
import sys, time, wave, threading
import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

sys.path.insert(0, '.')
RATE = 16000
CHUNK = 1280
SECONDS = 60

print('Loading OWW...')
oww = OWWModel(wakeword_models=['hey_jarvis'], inference_framework='onnx')
print('OWW loaded. Starting 60-second live test.')
print('>>> SPEAK HEY JARVIS clearly and repeatedly <<<')
print()

# Rolling buffer: last 5s = 62 chunks
BUFFER_CHUNKS = 62
ring = []
max_score = 0.0
chunk_count = 0
last_print = time.monotonic()
best_chunk_idx = 0
best_score = 0.0
saved_wav = False

def run():
    global max_score, chunk_count, last_print, best_chunk_idx, best_score, saved_wav, ring
    with sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=CHUNK) as stream:
        print('Stream open. Listen...')
        deadline = time.monotonic() + SECONDS
        while time.monotonic() < deadline:
            raw, _ = stream.read(CHUNK)
            chunk = raw.flatten()
            rms = float(np.sqrt(np.mean(chunk**2)))
            chunk_int16 = (chunk * 32767).astype(np.int16)
            scores = oww.predict(chunk_int16)
            score = float(scores.get('hey_jarvis', 0.0))
            chunk_count += 1

            ring.append(chunk.copy())
            if len(ring) > BUFFER_CHUNKS:
                ring.pop(0)

            if score > best_score:
                best_score = score
                best_chunk_idx = chunk_count

            now = time.monotonic()
            if rms > 0.003 and chunk_count % 3 == 0:
                print('  ch=' + str(chunk_count) + ' rms=' + str(round(rms,5)) + ' score=' + str(round(score,4)) + ' peak_i16=' + str(int(np.abs(chunk_int16).max())))
            elif now - last_print > 3.0:
                last_print = now
                print('  [silent] ch=' + str(chunk_count) + ' rms=' + str(round(rms,5)) + ' max_score_so_far=' + str(round(best_score,4)))

            if score > 0.05 and not saved_wav:
                saved_wav = True
                audio_save = np.concatenate(ring)
                audio_i16 = (audio_save * 32767).astype(np.int16)
                with wave.open('diag_oww_detection.wav', 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes(audio_i16.tobytes())
                print('  *** SAVED diag_oww_detection.wav at score=' + str(round(score,4)) + ' ***')

            if score >= 0.5:
                print()
                print('  *** WAKE WORD DETECTED! score=' + str(round(score,4)) + ' at chunk ' + str(chunk_count) + ' ***')
                print()

run()

print()
print('=== SUMMARY ===')
print('Total chunks: ' + str(chunk_count))
print('Max score: ' + str(round(best_score,4)) + ' at chunk ' + str(best_chunk_idx))
if best_score < 0.05:
    print()
    print('CONCLUSION: OWW never scored above 0.05.')
    print('This means the recording does not contain recognizable Hey Jarvis speech,')
    print('OR the mic volume is too low, OR there is audio processing stripping the signal.')
    print()
    print('NEXT STEPS:')
    print('  1. Open Windows Sound Settings -> Input device -> Microphone volume -> set to 80-100%')
    print('  2. Check if AI Noise Cancelling is active and try disabling it')
    print('  3. Try speaking into the mic while watching the rms values above - aim for rms > 0.02')
elif best_score < 0.5:
    print()
    print('PARTIAL: OWW scored ' + str(round(best_score,4)) + ' but never hit 0.5 threshold.')
    print('Speak louder and more clearly, or try lowering threshold in config.')
else:
    print()
    print('SUCCESS: Wake word detected at score ' + str(round(best_score,4)))
