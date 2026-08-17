
import sys, wave, numpy as np, sounddevice as sd
sys.path.insert(0, '.')

# List devices
print('=== INPUT DEVICES ===')
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print('  [' + str(i) + '] ' + d['name'] + ' | sr=' + str(int(d['default_samplerate'])))

default_dev = sd.query_devices(kind='input')
native_sr = int(default_dev['default_samplerate'])
print()
print('Default input: ' + default_dev['name'] + ' at ' + str(native_sr) + ' Hz')

# Quick 2s RMS test - NO PROMPTS (just analyze existing env audio)
print()
print('Recording 2s at 16000 Hz (background)...')
rec = sd.rec(16000 * 2, samplerate=16000, channels=1, dtype='float32')
sd.wait()
rms = float(np.sqrt(np.mean(rec**2)))
peak = float(np.abs(rec).max())
int16_peak = int(peak * 32767)
print('Background audio: RMS=' + str(round(rms, 6)) + ' peak=' + str(round(peak, 6)) + ' int16_peak=' + str(int16_peak))

# Now analyse existing diag_captured_audio.wav for OWW sensitivity at different amp levels
import os
wav_path = 'diag_captured_audio.wav'
if not os.path.exists(wav_path):
    print('ERROR: ' + wav_path + ' not found. Run diag_audio_deep.py first.')
    sys.exit(1)

with wave.open(wav_path, 'r') as wf:
    frames = wf.readframes(wf.getnframes())
    sr = wf.getframerate()
audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

print()
print('=== EXISTING WAV ANALYSIS ===')
print('SR=' + str(sr) + ' samples=' + str(len(audio)))
rms_wav = float(np.sqrt(np.mean(audio**2)))
peak_wav = float(np.abs(audio).max())
print('RMS(int16)=' + str(round(rms_wav, 1)) + ' peak=' + str(int(peak_wav)) + ' (' + str(round(peak_wav/32768*100, 2)) + '% full scale)')

# Test OWW with each amplification factor
from openwakeword.model import Model as OWWModel
CHUNK = 1280
print()
print('=== OWW SCORE VS AMPLIFICATION ===')
for amp in [1.0, 3.0, 6.0, 12.0, 20.0, 40.0]:
    oww = OWWModel(wakeword_models=['hey_jarvis'], inference_framework='onnx')
    amplified = np.clip(audio * amp, -32768, 32767).astype(np.int16)
    peak_amp = int(np.abs(amplified).max())
    rms_amp = float(np.sqrt(np.mean(amplified.astype(np.float32)**2)))
    max_s = 0.0
    for i in range(len(amplified) // CHUNK):
        chunk = amplified[i * CHUNK:(i + 1) * CHUNK]
        s = oww.predict(chunk)
        sc = float(s.get('hey_jarvis', 0.0))
        if sc > max_s:
            max_s = sc
    clip_pct = (np.abs(audio * amp) > 32767).sum() / max(1, len(audio)) * 100
    print('  amp=' + str(amp) + 'x | rms=' + str(round(rms_amp,1)) + ' | peak_i16=' + str(peak_amp) + ' | clipping=' + str(round(clip_pct,1)) + '% | max_oww_score=' + str(round(max_s, 4)))

print()
print('NOTE: If ALL OWW scores are < 0.05 regardless of amplitude, the WAV')
print('does not contain clear Hey Jarvis speech. Record again while speaking clearly.')
