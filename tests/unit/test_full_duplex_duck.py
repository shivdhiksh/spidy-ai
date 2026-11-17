import numpy as np

from spidy.core import playback_gain as g
from spidy.voice import barge_in as bi


def test_ducking_glides_without_clicks_and_restores():
    g.reset()
    x = np.ones(512, dtype=np.float32)
    assert np.allclose(g.apply(x), 1.0)
    g.duck(True)
    y = g.apply(x)
    assert y[0] > 0.9 and abs(y[-1] - g.DUCKED) < 1e-6 and np.all(np.diff(y) <= 0)   # smooth glide down
    assert np.allclose(g.apply(x), g.DUCKED)
    g.duck(False)
    z = g.apply(x)
    assert abs(z[-1] - 1.0) < 1e-6
    g.reset()


def test_voice_counts_as_speech_but_hum_does_not():
    t = np.arange(1280) / 16000
    voice = (np.sin(2 * np.pi * 300 * t) + 0.6 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    hum = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    assert bi._speechlike(voice)
    assert not bi._speechlike(hum)
