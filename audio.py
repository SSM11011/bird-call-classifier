"""Audio loading without librosa/numba: libsndfile (>=1.1 decodes mp3) + polyphase resampling."""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config as C


def load(path, max_seconds=None) -> np.ndarray:
    """Mono float32 at config.SR."""
    with sf.SoundFile(str(path)) as f:
        frames = -1 if max_seconds is None else int(max_seconds * f.samplerate)
        y = f.read(frames=frames, dtype="float32", always_2d=True).mean(axis=1)
        sr = f.samplerate
    if sr != C.SR:
        g = gcd(sr, C.SR)
        y = resample_poly(y, C.SR // g, sr // g).astype(np.float32)
    return y
