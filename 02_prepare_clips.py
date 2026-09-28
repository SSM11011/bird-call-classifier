"""Cut each recording into 5 s clips, tag difficult conditions, assign CV folds.

Xeno-canto labels are per recording, not per second, so we need to guess which
windows contain the bird. A cheap energy detector is used instead of Perch itself,
so window selection does not favour what Perch already recognises:

  * band energy (150 Hz - 11 kHz) per STFT frame, in dB
  * recording noise floor  = 20th percentile of frame energy
  * window "event level"   = 95th percentile of frame energy inside the window
  * window SNR proxy       = event level - noise floor

The top-K windows by SNR proxy (non-overlapping) become labelled clips. Because
we take top-K *per recording* rather than a global SNR cut, quiet and noisy
recordings still contribute their (quiet, noisy) best windows - they are not
filtered out. The lowest-SNR windows are kept separately as real background
noise for augmentation.

Condition tags (used for sliced evaluation):
  faint   - event level in the bottom quartile of all clips (quiet / distant call)
  noisy   - SNR proxy in the bottom quartile (call barely above the background)
  overlap - xeno-canto lists at least one other species calling in the recording

Writes data/clips.npy (float16, N x 160000), data/clips.csv, data/noise.npy.
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import stft
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

import audio
import config as C

N_FFT, HOP = 2048, 512
FRAMES_PER_SEC = C.SR / HOP


def band_energy_db(y: np.ndarray) -> np.ndarray:
    freqs, _, Z = stft(y, fs=C.SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                       boundary=None, padded=False)
    band = (freqs >= 150) & (freqs <= 11_000)
    # scipy scales the STFT so a full-scale sine has magnitude ~0.5: this is roughly dBFS
    return 10 * np.log10((np.abs(Z[band]) ** 2).sum(axis=0) / 0.25 + 1e-12)


def pick_windows(scores: np.ndarray, k: int, best: bool, exclude=()) -> list[int]:
    """Greedy non-overlapping choice of window indices (hop is half a window)."""
    order = np.argsort(-scores if best else scores)
    taken = set(exclude)
    out = []
    for i in order:
        if len(out) == k:
            break
        if {i - 1, i, i + 1} & taken:
            continue
        out.append(int(i))
        taken.add(int(i))
    return out


def main():
    rec = pd.read_csv(C.ROOT / "recordings.csv")
    rec = rec[rec.downloaded].reset_index(drop=True)
    clips, noise, rows = [], [], []
    hop = int(C.HOP_SECONDS * C.SR)
    for r in tqdm(rec.itertuples(), total=len(rec), desc="clips"):
        try:
            y = audio.load(C.RAW_DIR / f"XC{r.xc_id}.mp3", C.MAX_AUDIO_SECONDS)
        except Exception as e:  # corrupt / unsupported file
            print(f"skip XC{r.xc_id}: {e}")
            continue
        if len(y) < C.CLIP_SAMPLES:
            y = np.pad(y, (0, C.CLIP_SAMPLES - len(y)))
        e_db = band_energy_db(y)
        floor = np.percentile(e_db, 20)
        starts = np.arange(0, len(y) - C.CLIP_SAMPLES + 1, hop)
        f0 = (starts / HOP).astype(int)
        flen = int(C.CLIP_SECONDS * FRAMES_PER_SEC)
        level = np.array([np.percentile(e_db[a:a + flen], 95) for a in f0])
        snr = level - floor

        k = C.CLIPS_PER_RECORDING_RARE if r.species in C.RARE_CLASSES else C.CLIPS_PER_RECORDING
        chosen = pick_windows(snr, k, best=True)
        for i in chosen:
            clips.append(y[starts[i]:starts[i] + C.CLIP_SAMPLES].astype(np.float16))
            rows.append({"xc_id": r.xc_id, "species": r.species, "start_s": starts[i] / C.SR,
                         "level_db": level[i], "snr_db": snr[i], "quality": r.quality,
                         "n_background": r.n_background})
        for i in pick_windows(snr, C.NOISE_CLIPS_PER_RECORDING, best=False, exclude=chosen):
            noise.append((r.xc_id, y[starts[i]:starts[i] + C.CLIP_SAMPLES].astype(np.float16)))

    df = pd.DataFrame(rows)
    # Species without any audio (e.g. restricted ones) drop out of the class list.
    classes = [c for c in C.CLASSES if c in set(df.species)]
    missing = sorted(set(C.CLASSES) - set(classes))
    if missing:
        print(f"no audio for: {missing} -> training on {len(classes)} classes")
    (C.DATA / "classes.json").write_text(json.dumps(classes))
    df["label"] = df.species.map(classes.index)
    df["faint"] = df.level_db <= df.level_db.quantile(0.25)
    df["noisy"] = df.snr_db <= df.snr_db.quantile(0.25)
    df["overlap"] = df.n_background > 0

    # Folds are assigned per recording: clips of one recording never sit on both
    # sides of a train/test split (they share recordist, mic, site and individual bird).
    per_rec = df.groupby("xc_id").label.first()
    sgkf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    fold_of = {}
    for f, (_, te) in enumerate(sgkf.split(per_rec.index, per_rec.values, groups=per_rec.index)):
        fold_of.update({x: f for x in per_rec.index[te]})
    df["fold"] = df.xc_id.map(fold_of)

    C.DATA.mkdir(exist_ok=True)
    np.save(C.DATA / "clips.npy", np.stack(clips))
    np.save(C.DATA / "noise.npy", np.stack([n for _, n in noise]))
    pd.DataFrame({"xc_id": [i for i, _ in noise]}).assign(fold=lambda d: d.xc_id.map(fold_of)) \
        .to_csv(C.DATA / "noise.csv", index=False)
    df.to_csv(C.DATA / "clips.csv", index=False)

    print(f"thresholds: faint <= {df.level_db.quantile(0.25):.1f} dBFS, noisy <= {df.snr_db.quantile(0.25):.1f} dB SNR")
    print(df.groupby("species").agg(recordings=("xc_id", "nunique"), clips=("xc_id", "size"),
                                    faint=("faint", "sum"), noisy=("noisy", "sum"), overlap=("overlap", "sum")))


if __name__ == "__main__":
    main()
