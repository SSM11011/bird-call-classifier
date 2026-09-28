"""Build degraded copies of every clip, shared by all models in the comparison.

  aug     - training-time augmentation. Each clip gets several degraded copies:
              faint   : attenuated 14-26 dB, then real background noise at -6..3 dB SNR
              noisy   : real background noise (+ synthetic wind half the time) at 0..12 dB SNR
              overlap : a clip of a *different* species mixed in at -9..0 dB (multi-label target)
            Minority classes get proportionally more copies (oversampling by augmentation,
            not by duplicating identical examples).
  stress  - the same three degradations, one of each per clip, drawn with a different
            seed. Used only for evaluation, to measure robustness in a controlled way.

Leakage guard: noise clips and overlap partners are always drawn from the *same CV
fold* as the clip (and never from the same recording), so when fold k is held out,
no training example contains audio from fold k.

Writes data/{aug,stress}_waves.npy (float16, N x 160000) and data/augmented.npz
(source clip index, degradation kind, mixed-in species for every copy).
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from tqdm import tqdm

import config as C

AUG_TYPES = ("faint", "noisy", "overlap")
AUG_PER_CLIP = 2              # copies per clip for the largest class


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def db(x):
    return 10 ** (x / 20)


def wind(n, rng):
    """Brown-ish low-frequency rumble, a crude stand-in for wind on the microphone."""
    w = lfilter([1.0], [1.0, -0.995], rng.standard_normal(n))
    return w / rms(w)


class Augmenter:
    def __init__(self, clips, meta, noise, noise_meta, rng):
        self.clips, self.meta, self.noise, self.rng = clips, meta, noise, rng
        self.noise_by_fold = {f: g.index.values for f, g in noise_meta.groupby("fold")}
        self.noise_rec = noise_meta.xc_id.values

    def _noise(self, i):
        m = self.meta.iloc[i]
        pool = [j for j in self.noise_by_fold[m.fold] if self.noise_rec[j] != m.xc_id]
        return self.noise[self.rng.choice(pool)].astype(np.float32)

    def __call__(self, i, kind):
        """Return (waveform, index of the mixed-in species or -1)."""
        x = self.clips[i].astype(np.float32)
        partner = -1
        if kind == "faint":
            x = x * db(-self.rng.uniform(14, 26))
            n = self._noise(i)
            x = x + n / rms(n) * rms(x) / db(self.rng.uniform(-6, 3))
        elif kind == "noisy":
            n = self._noise(i)
            n = n / rms(n)
            if self.rng.random() < 0.5:
                n = n + 0.7 * wind(len(x), self.rng)
                n = n / rms(n)
            x = x + n * rms(x) / db(self.rng.uniform(0, 12))
        elif kind == "overlap":
            m = self.meta.iloc[i]
            pool = self.meta.index[(self.meta.fold == m.fold) & (self.meta.label != m.label)].values
            j = self.rng.choice(pool)
            y = self.clips[j].astype(np.float32)
            x = x + y / rms(y) * rms(x) * db(self.rng.uniform(-9, 0))
            partner = int(self.meta.label.iloc[j])
        peak = np.abs(x).max()
        return (x / peak * 0.9 if peak > 1 else x), partner


def build(aug, meta, copies_per_label, rng, one_of_each):
    waves, src, kinds, partners = [], [], [], []
    for i, lab in enumerate(tqdm(meta.label.values, leave=False)):
        kinds_i = AUG_TYPES if one_of_each else rng.choice(AUG_TYPES, copies_per_label[lab])
        for k in kinds_i:
            w, p = aug(i, k)
            waves.append(w.astype(np.float16))
            src.append(i)
            kinds.append(k)
            partners.append(p)
    return np.stack(waves), np.array(src), np.array(kinds), np.array(partners)


def main():
    meta = pd.read_csv(C.DATA / "clips.csv")
    classes = json.loads((C.DATA / "classes.json").read_text())
    clips = np.load(C.DATA / "clips.npy", mmap_mode="r")
    noise = np.load(C.DATA / "noise.npy")
    noise_meta = pd.read_csv(C.DATA / "noise.csv")

    counts = meta.label.value_counts()
    copies = {lab: int(round(AUG_PER_CLIP * counts.max() / n)) for lab, n in counts.items()}
    print("augmented copies per clip, by class:", {classes[k]: v for k, v in copies.items()})
    out = {}
    for name, seed, one_of_each in (("aug", C.SEED + 1, False), ("stress", C.SEED + 2, True)):
        rng = np.random.default_rng(seed)
        waves, out[f"{name}_src"], out[f"{name}_kind"], out[f"{name}_partner"] = \
            build(Augmenter(clips, meta, noise, noise_meta, rng), meta, copies, rng, one_of_each)
        np.save(C.DATA / f"{name}_waves.npy", waves)
        print(f"{name}: {len(waves)} clips")
        if name == "stress":  # one example of each degradation for the write-up figure
            kinds, src, partner = out["stress_kind"], out["stress_src"], out["stress_partner"]
            keep = [np.where(kinds == k)[0][0] for k in AUG_TYPES]
            np.save(C.DATA / "stress_examples.npy", waves[keep].astype(np.float32))
            (C.DATA / "stress_examples.json").write_text(json.dumps(
                [{"kind": str(kinds[i]), "src": int(src[i]), "partner": int(partner[i])} for i in keep]))
    np.savez(C.DATA / "augmented.npz", **out)


if __name__ == "__main__":
    main()
