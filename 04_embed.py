"""Embed real, augmented and stress clips with each frozen backbone.

    python 04_embed.py                 # all backbones
    python 04_embed.py birdnet panns   # a subset

All backbones see exactly the same waveforms (data/clips.npy, aug_waves.npy,
stress_waves.npy from 03_augment.py). Writes data/emb_<backbone>.npz with
  {clean,aug,stress}_emb      embeddings
  {clean,aug,stress}_logit    the backbone's own scores for our species (bird models only)
  {aug,stress}_{src,kind,partner}   copied from data/augmented.npz
and records the extraction time in data/emb_times.json.
"""
import json
import sys
import time

import numpy as np

import backbones as B
import config as C


def main(names):
    classes = json.loads((C.DATA / "classes.json").read_text())
    species = [C.SPECIES[c] for c in classes]
    waves = {"clean": np.load(C.DATA / "clips.npy", mmap_mode="r"),
             "aug": np.load(C.DATA / "aug_waves.npy", mmap_mode="r"),
             "stress": np.load(C.DATA / "stress_waves.npy", mmap_mode="r")}
    meta = dict(np.load(C.DATA / "augmented.npz"))
    times_path = C.DATA / "emb_times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}

    for name in names:
        model = B.BACKBONES[name](species)
        out, t0 = dict(meta), time.time()
        for split, w in waves.items():
            emb, logit = model(w)
            out[f"{split}_emb"] = emb.astype(np.float32)
            if logit is not None:
                out[f"{split}_logit"] = logit.astype(np.float32)
            print(f"{name} {split}: {emb.shape}")
        times[name] = {"seconds": time.time() - t0, "clips": int(sum(len(w) for w in waves.values()))}
        np.savez_compressed(C.DATA / f"emb_{name}.npz", **out)
        times_path.write_text(json.dumps(times, indent=1))
        del model


if __name__ == "__main__":
    main(sys.argv[1:] or list(B.BACKBONES))
