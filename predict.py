"""Classify a new recording with the trained head.

    python predict.py path/to/recording.mp3 [more files...]

Slides a 5 s window (2.5 s hop) over the file. A window gets a species only if the
head's confidence clears the threshold chosen during evaluation; otherwise it is
reported as "uncertain". Because the head is multi-label, the file summary lists
every species that confidently appears in at least one window (overlapping calls).
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import audio  # noqa: E402
import backbones  # noqa: E402
import config as C  # noqa: E402


def main(paths):
    ckpt = torch.load(C.RESULTS / "head.pt", weights_only=False)
    classes, tau = ckpt["classes"], ckpt["threshold"]
    net = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(len(ckpt["mu"]), len(classes)))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    model = backbones.Perch([C.SPECIES[c] for c in classes])

    for path in paths:
        y = audio.load(path)
        y = np.pad(y, (0, max(0, C.CLIP_SAMPLES - len(y))))
        hop = int(C.HOP_SECONDS * C.SR)
        starts = range(0, len(y) - C.CLIP_SAMPLES + 1, hop)
        emb, _ = model(np.stack([y[s:s + C.CLIP_SAMPLES] for s in starts]))
        with torch.no_grad():
            prob = torch.sigmoid(net(torch.tensor((emb - ckpt["mu"]) / ckpt["sd"], dtype=torch.float32))).numpy()

        print(f"\n{path}")
        for s, p in zip(starts, prob):
            best = p.argmax()
            label = classes[best] if p[best] >= tau else "uncertain"
            print(f"  {s / C.SR:7.1f}s  {label:24s} ({p[best]:.2f})")
        present = [f"{c} ({prob[:, j].max():.2f})" for j, c in enumerate(classes) if prob[:, j].max() >= tau]
        print("  species detected:", ", ".join(present) or "none above threshold")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
