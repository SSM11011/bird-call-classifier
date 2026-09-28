"""End-to-end baselines: CNNs trained directly on log-mel spectrograms.

    python 07_end_to_end.py                  # both models
    python 07_end_to_end.py cnn_small        # one

  cnn_small     4-block CNN from scratch (1.2M parameters)
  effnet_b0     EfficientNet-B0, ImageNet-pretrained (timm), whole network fine-tuned

These answer "is a pretrained *bird* model needed at all?". They get the same folds,
the same augmented copies (faint / noisy / overlap, multi-label), the same class
weights and AUG_WEIGHT as the Perch head, plus SpecAugment masking and random time
shifts because they have far more parameters to regularise. A fixed epoch budget is
used (no early stopping), so the held-out fold is never looked at during training.

Writes results/compare/oof_<model>.npz: out-of-fold probabilities for real clips (P)
and stress clips (S), training time and parameter count.
"""
import json
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import f1_score
from torch import nn

import config as C

TE = import_module("05_train_eval")
OUT = C.RESULTS / "compare"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"             # mixed precision on GPU only
BATCH = 32
CONFIGS = {  # epochs, peak learning rate
    "cnn_small": (40, 2e-3),
    "effnet_b0": (25, 5e-4),
}


class LogMel(nn.Module):
    """Per-clip standardised log-mel (128 bands, 150 Hz-15 kHz, 10 ms hop)."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(C.SR, n_fft=1024, hop_length=320, n_mels=128,
                                                        f_min=150, f_max=15000, power=2.0)

    def forward(self, wave):
        x = torch.log(self.mel(wave) + 1e-6)[:, None]              # (B, 1, 128, frames)
        return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-5)


def spec_augment(x, n_freq=2, f_width=16, n_time=2, t_width=50):
    B, _, Fq, T = x.shape
    for _ in range(n_freq):
        f0 = torch.randint(0, Fq - f_width, (B,), device=x.device)
        w = torch.randint(0, f_width, (B,), device=x.device)
        band = torch.arange(Fq, device=x.device)[None]
        x = x.masked_fill(((band >= f0[:, None]) & (band < (f0 + w)[:, None]))[:, None, :, None], 0)
    for _ in range(n_time):
        t0 = torch.randint(0, T - t_width, (B,), device=x.device)
        w = torch.randint(0, t_width, (B,), device=x.device)
        t = torch.arange(T, device=x.device)[None]
        x = x.masked_fill(((t >= t0[:, None]) & (t < (t0 + w)[:, None]))[:, None, None, :], 0)
    return x


class SmallCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        layers, cin = [], 1
        for cout in (32, 64, 128, 256):
            layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                       nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                       nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, k))

    def forward(self, x):
        x = self.features(x).mean(2)                    # average over frequency
        return self.head(x.max(2)[0] + x.mean(2))       # max + mean over time


def build(name, k):
    if name == "cnn_small":
        return SmallCNN(k)
    return timm.create_model("efficientnet_b0", pretrained=True, in_chans=1, num_classes=k, drop_rate=0.3)


def predict(net, frontend, waves, idx):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            spec = frontend(torch.from_numpy(np.asarray(waves[idx[s:s + 64]], dtype=np.float32)).to(DEV))
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                out.append(torch.sigmoid(net(spec).float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(name, k, waves, X_idx, Y, w, seed):
    """waves[X_idx[i]] is the i-th training clip; Y multi-hot targets, w loss weights."""
    torch.manual_seed(seed)
    epochs, lr = CONFIGS[name]
    net, frontend = build(name, k).to(DEV), LogMel().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(X_idx) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=AMP)
    Yt, wt = torch.tensor(Y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(X_idx)).split(BATCH):
            x = waves[X_idx[b.numpy()]].astype(np.float32)
            x = np.roll(x, rng.integers(0, C.CLIP_SAMPLES), axis=1)                # random circular shift
            with torch.no_grad():                   # frontend in fp32: fp16 power spectra overflow
                spec = spec_augment(frontend(torch.from_numpy(x).to(DEV)))
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                logits = net(spec)
            loss = F.binary_cross_entropy_with_logits(logits.float(), Yt[b].to(DEV), reduction="none")
            loss = (loss.sum(1) * wt[b].to(DEV)).sum() / wt[b].to(DEV).sum()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return net, frontend


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.DATA / "clips.csv")
    k = len(json.loads((C.DATA / "classes.json").read_text()))
    A = dict(np.load(C.DATA / "augmented.npz"))
    y, fold = meta.label.values, meta.fold.values
    n_clean = len(meta)
    print("loading waveforms into RAM ...")
    train_waves = np.concatenate([np.load(C.DATA / "clips.npy"), np.load(C.DATA / "aug_waves.npy")])
    stress_waves = np.load(C.DATA / "stress_waves.npy", mmap_mode="r")
    aug_fold, stress_fold = fold[A["aug_src"]], fold[A["stress_src"]]

    for name in names:
        P, S = np.zeros((n_clean, k)), np.zeros((len(A["stress_src"]), k))
        t0 = time.time()
        for f in range(C.N_FOLDS):
            tr, keep_aug = fold != f, aug_fold != f
            E = {"clean_emb": np.arange(n_clean)[:, None], "aug_emb": (n_clean + np.arange(len(aug_fold)))[:, None],
                 "aug_src": A["aug_src"], "aug_partner": A["aug_partner"]}
            X_idx, Y, w = TE.augmented_training_set(E, y, k, tr, keep_aug)   # indices stand in for embeddings
            net, frontend = train_fold(name, k, train_waves, X_idx[:, 0], Y, w, C.SEED + f)
            P[fold == f] = predict(net, frontend, train_waves, np.where(fold == f)[0])
            S[stress_fold == f] = predict(net, frontend, stress_waves, np.where(stress_fold == f)[0])
            print(f"{name} fold {f}: macro-F1 {f1_score(y[fold == f], P[fold == f].argmax(1), average='macro'):.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        params = sum(p.numel() for p in net.parameters())
        np.savez(OUT / f"oof_{name}.npz", P=P, S=S, train_seconds=time.time() - t0, params=params)
        print(f"{name}: macro-F1 {f1_score(y, P.argmax(1), average='macro'):.3f}, {params / 1e6:.2f}M params")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONFIGS))
