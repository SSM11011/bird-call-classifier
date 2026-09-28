"""Pretrained audio models used as frozen feature extractors.

Every backbone takes the same 5 s, 32 kHz clips and returns
  emb    : one fixed-size vector per clip
  logits : the model's own scores for our target species (bird models only), for zero-shot

  perch    Google Perch v2    bird/wildlife, 14.8k classes    32 kHz, 5 s  -> 1536-d
  birdnet  BirdNET v2.4       birds, 6.5k classes             48 kHz, 3 s  -> 1024-d (mean of 3 windows)
  yamnet   YAMNet             AudioSet (general), 521 classes 16 kHz, 0.96 s frames -> 1024-d (frame mean)
  panns    PANNs CNN14        AudioSet (general), 527 classes 32 kHz, whole clip -> 2048-d
  mfcc     no model           hand-crafted MFCC statistics    32 kHz -> 160-d (classical baseline)

TensorFlow models run on CPU (the local TF build does not match the machine's cuDNN);
CNN14 is re-implemented in plain PyTorch below so it needs no extra package.
"""
import csv
import os

import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

import config as C

MODELS = C.ROOT / "models"
PERCH = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
YAMNET = "google/yamnet/tensorFlow2/yamnet"
BIRDNET_DIR = MODELS / "birdnet"                 # Zenodo record 15050749, BirdNET_v2.4_protobuf.zip
CNN14_CKPT = MODELS / "Cnn14_mAP=0.431.pth"      # Zenodo record 3987831


def _tf():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")    # still needed if torch already initialised CUDA
    return tf


def _batches(waves, batch, desc):
    for s in tqdm(range(0, len(waves), batch), desc=desc, leave=False):
        yield np.asarray(waves[s:s + batch], dtype=np.float32)


class Perch:
    name, dim = "perch", 1536

    def __init__(self, species):
        import kagglehub
        self.tf = _tf()
        path = kagglehub.model_download(PERCH)
        self.model = self.tf.saved_model.load(path).signatures["serving_default"]
        with open(f"{path}/assets/labels.csv") as f:
            labels = [r[0] for r in csv.reader(f)][1:]
        self.target = [labels.index(s) for s in species]

    def __call__(self, waves, batch=32):
        embs, logits = [], []
        for x in _batches(waves, batch, self.name):
            out = self.model(inputs=self.tf.constant(x))
            embs.append(out["embedding"].numpy())
            logits.append(out["label"].numpy()[:, self.target])
        return np.concatenate(embs), np.concatenate(logits)


class BirdNET:
    """3 s model: embed the 5 s clip as three overlapping windows (0-3, 1-4, 2-5 s) and average."""
    name, dim = "birdnet", 1024
    SR, WIN, OFFSETS = 48000, 144000, (0, 48000, 96000)

    def __init__(self, species):
        self.tf = _tf()
        m = self.tf.saved_model.load(str(BIRDNET_DIR / "audio-model"))
        self.emb, self.score = m.signatures["embeddings"], m.signatures["basic"]
        labels = [line.split("_")[0] for line in (BIRDNET_DIR / "labels/en_us.txt").read_text().splitlines()]
        self.target = [labels.index(s) for s in species]

    def __call__(self, waves, batch=32):
        embs, logits = [], []
        for x in _batches(waves, batch, self.name):
            x = resample_poly(x, 3, 2, axis=1).astype(np.float32)        # 32k -> 48k
            win = np.concatenate([x[:, o:o + self.WIN] for o in self.OFFSETS])  # (3B, 144000)
            t = self.tf.constant(win)
            e = self.emb(inputs=t)["embeddings"].numpy().reshape(3, len(x), -1)
            s = self.score(inputs=t)["scores"].numpy()[:, self.target].reshape(3, len(x), -1)
            embs.append(e.mean(0))
            logits.append(s.max(0))            # a call anywhere in the clip counts
        return np.concatenate(embs), np.concatenate(logits)


class YAMNet:
    name, dim = "yamnet", 1024

    def __init__(self, species=None):
        import kagglehub
        self.model = _tf().saved_model.load(kagglehub.model_download(YAMNET))

    def __call__(self, waves, batch=64):
        embs = []
        for x in _batches(waves, batch, self.name):
            x = resample_poly(x, 1, 2, axis=1).astype(np.float32)        # 32k -> 16k
            for xi in x:                                                 # model takes one 1-D waveform
                _, e, _ = self.model(xi)
                embs.append(e.numpy().mean(0))
        return np.stack(embs), None


class PANNs:
    """CNN14 (Kong et al. 2020), weights from the official AudioSet checkpoint. The log-mel
    frontend reuses the checkpoint's own STFT kernels and mel matrix, so it matches training."""
    name, dim = "panns", 2048

    def __init__(self, species=None):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = _cnn14(torch.load(CNN14_CKPT, map_location="cpu", weights_only=False)["model"]).to(self.dev).eval()

    def __call__(self, waves, batch=32):
        embs = []
        with self.torch.no_grad():
            for x in _batches(waves, batch, self.name):
                embs.append(self.net(self.torch.from_numpy(x).to(self.dev)).cpu().numpy())
        return np.concatenate(embs), None


class MFCC:
    """Classical baseline, no pretraining: mean and std over time of 40 MFCCs and their deltas."""
    name, dim = "mfcc", 160

    def __init__(self, species=None):
        import torch
        import torchaudio
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mfcc = torchaudio.transforms.MFCC(C.SR, n_mfcc=40, melkwargs=dict(
            n_fft=1024, hop_length=320, n_mels=64, f_min=150, f_max=15000)).to(self.dev)
        self.delta = torchaudio.functional.compute_deltas

    def __call__(self, waves, batch=128):
        feats = []
        with self.torch.no_grad():
            for x in _batches(waves, batch, self.name):
                # (B, 1, T) input: torchaudio's top_db clamp is then per clip, not per batch
                m = self.mfcc(self.torch.from_numpy(x).to(self.dev)[:, None])[:, 0]   # (B, 40, frames)
                m = self.torch.cat([m, self.delta(m)], 1)
                feats.append(self.torch.cat([m.mean(2), m.std(2)], 1).cpu().numpy())
        return np.concatenate(feats), None


def _cnn14(state):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
            self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

        def forward(self, x, pool):
            x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
            return F.avg_pool2d(x, pool)

    class Cnn14(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("stft_re", state.pop("spectrogram_extractor.stft.conv_real.weight"))
            self.register_buffer("stft_im", state.pop("spectrogram_extractor.stft.conv_imag.weight"))
            self.register_buffer("melW", state.pop("logmel_extractor.melW"))
            self.bn0 = nn.BatchNorm2d(64)
            chans = [1, 64, 128, 256, 512, 1024, 2048]
            self.blocks = nn.ModuleList(Block(a, b) for a, b in zip(chans[:-1], chans[1:]))
            self.fc1 = nn.Linear(2048, 2048)
            self.fc_audioset = nn.Linear(2048, 527)
            renamed = {}                               # conv_blockN (1-based) -> blocks.N-1
            for k, v in state.items():
                if k.startswith("conv_block"):
                    n, rest = k[len("conv_block"):].split(".", 1)
                    k = f"blocks.{int(n) - 1}.{rest}"
                renamed[k] = v
            renamed.update(stft_re=self.stft_re, stft_im=self.stft_im, melW=self.melW)
            self.load_state_dict(renamed)             # strict: fails loudly on a wrong checkpoint

        def forward(self, wave):                       # (B, T) at 32 kHz
            x = F.pad(wave[:, None], (512, 512), mode="reflect")
            spec = F.conv1d(x, self.stft_re, stride=320) ** 2 + F.conv1d(x, self.stft_im, stride=320) ** 2
            mel = spec.transpose(1, 2) @ self.melW                          # (B, frames, 64)
            x = 10 * torch.log10(mel.clamp(min=1e-10))[:, None]              # (B, 1, frames, 64)
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            for i, b in enumerate(self.blocks):
                x = b(x, (2, 2) if i < 5 else (1, 1))
            x = x.mean(3)
            x = x.max(2)[0] + x.mean(2)
            return F.relu(self.fc1(x))

    return Cnn14()


BACKBONES = {"perch": Perch, "birdnet": BirdNET, "yamnet": YAMNet, "panns": PANNs, "mfcc": MFCC}
