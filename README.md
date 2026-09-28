# Bird Call Classifier (xeno-canto + Perch v2)

Classifies 5 s audio clips into Indian bird species using Google's **Perch v2**
embeddings and a small trained head. Write-up with results: [WRITEUP.md](WRITEUP.md).
Recordings used: [DATASET.md](DATASET.md) (also `recordings.csv`).

## Setup

```bash
pip install -r requirements.txt
echo "XC_API_KEY=<your xeno-canto key>" > .env      # from https://xeno-canto.org/account; .env is gitignored
```

Pretrained models:
- Perch v2 (`google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu`) and YAMNet (`google/yamnet/tensorFlow2/yamnet`)
  download automatically from Kaggle Models via `kagglehub` on first use. No login needed.
- Only needed for the comparison (`04_embed.py`); put them in `models/`:
  - BirdNET v2.4: unzip `BirdNET_v2.4_protobuf.zip` from https://zenodo.org/records/15050749 into `models/birdnet/`
  - PANNs CNN14: `Cnn14_mAP=0.431.pth` from https://zenodo.org/records/3987831
  Both are loaded directly (`backbones.py`); the `birdnet` / `panns_inference` pip packages are not needed, and they pin old TensorFlow/numpy versions.

## Run

| step | command | time* | output |
|---|---|---|---|
| 1. download recordings | `python 01_download.py` | ~30 min | `data/raw/*.mp3`, `recordings.csv` |
| 2. cut & tag 5 s clips, assign CV folds | `python 02_prepare_clips.py` | ~3 min | `data/clips.npy`, `data/clips.csv` |
| 3. faint / noisy / overlap copies, shared by all models | `python 03_augment.py` | ~2 min | `data/{aug,stress}_waves.npy` |
| 4. embeddings from every backbone | `python 04_embed.py [perch birdnet yamnet panns mfcc]` | Perch ~15 min (CPU), others 1–5 min | `data/emb_<model>.npz` |
| 5. main model: train + cross-validate + figures | `python 05_train_eval.py` | ~5 min | `results/` |
| 6. augmentation ablation | `python 06_ablation.py` | ~8 min | `results/ablation.csv` |
| 7. end-to-end CNN baselines (GPU) | `python 07_end_to_end.py` | ~70 min on RTX 4060 | `results/compare/oof_*.npz` |
| 8. comparative study | `python 08_compare.py` | ~10 min | `results/compare/` |
| 9. dataset listing | `python 09_dataset_table.py` | seconds | `DATASET.md` |

\*on an 8-core laptop. Every step caches its output; only 5 needs 1–4, and 8 needs 4 + 7.

Classify new audio with the final model (window-by-window, with an "uncertain" label):

```bash
python predict.py some_recording.mp3
```

## Results files

- `results/classification_report.txt` — per-class precision / recall / F1 (clip and recording level, plus zero-shot Perch)
- `results/confusion_matrix.png` — clip-level and recording-level confusion matrices
- `results/summary.csv`, `results/metrics.json` — all metrics for all variants, confusion matrix, threshold
- `results/ablation.csv` — effect of each augmentation type
- `results/oof_predictions.csv` — out-of-fold probability for every clip
- `results/head.pt` — final trained head (used by `predict.py`)
- `results/compare/comparison.{csv,md}` + `comparison_*.png` — the model comparison (Perch, BirdNET, PANNs, YAMNet, MFCC, CNNs)
- `results/logs/` — console output of each step's last run

## Files

```
config.py            species list, paths, windowing parameters
audio.py             mp3 loading + resampling (soundfile/scipy; no librosa/numba needed)
01_download.py       xeno-canto API v3 query, varied recording selection, download
02_prepare_clips.py  energy-based window selection, faint/noisy/overlap tags, grouped folds
03_augment.py        faint / noisy / overlap augmentation + stress set (same audio for every model)
backbones.py         Perch, BirdNET, YAMNet, PANNs CNN14, MFCC feature extractors
04_embed.py          runs the backbones over real, augmented and stress clips
05_train_eval.py     main model: linear heads, grouped 5-fold CV, thresholds, figures
06_ablation.py       which augmentation helps with what
07_end_to_end.py     small CNN and EfficientNet-B0 trained on log-mel spectrograms
08_compare.py        comparison table, bootstrap CIs, figures
09_dataset_table.py  writes DATASET.md
predict.py           inference on new files
```

Notes
- TensorFlow models (Perch, BirdNET, YAMNet) run on CPU on purpose: the TensorFlow build here did not match the machine's cuDNN. PyTorch models use the GPU if available.
- Oriental Magpie-Robin audio is withheld by xeno-canto (restricted species). If you obtain recordings,
  put them in `data/manual/Oriental Magpie-Robin/XC<id>.mp3` and rerun from step 1; it becomes the 6th class automatically.
