"""Comparative study: which model family works best on this data, and why.

Every method is scored on the same clips, the same recording-grouped 5 folds,
the same augmented / stress audio, with the same metrics (05_train_eval.metrics).

  zero-shot            the bird model's own classifier, restricted to our species
  frozen + linear      pretrained embedding -> class-weighted linear head (real clips only)
  frozen + linear +aug same, plus the faint / noisy / overlap copies (AUG_WEIGHT)
  end-to-end           CNN trained on log-mel (07_end_to_end.py), with the augmented copies

Backbones: Perch v2 and BirdNET v2.4 (bird-specific), PANNs CNN14 and YAMNet (general
AudioSet), MFCC statistics (no learned features), ImageNet EfficientNet-B0 and a small
CNN from scratch (no audio pretraining).

Writes results/compare/{comparison.csv, comparison.md, *.png}.
"""
import json
import time
from importlib import import_module

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TE = import_module("05_train_eval")
OUT = C.RESULTS / "compare"
N_BOOT = 1000

BACKBONES = {  # name -> (label, family, device the embeddings were extracted on)
    "perch": ("Perch v2", "bird", "CPU"),
    "birdnet": ("BirdNET v2.4", "bird", "CPU"),
    "panns": ("PANNs CNN14", "general", "GPU"),
    "yamnet": ("YAMNet", "general", "CPU"),
    "mfcc": ("MFCC stats", "none", "GPU"),
}
END_TO_END = {"effnet_b0": "EfficientNet-B0 (ImageNet)", "cnn_small": "Small CNN (scratch)"}
FAMILY = {"bird": ("bird-audio pretrained", TE.SERIES[0]),
          "general": ("general-audio pretrained", TE.SERIES[1]),
          "none": ("no audio pretraining", TE.SERIES[2])}


def linear_cv(E, meta, k, with_aug):
    y, fold = meta.label.values, meta.fold.values
    aug_fold, stress_fold = fold[E["aug_src"]], fold[E["stress_src"]]
    P, S = np.zeros((len(meta), k)), np.zeros((len(E["stress_src"]), k))
    t0 = time.time()
    for f in range(C.N_FOLDS):
        tr, te = fold != f, fold == f
        if with_aug:
            head = TE.train_head(*TE.augmented_training_set(E, y, k, tr, aug_fold != f))
        else:
            head = TE.train_head(E["clean_emb"][tr], TE.multi_hot(y[tr], k), TE.class_weights(y[tr], k))
        P[te] = head(E["clean_emb"][te])
        S[stress_fold == f] = head(E["stress_emb"][stress_fold == f])
    return P, S, (time.time() - t0) / C.N_FOLDS, E["clean_emb"].shape[1] * k + k


def bootstrap_ci(meta, prob):
    """95% interval of clip macro-F1, resampling whole recordings (clips of one recording are not independent)."""
    y, pred = meta.label.values, prob.argmax(1)
    recs = meta.xc_id.unique()
    by_rec = {r: np.where(meta.xc_id.values == r)[0] for r in recs}
    rng = np.random.default_rng(C.SEED)
    scores = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_rec[r] for r in rng.choice(recs, len(recs))])
        scores.append(f1_score(y[idx], pred[idx], average="macro"))
    return np.percentile(scores, [2.5, 97.5])


def run_all(meta, classes):
    k = len(classes)
    A = dict(np.load(C.DATA / "augmented.npz"))
    times = json.loads((C.DATA / "emb_times.json").read_text())
    runs = []   # (method, backbone label, family, P, S, head seconds/fold, trainable params)
    for name, (label, fam, dev) in BACKBONES.items():
        E = dict(np.load(C.DATA / f"emb_{name}.npz"))
        emb_rate = f"{times[name]['clips'] / times[name]['seconds']:.0f} ({dev})"
        if "clean_logit" in E:
            runs.append(dict(method=f"{label} · zero-shot", backbone=name, family=fam, kind="zero-shot",
                             P=TE.softmax(E["clean_logit"]), S=TE.softmax(E["stress_logit"]), train_s=0, params=0,
                             clips_per_s=emb_rate))
        for with_aug in (False, True):
            P, S, t, n = linear_cv(E, meta, k, with_aug)
            runs.append(dict(method=f"{label} · linear" + (" +aug" if with_aug else ""), backbone=name, family=fam,
                             kind="linear +aug" if with_aug else "linear", P=P, S=S, train_s=t, params=n,
                             clips_per_s=emb_rate))
            print(f"{runs[-1]['method']:32s} macro-F1 {f1_score(meta.label, P.argmax(1), average='macro'):.3f}")
    for name, label in END_TO_END.items():
        R = np.load(OUT / f"oof_{name}.npz")
        runs.append(dict(method=f"{label} · end-to-end +aug", backbone=name, family="none", kind="end-to-end +aug",
                         P=R["P"], S=R["S"], train_s=float(R["train_seconds"]) / C.N_FOLDS, params=int(R["params"]),
                         clips_per_s=""))
    return runs, A


def table(runs, meta, A, classes):
    rows = []
    for r in runs:
        m = TE.metrics(meta, A, r["P"], r["S"], classes)
        lo, hi = bootstrap_ci(meta, r["P"])
        rows.append({"method": r["method"], "family": FAMILY[r["family"]][0],
                     "clip macro-F1": m["clip_macro_f1"], "95% CI": f"{lo:.3f}-{hi:.3f}",
                     "recording macro-F1": m["recording_macro_f1"],
                     **{f"F1 {c}": m[f"f1 {c}"] for c in classes},
                     "natural faint": m["natural: faint"], "natural noisy": m["natural: noisy"],
                     "natural overlap": m["natural: overlap"],
                     "stress faint": m["stress: faint"], "stress noisy": m["stress: noisy"],
                     "stress overlap both top-2": m["stress: overlap, both species in top-2"],
                     "trainable params": r["params"], "train s/fold": r["train_s"],
                     "embedding clips/s": r["clips_per_s"]})
    return pd.DataFrame(rows).set_index("method")


def markdown(df, classes):
    cols = ["clip macro-F1", "95% CI", "recording macro-F1"] + [f"F1 {c}" for c in classes] + \
           ["stress faint", "stress noisy", "stress overlap both top-2", "trainable params"]
    head = ["method"] + [f"F1 {c.split()[-1]}" if c.startswith("F1 ") else c for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    best = df["clip macro-F1"].max()
    for method, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "trainable params":
                cells.append(f"{v / 1e6:.1f}M" if v >= 1e5 else (f"{v / 1e3:.1f}k" if v else "0"))
            elif isinstance(v, float):
                cells.append(f"**{v:.3f}**" if c == "clip macro-F1" and v == best else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ figures
def style(ax):
    ax.set_facecolor(TE.SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TE.GRID)
    ax.tick_params(colors=TE.INK2, labelsize=9)
    ax.grid(axis="x", color=TE.GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_scores(df, runs):
    """Clip and recording macro-F1 per method, bars coloured by pretraining family."""
    order = df.sort_values("clip macro-F1").index
    fam = {r["method"]: r["family"] for r in runs}
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.36 * len(order) + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    fig.patch.set_facecolor(TE.SURFACE)
    panels = [("clip macro-F1", "Clip macro-F1 (95% CI)"), ("recording macro-F1", "Recording macro-F1"),
              ("stress faint", "Accuracy on synthetic faint clips")]
    for ax, (col, title) in zip(axes, panels):
        style(ax)
        vals = df.loc[order, col].values
        colors = [FAMILY[fam[m]][1] for m in order]
        ax.barh(range(len(order)), vals, height=0.62, color=colors, edgecolor=TE.SURFACE, linewidth=1)
        if col == "clip macro-F1":
            ci = np.array([[float(x) for x in df.loc[m, "95% CI"].split("-")] for m in order])
            ax.errorbar(vals, range(len(order)), xerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt="none",
                        ecolor=TE.INK, elinewidth=1, capsize=2)
        for i, v in enumerate(vals):
            ax.text(0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=8, color="white", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, color=TE.INK, loc="left")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=9, color=TE.INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in FAMILY.values()]
    fig.legend(handles, [label for label, _ in FAMILY.values()], loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.6, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "comparison_scores.png", dpi=150, facecolor=TE.SURFACE)
    plt.close(fig)


def fig_per_class(df, classes):
    order = df.sort_values("clip macro-F1", ascending=False).index
    cols = [f"F1 {c}" for c in classes] + ["stress faint", "stress noisy", "stress overlap both top-2"]
    M = df.loc[order, cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(order) + 1.8))
    fig.patch.set_facecolor(TE.SURFACE)
    ax.imshow(M, cmap="Blues", vmin=0.3, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.75 else TE.INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"F1\n{c.split()[-1]}" if c.startswith("F1 ") else
                        c.replace("stress ", "stress\n").replace(" both top-2", "\n(both top-2)")
                        for c in cols], fontsize=8, color=TE.INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9, color=TE.INK)
    ax.axvline(len(classes) - 0.5, color=TE.SURFACE, lw=4)
    ax.set_title("Per-class F1 (real clips) and accuracy under synthetic degradation", fontsize=10, color=TE.INK,
                 loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_per_class.png", dpi=150, facecolor=TE.SURFACE)
    plt.close(fig)


def fig_confusions(runs, meta, classes, methods):
    y = meta.label.values
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    fig.patch.set_facecolor(TE.SURFACE)
    by = {r["method"]: r for r in runs}
    for ax, m in zip(axes.flat, methods):
        pred = by[m]["P"].argmax(1)
        TE.plot_confusion(ax, confusion_matrix(y, pred, labels=range(len(classes))), [c.split()[-1] for c in classes],
                          m.replace(" · ", "\n") + f"  (macro-F1 {f1_score(y, pred, average='macro'):.3f})")
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_confusion.png", dpi=130, facecolor=TE.SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.DATA / "clips.csv")
    classes = json.loads((C.DATA / "classes.json").read_text())
    runs, A = run_all(meta, classes)
    df = table(runs, meta, A, classes)
    df.to_csv(OUT / "comparison.csv", float_format="%.4f")
    (OUT / "comparison.md").write_text(markdown(df, classes))
    print(df[["clip macro-F1", "95% CI", "recording macro-F1", "stress faint", "stress noisy"]].to_string())

    fig_scores(df, runs)
    fig_per_class(df, classes)
    # confusion matrices for the best variant of each backbone, plus the best end-to-end CNN
    prefixes = [label for label, _, _ in BACKBONES.values()] + [END_TO_END["effnet_b0"]]
    best = [df[df.index.str.startswith(p)]["clip macro-F1"].idxmax() for p in prefixes]
    fig_confusions(runs, meta, classes, best)


if __name__ == "__main__":
    main()
