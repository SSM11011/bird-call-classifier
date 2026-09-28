"""Train classifier heads on Perch embeddings, evaluate with grouped 5-fold CV.

Every number reported is out-of-fold: each clip is predicted by a model that never
saw its recording (folds are split by recording, stratified by species). With only
21 Eagle-Owl recordings, a single hold-out split would leave ~4 test recordings;
cross-validation uses all of them as test data exactly once.

Variants (an ablation, so each fix for the messy data is measured separately):
  zero_shot      Perch v2's own classifier, restricted to our species. No training.
  probe          linear head on clean clips, plain loss.
  probe_weighted + class-balanced loss weights (imbalance).
  full           + augmented copies: faint / noisy / overlapping (multi-label targets),
                 with minority classes oversampled via extra augmented copies. Augmented
                 copies are down-weighted (AUG_WEIGHT) so they regularise rather than dominate.

The "full" model additionally gets a confidence threshold ("uncertain" below it)
and recording-level aggregation. Outputs go to results/.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_PRECISION = 0.98       # operating point for the "uncertain" threshold
AUG_WEIGHT = 0.3              # loss weight of an augmented copy relative to a real clip (see 06_ablation.py)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# --------------------------------------------------------------------------- model
class Head:
    """Linear layer on standardised Perch embeddings with sigmoid outputs (multi-label BCE),
    so a clip with two species calling can carry both labels."""

    def __init__(self, net, mu, sd):
        self.net, self.mu, self.sd = net, mu, sd

    def __call__(self, Z):
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor((Z - self.mu) / self.sd, dtype=torch.float32))).numpy()


def train_head(X, Y, w, epochs=120, seed=C.SEED):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt, wt = torch.tensor(Y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(X.shape[1], Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt)).split(256):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]), Yt[idx], reduction="none")
            loss = (loss.sum(1) * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return Head(net, mu, sd)


def class_weights(labels, k):
    counts = np.bincount(labels, minlength=k)
    return (len(labels) / (k * counts))[labels]


def augmented_training_set(E, y, k, keep_clean, keep_aug, aug_weight=AUG_WEIGHT):
    """Real clips + their augmented copies, with class-balanced, aug-down-weighted loss weights."""
    ya = y[E["aug_src"][keep_aug]]
    X = np.concatenate([E["clean_emb"][keep_clean], E["aug_emb"][keep_aug]])
    Y = np.concatenate([multi_hot(y[keep_clean], k), multi_hot(ya, k, E["aug_partner"][keep_aug])])
    w = class_weights(np.concatenate([y[keep_clean], ya]), k)
    w[keep_clean.sum():] *= aug_weight
    return X, Y, w


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def multi_hot(labels, k, partners=None):
    Y = np.eye(k)[labels]
    if partners is not None:
        has = partners >= 0
        Y[np.where(has)[0], partners[has]] = 1.0
    return Y


# ---------------------------------------------------------------------- experiment
def run_cv(E, meta, k):
    y, fold = meta.label.values, meta.fold.values
    aug_fold = fold[E["aug_src"]]
    variants = ["probe", "probe_weighted", "full"]
    P = {v: np.zeros((len(meta), k)) for v in variants}
    S = {v: np.zeros((len(E["stress_src"]), k)) for v in variants}
    stress_fold = fold[E["stress_src"]]
    for f in range(C.N_FOLDS):
        tr, te, ste = fold != f, fold == f, stress_fold == f
        Xc, yc = E["clean_emb"][tr], y[tr]
        heads = {
            "probe": train_head(Xc, multi_hot(yc, k), np.ones(len(yc))),
            "probe_weighted": train_head(Xc, multi_hot(yc, k), class_weights(yc, k)),
            "full": train_head(*augmented_training_set(E, y, k, tr, aug_fold != f)),
        }
        for v, h in heads.items():
            P[v][te] = h(E["clean_emb"][te])
            S[v][ste] = h(E["stress_emb"][ste])
        print(f"fold {f}: " + "  ".join(f"{v}={f1_score(y[te], P[v][te].argmax(1), average='macro'):.3f}" for v in variants))
    P["zero_shot"], S["zero_shot"] = softmax(E["clean_logit"]), softmax(E["stress_logit"])
    return P, S


def recordist_cv(E, meta, k):
    """Stricter check: folds grouped by *recordist*, not recording. 13 of the 21
    Eagle-Owl recordings come from one recordist, so a recording-level split can still
    put the same site/mic/individual on both sides. Uses the class-weighted probe on
    real clips only (augmentation partners were drawn per recording-fold, not per
    recordist-fold, so reusing them here would leak)."""
    rec = pd.read_csv(C.ROOT / "recordings.csv").set_index("xc_id").recordist
    groups = meta.xc_id.map(rec).values
    y = meta.label.values
    P = np.zeros((len(y), k))
    for tr, te in StratifiedGroupKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(y, y, groups):
        P[te] = train_head(E["clean_emb"][tr], multi_hot(y[tr], k), class_weights(y[tr], k))(E["clean_emb"][te])
    return P


def recording_level(meta, prob):
    """Average log-probabilities of a recording's clips -> one prediction per recording."""
    g = pd.DataFrame(np.log(prob + 1e-9)).groupby(meta.xc_id.values).mean()
    truth = meta.groupby("xc_id").label.first().loc[g.index]
    return truth.values, g.values.argmax(1)


def metrics(meta, E, prob, stress_prob, classes):
    """All reported numbers for one model, from its out-of-fold probabilities."""
    y = meta.label.values
    pred = prob.argmax(1)
    ry, rp = recording_level(meta, prob)
    sy, sp = y[E["stress_src"]], stress_prob.argmax(1)
    r = {"clip_accuracy": accuracy_score(y, pred), "clip_balanced_accuracy": balanced_accuracy_score(y, pred),
         "clip_macro_f1": f1_score(y, pred, average="macro"),
         "recording_accuracy": accuracy_score(ry, rp), "recording_macro_f1": f1_score(ry, rp, average="macro"),
         "fold_macro_f1_std": float(np.std([f1_score(y[meta.fold == f], pred[meta.fold == f], average="macro")
                                            for f in range(C.N_FOLDS)]))}
    for c, name in enumerate(classes):
        r[f"f1 {name}"] = f1_score(y == c, pred == c)
    clean = ~(meta.faint | meta.noisy | meta.overlap)
    for name, mask in [("none of the three", clean), ("faint", meta.faint), ("noisy", meta.noisy),
                       ("overlap", meta.overlap)]:
        r[f"natural: {name}"] = accuracy_score(y[mask], pred[mask])
    for kind in ["faint", "noisy", "overlap"]:
        m = E["stress_kind"] == kind
        r[f"stress: {kind}"] = accuracy_score(sy[m], sp[m])
    m = E["stress_kind"] == "overlap"
    top2 = np.argsort(-stress_prob[m], 1)[:, :2]
    both = [(a in t) and (b in t) for a, b, t in zip(sy[m], E["stress_partner"][m], top2)]
    r["stress: overlap, both species in top-2"] = float(np.mean(both))
    return r


def summarise(meta, E, P, S, classes):
    return pd.DataFrame({v: metrics(meta, E, P[v], S[v], classes)
                         for v in ["zero_shot", "probe", "probe_weighted", "full"]})


def threshold_analysis(meta, prob, classes):
    y = meta.label.values
    conf, pred = prob.max(1), prob.argmax(1)
    order = np.argsort(-conf)
    correct = (pred == y)[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    acc = np.cumsum(correct) / np.arange(1, len(y) + 1)
    ok = np.where(acc >= TARGET_PRECISION)[0]
    i = ok.max() if len(ok) else 0
    tau = float(conf[order][i])
    kept = conf >= tau
    per_class = {c: {"coverage": float(kept[y == j].mean()),
                     "accuracy_when_confident": float((pred == y)[kept & (y == j)].mean()) if (kept & (y == j)).any() else None}
                 for j, c in enumerate(classes)}
    return {"threshold": tau, "coverage": float(kept.mean()), "accuracy_when_confident": float(acc[i]),
            "per_class": per_class}, cov, acc


# ------------------------------------------------------------------------ figures
def plot_confusion(ax, cm, classes, title):
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)), [c.replace(" ", "\n", 1) for c in classes], fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)


def figures(meta, P, summary, cov, acc, thr, classes):
    y = meta.label.values
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    plot_confusion(axes[0], confusion_matrix(y, P["full"].argmax(1)), classes, "Per 5 s clip (out-of-fold)")
    ry, rp = recording_level(meta, P["full"])
    plot_confusion(axes[1], confusion_matrix(ry, rp, labels=range(len(classes))), classes, "Per recording (clips averaged)")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    names = {"zero_shot": "Perch zero-shot", "probe": "Linear probe", "probe_weighted": "+ class weights",
             "full": "+ augmentation (final)"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    h = 0.8 / len(names)
    for i, (v, label) in enumerate(names.items()):
        vals = [summary.loc[f"f1 {c}", v] for c in classes]
        ax.barh(np.arange(len(classes)) + (i - 1.5) * h, vals, height=h - 0.03, color=SERIES[i], label=label)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (out-of-fold, per clip)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
    ax.set_title("Per-class F1 by variant", pad=28)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "per_class_f1.png", dpi=150)
    plt.close(fig)

    conds = ["natural: none of the three", "natural: faint", "natural: noisy", "natural: overlap",
             "stress: faint", "stress: noisy", "stress: overlap, both species in top-2"]
    shown = ["zero_shot", "probe_weighted", "full"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x, w = np.arange(len(conds)), 0.27
    for i, v in enumerate(shown):
        vals = summary.loc[conds, v].values
        off = (i - 1) * w
        ax.bar(x + off, vals, width=w - 0.02, color=SERIES[list(names).index(v)], label=names[v])
        for xi, val in zip(x, vals):
            ax.text(xi + off, val + 0.01, f"{val:.2f}", ha="center", fontsize=7, color=INK2)
    labels = [c.replace(": ", "\n").replace(", both species in top-2", "\n(both in top-2)") for c in conds]
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.axvline(3.5, color=GRID, lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("clip accuracy")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    ax.set_title("Accuracy by condition: real slices of the data (left) vs controlled degradations (right)", pad=26)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "robustness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(cov, acc, color=SERIES[0], lw=2)
    ax.axhline(TARGET_PRECISION, color=INK2, lw=1, ls="--")
    ax.text(0.01, TARGET_PRECISION + 0.002, f"target {TARGET_PRECISION:.0%}", fontsize=8, color=INK2)
    ax.plot([thr["coverage"]], [thr["accuracy_when_confident"]], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2)
    ax.annotate(f"threshold {thr['threshold']:.2f}\nkeeps {thr['coverage']:.0%} of clips",
                (thr["coverage"], thr["accuracy_when_confident"]), xytext=(10, -34), textcoords="offset points",
                fontsize=9, color=INK)
    ax.set_xlabel("coverage (fraction of clips given a label)")
    ax.set_ylabel("accuracy on those clips")
    ax.set_ylim(min(acc.min(), 0.8) - 0.02, 1.005)
    ax.grid(color=GRID, lw=0.8)
    ax.set_title("Abstaining on low-confidence clips")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "coverage_accuracy.png", dpi=150)
    plt.close(fig)


def example_spectrograms(meta, classes):
    from scipy.signal import spectrogram
    ex = np.load(C.DATA / "stress_examples.npy")
    info = json.loads((C.DATA / "stress_examples.json").read_text())
    clips = np.load(C.DATA / "clips.npy", mmap_mode="r")
    panels = [(clips[info[0]["src"]].astype(np.float32), f"original: {meta.species[info[0]['src']]}")]
    for w, d in zip(ex, info):
        extra = f" + {classes[d['partner']]}" if d["partner"] >= 0 else ""
        panels.append((w, f"{d['kind']}: {meta.species[d['src']]}{extra}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.2), sharey=True)
    for ax, (w, title) in zip(axes, panels):
        f, t, S = spectrogram(w, fs=C.SR, nperseg=1024, noverlap=768)
        keep = f <= 12_000
        ax.pcolormesh(t, f[keep] / 1000, 10 * np.log10(S[keep] + 1e-14), cmap="magma", vmin=-120, vmax=-50,
                      shading="auto", rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("s")
    axes[0].set_ylabel("kHz")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "augmentation_examples.png", dpi=130)
    plt.close(fig)


def main():
    C.RESULTS.mkdir(exist_ok=True)
    meta = pd.read_csv(C.DATA / "clips.csv")
    classes = json.loads((C.DATA / "classes.json").read_text())
    E = dict(np.load(C.DATA / "emb_perch.npz"))
    P, S = run_cv(E, meta, len(classes))

    summary = summarise(meta, E, P, S, classes)
    strict = recordist_cv(E, meta, len(classes))
    y = meta.label.values
    summary.loc["clip_macro_f1 (leave-recordist-out, probe_weighted)", "probe_weighted"] = \
        f1_score(y, strict.argmax(1), average="macro")
    for c, name in enumerate(classes):
        summary.loc[f"f1 {name} (leave-recordist-out)", "probe_weighted"] = f1_score(y == c, strict.argmax(1) == c)
    thr, cov, acc = threshold_analysis(meta, P["full"], classes)
    ry, rp = recording_level(meta, P["full"])
    report = ("CLIP LEVEL (5 s windows, out-of-fold)\n"
              + classification_report(y, P["full"].argmax(1), target_names=classes, digits=3)
              + "\nRECORDING LEVEL (clip log-probs averaged)\n"
              + classification_report(ry, rp, labels=range(len(classes)), target_names=classes, digits=3, zero_division=0)
              + "\nZERO-SHOT PERCH, CLIP LEVEL\n"
              + classification_report(y, P["zero_shot"].argmax(1), target_names=classes, digits=3))
    (C.RESULTS / "classification_report.txt").write_text(report)
    summary.round(3).to_csv(C.RESULTS / "summary.csv")
    cm = confusion_matrix(y, P["full"].argmax(1))
    (C.RESULTS / "metrics.json").write_text(json.dumps({
        "classes": classes, "n_clips": int(len(meta)), "n_recordings": int(meta.xc_id.nunique()),
        "summary": summary.round(4).to_dict(), "confidence_threshold": thr,
        "confusion_matrix_clip": cm.tolist()}, indent=2))
    pd.DataFrame(P["full"], columns=classes).assign(xc_id=meta.xc_id, true=meta.species, fold=meta.fold) \
        .to_csv(C.RESULTS / "oof_predictions.csv", index=False)

    figures(meta, P, summary, cov, acc, thr, classes)
    example_spectrograms(meta, classes)

    # Final model on all clips + augmentations, used by predict.py.
    everything = np.ones(len(y), bool)
    head = train_head(*augmented_training_set(E, y, len(classes), everything, np.ones(len(E["aug_src"]), bool)))
    torch.save({"state_dict": head.net.state_dict(), "mu": head.mu, "sd": head.sd,
                "classes": classes, "threshold": thr["threshold"]}, C.RESULTS / "head.pt")

    print(report)
    print(summary.round(3).to_string())
    print(json.dumps(thr, indent=1))


if __name__ == "__main__":
    main()
