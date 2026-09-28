"""Which augmentation helps with what? Same grouped 5-fold CV as 05_train_eval.py,
varying the augmentation types used in training and the weight of an augmented copy.

Writes results/ablation.csv.
"""
import json
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config as C

T = import_module("05_train_eval")

SETTINGS = [  # (augmentation types, loss weight of an augmented copy)
    ((), 0.0),
    (("faint", "noisy", "overlap"), 1.0),
    (("faint", "noisy", "overlap"), 0.3),
    (("faint",), 0.3),
    (("noisy",), 0.3),
    (("overlap",), 0.3),
    (("faint", "noisy"), 0.3),
]


def main():
    meta = pd.read_csv(C.DATA / "clips.csv")
    k = len(json.loads((C.DATA / "classes.json").read_text()))
    E = dict(np.load(C.DATA / "emb_perch.npz"))
    y, fold = meta.label.values, meta.fold.values
    sy, sfold, skind = y[E["stress_src"]], fold[E["stress_src"]], E["stress_kind"]
    rows = []
    for kinds, weight in SETTINGS:
        P, S = np.zeros((len(y), k)), np.zeros((len(sy), k))
        for f in range(C.N_FOLDS):
            keep_aug = (fold[E["aug_src"]] != f) & np.isin(E["aug_kind"], kinds)
            head = T.train_head(*T.augmented_training_set(E, y, k, fold != f, keep_aug, weight))
            P[fold == f], S[sfold == f] = head(E["clean_emb"][fold == f]), head(E["stress_emb"][sfold == f])
        pred, spred = P.argmax(1), S.argmax(1)
        r = {"augmentation": "+".join(kinds) or "none", "aug_weight": weight,
             "clip_macro_f1": f1_score(y, pred, average="macro")}
        for c in ("faint", "noisy", "overlap"):
            r[f"natural {c}"] = accuracy_score(y[meta[c]], pred[meta[c]])
        for c in ("faint", "noisy", "overlap"):
            r[f"stress {c}"] = accuracy_score(sy[skind == c], spred[skind == c])
        m = skind == "overlap"
        top2 = np.argsort(-S[m], 1)[:, :2]
        r["stress overlap both in top-2"] = np.mean([(a in t) and (b in t)
                                                     for a, b, t in zip(sy[m], E["stress_partner"][m], top2)])
        rows.append(r)
        print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    pd.DataFrame(rows).round(3).to_csv(C.RESULTS / "ablation.csv", index=False)


if __name__ == "__main__":
    main()
