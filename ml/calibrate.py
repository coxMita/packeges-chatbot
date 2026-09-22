#!/usr/bin/env python3
"""Turn the two raw scores into "chance this is really malware".

Neither raw score is that probability. The classifier's output and the
similarity share are only ranks, and both were measured on a test set that is
26% malware -- far above the rate on real PyPI. This script fits a tiny
logistic calibrator on the held-out split, where neither model has seen the
labels, and records the test base rate so the server can re-weight the answer
to a realistic one (Bayes' rule on the odds).

It also fixes the similarity flag threshold used by the three-tier verdict:

    malicious   classifier and similarity both flag it
    suspicious  exactly one of them does
    clean       neither does

No model is retrained. Similarity is scored against training packages only,
so a held-out package can never find itself.

Usage:
    python ml/calibrate.py     -> ml/models/calibration.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from config import FEATURES_PARQUET, MODELS  # noqa: E402
from similarity import SimilarityIndex, neighbour_vote  # noqa: E402
from split import grouped_split  # noqa: E402

SIM_FLAG = 0.60   # similarity share at or above which similarity "flags"
EPS = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main() -> int:
    df = pd.read_parquet(FEATURES_PARQUET)
    train, test = grouped_split(df)

    gbdt = pickle.load(open(MODELS / "gbdt.pkl", "rb"))
    idx = SimilarityIndex()
    if not idx.ready:
        print("[error] similarity index unavailable -- run train_embed.py first")
        return 1

    pos = {k: i for i, k in enumerate(idx.keys)}
    key = lambda d: [f"{p}@{v}" for p, v in zip(d["package"], d["version"])]  # noqa: E731
    train_rows = np.array([pos[k] for k in key(train) if k in pos])
    matrix = idx.matrix[train_rows]
    is_mal = idx._is_malicious[train_rows]

    test = test[[k in pos for k in key(test)]].reset_index(drop=True)
    sims = idx.matrix[[pos[k] for k in key(test)]] @ matrix.T
    sim = np.array([neighbour_vote(row, matrix, is_mal)[2] for row in sims])
    clf = gbdt["model"].predict(test[gbdt["feature_names"]])
    y = test["label"].to_numpy()
    base_rate = float(y.mean())

    X_both = np.column_stack([logit(clf), sim])
    X_clf = logit(clf).reshape(-1, 1)

    # Out-of-fold check that the calibrator is honest before fitting on all of it.
    oof = cross_val_predict(LogisticRegression(), X_both, y, cv=5, method="predict_proba")[:, 1]
    print(f"[data] {len(y)} held-out packages, {base_rate:.1%} malware\n")
    print(" predicted   n   actually malware   (5-fold, held-out)")
    for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.01)]:
        m = (oof >= lo) & (oof < hi)
        if m.any():
            print(f" {lo:>4.0%}-{min(hi, 1):<4.0%} {m.sum():>5}   {y[m].mean():>6.1%}")

    both = LogisticRegression().fit(X_both, y)
    clf_only = LogisticRegression().fit(X_clf, y)

    clf_flag = clf >= gbdt["threshold"]
    sim_flag = sim >= SIM_FLAG
    print("\n tier         benign   malware")
    for name, m in [("malicious", clf_flag & sim_flag), ("suspicious", clf_flag ^ sim_flag),
                    ("clean", ~clf_flag & ~sim_flag)]:
        print(f" {name:11} {int(m[y == 0].sum()):>6}   {int(m[y == 1].sum()):>7}")

    out = {
        "base_rate": base_rate,
        "sim_flag": SIM_FLAG,
        "classifier_threshold": float(gbdt["threshold"]),
        "both": {"coef": both.coef_[0].tolist(), "intercept": float(both.intercept_[0])},
        "classifier_only": {"coef": clf_only.coef_[0].tolist(),
                            "intercept": float(clf_only.intercept_[0])},
        "n_heldout": int(len(y)),
    }
    path = MODELS / "calibration.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
