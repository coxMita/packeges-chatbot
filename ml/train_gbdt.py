#!/usr/bin/env python3
"""Model A -- LightGBM over the engineered static features.

This is the production model. It is chosen over the embedding model for three
reasons that all matter to the end product:

  * it trains in seconds on CPU and scores a package in ~1ms;
  * SHAP gives exact per-feature attribution for a single prediction, which is
    what the chatbot's LLM narrates;
  * the features are named and documented, so an explanation cites
    "network call at setup.py module level" rather than "dimension 412".

The decision threshold is tuned on out-of-fold predictions rather than left at
0.5. A malicious-package scanner that cries wolf is ignored, so we pick the
threshold that maximises F-beta with beta<1, which weights precision above
recall.

Usage:
    python ml/train_gbdt.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))

from behaviour import MONOTONE_PREFIXES
from config import FEATURES_PARQUET, MODELS, PROCESSED, RANDOM_SEED, REPORTS, ensure_dirs
from features import feature_names
from split import dataset_splits, with_synthetic, xy

# beta < 1 weights precision over recall: a false positive on a legitimate
# package is more damaging to trust than missing one sample in a corpus.
FBETA = 0.5

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": 8,  # not all 12 cores: the laptop runs hot
}
# A tuned parameter set from `ml/experiment.py --tune`, if one has been saved.
TUNED = REPORTS / "gbdt_params.json"
# Out-of-fold benign false-positive rate for the lower "review" threshold:
# above it a package is sent to a human (suspicious) rather than cleared.
REVIEW_FPR = 0.01
# ...and for the upper "strict" threshold: above it the classifier alone is
# enough for the malicious tier, without a second vote from similarity.
STRICT_FPR = 0.001

N_ROUNDS = 800
EARLY_STOPPING = 50


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability cut-off maximising F-beta on out-of-fold scores."""
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        s = fbeta_score(y_true, (y_prob >= t).astype(int), beta=FBETA, zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t


def cv_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: dict = PARAMS,
           folds: int = 5, log: bool = True) -> tuple[np.ndarray, int]:
    """Grouped K-fold out-of-fold scores, and the average early-stopped round count."""
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y))
    best_rounds: list[int] = []

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        pos, neg = int(y[tr].sum()), int((1 - y[tr]).sum())
        booster = lgb.train(
            {**params, "scale_pos_weight": neg / max(pos, 1)},
            lgb.Dataset(X[tr], label=y[tr]),
            num_boost_round=N_ROUNDS,
            valid_sets=[lgb.Dataset(X[va], label=y[va])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        best_rounds.append(booster.best_iteration or N_ROUNDS)
        if log:
            print(f"[cv] fold {fold}: PR-AUC {average_precision_score(y[va], oof[va]):.4f} "
                  f"({booster.best_iteration} rounds)")
    return oof, max(int(np.mean(best_rounds)), 50)


def fit_final(X: np.ndarray, y: np.ndarray, rounds: int, params: dict = PARAMS) -> lgb.Booster:
    pos, neg = int(y.sum()), int((1 - y).sum())
    return lgb.train({**params, "scale_pos_weight": neg / max(pos, 1)},
                     lgb.Dataset(X, label=y), num_boost_round=rounds)


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float) -> float:
    """Lowest threshold whose out-of-fold benign false-positive rate is <= fpr."""
    benign = np.sort(p[y == 0])
    k = int(np.floor(len(benign) * (1 - fpr)))
    return float(benign[min(k, len(benign) - 1)]) + 1e-9


def save_reference(train: pd.DataFrame, cols: list[str], path: Path) -> None:
    """Sorted per-class values of every feature on the real training packages.

    The server compares a new package's value against these to say how common
    it was among malware vs. benign packages the model learned from.
    """
    arrays = {}
    for label, name in ((1, "malicious"), (0, "benign")):
        part = train[train["label"] == label]
        for c in cols:
            arrays[f"{name}/{c}"] = np.sort(part[c].to_numpy(np.float32))
    np.savez_compressed(path, **arrays)
    print(f"[reference] per-class feature distributions -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--include-future", action=argparse.BooleanOptionalAction, default=True,
                    help="also train on the future-holdout malware (default). Its unseen-data "
                         "performance is measured by ml/generalize.py's monthly retraining.")
    ap.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=False,
                    help="add ml/augment.py's trojanised packages (hurt on unseen families)")
    args = ap.parse_args()

    ensure_dirs()
    df = pd.read_parquet(args.data)
    cols = feature_names()

    if df["label"].nunique() < 2:
        print("[error] the dataset contains only one class -- acquire both before training")
        return 1

    splits = dataset_splits(df)
    # Validation and test stay held out either way; the future malware is the
    # newest data there is, and a deployed model should know it.
    train_df = (pd.concat([splits.train, splits.future], ignore_index=True)
                if args.include_future else splits.train)
    n_real = len(train_df)
    if args.synthetic:
        # Trojanised libraries (ml/augment.py): training only, and only those
        # whose host and donor code both sit in this training split.
        train_df = with_synthetic(train_df, PROCESSED / "augmented.parquet")
        print(f"[data] + {len(train_df) - n_real} synthetic packages for training")
    params = {**PARAMS, **json.loads(TUNED.read_text())} if TUNED.exists() else dict(PARAMS)
    if TUNED.exists():
        print(f"[params] tuned set from {TUNED.name}")
    # A danger signal may only raise the score (ml/behaviour.py): on unseen
    # families this beat the unconstrained model on the dev set.
    params["monotone_constraints"] = [1 if c.startswith(MONOTONE_PREFIXES) else 0 for c in cols]
    params["monotone_constraints_method"] = "advanced"
    X, y = xy(train_df, cols)
    groups = train_df["group"].to_numpy()

    print(f"[data] {len(df)} packages | train {n_real} (future malware "
          f"{'included' if args.include_future else 'held out'}) / val {len(splits.val)} / "
          f"test {len(splits.test)}")
    print(f"[data] positives: train {int(y.sum())}, val {int(splits.val['label'].sum())}, "
          f"test {int(splits.test['label'].sum())}")

    # --- cross-validated fit, for an honest threshold and a stable round count
    oof, rounds = cv_oof(X, y, groups, params, folds=args.folds)

    # Thresholds are set on real packages only: the synthetic benign controls
    # are harder than real code and would push every threshold up.
    real = (train_df["pool"] != "synthetic").to_numpy()
    y_real, oof_real = y[real], oof[real]
    print(f"\n[cv] out-of-fold (real packages) ROC-AUC {roc_auc_score(y_real, oof_real):.4f} | "
          f"PR-AUC {average_precision_score(y_real, oof_real):.4f}")

    threshold = tune_threshold(y_real, oof_real)
    print(f"[cv] tuned threshold {threshold:.3f} (F{FBETA}-optimal, precision-weighted)")
    review = min(threshold_at_fpr(y_real, oof_real, REVIEW_FPR), threshold)
    print(f"[cv] review threshold {review:.3f} ({REVIEW_FPR:.0%} out-of-fold false-positive rate)")
    strict = max(threshold_at_fpr(y_real, oof_real, STRICT_FPR), threshold)
    print(f"[cv] strict threshold {strict:.3f} ({STRICT_FPR:.1%} out-of-fold false-positive rate)")

    # --- final fit on all training data, at the CV-average round count
    t0 = time.perf_counter()
    model = fit_final(X, y, rounds, params)
    print(f"[fit] final model: {rounds} rounds in {time.perf_counter() - t0:.1f}s")

    # --- leakage check ------------------------------------------------------
    gain = model.feature_importance("gain")
    ranked = sorted(zip(cols, gain), key=lambda kv: -kv[1])
    print("\n[importance] top 15 by gain:")
    for name, g in ranked[:15]:
        print(f"    {g:12.1f}  {name}")

    size_features = {"pkg_n_files", "pkg_loc", "pkg_n_py_files", "pkg_loc_per_file"}
    top2 = {n for n, _ in ranked[:2]}
    if top2 & size_features:
        print("\n[WARNING] a raw size feature is in the top 2 by gain. The model may be "
              "learning 'small package == malicious' rather than behaviour. Grow the "
              "obscure/hard-negative pool in acquire_benign.py before trusting these "
              "numbers -- see the note in evaluate.py.")

    # --- persist ------------------------------------------------------------
    MODELS.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": cols,
        "threshold": threshold,
        "review_threshold": review,
        "strict_threshold": strict,
        "params": params,
        "fbeta": FBETA,
        "cv_roc_auc": float(roc_auc_score(y_real, oof_real)),
        "cv_pr_auc": float(average_precision_score(y_real, oof_real)),
        "include_future": bool(args.include_future),
        "feature_version": 3,
    }
    with open(MODELS / "gbdt.pkl", "wb") as fh:
        pickle.dump(artifact, fh)

    save_reference(train_df[real], cols, MODELS / "feature_reference.npz")

    (REPORTS / "gbdt_importance.json").write_text(json.dumps(
        [{"feature": n, "gain": float(g)} for n, g in ranked], indent=2
    ))

    print(f"\n[done] model -> {MODELS / 'gbdt.pkl'}")
    print(f"       importances -> {REPORTS / 'gbdt_importance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
