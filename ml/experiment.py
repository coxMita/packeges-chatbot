#!/usr/bin/env python3
"""Model A ablations: what did the new data and the new features each buy?

Every configuration is scored on the same held-out packages, plus a "future"
set: malware first reported on or after --cutoff whose behaviour fingerprint
never appears before it. Future packages are excluded from training *and* the
regular test split, so their recall is the closest thing we have to "how does
this do on malware nobody has trained on yet".

Usage:
    python ml/experiment.py                  # ablation table
    python ml/experiment.py --tune 24        # plus a random hyperparameter search
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))

from config import FEATURES_PARQUET, PROCESSED, RANDOM_SEED, REPORTS
from features import feature_names
from split import FUTURE_CUTOFF, dataset_splits, with_synthetic
from train_gbdt import PARAMS, cv_oof, fit_final, threshold_at_fpr, tune_threshold

NEW_FEATURES = ["call_computed_import", "call_obfuscated_import", "obf_char_code_build",
                "worst_file_categories", "worst_file_suspicious_per_kloc"]
PROBES = Path(__file__).parent / "reports" / "probes.parquet"


def evaluate(name, cols, train, test, future, probes, params=PARAMS) -> dict:
    X, y = train[cols].to_numpy(np.float32), train["label"].to_numpy()
    oof, rounds = cv_oof(X, y, train["group"].to_numpy(), params, log=False)
    # Thresholds and OOF scores on real packages only: synthetic controls are
    # harder than real benign code and would skew both.
    real = (train["pool"] != "synthetic").to_numpy()
    y_real, oof_real = y[real], oof[real]
    t_f05 = tune_threshold(y_real, oof_real)
    t_1 = threshold_at_fpr(y_real, oof_real, 0.01)
    model = fit_final(X, y, rounds, params)

    p = model.predict(test[cols].to_numpy(np.float32))
    yt = test["label"].to_numpy()
    ben, mal = yt == 0, yt == 1
    big = mal & (test["pkg_n_files"].to_numpy() >= 11)
    mr = mal & (test["source"] == "malregistry").to_numpy()
    dd = mal & (test["source"] == "datadog").to_numpy()
    obs = (test["pool"] == "obscure").to_numpy()
    pf = model.predict(future[cols].to_numpy(np.float32)) if len(future) else np.array([])

    def rec(mask, t, probs=p):
        return float((probs[mask] >= t).mean()) if mask.any() else float("nan")

    row = {
        "config": name,
        "oof_pr_auc": average_precision_score(y_real, oof_real),
        "test_pr_auc": average_precision_score(yt, p),
        "thr_f05": t_f05,
        "precision_f05": float(yt[p >= t_f05].mean()) if (p >= t_f05).any() else float("nan"),
        "recall_f05": rec(mal, t_f05),
        "fpr_f05": rec(ben, t_f05),
        "obscure_fpr_f05": rec(obs, t_f05),
        "recall_fpr1": rec(mal, t_1),
        "fpr_fpr1": rec(ben, t_1),
        "recall_big_f05": rec(big, t_f05),
        "recall_big_fpr1": rec(big, t_1),
        "recall_datadog_f05": rec(dd, t_f05),
        "recall_malreg_f05": rec(mr, t_f05),
        "future_recall_f05": float((pf >= t_f05).mean()) if len(pf) else float("nan"),
        "future_recall_fpr1": float((pf >= t_1).mean()) if len(pf) else float("nan"),
    }
    if probes is not None:
        for _, r in probes.iterrows():
            row[f"probe_{r['package']}"] = float(model.predict(r[cols].to_numpy(np.float32)[None])[0])
    return row


def load_probes(cols: list[str]) -> pd.DataFrame | None:
    """Live packages kept out of every dataset (see `--refresh-probes`)."""
    return pd.read_parquet(PROBES) if PROBES.exists() else None


def refresh_probes() -> None:
    import asyncio
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from acquire_benign import fetch_top_packages, normalise
    from features import extract_features
    from fetch import fetch_package
    popular = {normalise(p) for p in fetch_top_packages()[:2000]}
    rows = []
    for spec in ["index-forum==2.5.4"]:
        pkg = asyncio.run(fetch_package(spec))
        try:
            rows.append({"package": pkg.name, **extract_features(pkg.root, pkg.name, popular)})
        finally:
            pkg.cleanup()
    PROBES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(PROBES, index=False)


SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [5, 10, 20, 40],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.7, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0],
    "min_gain_to_split": [0.0, 0.1],
}


def tune(train: pd.DataFrame, cols: list[str], trials: int) -> dict:
    """Random search on grouped-CV PR-AUC. The test split is never touched."""
    X, y = train[cols].to_numpy(np.float32), train["label"].to_numpy()
    groups = train["group"].to_numpy()
    real = (train["pool"] != "synthetic").to_numpy()

    def score(params: dict) -> float:
        # Scored on real packages; synthetic rows only help the model learn.
        return average_precision_score(y[real], cv_oof(X, y, groups, params, log=False)[0][real])

    rng = random.Random(RANDOM_SEED)
    best, best_score = PARAMS, score(PARAMS)
    print(f"[tune] baseline CV PR-AUC {best_score:.4f}")
    for i in range(trials):
        params = {**PARAMS, **{k: rng.choice(v) for k, v in SEARCH_SPACE.items()}}
        score_i = score(params)
        mark = ""
        if score_i > best_score:
            best, best_score, mark = params, score_i, "  <- best"
        print(f"[tune] {i + 1:>2}/{trials} PR-AUC {score_i:.4f}{mark}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", default=FUTURE_CUTOFF)
    ap.add_argument("--tune", type=int, default=0, help="random-search trials (0 = skip)")
    ap.add_argument("--only", default="", help="comma-separated config prefixes, e.g. A3,A5")
    ap.add_argument("--refresh-probes", action="store_true")
    args = ap.parse_args()

    if args.refresh_probes or not PROBES.exists():
        refresh_probes()

    df = pd.read_parquet(FEATURES_PARQUET)
    cols = feature_names()
    old_cols = [c for c in cols if c not in NEW_FEATURES]
    splits = dataset_splits(df, cutoff=args.cutoff)
    train, test, future = splits.train, splits.test, splits.future
    probes = load_probes(cols)
    dd_train = train[train["source"] != "malregistry"]

    print(f"[data] {len(df)} packages; future (reported >= {args.cutoff}, no older copy): "
          f"{len(future)}; train {len(train)} ({int(train.label.sum())} malicious), "
          f"test {len(test)} ({int(test.label.sum())} malicious)")

    configs = [
        ("A0 old features, old data", old_cols, dd_train, PARAMS),
        ("A1 new features, old data", cols, dd_train, PARAMS),
        ("A2 old features, new data", old_cols, train, PARAMS),
        ("A3 new features, new data", cols, train, PARAMS),
    ]
    aug_train = with_synthetic(train, PROCESSED / "augmented.parquet")
    if len(aug_train) > len(train):
        n_syn = len(aug_train) - len(train)
        print(f"[data] + {n_syn} synthetic trojanized/control packages usable for training")
        configs.append(("A5 = A3 + synthetic", cols, aug_train, PARAMS))

    if args.tune:
        best = tune(aug_train, cols, args.tune)
        (REPORTS / "gbdt_params.json").write_text(json.dumps(best, indent=2))
        configs.append(("A6 = A5 + tuned params", cols, aug_train, best))

    if args.only:
        keep = tuple(args.only.split(","))
        configs = [c for c in configs if c[0].startswith(keep)]
    rows = []
    for name, c, tr, params in configs:
        rows.append(evaluate(name, c, tr, test, future, probes, params))
        print(f"[run] {name}: test PR-AUC {rows[-1]['test_pr_auc']:.4f}, "
              f"future recall {rows[-1]['future_recall_f05']:.3f}")

    table = pd.DataFrame(rows).set_index("config").T
    pd.set_option("display.width", 200)
    print("\n" + table.round(4).to_string())
    out = REPORTS / "experiment.md"
    out.write_text(f"# Model A ablations\n\nFuture cutoff: {args.cutoff}, "
                   f"{len(future)} future packages. Test: {len(test)} packages.\n\n"
                   + "```\n" + table.round(4).to_string() + "\n```\n")
    print(f"\n[done] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
