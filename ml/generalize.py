#!/usr/bin/env python3
"""Generalisation experiments: what makes the detector work on *unseen* data?

"Unseen" means two things here, and both are kept out of every decision:

  DEV    (chooses the configuration and the threshold)
           validation split (15%)  +  future malware reported 2025-06 .. 2025-12
  FINAL  (reported once, after the choice is made)
           test-split benign       +  future malware reported 2026-01 onwards
           + live PyPI packages absent from the dataset (ml/live.py)

Future malware comes from families whose behaviour fingerprint never appears
before the cutoff (ml/split.py), so it is the closest thing to "the next
campaign". Splitting it by time -- earlier half to choose, later half to
report -- stops the final number from being tuned on itself.

Configurations (each adds one idea, so the table reads as an ablation):

  C0  v1 features (the shipped model's)                 baseline
  C1  v2 features: + autorun reachability, data-flow, process tradecraft,
      binary/.pth/script payloads, worst-file pooling    (ml/behaviour.py)
  C2  C1 + monotone constraints: a danger signal can only raise the score,
      so a big benign-looking package cannot cancel "downloads and runs a
      binary on import"
  C3  C2 without package-shape features (file counts, LOC, README...), which
      describe the training families rather than behaviour
  C4  + the synthetic trojanised packages (ml/augment.py, incl. __init__.py
      injection): on the v2 features this LOWERED dev accuracy 95.3% -> 94.6%
      and final recall -- the model learned the injection shape. Not used.
  C5  + recency weighting (newer malware up to 2x): no measurable effect. Not used.

Decision rules (applied to the monthly-retraining scores, see rolling_scores):
  a capability *soft gate* -- the normal threshold when the code shows a
  concrete capability (behaviour.CAPABILITY), the strict 0.1%-FPR threshold
  when it shows none -- raised precision on unseen families from ~85% to ~93%
  at a few points of recall. Chosen on 2025 months, reported on 2026 months.

Usage:
    python ml/generalize.py                   # -> ml/reports/generalize.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA, PROCESSED, RANDOM_SEED, REPORTS  # noqa: E402
from behaviour import CAPABILITY, MONOTONE_PREFIXES  # noqa: E402
from split import dataset_splits, with_synthetic  # noqa: E402
from train_gbdt import PARAMS, threshold_at_fpr, tune_threshold  # noqa: E402

V1 = PROCESSED / "features_v1.parquet"
V2 = PROCESSED / "features.parquet"
AUG_V1 = PROCESSED / "augmented_v1.parquet"
AUG_V2 = PROCESSED / "augmented.parquet"
LIVE_V1 = DATA / "cache" / "live_eval.parquet"
LIVE_V2 = PROCESSED / "live_features.parquet"
DEV_END = "2026-01-01"          # future malware before this chooses; after it reports

SHAPE = {"pkg_n_files", "pkg_n_py_files", "pkg_loc", "pkg_loc_per_file", "pkg_py_file_ratio",
         "pkg_has_readme", "pkg_has_license", "pkg_has_tests", "pkg_parse_failures"}
N_ROUNDS, EARLY = 800, 50


def columns(df: pd.DataFrame) -> list[str]:
    meta = {"package", "version", "label", "pool", "intent_class", "source", "reported", "group",
            "host_group", "donor_group", "set", "path", "sim_share"}
    return sorted(c for c in df.columns if c not in meta)


def live_features() -> pd.DataFrame:
    """v2 features for the stored live packages (cached)."""
    from features import extract_features
    from live import load_index
    from acquire_benign import fetch_top_packages, normalise
    idx = load_index()
    if LIVE_V2.exists():
        cached = pd.read_parquet(LIVE_V2)
        if len(cached) == len(idx):
            return cached
    popular = {normalise(p) for p in fetch_top_packages()[:2000]}
    rows = [{"package": r["package"], "version": r["version"], "set": r["set"], "label": r["label"],
             **extract_features(DATA / r["path"], r["package"], popular)} for r in idx]
    out = pd.DataFrame(rows)
    out.to_parquet(LIVE_V2, index=False)
    return out


def recency_weight(df: pd.DataFrame) -> np.ndarray:
    """1 for benign and undated rows; malware weighted up to 2x by report date."""
    w = np.ones(len(df))
    dated = (df["label"] == 1) & (df["reported"].fillna("") != "")
    days = (pd.to_datetime(df.loc[dated, "reported"], errors="coerce")
            - pd.Timestamp("2022-01-01")).dt.days.clip(lower=0).fillna(0)
    w[dated.to_numpy()] = 1 + days.to_numpy() / max(days.max(), 1)
    return w


def fit(train: pd.DataFrame, cols: list[str], monotone: bool, weights: np.ndarray | None):
    """Grouped 5-fold CV for thresholds and rounds, then one final fit."""
    X = train[cols].to_numpy(np.float32)
    y = train["label"].to_numpy()
    g = train["group"].to_numpy()
    w = np.ones(len(y)) if weights is None else weights
    params = {**PARAMS}
    if monotone:
        params["monotone_constraints"] = [1 if c.startswith(MONOTONE_PREFIXES) else 0 for c in cols]
        params["monotone_constraints_method"] = "advanced"
    oof, rounds = np.zeros(len(y)), []
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_SEED).split(X, y, g):
        spw = (1 - y[tr]).sum() / max(y[tr].sum(), 1)
        b = lgb.train({**params, "scale_pos_weight": spw},
                      lgb.Dataset(X[tr], y[tr], weight=w[tr]), N_ROUNDS,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        oof[va] = b.predict(X[va], num_iteration=b.best_iteration)
        rounds.append(b.best_iteration or N_ROUNDS)
    real = (train["pool"] != "synthetic").to_numpy()
    thr = {"f05": tune_threshold(y[real], oof[real]),
           "fpr1": threshold_at_fpr(y[real], oof[real], 0.01),
           "fpr05": threshold_at_fpr(y[real], oof[real], 0.005),
           "strict": threshold_at_fpr(y[real], oof[real], 0.001)}
    spw = (1 - y).sum() / max(y.sum(), 1)
    model = lgb.train({**params, "scale_pos_weight": spw}, lgb.Dataset(X, y, weight=w),
                      max(int(np.mean(rounds)), 50))
    return model, thr, float(average_precision_score(y[real], oof[real]))


def metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    tn, fn = int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"acc": (tp + tn) / max(len(y), 1), "prec": prec, "rec": rec,
            "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else float("nan"), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def sets(df: pd.DataFrame, live: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = dataset_splits(df)
    fut = s.future
    dev_fut, fin_fut = fut[fut["reported"] < DEV_END], fut[fut["reported"] >= DEV_END]
    return {
        "train": s.train,
        "dev": pd.concat([s.val, dev_fut], ignore_index=True),
        "final": pd.concat([s.test[s.test["label"] == 0], fin_fut,
                            live.assign(group="live", pool="live", source="live", reported="")],
                           ignore_index=True),
        "final_future": fin_fut,
        "final_live_mal": live[live["label"] == 1],
        "final_live_ben": live[live["label"] == 0],
        "dev_future": dev_fut,
    }


def run(name, df, aug, live, drop=(), monotone=False, recency=False) -> dict:
    S = sets(df, live)
    cols = [c for c in columns(df) if c not in drop and c in live.columns]
    train = with_synthetic(S["train"], aug) if aug else S["train"]
    model, thr, oof_ap = fit(train, cols, monotone, recency_weight(train) if recency else None)
    row = {"config": name, "n_features": len(cols), "oof_pr_auc": oof_ap}
    pred = {k: model.predict(v[cols].to_numpy(np.float32)) for k, v in S.items() if k != "train"}
    for tname, t in {k: thr[k] for k in ("f05", "fpr1")}.items():
        for k in ("dev", "final"):
            m = metrics(S[k]["label"].to_numpy(), pred[k], t)
            for mk in ("acc", "prec", "rec", "fpr"):
                row[f"{k}_{mk}@{tname}"] = m[mk]
        for k in ("dev_future", "final_future", "final_live_mal"):
            row[f"{k}_rec@{tname}"] = float((pred[k] >= t).mean()) if len(pred[k]) else float("nan")
        row[f"final_live_ben_fpr@{tname}"] = float((pred["final_live_ben"] >= t).mean())
    row["thr_f05"], row["thr_fpr1"] = thr["f05"], thr["fpr1"]
    print(f"[{name}] dev acc {row['dev_acc@f05']:.3f} prec {row['dev_prec@f05']:.3f} "
          f"future-dev rec {row['dev_future_rec@f05']:.3f}", flush=True)
    return row


def main() -> int:
    v1, v4 = pd.read_parquet(V1), pd.read_parquet(V2)
    live1, live4 = pd.read_parquet(LIVE_V1), live_features()
    rows = [run("C0 v1 features (shipped before)", v1, AUG_V1, live1),
            run("C1 v4 features", v4, None, live4),
            run("C2 v4 + monotone (chosen)", v4, None, live4, monotone=True),
            run("C3 v4 + monotone - shape", v4, None, live4, drop=SHAPE, monotone=True)]
    table = pd.DataFrame(rows).set_index("config").T
    (REPORTS / "generalize.json").write_text(json.dumps(rows, indent=2, default=float))

    scores = rolling_scores(v4)
    scores.to_parquet(PROCESSED / "rolling_scores.parquet", index=False)
    dev, fin = scores[scores["month"] < DEV_END[:7]], scores[scores["month"] >= DEV_END[:7]]
    pd.set_option("display.width", 250)
    txt = ("# Generalisation experiments\n\n"
           f"DEV = validation + future malware reported before {DEV_END}; FINAL = test benign + "
           f"future malware from {DEV_END} + live PyPI. Thresholds come from grouped CV on the "
           "training split only.\n\n```\n" + table.round(4).to_string() + "\n```\n\n"
           "## Monthly retraining, decision rules (new-family malware vs ~1,100 benign)\n\n"
           "Choose (2025-07..12):\n\n```\n" + policy_table(dev).round(4).to_string() + "\n```\n\n"
           "Report (2026-01..08):\n\n```\n" + policy_table(fin).round(4).to_string() + "\n```\n")
    (REPORTS / "generalize.md").write_text(txt)
    print(txt)
    return 0




# ---- time-aware evaluation: monthly retraining -------------------------------

def rolling(df: pd.DataFrame, live: pd.DataFrame, start: str = "2025-07", end: str = "2026-08",
            monotone: bool = True, drop=()) -> tuple[pd.DataFrame, dict]:
    """Retrain at the start of every month on everything reported before it;
    score that month's malware from never-seen families against the fixed
    test-split benign packages (about 10x as many, so precision is tested hard).

    This is the deployment the numbers should describe: a detector retrained
    monthly meeting next month's new packages. Returns per-month rows and the
    pooled confusion matrix.
    """
    s = dataset_splits(df)
    test_benign = s.test[s.test["label"] == 0]
    benign_pool = pd.concat([s.train, s.val], ignore_index=True)
    benign_pool = benign_pool[benign_pool["label"] == 0]
    mal = df[df["label"] == 1].copy()
    mal["month"] = mal["reported"].fillna("").str[:7]
    cols = [c for c in columns(df) if c not in drop]
    rows, pooled = [], dict(tp=0, fp=0, tn=0, fn=0)
    gated = dict(tp=0, fp=0, tn=0, fn=0)
    for month in pd.period_range(start, end, freq="M").astype(str):
        before = mal[(mal["month"] < month)]          # undated ("") sorts first: treated as old
        seen = set(before["group"])
        test_mal = mal[(mal["month"] == month) & ~mal["group"].isin(seen)]
        if test_mal.empty:
            continue
        train = pd.concat([benign_pool, before[~before["group"].isin(set(test_mal["group"]))]],
                          ignore_index=True)
        model, thr, _ = fit(train, cols, monotone, None)
        t = thr["fpr1"]
        test = pd.concat([test_benign, test_mal], ignore_index=True)
        prob = model.predict(test[cols].to_numpy(np.float32))
        m = metrics(test["label"].to_numpy(), prob, t)
        mg = metrics(test["label"].to_numpy(), np.where(has_capability(test), prob, 0.0), t)
        for k in pooled:
            pooled[k] += m[k]
            gated[k] += mg[k]
        rows.append({"month": month, "n_new_family_malware": len(test_mal), "n_benign": len(test_benign),
                     "recall": m["rec"], "precision": m["prec"], "fpr": m["fpr"], "accuracy": m["acc"],
                     "threshold": t, "gated_recall": mg["rec"], "gated_precision": mg["prec"],
                     "gated_fpr": mg["fpr"]})
        print(f"[rolling] {month}: {len(test_mal)} new-family malware, recall {m['rec']:.3f}, "
              f"precision {m['prec']:.3f}, FPR {m['fpr']:.4f}", flush=True)
    return pd.DataFrame(rows), {"model only": _pooled(pooled), "with capability gate": _pooled(gated)}


def _pooled(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "fpr": fp / max(fp + tn, 1), "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
            # The same rates on a balanced 50/50 mix of malware and benign.
            "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))}


# ---- capability gate -----------------------------------------------------------

def has_capability(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in CAPABILITY if c in df.columns]
    return (df[cols].to_numpy(np.float32) > 0).any(axis=1)


# ---- decision policies, evaluated offline on recorded monthly scores ---------------

def rolling_scores(df: pd.DataFrame, start: str = "2025-07", end: str = "2026-08",
                   monotone: bool = True) -> pd.DataFrame:
    """Like rolling(), but return every scored row -- month, label, probability,
    capability flag and that month's thresholds -- so decision rules can be
    compared without retraining."""
    s = dataset_splits(df)
    test_benign = s.test[s.test["label"] == 0]
    benign_pool = pd.concat([s.train, s.val], ignore_index=True)
    benign_pool = benign_pool[benign_pool["label"] == 0]
    mal = df[df["label"] == 1].copy()
    mal["month"] = mal["reported"].fillna("").str[:7]
    cols = columns(df)
    out = []
    for month in pd.period_range(start, end, freq="M").astype(str):
        before = mal[mal["month"] < month]
        test_mal = mal[(mal["month"] == month) & ~mal["group"].isin(set(before["group"]))]
        if test_mal.empty:
            continue
        train = pd.concat([benign_pool, before[~before["group"].isin(set(test_mal["group"]))]],
                          ignore_index=True)
        model, thr, _ = fit(train, cols, monotone, None)
        test = pd.concat([test_benign, test_mal], ignore_index=True)
        rec = pd.DataFrame({"month": month, "package": test["package"].to_numpy(),
                            "label": test["label"].to_numpy(),
                            "prob": model.predict(test[cols].to_numpy(np.float32)),
                            "cap": has_capability(test)})
        for k, v in thr.items():
            rec[f"t_{k}"] = v
        out.append(rec)
        print(f"[scores] {month}: {len(test_mal)} new-family malware", flush=True)
    return pd.concat(out, ignore_index=True)


POLICIES = {
    "model @F0.5":                lambda r: r.prob >= r.t_f05,
    "model @1% FPR":              lambda r: r.prob >= r.t_fpr1,
    "model @0.5% FPR":            lambda r: r.prob >= r.t_fpr05,
    "hard gate @1% FPR":          lambda r: (r.prob >= r.t_fpr1) & r.cap,
    "soft gate @1% / strict":     lambda r: ((r.prob >= r.t_fpr1) & r.cap) | (r.prob >= r.t_strict),
    "soft gate @F0.5 / strict":   lambda r: ((r.prob >= r.t_f05) & r.cap) | (r.prob >= r.t_strict),
    "soft gate @0.5% / strict":   lambda r: ((r.prob >= r.t_fpr05) & r.cap) | (r.prob >= r.t_strict),
}


def policy_table(scores: pd.DataFrame, balanced: bool = False) -> pd.DataFrame:
    """Pooled metrics per policy. `balanced` re-weights benign rows so each
    month is a 50/50 mix -- the same rates, a different base rate."""
    scores = scores.reset_index(drop=True)
    rows = []
    for name, rule in POLICIES.items():
        pred = rule(scores).to_numpy()
        y = scores["label"].to_numpy()
        w = np.ones(len(y))
        if balanced:
            for m, g in scores.groupby("month").groups.items():
                g = np.asarray(g)
                n_mal, n_ben = (y[g] == 1).sum(), (y[g] == 0).sum()
                w[g[y[g] == 0]] = n_mal / max(n_ben, 1)
        tp = w[(pred) & (y == 1)].sum(); fp = w[(pred) & (y == 0)].sum()
        tn = w[(~pred) & (y == 0)].sum(); fn = w[(~pred) & (y == 1)].sum()
        rows.append({"policy": name, "accuracy": (tp + tn) / (tp + fp + tn + fn),
                     "precision": tp / max(tp + fp, 1e-9), "recall": tp / max(tp + fn, 1e-9),
                     "fpr": fp / max(fp + tn, 1e-9)})
    return pd.DataFrame(rows).set_index("policy")


if __name__ == "__main__":
    raise SystemExit(main())
