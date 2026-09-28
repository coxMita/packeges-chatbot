#!/usr/bin/env python3
"""Performance report: in-distribution, unseen families, and live PyPI.

Three questions, three kinds of evidence:

  1. In distribution -- validation (15%) and test (15%) of the 70/15/15
     grouped split, scored by the production model.
  2. Unseen malware families, as deployed -- ml/generalize.py retrains every
     month on everything reported before it and scores that month's malware
     from families never seen before, against ~1,100 benign test packages. The
     decision rule was chosen on 2025-07..12 and is reported on 2026-01..08.
  3. Outside the dataset -- live PyPI packages that appear nowhere in it
     (ml/live.py): recent OSSF malware still downloadable, and random plus
     mid-popularity benign projects. Scored through the server's own code
     path: features -> classifier -> similarity -> tier, with the capability
     gate.

The production model trains on the training split plus the future-holdout
malware (the newest data there is), so it has no "future" set of its own;
question 2 is how its unseen-data performance is measured.

Usage:
    python ml/report.py              # uses data/processed/rolling_scores*.parquet
    python ml/report.py --rolling    # recompute the monthly retraining first (~20 min)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from behaviour import capabilities  # noqa: E402
from config import DATA, FEATURES_PARQUET, MODELS, PROCESSED, REPORTS  # noqa: E402
from split import dataset_splits  # noqa: E402

ROLLING = PROCESSED / "rolling_scores.parquet"
ROLLING_V1 = PROCESSED / "rolling_scores_v1.parquet"
# Chosen on the 2025 months as the most accurate rule with precision >= 95%
# (a margin over the 90% target, for drift); reported on the 2026 months.
CHOSEN_POLICY = "soft gate @F0.5 / strict"
DEV_END = "2026-01"


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    return confusion(y, p >= thr, p)


def confusion(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    pred = np.asarray(pred, dtype=bool)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    both = score is not None and len(np.unique(y)) == 2
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {"n": int(len(y)), "n_malicious": int((y == 1).sum()),
            "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": float(np.nanmean([rec, tnr])),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "roc_auc": float(roc_auc_score(y, score)) if both else float("nan"),
            "pr_auc": float(average_precision_score(y, score)) if both else float("nan"),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def fmt(v, pct=True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def metric_table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| set | n (malware) | accuracy | balanced acc. | precision | recall | FPR | PR-AUC |",
           "|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        out.append(f"| {name} | {m['n']:,} ({m['n_malicious']:,}) | {fmt(m['accuracy'])} | "
                   f"{fmt(m['balanced_accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                   f"{fmt(m['fpr'])} | {fmt(m['pr_auc'], False)} |")
    return out


def tier_counts(tiers, y) -> dict:
    t = np.asarray(tiers)
    return {k: {"benign": int(((t == k) & (y == 0)).sum()), "malware": int(((t == k) & (y == 1)).sum())}
            for k in ("malicious", "suspicious", "clean")}


def tier_table(rows) -> list[str]:
    out = ["| set | malicious (benign / malware) | suspicious | clean |", "|---|---|---|---|"]
    for name, t in rows:
        cell = lambda k: f"{t[k]['benign']:,} / {t[k]['malware']:,}"  # noqa: E731
        out.append(f"| {name} | {cell('malicious')} | {cell('suspicious')} | {cell('clean')} |")
    return out


def assess_rows(assessor, probs, sims, caps) -> list[str]:
    return [assessor.assess(float(p), None if np.isnan(s) else float(s), bool(c))["tier"]
            for p, s, c in zip(probs, sims, caps)]


def train_similarity(train: pd.DataFrame):
    """Similarity vote against training rows only (as ml/calibrate.py)."""
    from similarity import SimilarityIndex, neighbour_vote
    idx = SimilarityIndex()
    if not idx.ready:
        return idx, lambda d: np.full(len(d), np.nan)
    pos = {k: i for i, k in enumerate(idx.keys)}
    rows = np.array([pos[f"{p}@{v}"] for p, v in zip(train["package"], train["version"])
                     if f"{p}@{v}" in pos])
    M, mal = idx.matrix[rows], idx._is_malicious[rows]

    def shares(d: pd.DataFrame) -> np.ndarray:
        out = np.full(len(d), np.nan)
        for i, (p, v) in enumerate(zip(d["package"], d["version"])):
            j = pos.get(f"{p}@{v}")
            if j is not None:
                out[i] = neighbour_vote(idx.matrix[j] @ M.T, M, mal)[2]
        return out
    return idx, shares


def live_rows(idx) -> pd.DataFrame:
    """Live packages with features and a similarity share, as the server sees them."""
    import generalize
    from live import load_index
    feats = generalize.live_features()
    paths = {(r["package"], r["version"]): DATA / r["path"] for r in load_index()}
    sims = []
    for r in feats.itertuples():
        found = idx.search(paths[(r.package, r.version)]) if idx.ready else None
        sims.append(found.malicious_percent / 100 if found else np.nan)
    return feats.assign(sim_share=sims)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rolling", action="store_true", help="recompute monthly retraining scores")
    args = ap.parse_args()

    import generalize
    from assess import Assessor

    df = pd.read_parquet(FEATURES_PARQUET)
    s = dataset_splits(df)
    gbdt = pickle.load(open(MODELS / "gbdt.pkl", "rb"))
    cols, thr, model = gbdt["feature_names"], gbdt["threshold"], gbdt["model"]
    assessor = Assessor()
    idx, shares = train_similarity(s.train)

    def cap_of(d: pd.DataFrame) -> np.ndarray:
        return np.array([bool(capabilities(r)) for r in d[cols].to_dict("records")])

    out: dict = {"threshold": thr, "policy": CHOSEN_POLICY, "sets": {}}

    # 1. in distribution
    clf_rows, tiers_rows = [], []
    for name, d in (("validation", s.val), ("test", s.test)):
        y, p = d["label"].to_numpy(), model.predict(d[cols].to_numpy(np.float32))
        clf_rows.append((name, binary_metrics(y, p, thr)))
        tiers_rows.append((name, tier_counts(assess_rows(assessor, p, shares(d), cap_of(d)), y)))
        out["sets"][name] = {"classifier": clf_rows[-1][1], "tiers": tiers_rows[-1][1]}

    # 2. unseen families, monthly retraining
    if args.rolling or not ROLLING.exists():
        generalize.rolling_scores(df).to_parquet(ROLLING, index=False)
    if args.rolling or not ROLLING_V1.exists():
        v1 = pd.read_parquet(generalize.V1)
        generalize.rolling_scores(v1, monotone=False).to_parquet(ROLLING_V1, index=False)
    roll, roll1 = pd.read_parquet(ROLLING), pd.read_parquet(ROLLING_V1)
    rule = generalize.POLICIES[CHOSEN_POLICY]
    unseen = []
    for label, sc, rname in (("v1 features, model only (before)", roll1, "model @F0.5"),
                             ("v4 features + monotone, model only", roll, "model @1% FPR"),
                             ("v4 + capability soft gate (chosen)", roll, CHOSEN_POLICY)):
        r = generalize.POLICIES[rname]
        for period, part in (("choose: 2025-07..12", roll_part(sc, dev=True)),
                             ("report: 2026-01..08", roll_part(sc, dev=False))):
            m = confusion(part["label"].to_numpy(), r(part).to_numpy(), part["prob"].to_numpy())
            unseen.append((f"{label} · {period}", m))
    fin_bal = generalize.policy_table(roll_part(roll, dev=False), balanced=True).loc[CHOSEN_POLICY]
    per_month = [(month, confusion(part["label"].to_numpy(), rule(part).to_numpy()))
                 for month, part in roll.groupby("month")]
    out["unseen"] = {n: m for n, m in unseen}
    out["unseen_final_balanced"] = fin_bal.to_dict()

    # 3. live PyPI, outside the dataset
    live = live_rows(idx)
    y = live["label"].to_numpy()
    p = model.predict(live[cols].to_numpy(np.float32))
    caps = cap_of(live)
    tiers = np.asarray(assess_rows(assessor, p, live["sim_share"].to_numpy(float), caps))
    live_metric_rows = [("live · classifier @ threshold", binary_metrics(y, p, thr)),
                        ("live · tier is malicious", confusion(y, tiers == "malicious", p)),
                        ("live · tier is malicious or suspicious",
                         confusion(y, np.isin(tiers, ["malicious", "suspicious"]), p))]
    by_set = []
    for g in ("live-malware", "live-benign-random", "live-benign-midpop"):
        k = (live["set"] == g).to_numpy()
        by_set.append((g, tier_counts(tiers[k], y[k])))
    out["sets"]["live"] = {n: m for n, m in live_metric_rows}

    # --- write -------------------------------------------------------------------
    n_scheme = len(s.train) + len(s.val) + len(s.test)
    L = ["# Performance report", "",
         "Generated by `ml/report.py`. Classifier: LightGBM over the v1 static features plus the",
         "behaviour features of `ml/behaviour.py`, monotone in every danger signal. Verdict: the",
         "three tiers of `backend/assess.py`, where *malicious* also requires a concrete",
         "capability found in the code.", "",
         "## Data", "",
         "| set | packages | role |", "|---|---|---|",
         f"| train | {len(s.train):,} ({len(s.train) / n_scheme:.0%}) | fit, together with the "
         f"{len(s.future):,} future-holdout malware |",
         f"| validation | {len(s.val):,} ({len(s.val) / n_scheme:.0%}) | calibrator, similarity flag |",
         f"| test | {len(s.test):,} ({len(s.test) / n_scheme:.0%}) | touched only here |",
         f"| live PyPI | {len(live):,} | outside the dataset entirely |", "",
         "## 1. In distribution", ""] + metric_table(clf_rows) + ["", "Tiers:", ""] + tier_table(tiers_rows)
    L += ["", "## 2. Unseen malware families, retrained monthly", "",
          "Each month's model is trained on everything reported before that month and scored on",
          "that month's malware **from families never seen before**, mixed with ~1,100 benign",
          "test packages (about 12 benign per malware, so precision is tested hard). The decision",
          "rule was chosen on the 2025 months and is reported on the 2026 months.", ""]
    L += metric_table(unseen)
    L += ["", f"The same 2026 rates on a balanced 50/50 mix: accuracy {fmt(fin_bal['accuracy'])}, "
          f"precision {fmt(fin_bal['precision'])}, recall {fmt(fin_bal['recall'])}.", "",
          "Per month, chosen rule:", "", "| month | new-family malware | recall | precision | FPR |",
          "|---|---|---|---|---|"]
    L += [f"| {mo} | {m['n_malicious']} | {fmt(m['recall'])} | {fmt(m['precision'])} | {fmt(m['fpr'])} |"
          for mo, m in per_month]
    L += ["", "## 3. Live PyPI packages outside the dataset", "",
          "Scored exactly as the server does, by the production model. Live malware is what is",
          "still on PyPI after being reported -- by selection, malware that evaded removal.", ""]
    L += metric_table(live_metric_rows) + ["", "Tiers per set:", ""] + tier_table(by_set)
    L += ["", "| package | version | set | classifier p | similarity | capability | tier |",
          "|---|---|---|---|---|---|---|"]
    for i in np.argsort(-p):
        r = live.iloc[i]
        sim = "—" if np.isnan(r.sim_share) else f"{100 * r.sim_share:.0f}%"
        L.append(f"| {r.package} | {r.version} | {r.set} | {p[i]:.3f} | {sim} | "
                 f"{'yes' if caps[i] else 'no'} | {tiers[i]} |")
    (REPORTS / "performance.md").write_text("\n".join(L) + "\n")
    (REPORTS / "performance.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(L[:75]))
    return 0


def roll_part(scores: pd.DataFrame, dev: bool) -> pd.DataFrame:
    return scores[scores["month"] < DEV_END] if dev else scores[scores["month"] >= DEV_END]


if __name__ == "__main__":
    raise SystemExit(main())
