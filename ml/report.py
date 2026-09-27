#!/usr/bin/env python3
"""Performance report: validation, test, and packages from outside the dataset.

Four sets, none of which the classifier was fitted on:

  val        15% of the dataset. The calibrator and the similarity flag were fit
             here, so the three-tier numbers on it are optimistic by design.
  test       15% of the dataset, touched only by this script.
  future     malware first reported on or after the split's cutoff whose
             behaviour fingerprint never appears earlier (ml/split.py). Kept out
             of the 70/15/15 scheme entirely.
  live       fetched from PyPI now, and absent from the dataset by name:
               live-malware  OSSF reports newer than the dataset whose reported
                             version is still downloadable
               live-benign   random PyPI projects, plus mid-popularity ones
                             (download rank 5,001-15,000), not in the dataset
             Scored through the server's own code path (fetch -> features ->
             classifier -> similarity -> tier), so this is the product's number.

Live packages are downloaded and parsed exactly like the server does: never
installed, never run. Results are cached in data/cache/live_eval.parquet; pass
--refresh-live to fetch again.

Usage:
    python ml/report.py                   # uses the live cache if present
    python ml/report.py --refresh-live    # re-fetch the live sets (~15 min)
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from config import CACHE, FEATURES_PARQUET, MODELS, RANDOM_SEED, REPORTS  # noqa: E402
from split import FUTURE_CUTOFF, dataset_splits  # noqa: E402

LIVE_CACHE = CACHE / "live_eval.parquet"
OSSF_DIR = CACHE / "ossf-malicious-packages" / "osv" / "malicious" / "pypi"
N_LIVE_RANDOM = 160
N_LIVE_MIDPOP = 60
CONCURRENCY = 6          # parallel downloads; scoring itself is serialised


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ---- metrics -----------------------------------------------------------------

def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(y)), "n_malicious": int(y.sum()),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": float(np.nanmean([rec, tnr])),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def tier_counts(tiers: list[str], y: np.ndarray) -> dict:
    t = np.array(tiers)
    return {name: {"benign": int(((t == name) & (y == 0)).sum()),
                   "malware": int(((t == name) & (y == 1)).sum())}
            for name in ("malicious", "suspicious", "clean")}


# ---- similarity against training packages only ------------------------------

class TrainSimilarity:
    """Neighbour vote restricted to training rows, so a held-out package can
    never find itself (same rule as ml/calibrate.py)."""

    def __init__(self, train: pd.DataFrame) -> None:
        from similarity import SimilarityIndex, neighbour_vote
        self.vote = neighbour_vote
        self.idx = SimilarityIndex()
        if not self.idx.ready:
            self.pos = {}
            return
        self.pos = {k: i for i, k in enumerate(self.idx.keys)}
        rows = np.array([self.pos[f"{p}@{v}"] for p, v in zip(train["package"], train["version"])
                         if f"{p}@{v}" in self.pos])
        self.matrix = self.idx.matrix[rows]
        self.is_mal = self.idx._is_malicious[rows]

    def shares(self, df: pd.DataFrame) -> np.ndarray:
        """Malicious-weighted share per row, NaN where the package has no vector."""
        out = np.full(len(df), np.nan)
        if not self.pos:
            return out
        for i, (p, v) in enumerate(zip(df["package"], df["version"])):
            j = self.pos.get(f"{p}@{v}")
            if j is not None:
                out[i] = self.vote(self.idx.matrix[j] @ self.matrix.T, self.matrix, self.is_mal)[2]
        return out


def tiers_for(assessor, clf: np.ndarray, sim: np.ndarray) -> list[str]:
    return [assessor.assess(float(c), None if np.isnan(s) else float(s))["tier"]
            for c, s in zip(clf, sim)]


# ---- live packages -----------------------------------------------------------

def recent_ossf(dataset_names: set[str]) -> list[tuple[str, str | None, str]]:
    """(name, reported version or None, published) for OSSF PyPI reports newer
    than anything in the dataset and not in it by name."""
    out, seen = [], set()
    for f in sorted(glob.glob(str(OSSF_DIR / "*" / "*.json"))):
        d = json.loads(Path(f).read_text())
        pub = d.get("published", "")
        if pub < FUTURE_CUTOFF:
            continue
        for aff in d.get("affected", []):
            name = aff["package"]["name"]
            if norm(name) in dataset_names or norm(name) in seen:
                continue
            seen.add(norm(name))
            versions = aff.get("versions") or [None]
            # A report without versions covers the whole project.
            out.append((name, versions[-1], pub))
    return out


def benign_candidates(dataset_names: set[str], bad: set[str]) -> list[tuple[str, str]]:
    from acquire_benign import fetch_all_package_names, fetch_top_packages
    rng = random.Random(RANDOM_SEED + 7)
    top = fetch_top_packages()
    exclude = dataset_names | bad
    mid = [n for n in top[5000:15000] if norm(n) not in exclude]
    exclude |= {norm(n) for n in top[:15000]}
    every = [n for n in fetch_all_package_names() if norm(n) not in exclude]
    # Oversample: many projects publish wheels only and are skipped.
    return ([(n, "live-benign-midpop") for n in rng.sample(mid, N_LIVE_MIDPOP * 3)]
            + [(n, "live-benign-random") for n in rng.sample(every, N_LIVE_RANDOM * 3)])


async def collect_live(dataset_names: set[str]) -> pd.DataFrame:
    import threading

    from acquire_benign import fetch_top_packages
    from features import extract_features
    from fetch import PackageNotFound, fetch_package
    from similarity import SimilarityIndex

    popular = {norm(p) for p in fetch_top_packages()[:2000]}
    sim_idx = SimilarityIndex()
    lock = threading.Lock()
    ossf_names = {norm(p.name) for p in OSSF_DIR.iterdir()}

    jobs: list[tuple[str, str | None, str, int]] = []
    for name, ver, _ in recent_ossf(dataset_names):
        jobs.append((name, ver, "live-malware", 1))
    for name, group in benign_candidates(dataset_names, ossf_names):
        jobs.append((name, None, group, 0))
    quota = {"live-benign-random": N_LIVE_RANDOM, "live-benign-midpop": N_LIVE_MIDPOP}
    got = {k: 0 for k in quota}

    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict] = []
    skipped = {"gone": 0, "no_sdist": 0, "error": 0}

    def score(pkg, group, label):
        feats = extract_features(pkg.root, pkg.name, popular)
        share = None
        if sim_idx.ready:
            with lock:
                found = sim_idx.search(pkg.root)
            share = found.malicious_percent / 100 if found else None
        return {"package": pkg.name, "version": pkg.version, "set": group, "label": label,
                "sim_share": np.nan if share is None else share, **feats}

    async def one(name, ver, group, label):
        if group in quota and got[group] >= quota[group]:
            return
        spec = f"{name}=={ver}" if ver else name
        async with sem:
            if group in quota and got[group] >= quota[group]:
                return
            try:
                pkg = await fetch_package(spec)
            except PackageNotFound as exc:
                skipped["no_sdist" if "wheels" in str(exc) else "gone"] += 1
                return
            except Exception:
                skipped["error"] += 1
                return
            try:
                row = await asyncio.to_thread(score, pkg, group, label)
                if group in quota:
                    if got[group] >= quota[group]:
                        return
                    got[group] += 1
                rows.append(row)
                if len(rows) % 20 == 0:
                    print(f"[live] {len(rows)} scored ({got}) skipped {skipped}", flush=True)
            except Exception as exc:
                print(f"[live] {spec}: {exc}")
            finally:
                pkg.cleanup()

    await asyncio.gather(*(one(*j) for j in jobs))
    print(f"[live] done: {len(rows)} scored; skipped {skipped}")
    return pd.DataFrame(rows)


# ---- report ------------------------------------------------------------------

def fmt(v, pct=True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def metric_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = ["| set | n (malware) | accuracy | balanced acc. | precision | recall | F1 | FPR | ROC-AUC | PR-AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['n']} ({m['n_malicious']}) | {fmt(m['accuracy'])} | "
                     f"{fmt(m['balanced_accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                     f"{fmt(m['f1'])} | {fmt(m['fpr'])} | {fmt(m['roc_auc'], False)} | "
                     f"{fmt(m['pr_auc'], False)} |")
    return lines


def tier_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = ["| set | malicious (benign / malware) | suspicious | clean |", "|---|---|---|---|"]
    for name, t in rows:
        cell = lambda k: f"{t[k]['benign']} / {t[k]['malware']}"  # noqa: E731
        lines.append(f"| {name} | {cell('malicious')} | {cell('suspicious')} | {cell('clean')} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-live", action="store_true")
    args = ap.parse_args()

    from assess import Assessor

    df = pd.read_parquet(FEATURES_PARQUET)
    splits = dataset_splits(df)
    gbdt = pickle.load(open(MODELS / "gbdt.pkl", "rb"))
    cols, thr = gbdt["feature_names"], gbdt["threshold"]
    review = gbdt.get("review_threshold", thr)
    model = gbdt["model"]
    assessor = Assessor()
    sim = TrainSimilarity(splits.train)

    out: dict = {"threshold": thr, "review_threshold": review,
                 "sizes": {k: len(getattr(splits, k)) for k in ("train", "val", "test", "future")},
                 "sets": {}}

    clf_rows, clf_review_rows, tier_rows, slice_rows = [], [], [], []
    for name in ("val", "test", "future"):
        d = getattr(splits, name)
        y, p = d["label"].to_numpy(), model.predict(d[cols].to_numpy(np.float32))
        m = binary_metrics(y, p, thr)
        clf_rows.append((name, m))
        clf_review_rows.append((name, binary_metrics(y, p, review)))
        tiers = tiers_for(assessor, p, sim.shares(d))
        tier_rows.append((name, tier_counts(tiers, y)))
        out["sets"][name] = {"classifier": m, "tiers": tier_rows[-1][1]}
        if name == "test":
            for sl, mask in (("test · malware vs popular", d["pool"].isin(["malicious", "popular"])),
                             ("test · malware vs obscure", d["pool"].isin(["malicious", "obscure"])),
                             ("test · DataDog malware only", (d["label"] == 0) | (d["source"] == "datadog")),
                             ("test · pypi_malregistry only", (d["label"] == 0) | (d["source"] == "malregistry")),
                             ("test · malware with 11+ files", (d["label"] == 0) | (d["pkg_n_files"] >= 11))):
                mk = mask.to_numpy()
                slice_rows.append((sl, binary_metrics(y[mk], p[mk], thr)))

    # Model B on the embedded part of the test split, for comparison.
    embed_rows = []
    emb_path, cache_path = MODELS / "embed.pkl", CACHE / "embeddings.npz"
    if emb_path.exists() and cache_path.exists():
        art = pickle.load(open(emb_path, "rb"))
        with np.load(cache_path) as z:
            for name in ("val", "test"):
                d = getattr(splits, name)
                keys = [f"{a}@{b}" for a, b in zip(d["package"], d["version"])]
                has = np.array([k in z.files for k in keys])
                E = np.vstack([z[k] for k, h in zip(keys, has) if h])
                pb = art["head"].predict_proba(art["scaler"].transform(E))[:, 1]
                pa = model.predict(d[cols].to_numpy(np.float32))[has]
                yy = d["label"].to_numpy()[has]
                embed_rows += [(f"{name} · Model A (LightGBM), embedded subset", binary_metrics(yy, pa, thr)),
                               (f"{name} · Model B (embeddings), embedded subset",
                                binary_metrics(yy, pb, art["threshold"]))]

    # --- live, outside the dataset -----------------------------------------
    if args.refresh_live or not LIVE_CACHE.exists():
        names = {norm(n) for n in df["package"]}
        live = asyncio.run(collect_live(names))
        LIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        live.to_parquet(LIVE_CACHE, index=False)
    live = pd.read_parquet(LIVE_CACHE)
    live_rows, live_tier_rows, live_detail = [], [], []
    if len(live):
        p = model.predict(live[cols].to_numpy(np.float32))
        live = live.assign(prob=p, tier=tiers_for(assessor, p, live["sim_share"].to_numpy(float)))
        y = live["label"].to_numpy()
        live_rows.append(("live · all", binary_metrics(y, p, thr)))
        for group in ("live-malware", "live-benign-random", "live-benign-midpop"):
            g = (live["set"] == group).to_numpy()
            if g.any():
                live_rows.append((f"live · {group}", binary_metrics(y[g], p[g], thr)))
        live_tier_rows.append(("live · all", tier_counts(list(live["tier"]), y)))
        out["sets"]["live"] = {"classifier": live_rows[0][1], "tiers": live_tier_rows[0][1],
                               "by_group": {n: m for n, m in live_rows[1:]}}
        live_detail = live.sort_values("prob", ascending=False)

    # --- write --------------------------------------------------------------
    s = out["sizes"]
    n_scheme = s["train"] + s["val"] + s["test"]
    L = ["# Performance report", "",
         "Generated by `ml/report.py`. Model A (LightGBM over 69 static features) is the",
         "classifier; the tier combines it with code similarity (`backend/assess.py`).", "",
         "## Data scheme", "",
         f"| set | packages | share | role |", "|---|---|---|---|",
         f"| train | {s['train']:,} | {s['train'] / n_scheme:.0%} | fit the model; thresholds from grouped 5-fold CV inside it |",
         f"| validation | {s['val']:,} | {s['val'] / n_scheme:.0%} | fit the calibrator and similarity flag |",
         f"| test | {s['test']:,} | {s['test'] / n_scheme:.0%} | touched only here |",
         f"| future malware | {s['future']:,} | outside | reported on/after {FUTURE_CUTOFF}, family never seen earlier |",
         f"| live PyPI | {len(live):,} | outside | fetched now, not in the dataset by name |", "",
         "Splits are grouped by package family (name, merged with identical behaviour",
         "fingerprints) and stratified by class, so no family straddles two sets.", "",
         f"## Classifier at its decision threshold ({thr:.3f})", ""]
    L += metric_table(clf_rows + live_rows)
    L += ["", "Accuracy depends on the class mix (the dataset is ~60% malware; real PyPI is",
          "well under 1%), so read balanced accuracy, precision and FPR alongside it.",
          "Future and live-malware sets contain only malware, so only recall applies there.", "",
          f"## Classifier at the review threshold ({review:.3f}, 1% out-of-fold FPR)", ""]
    L += metric_table(clf_review_rows)
    L += ["", "## Test slices", ""] + metric_table(slice_rows)
    if embed_rows:
        L += ["", "## Model A vs Model B (packages both can score)", ""] + metric_table(embed_rows)
    L += ["", "## Three-tier verdict shown in the chat (benign / malware per tier)", ""]
    L += tier_table(tier_rows + live_tier_rows)
    L += ["", "The validation tiers are optimistic: the calibrator and the similarity flag were",
          "fit on that set. Similarity is only available for DataDog and PyPI packages",
          "(pypi_malregistry samples were never embedded), so most future malware is tiered",
          "by the classifier alone."]
    if len(live_detail):
        L += ["", "## Live packages, highest score first", "",
              "| package | version | set | classifier p | similarity | tier |", "|---|---|---|---|---|---|"]
        for r in live_detail.itertuples():
            simv = "n/a" if np.isnan(r.sim_share) else f"{100 * r.sim_share:.0f}%"
            L.append(f"| {r.package} | {r.version} | {r.set} | {r.prob:.3f} | {simv} | {r.tier} |")
    (REPORTS / "performance.md").write_text("\n".join(L) + "\n")
    (REPORTS / "performance.json").write_text(json.dumps(out, indent=2, default=float))

    print("\n".join(metric_table(clf_rows + live_rows)))
    print()
    print("\n".join(tier_table(tier_rows + live_tier_rows)))
    print(f"\n[done] -> {REPORTS / 'performance.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
