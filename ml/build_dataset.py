#!/usr/bin/env python3
"""Turn the downloaded packages into a feature matrix.

Reads the two acquisition indexes, runs `features.extract_features` over every
package directory in parallel, and writes a single parquet file.

Two columns exist purely to keep the evaluation honest:

* `pool` -- 'malicious', 'popular' or 'obscure'. `evaluate.py` reports metrics
  on the obscure (hard-negative) slice separately, because overall accuracy on
  a popular-vs-malware split is a flattering and largely meaningless number.

* `group` -- the package name. The train/test split is grouped on it so that two
  versions of the same compromised library can never straddle the split. Without
  this, a model can memorise a package rather than learn a behaviour and the
  held-out score is inflated.

Usage:
    python ml/build_dataset.py
    python ml/build_dataset.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from acquire_benign import fetch_top_packages, normalise
from config import BENIGN_DIR, FEATURES_PARQUET, MALICIOUS_DIR, RAW, ensure_dirs
from features import extract_features, feature_names

# The typosquat feature compares against this many top names. More is slower
# (it is an edit-distance sweep per package) with rapidly diminishing returns.
N_POPULAR_FOR_TYPOSQUAT = 2000


def _one(args: tuple[dict, str, set[str]]) -> dict | None:
    """Worker: extract features for a single package directory."""
    row, root_str, popular = args
    root = Path(root_str)
    if not root.is_dir():
        return None
    try:
        feats = extract_features(root, row["package"], popular)
    except (OSError, RecursionError, MemoryError) as exc:
        print(f"[skip] {row['package']}: {type(exc).__name__}: {exc}")
        return None

    return {
        "package": row["package"],
        "version": row.get("version", ""),
        "label": row["label"],
        "pool": row.get("pool") or "malicious",
        "intent_class": row.get("intent_class", ""),
        "group": normalise(row["package"]),
        **feats,
    }


def load_index(path: Path, base: Path) -> list[tuple[dict, str]]:
    if not path.exists():
        print(f"[warn] {path} not found -- run the matching acquire script first")
        return []
    rows = json.loads(path.read_text())
    out = []
    for r in rows:
        # Index rows store a repo-relative path; resolve it against the root.
        d = base.parent.parent / r["path"] if "path" in r else base / r["package"]
        out.append((r, str(d)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=FEATURES_PARQUET)
    args = ap.parse_args()

    ensure_dirs()

    popular = {normalise(p) for p in fetch_top_packages()[:N_POPULAR_FOR_TYPOSQUAT]}

    jobs = (load_index(RAW / "malicious_index.json", MALICIOUS_DIR)
            + load_index(RAW / "benign_index.json", BENIGN_DIR))
    if not jobs:
        print("[error] nothing to build -- no packages have been acquired yet")
        return 1

    print(f"[build] extracting features from {len(jobs)} packages "
          f"on {args.workers} workers")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_one, (r, p, popular)) for r, p in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row:
                rows.append(row)
            if i % 500 == 0:
                print(f"[progress] {i}/{len(jobs)}")

    df = pd.DataFrame(rows)

    # Guarantee a stable column order and no silent NaNs -- LightGBM tolerates
    # NaN, but a NaN here means a feature failed to compute, which we want loud.
    cols = feature_names()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"features missing from the matrix: {missing}")
    df[cols] = df[cols].fillna(0.0).astype("float32")

    meta = ["package", "version", "label", "pool", "intent_class", "group"]
    df = df[meta + cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n[done] {len(df)} packages x {len(cols)} features -> {args.out}")
    print("\nclass balance:")
    print(df.groupby(["label", "pool"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
