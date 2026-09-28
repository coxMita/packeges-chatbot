#!/usr/bin/env python3
"""Live packages from outside the dataset, kept on disk for re-featurising.

ml/report.py first scored live PyPI packages straight from a temp dir, so
after a feature change they had to be fetched again -- and live malware is
deleted from PyPI within days. This keeps them:

    data/raw/live/<set>/<name>-<version>/    unpacked sdist, never executed
    data/raw/live_index.json                 package, version, set, label, path

Sets: live-malware (OSSF reports newer than the dataset, still downloadable),
live-benign-random, live-benign-midpop (see ml/report.py).

Usage:
    python ml/live.py --fresh           # source new live packages outside the dataset
    python ml/live.py --from-cache      # re-download the packages in live_eval.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from config import CACHE, DATA, RANDOM_SEED, RAW  # noqa: E402
from split import FUTURE_CUTOFF  # noqa: E402

LIVE_DIR = RAW / "live"
OSSF_DIR = CACHE / "ossf-malicious-packages" / "osv" / "malicious" / "pypi"
N_LIVE_RANDOM = 160
N_LIVE_MIDPOP = 60
LIVE_INDEX = RAW / "live_index.json"
CONCURRENCY = 6


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:120]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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


def load_index() -> list[dict]:
    return json.loads(LIVE_INDEX.read_text()) if LIVE_INDEX.exists() else []


async def download(jobs: list[dict], quota: dict[str, int] | None = None) -> list[dict]:
    """Fetch `jobs` ({package, version, set, label}); returns the stored rows.
    `quota` caps how many are stored per set (candidates are oversampled)."""
    quota = quota or {}
    got = dict.fromkeys(quota, 0)
    from fetch import PackageNotFound, fetch_package

    have = {(r["package"], r["version"]) for r in load_index()}
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict] = []
    gone = 0

    async def one(job: dict) -> None:
        nonlocal gone
        if (job["package"], job["version"]) in have:
            return
        spec = f"{job['package']}=={job['version']}" if job.get("version") else job["package"]
        async with sem:
            if job["set"] in quota and got[job["set"]] >= quota[job["set"]]:
                return
            try:
                pkg = await fetch_package(spec)
            except PackageNotFound:
                gone += 1
                return
            except Exception as exc:
                print(f"[live] {spec}: {exc}")
                return
        if job["set"] in quota:
            if got[job["set"]] >= quota[job["set"]]:
                pkg.cleanup()
                return
            got[job["set"]] += 1
        dest = LIVE_DIR / job["set"] / _safe(f"{pkg.name}-{pkg.version}")
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pkg.root), dest)
        shutil.rmtree(pkg.root.parent, ignore_errors=True)
        rows.append({"package": pkg.name, "version": pkg.version, "set": job["set"],
                     "label": job["label"], "path": str(dest.relative_to(DATA))})

    await asyncio.gather(*(one(j) for j in jobs))
    index = load_index() + rows
    LIVE_INDEX.write_text(json.dumps(index, indent=1))
    print(f"[live] stored {len(rows)} new, {gone} no longer on PyPI; index holds {len(index)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-cache", action="store_true",
                    help="re-download the packages listed in data/cache/live_eval.parquet")
    ap.add_argument("--fresh", action="store_true",
                    help="source new packages: recent OSSF malware still on PyPI, plus random "
                         "and mid-popularity benign ones, none of them in the dataset")
    args = ap.parse_args()
    if args.fresh:
        from config import FEATURES_PARQUET
        names = {norm(n) for n in pd.read_parquet(FEATURES_PARQUET)["package"]}
        names |= {norm(r["package"]) for r in load_index()}
        ossf = {norm(p.name) for p in OSSF_DIR.iterdir()}
        jobs = [{"package": n, "version": v, "set": "live-malware", "label": 1}
                for n, v, _ in recent_ossf(names)]
        cands = benign_candidates(names, ossf)
        jobs += [{"package": n, "version": "", "set": g, "label": 0} for n, g in cands]
        asyncio.run(download(jobs, {"live-benign-random": N_LIVE_RANDOM,
                                    "live-benign-midpop": N_LIVE_MIDPOP}))
    if args.from_cache:
        old = pd.read_parquet(CACHE / "live_eval.parquet")
        jobs = [{"package": r.package, "version": r.version, "set": r.set, "label": int(r.label)}
                for r in old.itertuples()]
        asyncio.run(download(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
