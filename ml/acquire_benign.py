#!/usr/bin/env python3
"""Download the benign PyPI class -- in two deliberately different flavours.

The single most important methodological decision in this project lives here.

Popular PyPI packages are large, mature, well-tooled projects. Malicious packages
are tiny throwaways. If the benign class is drawn only from the top of the
download charts, the model does not learn "malicious behaviour" -- it learns
"small package". It then scores ~99% on the held-out split and is useless against
any real upload, because every new legitimate package by a first-time author
looks exactly like the malicious class.

So we sample two pools:

  1. POPULAR  (`--n-top`, default 5000) -- the realistic negative class, taken
     from the top-PyPI download rankings.

  2. OBSCURE  (`--n-random`, default 3000) -- the HARD NEGATIVES. Sampled
     uniformly at random from the full PyPI simple index, excluding anything in
     the top 15000. These are small, amateur, sparsely documented, often
     single-file packages that are nonetheless perfectly benign. They are what
     forces the model to learn behaviour instead of size.

`evaluate.py` reports metrics on the obscure slice separately. That number, not
the headline F1, is the honest measure of whether this model works.

Every candidate is cross-checked against the OSSF malicious-packages advisory
feed, so a known-bad package cannot silently land in the benign class.

Only source distributions (sdists) are downloaded -- wheels are built artifacts
and often omit `setup.py`, which is exactly where install-time attacks live.
Nothing in this script executes package code.

Usage:
    python ml/acquire_benign.py
    python ml/acquire_benign.py --n-top 200 --n-random 100   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BENIGN_DIR,
    CACHE,
    DOWNLOAD_TIMEOUT,
    HTTP_USER_AGENT,
    MAX_ARCHIVE_BYTES,
    N_RANDOM_PACKAGES,
    N_TOP_PACKAGES,
    OSSF_MALICIOUS_REPO,
    PYPI_JSON_API,
    PYPI_SIMPLE_INDEX,
    RANDOM_SEED,
    TOP_PYPI_URL,
    TOP_RANK_EXCLUSION,
    ensure_dirs,
)
from safe_extract import UnsafeArchive, extract_any

SDIST_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar.bz2")

_session_local = threading.local()


def session() -> requests.Session:
    """One requests.Session per worker thread (Session is not thread-safe)."""
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = HTTP_USER_AGENT
        _session_local.s = s
    return s


# --- candidate selection ----------------------------------------------------


def fetch_top_packages() -> list[str]:
    """Ranked list of the most-downloaded PyPI projects (cached on disk)."""
    cache = CACHE / "top-pypi-packages.json"
    if not cache.exists():
        print("[top] fetching download rankings")
        r = requests.get(TOP_PYPI_URL, timeout=DOWNLOAD_TIMEOUT,
                         headers={"User-Agent": HTTP_USER_AGENT})
        r.raise_for_status()
        cache.write_text(r.text)
    rows = json.loads(cache.read_text())["rows"]
    return [row["project"] for row in rows]


def fetch_all_package_names() -> list[str]:
    """Every project name on PyPI, from the simple index (cached; ~25MB)."""
    cache = CACHE / "pypi-simple-index.txt"
    if not cache.exists():
        print("[index] fetching the full PyPI simple index (~25MB, one-off)")
        r = requests.get(PYPI_SIMPLE_INDEX, timeout=120,
                         headers={"User-Agent": HTTP_USER_AGENT})
        r.raise_for_status()
        names = re.findall(r">([^<]+)</a>", r.text)
        cache.write_text("\n".join(names))
    return cache.read_text().splitlines()


def fetch_known_malicious_names() -> set[str]:
    """Normalised names from the OSSF malicious-packages advisory feed.

    Used as a veto list so a known-bad package cannot end up labelled benign.

    We take these from a blobless sparse clone rather than the GitHub contents
    API: that API silently caps at 1000 entries and ignores pagination, which
    would hand us a partial veto list with no error -- the worst possible
    failure mode for a safety check. The clone only needs the directory tree
    (one directory per package), so --filter=blob:none keeps it small.

    Falls back to an empty set, loudly, if git or the network is unavailable.
    """
    cache = CACHE / "ossf-malicious-pypi.json"
    if cache.exists():
        return {normalise(n) for n in json.loads(cache.read_text())}

    clone = CACHE / "ossf-malicious-packages"
    subdir = "osv/malicious/pypi"
    try:
        if not (clone / ".git").exists():
            print("[ossf] sparse-cloning the advisory feed (one-off)")
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout",
                 "--depth=1", "--sparse", OSSF_MALICIOUS_REPO, str(clone)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for cmd in (["git", "sparse-checkout", "init", "--cone"],
                        ["git", "sparse-checkout", "set", subdir],
                        ["git", "checkout"]):
                subprocess.run(cmd, cwd=clone, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        base = clone / subdir
        names = sorted(d.name for d in base.iterdir() if d.is_dir())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print(f"[ossf] WARNING: advisory cross-check unavailable ({exc}). "
              "Benign candidates will rely on the DataDog manifest only.")
        return set()

    print(f"[ossf] {len(names)} known-malicious PyPI names in the veto list")
    cache.write_text(json.dumps(names))
    return {normalise(n) for n in names}


def normalise(name: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def select_candidates(n_top: int, n_random: int) -> list[tuple[str, str]]:
    """Return (package_name, pool) pairs, where pool is 'popular' or 'obscure'."""
    top = fetch_top_packages()
    veto = fetch_known_malicious_names()

    popular = [p for p in top[:n_top] if normalise(p) not in veto]

    excluded = {normalise(p) for p in top[:TOP_RANK_EXCLUSION]} | veto
    obscure_pool = [p for p in fetch_all_package_names() if normalise(p) not in excluded]

    rng = random.Random(RANDOM_SEED)
    obscure = rng.sample(obscure_pool, min(n_random, len(obscure_pool)))

    print(f"[select] {len(popular)} popular + {len(obscure)} obscure (hard negatives) "
          f"from a pool of {len(obscure_pool)}")

    # Interleave the two pools rather than returning popular-then-obscure.
    # Downloads are submitted in list order, so a concatenated list means an
    # interrupted or resumed run acquires 5000 popular packages before touching
    # a single hard negative -- which is precisely the half that makes the
    # evaluation meaningful. Shuffling keeps both pools growing together.
    candidates = [(p, "popular") for p in popular] + [(p, "obscure") for p in obscure]
    rng.shuffle(candidates)
    return candidates


# --- download ---------------------------------------------------------------


def pick_sdist(release_files: list[dict]) -> dict | None:
    """Choose the source distribution from a PyPI release's file list."""
    for f in release_files:
        if f.get("packagetype") == "sdist" and f["filename"].endswith(SDIST_SUFFIXES):
            return f
    return None


def acquire_one(name: str, pool: str, force: bool) -> dict | None:
    """Download and unpack one package's latest sdist. Returns an index row."""
    dest = BENIGN_DIR / normalise(name)

    if dest.exists() and not force:
        meta = dest / ".meta.json"
        if meta.exists():
            return json.loads(meta.read_text())
        shutil.rmtree(dest, ignore_errors=True)

    try:
        r = session().get(PYPI_JSON_API.format(name=name), timeout=DOWNLOAD_TIMEOUT)
        if r.status_code != 200:
            return None
        info = r.json()
    except (requests.RequestException, ValueError):
        return None

    version = info["info"]["version"]
    sdist = pick_sdist(info["urls"])
    if sdist is None or sdist["size"] > MAX_ARCHIVE_BYTES:
        return None  # wheel-only release, or implausibly large

    archive = CACHE / "sdists" / sdist["filename"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session().get(sdist["url"], timeout=DOWNLOAD_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            with open(archive, "wb") as out:
                for chunk in resp.iter_content(1 << 16):
                    out.write(chunk)

        shutil.rmtree(dest, ignore_errors=True)
        extract_any(archive, dest)
    except (requests.RequestException, UnsafeArchive, OSError, ValueError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[skip] {name}: {type(exc).__name__}: {exc}")
        return None
    finally:
        archive.unlink(missing_ok=True)

    row = {
        "package": name,
        "version": version,
        "label": 0,
        "source": "pypi",
        "pool": pool,
        "path": str(dest.relative_to(BENIGN_DIR.parent.parent)),
    }
    (dest / ".meta.json").write_text(json.dumps(row))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-top", type=int, default=N_TOP_PACKAGES)
    ap.add_argument("--n-random", type=int, default=N_RANDOM_PACKAGES)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    BENIGN_DIR.mkdir(parents=True, exist_ok=True)

    candidates = select_candidates(args.n_top, args.n_random)
    index: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {
            pool_exec.submit(acquire_one, name, pool, args.force): name
            for name, pool in candidates
        }
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                index.append(row)
            done += 1
            if done % 250 == 0:
                print(f"[progress] {done}/{len(candidates)} attempted, {len(index)} acquired")

    index_path = BENIGN_DIR.parent / "benign_index.json"
    index_path.write_text(json.dumps(index, indent=2))

    n_pop = sum(1 for r in index if r["pool"] == "popular")
    n_obs = len(index) - n_pop
    print(f"\n[done] {len(index)} benign packages: {n_pop} popular, {n_obs} obscure")
    print(f"       index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
