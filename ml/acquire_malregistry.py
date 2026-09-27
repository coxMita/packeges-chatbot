#!/usr/bin/env python3
"""Add the pypi_malregistry corpus to the malicious class.

Source: https://github.com/lxyeternal/pypi_malregistry -- ~12k malicious PyPI
sdists collected for "An Empirical Study of Malicious Code In PyPI Ecosystem"
(ASE 2023), still being extended. The repository declares no license; it is
used here for local research only and, like everything under data/, is never
committed or redistributed.

Layout: <package>/<version>/<archive>.tar.gz (a few .zip). Archives are plain,
not password-protected. Wheels are skipped for the same reason as everywhere
else in this project: install-time attacks live in setup.py, which wheels drop.

Packages whose name DataDog already covers are skipped, so the two sources do
not double-count one release. Every row gets a `reported` date from the OSSF
malicious-packages reports where one exists, so the evaluation can hold out the
newest malware as a "future" set.

Nothing in this script executes package code.

Usage:
    python ml/acquire_malregistry.py            # clone must already be checked out
    python ml/acquire_malregistry.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from acquire_benign import normalise
from config import CACHE, MALICIOUS_DIR, RAW, ensure_dirs
from safe_extract import UnsafeArchive, extract_any

REPO_DIR = CACHE / "pypi_malregistry"
REPO_URL = "https://github.com/lxyeternal/pypi_malregistry.git"
OSSF_PYPI = CACHE / "ossf-malicious-packages" / "osv" / "malicious" / "pypi"
INDEX = RAW / "malregistry_index.json"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar.bz2")


def ossf_report_dates() -> dict[str, str]:
    """Earliest OSSF report date (YYYY-MM-DD) per normalised package name."""
    dates: dict[str, str] = {}
    for f in OSSF_PYPI.glob("*/*.json"):
        try:
            j = json.loads(f.read_text())
            name = normalise(j["affected"][0]["package"]["name"])
            day = (j.get("published") or j.get("modified") or "")[:10]
        except (OSError, ValueError, KeyError, IndexError):
            continue
        if day and (name not in dates or day < dates[name]):
            dates[name] = day
    return dates


def iter_archives(repo: Path):
    """Yield (package, version, archive) for every sdist-shaped archive."""
    for pkg_dir in sorted(p for p in repo.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for ver_dir in sorted(p for p in pkg_dir.iterdir() if p.is_dir()):
            archives = sorted(a for a in ver_dir.iterdir()
                              if a.is_file() and a.name.lower().endswith(ARCHIVE_SUFFIXES))
            if archives:
                yield pkg_dir.name, ver_dir.name, archives[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-extract samples already on disk")
    args = ap.parse_args()

    ensure_dirs()
    if not REPO_DIR.is_dir() or not any(REPO_DIR.iterdir()):
        print(f"[error] clone {REPO_URL} into {REPO_DIR} first")
        return 1

    datadog = json.loads((RAW / "malicious_index.json").read_text())
    covered = {normalise(r["package"]) for r in datadog}
    dates = ossf_report_dates()

    index: list[dict] = []
    seen: set[str] = set()
    extracted = cached = failed = duplicate = 0

    for package, version, archive in iter_archives(REPO_DIR):
        name = normalise(package)
        # One release per name: later versions of a throwaway are near-copies,
        # and DataDog's copy wins where both have it.
        if name in covered or name in seen:
            duplicate += 1
            continue
        seen.add(name)

        dest = MALICIOUS_DIR / f"mr_{name}_{version}"[:150]
        if dest.exists() and not args.force:
            cached += 1
        else:
            shutil.rmtree(dest, ignore_errors=True)
            try:
                extract_any(archive, dest)
                extracted += 1
            except (UnsafeArchive, OSError, EOFError, ValueError, RuntimeError) as exc:
                # Malformed archives are expected in a malware corpus.
                print(f"[skip] {package}@{version}: {type(exc).__name__}: {exc}")
                shutil.rmtree(dest, ignore_errors=True)
                failed += 1
                continue

        index.append({
            "package": package,
            "version": version,
            "label": 1,
            "source": "malregistry",
            "intent_class": "malicious_intent",
            "reported": dates.get(name, ""),
            "path": str(dest.relative_to(MALICIOUS_DIR.parent.parent)),
        })
        if len(index) % 1000 == 0:
            print(f"[progress] {len(index)} packages ({extracted} new, {cached} cached)")
        if args.limit and len(index) >= args.limit:
            break

    INDEX.write_text(json.dumps(index, indent=2))
    dated = sum(1 for r in index if r["reported"])
    print(f"\n[done] {len(index)} malicious packages indexed ({dated} with an OSSF report date)")
    print(f"       {extracted} extracted, {cached} already present, {failed} failed, "
          f"{duplicate} skipped as already covered")
    print(f"       index -> {INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
