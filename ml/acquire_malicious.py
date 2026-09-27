#!/usr/bin/env python3
"""Download the malicious PyPI class from DataDog's open dataset.

Source: https://github.com/DataDog/malicious-software-packages-dataset (Apache-2.0)

The repository is ~20GB, but 17GB of that is the npm half we do not need. We use
a blobless sparse checkout of `samples/pypi` only, which keeps the download to a
couple of gigabytes.

Samples are stored as zips encrypted with the password `infected` -- a convention
the dataset authors use so that scanners on developer machines do not quarantine
the repo. Layout:

    samples/pypi/<malicious_intent|compromised_lib>/<package>/<version>/*.zip

As of this writing the PyPI half holds ~2,530 sample archives across 1,832
package names (the npm half is far larger -- hence the sparse checkout).

`malicious_intent` packages exist only to attack; `compromised_lib` are real
libraries whose maintainer account was hijacked for one or more releases. We keep
the distinction as a column -- compromised libraries are the harder, more
interesting cases, since the malicious code is a few lines hidden in an otherwise
legitimate project.

Nothing in this script executes package code.

Usage:
    python ml/acquire_malicious.py            # clone (or update) and extract
    python ml/acquire_malicious.py --limit 50 # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE, DATADOG_REPO, DATADOG_SAMPLE_PASSWORD, MALICIOUS_DIR, ensure_dirs
from safe_extract import UnsafeArchive, extract_zip

CLONE_DIR = CACHE / "datadog-dataset"
SAMPLES_SUBDIR = "samples/pypi"
INTENT_CLASSES = ("malicious_intent", "compromised_lib")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def sparse_clone() -> Path:
    """Blobless sparse checkout of just the PyPI samples. Idempotent."""
    if (CLONE_DIR / ".git").exists():
        print(f"[clone] updating existing checkout at {CLONE_DIR}")
        _run(["git", "fetch", "--depth=1", "origin", "main"], cwd=CLONE_DIR)
        _run(["git", "reset", "--hard", "origin/main"], cwd=CLONE_DIR)
        return CLONE_DIR

    print(f"[clone] blobless sparse clone of {SAMPLES_SUBDIR} (a few GB, be patient)")
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "git", "clone",
        "--filter=blob:none",   # do not fetch file contents up front
        "--no-checkout",
        "--depth=1",
        "--sparse",
        DATADOG_REPO, str(CLONE_DIR),
    ])
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=CLONE_DIR)
    _run(["git", "sparse-checkout", "set", SAMPLES_SUBDIR], cwd=CLONE_DIR)
    _run(["git", "checkout"], cwd=CLONE_DIR)
    return CLONE_DIR


def iter_samples(repo: Path, limit: int | None = None):
    """Yield (intent_class, package, version, zip_path) for every PyPI sample.

    The corpus uses two layouts, and missing the second silently drops ~10% of
    the malicious class:

        samples/pypi/<intent>/<package>/<version>/<file>.zip   (2271 samples)
        samples/pypi/<intent>/<package>/<file>.zip             ( 259 samples)

    Packages in the second form were archived without a resolvable version.
    """
    n = 0
    for intent in INTENT_CLASSES:
        base = repo / SAMPLES_SUBDIR / intent
        if not base.is_dir():
            continue
        for zip_path in sorted(base.rglob("*.zip")):
            rel = zip_path.relative_to(base).parts
            if len(rel) >= 3:              # <package>/<version>/<file>.zip
                package, version = rel[0], rel[1]
            elif len(rel) == 2:            # <package>/<file>.zip
                package, version = rel[0], "unknown"
            else:
                continue
            yield intent, package, version, zip_path
            n += 1
            if limit is not None and n >= limit:
                return


def slug(package: str, version: str, archive: Path | None = None) -> str:
    """Filesystem-safe identifier for one package release.

    A handful of (package, version) pairs ship more than one archive. Without a
    disambiguator they would overwrite each other on disk, so the archive stem
    is folded in when a collision is possible.
    """
    ident = f"{package}@{version}"
    if archive is not None:
        ident = f"{ident}@{archive.stem}"
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in ident)
    return safe[:150]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="stop after N samples")
    ap.add_argument("--force", action="store_true", help="re-extract samples already on disk")
    args = ap.parse_args()

    ensure_dirs()
    repo = sparse_clone()

    MALICIOUS_DIR.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    extracted = skipped = failed = 0

    for intent, package, version, zip_path in iter_samples(repo, args.limit):
        dest = MALICIOUS_DIR / slug(package, version, zip_path)

        if dest.exists() and not args.force:
            skipped += 1
        else:
            if dest.exists():
                shutil.rmtree(dest)
            try:
                extract_zip(zip_path, dest, password=DATADOG_SAMPLE_PASSWORD)
                extracted += 1
            except (UnsafeArchive, OSError, RuntimeError, ValueError) as exc:
                # Malformed or hostile archives are expected in a malware corpus.
                print(f"[skip] {package}@{version}: {type(exc).__name__}: {exc}")
                shutil.rmtree(dest, ignore_errors=True)
                failed += 1
                continue

        index.append({
            "package": package,
            "version": version,
            "label": 1,
            "source": "datadog",
            "intent_class": intent,
            "path": str(dest.relative_to(MALICIOUS_DIR.parent.parent)),
        })

        total = extracted + skipped
        if total and total % 500 == 0:
            print(f"[progress] {total} samples ({extracted} new, {skipped} cached)")

    index_path = MALICIOUS_DIR.parent / "malicious_index.json"
    index_path.write_text(json.dumps(index, indent=2))

    print(f"\n[done] {len(index)} malicious releases indexed")
    print(f"       {extracted} newly extracted, {skipped} already present, {failed} failed")
    print(f"       index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
