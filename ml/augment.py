#!/usr/bin/env python3
"""Synthetic trojanized libraries: teach the model that big can still be bad.

The training set holds ~5,500 large benign projects but only ~500 large
malicious ones, and just 26 real compromised libraries -- a genuine project
with a payload slipped in. So the model learned "large, well-structured means
benign", and a trojanized fork like index-forum scores 0.002.

This builds what those attacks look like. A benign host package gets real
malicious code from a training sample, injected one of three ways:

  append   payload appended to an existing module (index-forum's shape)
  setup    payload appended to setup.py, so it runs at install time
  module   payload dropped in as a new module next to the host's code
  init     payload appended to a package __init__.py, so it runs on import

Every malicious sample has a *control*: the same host and the same injection,
but with code from another benign package, labelled benign. Without controls
the model could learn "has an injected-looking extra blob" instead of
"the extra code is malicious". Most controls come from benign code that uses
the same risky APIs malware does (network, subprocess, decoding, dynamic
imports); easy controls taught the model to flag real client libraries.

Rows record `host_group` and `donor_group`. Training code must only use a
synthetic row when both groups sit in its training split, and synthetic rows
are never scored as test data.

Hosts are hard-linked into a scratch folder, modified by replacing files (so
the originals are untouched), featurised and deleted. Nothing is executed.

Usage:
    python -u ml/augment.py              # -> data/processed/augmented.parquet
    python -u ml/augment.py --n 3000
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from acquire_benign import fetch_top_packages, normalise
from build_dataset import BENIGN_DIR, MALICIOUS_DIR, RAW, load_index
from config import CACHE, FEATURES_PARQUET, PROCESSED, RANDOM_SEED
from features import extract_features, feature_names, iter_python_files, read_text

OUT = PROCESSED / "augmented.parquet"
SCRATCH = CACHE / "augment_tmp"
MAX_DONOR_CHARS = 20_000
HOST_FILES = (5, 300)        # inject into real projects, but not giants
KINDS = ("append", "setup", "module", "init")
RISKY_CONTROL_SHARE = 0.75   # of benign controls, drawn from risky-API donors


def donor_code(root: Path) -> str:
    """All Python text of a donor package, capped to a realistic payload size."""
    parts = []
    for path in iter_python_files(root):
        text = read_text(path)
        if text:
            parts.append(text)
    return "\n\n".join(parts)[:MAX_DONOR_CHARS]


def _replace(path: Path, text: str) -> None:
    """Write via a new inode so the hard-linked original is never modified."""
    tmp = path.with_name(path.name + ".aug")
    tmp.write_text(text)
    os.replace(tmp, path)


def inject(host: Path, work: Path, code: str, kind: str, rng: random.Random) -> bool:
    shutil.copytree(host, work, copy_function=os.link, symlinks=False)
    py = [p for p in iter_python_files(work) if p.name != "setup.py"]
    setup = [p for p in iter_python_files(work) if p.name == "setup.py"]

    inits = [p for p in py if p.name == "__init__.py"]
    if kind == "init" and inits:
        target = rng.choice(inits)
        _replace(target, (read_text(target) or "") + "\n\n" + code + "\n")
    elif kind == "setup" and setup:
        target = setup[0]
        _replace(target, (read_text(target) or "") + "\n\n" + code + "\n")
    elif kind == "append" and py:
        target = rng.choice(py)
        _replace(target, (read_text(target) or "") + "\n\n" + code + "\n")
    elif py or setup:
        pkg_dir = (py or setup)[0].parent
        (pkg_dir / f"_{rng.choice(['compat', 'utils', 'helpers', 'diag', 'core'])}"
                   f"{rng.randint(0, 99)}.py").write_text(code + "\n")
    else:
        return False
    return True


def _one(job: dict) -> dict | None:
    rng = random.Random(job["seed"])
    work = SCRATCH / f"job{job['seed']}"
    shutil.rmtree(work, ignore_errors=True)
    try:
        code = donor_code(Path(job["donor_path"]))
        if len(code.strip()) < 40:
            return None
        if not inject(Path(job["host_path"]), work, code, job["kind"], rng):
            return None
        feats = extract_features(work, job["host_package"], job["popular"])
    except (OSError, RecursionError, MemoryError, shutil.Error):
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "package": f"aug:{job['host_package']}+{job['donor_package']}:{job['kind']}",
        "version": "", "label": job["label"], "pool": "synthetic",
        "intent_class": f"synthetic_{job['kind']}", "source": "synthetic", "reported": "",
        "group": job["host_group"], "host_group": job["host_group"],
        "donor_group": job["donor_group"], **feats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3500, help="malicious samples (plus as many controls)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES_PARQUET)
    paths = {}
    for index, base in [(RAW / "malicious_index.json", MALICIOUS_DIR),
                        (RAW / "malregistry_index.json", MALICIOUS_DIR),
                        (RAW / "benign_index.json", BENIGN_DIR)]:
        for row, path in load_index(index, base):
            paths[(row["package"], row.get("version", ""))] = path
    df["path"] = [paths.get((p, v)) for p, v in zip(df["package"], df["version"])]
    df = df[df["path"].notna()]

    lo, hi = HOST_FILES
    hosts = df[(df.label == 0) & df.pkg_n_files.between(lo, hi)]
    mal_donors = df[(df.label == 1) & (df.pkg_n_py_files >= 1)]
    ben_donors = df[(df.label == 0) & (df.pkg_n_py_files >= 1) & (df.pkg_loc <= 2000)]
    # Controls must look risky too. Benign code that never touches the network
    # or a subprocess teaches "injected code that does is malware" -- and then
    # every real client library (Telegram, DNS, trading APIs) gets flagged.
    risky = ben_donors[(ben_donors[["call_network", "call_process", "decode_calls",
                                    "call_exec", "call_computed_import"]] > 0).any(axis=1)]
    print(f"[augment] {len(hosts)} hosts, {len(mal_donors)} malicious donors, "
          f"{len(ben_donors)} benign donors ({len(risky)} using risky APIs)")

    rng = random.Random(RANDOM_SEED)
    popular = {normalise(p) for p in fetch_top_packages()[:2000]}
    jobs = []
    for i in range(args.n):
        host = hosts.iloc[rng.randrange(len(hosts))]
        kind = rng.choice(KINDS)
        for label, donors in [(1, mal_donors), (0, ben_donors)]:
            if label == 0 and rng.random() < RISKY_CONTROL_SHARE:
                donors = risky
            donor = donors.iloc[rng.randrange(len(donors))]
            if donor["group"] == host["group"]:
                continue
            jobs.append({
                "seed": len(jobs), "label": label, "kind": kind,
                "host_path": host["path"], "host_package": host["package"],
                "host_group": host["group"], "donor_path": donor["path"],
                "donor_package": donor["package"], "donor_group": donor["group"],
                "popular": popular,
            })

    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            if (row := fut.result()) is not None:
                rows.append(row)
            if i % 250 == 0:
                print(f"[progress] {i}/{len(jobs)}")
    shutil.rmtree(SCRATCH, ignore_errors=True)

    out = pd.DataFrame(rows)
    cols = feature_names()
    out[cols] = out[cols].fillna(0.0).astype("float32")
    out.to_parquet(OUT, index=False)
    print(f"\n[done] {len(out)} synthetic packages -> {OUT}")
    print(out.groupby(["label", "intent_class"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
