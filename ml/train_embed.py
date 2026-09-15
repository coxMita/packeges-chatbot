#!/usr/bin/env python3
"""Model B -- code embeddings + a classifier head.

The contrast with Model A is the point of the comparison. Where Model A is told
what to look for (subprocess calls, entropy, install hooks), Model B is told
nothing: it reads the source through a pretrained code encoder and lets a linear
head find the boundary. It should, in principle, generalise better to obfuscation
patterns nobody wrote a feature for -- at the cost of being far slower and
essentially unexplainable at the feature level.

Practical constraints shaped three choices here:

  * **Encoder size.** CodeBERT-base (125M) would take many hours on CPU for a
    15k-package corpus. We use a small code-specialised encoder instead and
    fall back automatically if it cannot be fetched.
  * **What to embed.** Embedding every file is wasteful -- most of a package is
    boilerplate. We embed the files most likely to carry a payload (setup.py
    first, then the largest/most suspicious modules), capped per package, and
    mean-pool them into a single package vector.
  * **Caching.** The embedding pass is the expensive step, so vectors are cached
    to an .npz keyed by package. Re-running only embeds what is new.

Usage:
    python ml/train_embed.py                  # embed (cached) then fit the head
    python ml/train_embed.py --embed-only     # just build the cache
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE, FEATURES_PARQUET, MODELS, RANDOM_SEED, ensure_dirs
from features import iter_python_files, read_text
from split import grouped_split

PRIMARY_MODEL = "jinaai/jina-embeddings-v2-small-code"
FALLBACK_MODEL = "microsoft/codebert-base"

MAX_FILES_PER_PACKAGE = 10   # mean-pooled into one package vector
MAX_TOKENS = 512
BATCH_SIZE = 16

EMBED_CACHE = CACHE / "embeddings.npz"


def load_encoder(name: str):
    """Load a code encoder, falling back if the primary is unavailable."""
    from transformers import AutoModel, AutoTokenizer

    for candidate in (name, FALLBACK_MODEL):
        try:
            print(f"[encoder] loading {candidate}")
            tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=False)
            mdl = AutoModel.from_pretrained(candidate, trust_remote_code=False)
            mdl.eval()
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
            return tok, mdl, candidate
        except Exception as exc:  # network, auth, or unsupported architecture
            print(f"[encoder] {candidate} unavailable: {type(exc).__name__}: {exc}")
    raise RuntimeError("no usable code encoder could be loaded")


def select_files(root: Path) -> list[str]:
    """Pick the files most likely to carry a payload, as text.

    `iter_python_files` already sorts setup.py and top-level modules first --
    exactly where install-time attacks live -- so we take its head.
    """
    texts: list[str] = []
    for path in iter_python_files(root)[:MAX_FILES_PER_PACKAGE]:
        text = read_text(path)
        if text and text.strip():
            texts.append(text)
    return texts


@torch.no_grad()
def embed_texts(texts: list[str], tok, mdl) -> np.ndarray:
    """Mean-pooled embedding per text, batched."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=MAX_TOKENS, return_tensors="pt")
        hidden = mdl(**enc).last_hidden_state
        # Mask-aware mean pooling: padding tokens must not dilute the vector.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, mdl.config.hidden_size), np.float32)


def build_embeddings(df: pd.DataFrame, root: Path) -> dict[str, np.ndarray]:
    """Embed every package, resuming from the on-disk cache."""
    cache: dict[str, np.ndarray] = {}
    if EMBED_CACHE.exists():
        with np.load(EMBED_CACHE) as z:
            cache = {k: z[k] for k in z.files}
        print(f"[cache] {len(cache)} package vectors already embedded")

    todo = [r for r in df.itertuples() if _key(r) not in cache]
    if not todo:
        return cache

    tok, mdl, used = load_encoder(PRIMARY_MODEL)
    dim = mdl.config.hidden_size
    print(f"[embed] {len(todo)} packages to embed with {used} (dim {dim}) -- "
          "this is the slow step on CPU")

    t0 = time.perf_counter()
    for i, row in enumerate(todo, 1):
        pkg_dir = root / row.path if hasattr(row, "path") else None
        texts = select_files(pkg_dir) if pkg_dir and pkg_dir.is_dir() else []
        if texts:
            vecs = embed_texts(texts, tok, mdl)
            cache[_key(row)] = vecs.mean(axis=0).astype(np.float32)
        else:
            cache[_key(row)] = np.zeros(dim, np.float32)

        if i % 100 == 0:
            rate = i / (time.perf_counter() - t0)
            eta = (len(todo) - i) / max(rate, 1e-9) / 60
            print(f"[embed] {i}/{len(todo)} ({rate:.1f}/s, ~{eta:.0f} min left)")
            np.savez_compressed(EMBED_CACHE, **cache)

    np.savez_compressed(EMBED_CACHE, **cache)
    print(f"[embed] done in {(time.perf_counter() - t0) / 60:.1f} min -> {EMBED_CACHE}")
    return cache


def _key(row) -> str:
    return f"{row.package}@{row.version}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--embed-only", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    df = pd.read_parquet(args.data)

    # Package directories are recorded relative to the repo root in the indexes;
    # rebuild that mapping so the embedder knows where each package lives.
    import json
    from config import RAW
    paths: dict[str, str] = {}
    for idx in ("malicious_index.json", "benign_index.json"):
        f = RAW / idx
        if f.exists():
            for r in json.loads(f.read_text()):
                paths[f"{r['package']}@{r.get('version', '')}"] = r["path"]
    df = df.assign(path=[paths.get(f"{p}@{v}", "") for p, v in
                         zip(df["package"], df["version"])])
    df = df[df["path"] != ""].reset_index(drop=True)

    cache = build_embeddings(df, Path(__file__).resolve().parents[1])
    if args.embed_only:
        return 0

    if df["label"].nunique() < 2:
        print("[error] the dataset contains only one class -- acquire both before training")
        return 1

    dim = len(next(iter(cache.values())))
    E = np.vstack([cache.get(_key(r), np.zeros(dim, np.float32)) for r in df.itertuples()])

    train_df, test_df = grouped_split(df)
    tr_idx = train_df.index.to_numpy()
    y_tr = train_df["label"].to_numpy()

    # Re-derive positional indices, since grouped_split resets the index.
    keys = {_key(r): i for i, r in enumerate(df.itertuples())}
    tr_pos = np.array([keys[_key(r)] for r in train_df.itertuples()])
    te_pos = np.array([keys[_key(r)] for r in test_df.itertuples()])

    scaler = StandardScaler().fit(E[tr_pos])
    head = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED
    ).fit(scaler.transform(E[tr_pos]), y_tr)

    te_prob = head.predict_proba(scaler.transform(E[te_pos]))[:, 1]
    y_te = test_df["label"].to_numpy()

    print(f"\n[test] ROC-AUC {roc_auc_score(y_te, te_prob):.4f} | "
          f"PR-AUC {average_precision_score(y_te, te_prob):.4f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "embed.pkl", "wb") as fh:
        pickle.dump({"scaler": scaler, "head": head, "dim": dim,
                     "encoder": PRIMARY_MODEL, "threshold": 0.5}, fh)
    print(f"[done] model -> {MODELS / 'embed.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
