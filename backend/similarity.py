"""Nearest-neighbour lookup against every package the models were trained on.

A second opinion that needs no retraining: embed the new package with the same
code encoder Model B used, find the known packages whose code is closest, and
report how many of them are malware. Unlike the LightGBM score, the answer
comes with receipts -- the names of the closest known packages, and the pair of
files (new package vs. closest known malware) that look most alike.

Two caveats the UI must carry:

  * Similarity is likeness, not proof. A benign package that happens to share
    boilerplate with a dropper will sit near it.
  * The encoder reads the first MAX_TOKENS tokens of each file, so the code
    shown is the start of each file -- exactly the part that was compared.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from config import DATA, RAW  # noqa: E402
from features import iter_python_files, read_text  # noqa: E402

K_NEIGHBOURS = 10
# Both tuned on the held-out split (see ml/calibrate.py). Together they cut
# false alarms at the 60% mark from 25 to 7 of 1,448 benign packages at the
# same recall (0.81 -> 0.80):
#   * DUPLICATE_SIM -- DataDog holds many near-copies of one campaign; without
#     this, "9 of 10 neighbours are malware" can mean one campaign nine times.
#   * TEMPERATURE -- a neighbour at 0.97 should outvote one at 0.86. Weights
#     are exp((sim - best) / T), so each 0.02 of distance costs a factor of e.
DUPLICATE_SIM = 0.99
TEMPERATURE = 0.02
CANDIDATES = 60         # how far down the ranking to look for distinct families
MAX_FILES = 10          # same cap Model B used when building the index
SNIPPET_LINES = 40
SNIPPET_CHARS = 3000
ENCODER_THREADS = 6     # a request should not pin every core on a laptop

EMBED_CACHE = DATA / "cache" / "embeddings.npz"


@dataclass
class Neighbour:
    package: str
    version: str
    label: str          # "malicious" | "benign"
    pool: str           # "malicious" | "popular" | "obscure"
    similarity: float
    weight: float = 0.0  # share of the vote after distance weighting


@dataclass
class CodeMatch:
    similarity: float
    query_file: str
    query_code: str
    known_package: str
    known_version: str
    known_label: str
    known_file: str
    known_code: str


def neighbour_vote(sims: np.ndarray, matrix: np.ndarray, is_malicious: np.ndarray,
                   k: int = K_NEIGHBOURS) -> tuple[list[int], np.ndarray, float]:
    """Pick k distinct-family neighbours and score them.

    `sims` holds one query's cosine similarity to every row of `matrix`. Returns
    the chosen row indices, their weights, and the weighted malicious share in
    [0, 1]. Shared by the server and ml/calibrate.py so both score identically.
    """
    chosen: list[int] = []
    for j in np.argsort(-sims)[:CANDIDATES]:
        if chosen and float((matrix[chosen] @ matrix[j]).max()) >= DUPLICATE_SIM:
            continue  # a near-copy of a family already counted
        chosen.append(int(j))
        if len(chosen) == k:
            break
    s = sims[chosen]
    weights = np.exp((s - s.max()) / TEMPERATURE)
    share = float((weights * is_malicious[chosen]).sum() / weights.sum())
    return chosen, weights / weights.sum(), share


@dataclass
class SimilarityResult:
    malicious_percent: float
    k: int
    n_malicious: int
    neighbours: list[Neighbour] = field(default_factory=list)
    matches: list[CodeMatch] = field(default_factory=list)
    n_files_compared: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _snippet(text: str) -> str:
    lines = text.splitlines()[:SNIPPET_LINES]
    out = "\n".join(lines)
    if len(out) > SNIPPET_CHARS:
        out = out[:SNIPPET_CHARS] + "\n…"
    elif len(text.splitlines()) > SNIPPET_LINES:
        out += "\n…"
    return out


def _files(root: Path) -> list[tuple[str, str]]:
    """(relative path, text) for the files Model B would embed, setup.py first."""
    out = []
    for path in iter_python_files(root)[:MAX_FILES]:
        text = read_text(path)
        if text and text.strip():
            out.append((str(path.relative_to(root)), text))
    return out


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


class SimilarityIndex:
    """Package-level vectors for the whole training corpus, plus the encoder."""

    def __init__(self) -> None:
        self.ready = False
        self._encoder = None
        self._lock = threading.Lock()  # torch models are not re-entrant
        try:
            self._load()
        except Exception as exc:
            print(f"[similarity] index unavailable ({exc}); "
                  "run `python ml/train_embed.py --embed-only` to build it")

    def _load(self) -> None:
        meta: dict[str, dict] = {}
        for name in ("malicious_index.json", "benign_index.json"):
            for r in json.loads((RAW / name).read_text()):
                meta[f"{r['package']}@{r.get('version', '')}"] = r

        keys, vecs = [], []
        with np.load(EMBED_CACHE) as z:
            for k in z.files:
                v = z[k]
                if k in meta and v.any():  # zero vector = nothing was embeddable
                    keys.append(k)
                    vecs.append(v)

        self.keys = keys
        self.meta = [meta[k] for k in keys]
        self.matrix = _unit(np.vstack(vecs).astype(np.float32))
        self._is_malicious = np.array([m["label"] == 1 for m in self.meta])
        self.ready = True
        n_mal = int(self._is_malicious.sum())
        print(f"[similarity] {len(keys)} known packages indexed ({n_mal} malicious)")

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch
        from train_embed import PRIMARY_MODEL, embed_texts, load_encoder

        if self._encoder is None:
            torch.set_num_threads(ENCODER_THREADS)
            tok, mdl, used = load_encoder(PRIMARY_MODEL)
            if used != PRIMARY_MODEL:
                # Vectors from a different encoder live in a different space;
                # comparing them to the index would be meaningless.
                raise RuntimeError(f"index was built with {PRIMARY_MODEL}, got {used}")
            self._encoder = (tok, mdl)
        return embed_texts(texts, *self._encoder)

    def search(self, root: Path) -> SimilarityResult | None:
        files = _files(root)
        if not files:
            return None

        with self._lock:
            file_vecs = self._encode([t for _, t in files])
            pkg_vec = _unit(file_vecs.mean(axis=0))

            sims = self.matrix @ pkg_vec
            top, weights, share = neighbour_vote(sims, self.matrix, self._is_malicious)
            neighbours = [
                Neighbour(
                    package=self.meta[i]["package"],
                    version=str(self.meta[i].get("version", "")),
                    label="malicious" if self.meta[i]["label"] == 1 else "benign",
                    pool=self.meta[i].get("pool", "malicious"),
                    similarity=round(float(sims[i]), 4),
                    weight=round(float(w), 4),
                )
                for i, w in zip(top, weights)
            ]
            is_mal = np.array([n.label == "malicious" for n in neighbours])
            pct = 100.0 * share

            matches = self._code_matches(files, file_vecs, sims)

        return SimilarityResult(
            malicious_percent=round(pct, 1),
            k=len(neighbours),
            n_malicious=int(is_mal.sum()),
            neighbours=neighbours,
            matches=matches,
            n_files_compared=len(files),
        )

    def _code_matches(self, files, file_vecs, sims) -> list[CodeMatch]:
        """File-to-file comparison against the closest known malware and the
        closest known benign package in the whole index -- not just the top k,
        so a clean-looking package still shows its most malware-like code."""
        q = _unit(file_vecs)
        is_mal = self._is_malicious
        out: list[CodeMatch] = []
        for want in ("malicious", "benign"):
            mask = is_mal if want == "malicious" else ~is_mal
            if not mask.any():
                continue
            idx = int(np.flatnonzero(mask)[np.argmax(sims[mask])])
            known = self.meta[idx]
            known_files = _files(DATA / known["path"])
            if not known_files:
                continue
            k = _unit(self._encode([t for _, t in known_files]))
            pair = q @ k.T
            qi, ki = np.unravel_index(int(np.argmax(pair)), pair.shape)
            out.append(CodeMatch(
                similarity=round(float(pair[qi, ki]), 4),
                query_file=files[qi][0],
                query_code=_snippet(files[qi][1]),
                known_package=known["package"],
                known_version=str(known.get("version", "")),
                known_label=want,
                known_file=known_files[ki][0],
                known_code=_snippet(known_files[ki][1]),
            ))
        return out
