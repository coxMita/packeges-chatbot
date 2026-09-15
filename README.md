# packeges-chatbot — Malicious PyPI Package Detector

A machine-learning classifier that decides whether a PyPI package is **malicious**,
with a confidence score, wrapped in a React chatbot where a **local LLM explains the
model's reasoning** in plain English.

> **The LLM never classifies.** The trained model produces the verdict and a ranked
> list of the features that drove it. The LLM is given *only* that structured
> evidence — never the package source — so it cannot form an opinion of its own. It
> turns attributions into readable prose, nothing more.
>
> This is an ML project with an LLM presentation layer, not an "ask an LLM if it's
> malware" wrapper.

---

## How it works

```
  "requests==2.31.0"
          │
          ▼
  fetch sdist from PyPI ──────► unpack under extraction guards
          │                     (never installed, never executed)
          ▼
  ml/features.py ─────────────► 64 named static features
          │                     AST walk + text statistics
          ▼
  LightGBM ───────────────────► probability → verdict + confidence
          │
          ├── SHAP ───────────► ranked per-feature attribution
          │                         │
          ▼                         ▼
     verdict card            local LLM (Ollama)
     + evidence table        streams the explanation
```

The verdict is deterministic and takes ~1ms. The explanation streams separately, so
the UI paints the result immediately rather than waiting on a 4B model on CPU.

---

## Quick start

```bash
./setup.sh                          # Python 3.12 venv + CPU-only torch + deps

python ml/acquire_malicious.py      # ~6.7k real malicious packages (a few GB)
python ml/acquire_benign.py         # 5k popular + 3k obscure packages
python ml/build_dataset.py          # → data/processed/features.parquet
python ml/train_gbdt.py             # Model A  (seconds)
python ml/train_embed.py            # Model B  (1-2h on CPU, cached)
python ml/evaluate.py               # head-to-head → ml/reports/comparison.md

ollama serve &                      # local LLM for explanations
ollama pull qwen3.5:4b

./run.sh                            # backend :8000 + frontend :5173
```

Everything except the LLM explanation works without Ollama running — the verdict and
evidence come from the classifier.

---

## Data

| Source | Role | License |
|---|---|---|
| [DataDog/malicious-software-packages-dataset](https://github.com/DataDog/malicious-software-packages-dataset) | Malicious class — ~6.7k real PyPI packages caught in the wild, human-vetted | Apache-2.0 |
| [top-pypi-packages](https://hugovk.github.io/top-pypi-packages/) | Benign class — popular packages | public data |
| PyPI simple index (random sample) | **Hard negatives** — obscure but benign | public data |
| [ossf/malicious-packages](https://github.com/ossf/malicious-packages) | Veto list, so known-bad packages stay out of the benign class | Apache-2.0 |

Only **source distributions** are used. Wheels are built artifacts and frequently
omit `setup.py` — exactly where install-time attacks live.

### The sampling trap, and what this project does about it

Popular PyPI packages are large, mature, well-tooled projects. Malicious packages are
tiny throwaways. Train on those two pools alone and the model does not learn
*malicious behaviour* — it learns *small package*. It then scores ~99% on the held-out
split and is useless against any real upload, because every new legitimate package by
a first-time author looks exactly like the malicious class.

So the benign class is drawn from two pools:

- **popular** — 5,000 top-download projects, the realistic negative class;
- **obscure** — 3,000 packages sampled uniformly at random from the full PyPI index,
  excluding the top 15,000. Small, amateur, sparsely documented, often single-file —
  and perfectly benign. These are the **hard negatives**.

`ml/evaluate.py` reports the `obscure` slice separately. **That number, not the
headline F1, is the honest measure of whether this model works.** `train_gbdt.py`
additionally warns if a raw size feature lands in the top 2 by gain, which is the
symptom of the model having learned size after all.

---

## Features (`ml/features.py`)

64 numeric features, every one named and documented, derived from `ast.parse` plus
text statistics. `setup.py` is parsed, **never executed** — running the file you are
trying to judge would defeat the purpose.

| Group | What it captures |
|---|---|
| `install_*` | Code that runs at `pip install` time: `cmdclass` overrides, install-command subclasses, module-level statements, network/subprocess calls in `setup.py`. **Highest-signal group.** |
| `call_*` | `eval`/`exec`/`compile`, subprocess and shell execution, sockets and HTTP, `pickle`/`marshal`, dynamic `getattr`, `except: pass`, imports hidden in function bodies |
| `decode_*` | base64/hex/zlib decoding, and specifically *decode-then-exec* chains — the packed-payload signature |
| `obf_*` | String-literal entropy, longest literal, base64/hex-shaped strings, non-ASCII ratio, very long lines, identifier-length distribution |
| `exfil_*` | References to SSH keys, cloud credentials, `.env`/`.pypirc`, browser cookie and password stores, crypto wallets, Discord/Telegram session data |
| `net_*` | Hardcoded routable IPs, URL counts, Discord webhooks, Telegram bot API, paste sites, tunnels, OAST hosts, throwaway TLDs |
| `pkg_*` | File and line counts, README/LICENSE presence, bundled binaries, and edit distance to the nearest popular package name (**typosquatting**) |

Raw counts are paired with **per-kLOC densities**, so a 40-line dropper with three
subprocess calls outranks a 50k-line project with the same three.

---

## Models

**Model A — LightGBM over the engineered features.** The production model. Grouped
5-fold CV; the decision threshold is tuned on out-of-fold predictions to maximise
F-beta with β=0.5, weighting precision over recall — a scanner that cries wolf gets
ignored. SHAP gives exact per-prediction attribution, which is what the chatbot
narrates.

**Model B — code embeddings + logistic head.** Told nothing about what to look for;
reads source through a pretrained code encoder and lets a linear head find the
boundary. Slower, essentially unexplainable at the feature level, but not limited to
patterns someone wrote a feature for.

Run `python ml/evaluate.py` to regenerate `ml/reports/comparison.md` with both models
scored on the same grouped split.

---

## Safety

- Packages are **downloaded, unpacked and parsed as text** — never installed, never
  imported, never executed.
- `ml/safe_extract.py` guards every unpack against path traversal, absolute member
  paths, symlinks and hardlinks, and archive bombs. Those guards are deliberately set
  generously rather than tightly: a cap tight enough to reject numpy's sdist would
  bias the benign class toward small packages and reintroduce the exact artifact the
  hard-negative pool exists to prevent.
- No malware is committed to this repo. `data/` is gitignored; the acquisition
  scripts re-download from DataDog on demand.
- The train/test split is **grouped on package name**, so two releases of the same
  compromised library can never straddle it.

---

## Layout

```
ml/
  config.py            paths, constants, sampling sizes
  safe_extract.py      guarded archive extraction
  acquire_malicious.py DataDog corpus, blobless sparse checkout
  acquire_benign.py    popular + hard-negative pools, OSSF veto
  features.py          64-feature static extractor  ← core
  build_dataset.py     → features.parquet
  split.py             shared grouped split
  train_gbdt.py        Model A
  train_embed.py       Model B
  evaluate.py          head-to-head, three slices
  tests/
backend/
  fetch.py             PyPI resolution + safe unpack
  predict.py           scoring + SHAP evidence
  llm.py               Ollama; explains, never decides
  main.py              FastAPI: /analyze, /explain (SSE), /health
frontend/              React + Vite + TypeScript chat UI
```

## Limitations

- **Static analysis only.** No sandboxed execution, so behaviour that only manifests
  at runtime is invisible.
- **PyPI only.** The pipeline is structured to take npm next, but nothing here has
  been validated against JavaScript.
- **Novel obfuscation.** Model A can only see what someone wrote a feature for. A
  genuinely new packing scheme will evade it; that is the gap Model B exists to probe.
- Trained on packages caught between roughly 2018 and 2026.

## License

Apache-2.0
