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

**Follow-up questions.** After a result, the same box takes questions: "why is it
suspicious?", "is a missing README really a red flag?". Input that names a package
(`flask`, `flask==3.0.0`, `check flask`, `pip install flask`, `is flask safe?`,
`what about flask?`) is analysed; anything else ("is it safe?", "which file should
I read?") is answered by the LLM about the **most recently analysed package**, from
the evidence of the last three analyses plus the recent conversation
(`POST /api/chat`). The UI shows which package follow-ups refer to.

**Training-data context.** Every evidence signal is set against the training
data. The UI draws the share of malicious and benign training packages with the
same value or beyond, for example "decoded data passed to exec: 4% of malware, 0.02% of
benign → 39× more common in malware". The explanation cites these shares and names
the suspect file, the calls and their line numbers. The LLM gets call names only,
never source lines. The same rules apply: it still never sees source
code and cannot override a verdict. Answers take ~1 minute on CPU.

---

## Quick start

```bash
./setup.sh                          # Python 3.12 venv + CPU-only torch + deps

python ml/acquire_malicious.py      # ~2.5k real malicious packages (a few GB)
python ml/acquire_benign.py         # 5k popular + 3k obscure packages
python ml/build_dataset.py          # → data/processed/features.parquet
python ml/train_gbdt.py             # Model A  (seconds; train split + future malware)
python ml/train_embed.py            # Model B  (~2.5h on 8 CPU threads, cached, resumable)
python ml/evaluate.py               # head-to-head → ml/reports/comparison.md
python ml/calibrate.py              # tiers + "chance it is really malware"
python ml/live.py --fresh           # optional: fetch live PyPI packages outside the dataset
python ml/generalize.py             # unseen-family experiments + monthly retraining (~30 min)
python ml/report.py                 # → ml/reports/performance.md

ollama serve &                      # local LLM for explanations
ollama pull qwen3.5:4b

./run.sh                            # backend :8000 + frontend :5173
```

Everything except the LLM explanation works without Ollama running — the verdict and
evidence come from the classifier.

### Docker

Three containers: `frontend` (nginx serving the built UI and proxying `/api`),
`backend` (FastAPI, models loaded once) and `ollama` (the local LLM).

```bash
python ml/train_gbdt.py && python ml/calibrate.py   # artifacts the backend mounts
docker compose up -d --build                        # → http://localhost:8080
docker compose logs -f ollama                       # first start pulls qwen3.5:4b (~3.4 GB)
```

- Only the UI is published, on `127.0.0.1:${APP_PORT:-8080}`. The API and the LLM sit
  on internal networks (`web`, `llm`).
- `ml/models/` and `data/` are bind-mounted **read-only**. `data/` is optional: without
  it the similarity second opinion is off and everything else works.
- Every container runs as a non-root user with `cap_drop: ALL` and
  `no-new-privileges`, with CPU and memory limits. The frontend and backend
  also get a read-only root filesystem, with a tmpfs `/tmp` for unpacking packages.
- The CodeBERTa encoder weights are baked into the backend image, which runs with
  `HF_HUB_OFFLINE=1`.
- The backend starts without waiting for the model download; verdicts work
  immediately and explanations start once Ollama reports healthy.
- NVIDIA GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.
- Settings via `.env`: `OLLAMA_MODEL`, `APP_PORT`, `APP_BIND`, `OLLAMA_VERSION`.

### Diagrams

draw.io files at the repo root (open in [app.diagrams.net](https://app.diagrams.net)
or the VS Code Draw.io extension):

| File | Shows |
|---|---|
| `architecture.drawio` | software components: UI, nginx, API modules, LLM, artifacts, offline pipeline |
| `deployment.drawio` | Docker Compose deployment: containers, networks, volumes, ports, egress |
| `sequence-analysis.drawio` | analyse → explain (SSE) → follow-up question |
| `ml-pipeline.drawio` | data sources, 70/15/15 split + future holdout, training, calibration, evaluation |
| `chat-routing.drawio` | how a chat message is routed to analysis or to a follow-up answer |

---

## Data

| Source | Role | License |
|---|---|---|
| [DataDog/malicious-software-packages-dataset](https://github.com/DataDog/malicious-software-packages-dataset) | Malicious class — ~2.5k real PyPI sample archives caught in the wild, human-vetted | Apache-2.0 |
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

## Features (`ml/features.py`, `ml/behaviour.py`)

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

Plus 38 **behaviour** features (`autorun_*`, `flow_*`, `proc_*`, `bin_*`, `pth_*`), which
describe what the code does by itself and where data flows. See *Generalising to unseen
packages*. The verdict card lists the ones found under **What the code can do**.

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
reads source through a pretrained code encoder
([CodeBERTa-small](https://huggingface.co/huggingface/CodeBERTa-small-v1), chosen
partly because it loads without `trust_remote_code`) and lets a linear head find the
boundary. Slower, essentially unexplainable at the feature level, but not limited to
patterns someone wrote a feature for.

Run `python ml/evaluate.py` to regenerate `ml/reports/comparison.md` with both models
scored on the same grouped split.

---

## Results

18,874 packages: 11,538 malicious (DataDog, pypi_malregistry), 4,709 popular and 2,627
obscure benign. They're split **70 / 15 / 15**, grouped by package family and stratified
by class. Full tables: [`ml/reports/performance.md`](ml/reports/performance.md);
experiments: [`ml/reports/generalize.md`](ml/reports/generalize.md).

**In distribution** (test split, production model):
98.3% accuracy · 99.4% precision · 97.7% recall · 0.8% FPR.

**Unseen malware families** is the number that matters. Every month from Jul 2025 to
Aug 2026, a model is trained on everything reported before that month. It is scored on
that month's malware *from families never seen before*, mixed with ~1,100 benign packages
(about 12 benign per malware, so precision is tested hard). The decision rule was chosen
on the 2025 months and is reported on the 2026 months:

| | Accuracy | Precision | Recall | FPR |
|---|---|---|---|---|
| Before (v1 features) | 97.3% | 82.8% | 71.4% | 1.0% |
| **Now (behaviour features + capability gate)** | **97.9%** | **94.6%** | 69.6% | **0.26%** |

On a balanced 50/50 mix the same rates give 99.6% precision and 84.7% accuracy.
Recall is the remaining gap; see *Generalising to unseen packages* below.

**Outside the dataset** means 247 live PyPI packages that appear nowhere in the data:
- **27 malware** reported to OSSF and still downloadable. The detector flags 9 (7 malicious,
  2 suspicious). This is by selection the malware that has evaded removal, and several of
  the rest show no payload in the version still published.
- **220 benign** (random and mid-popularity). The detector calls 1 malicious and 6
  suspicious. These labels are unverified.

### Generalising to unseen packages

The first model reached 96% on its test split but caught only 64% of malware from new
families. Error analysis on those misses found five causes, and each has a fix in
[`ml/behaviour.py`](ml/behaviour.py):

| Why new malware was missed | Fix |
|---|---|
| **Dilution:** package-wide counts average away one bad file. Recall fell from 94% (≤ 3 files) to 9% (30+ files) | per-file analysis, pooled by *max* (`worst_file_danger`, `n_danger_files`) |
| **Only top-level `setup.py` counted as "runs by itself"**: import-time droppers (`__init__.py`, `threading.Thread(target=…)`), install-command `run()` methods and DNS beacons looked like library code | **autorun reachability**: module-level code plus the local functions, thread/atexit targets and install hooks it reaches, per operation kind (`autorun_install_*`, `autorun_import_*`) |
| **No data flow:** `x = b64decode(…); exec(x)` was invisible | flow-insensitive taint per scope: decode/network/char-code → exec, process or file; download-then-execute; hidden endpoints; wallet secrets sent out |
| **Payloads outside Python:** compiled `__init__.so`, `.pth` start-up hooks, `ctypes.CDLL` of a bundled library | printable strings of binaries, orphan compiled modules, `.pth` code lines, native loads |
| **A 400-file cap** skipped the injected module in compromised 5,000-file libraries (`telnyx`, `litellm`) | a token pre-filter over *all* `.py` files for behaviour analysis |

Two modelling changes on top:
- **Monotone constraints.** A danger signal can only raise the score, so a large,
  well-documented package can't cancel "downloads and runs a binary on import".
- **A capability soft gate.** *Malicious* requires the classifier **and** a concrete
  capability, unless the classifier clears its strict 0.1%-false-alarm threshold. Most
  false alarms were tiny benign packages flagged on shape alone; the gate cut false
  alarms by 3–4×.

Tried and rejected, by the same dev-set rule:
- Synthetic trojanised packages: slightly worse on unseen families.
- Dropping shape features: worse.
- Recency weighting: no effect.

**Method.** Nothing was tuned on the numbers reported above:
- Configurations were chosen on the validation split plus 2025 future-family malware.
- The decision rule was chosen on the 2025 months, using a criterion fixed beforehand:
  the most accurate rule with precision ≥ 95%.
- The 2026 months and the live packages were used only for reporting.
- Evaluating with monthly retraining follows TESSERACT (Pendlebury et al., USENIX
  Security 2019). A random split hides concept drift, and a single frozen model
  understates what a regularly retrained scanner does.

<details><summary>Earlier results (v1 features, kept for the record)</summary>

70/15/15 split, v1 features: test accuracy 96.3%, precision 99.5%, recall 94.3%. Future
families: recall 63.8%. Live malware: 7 of 27 flagged. On the older 80/20, DataDog-only
dataset: PR-AUC 0.981, precision 0.996, recall 0.894.

</details>

### What the headline hides

**A label leak, found and fixed.** The first model ranked `pkg_n_files` top by 10×
gain. Part of that was a pipeline artifact: DataDog wraps every sample with a
`package_info-*.json` that no package fetched live from PyPI has. That file, the
benign-side `.meta.json`, and setuptools' `PKG-INFO`/`*.egg-info/` are now excluded
from shape counts. The size check in `train_gbdt.py` still fires afterwards, because
malicious packages really are smaller (median 5 files vs 11 for obscure).

**Size shapes recall, not false positives.** Broken down by file count on the test set:

| Malicious package size | Model A recall | Model B recall | A or B |
|---|---|---|---|
| < 4 files | 0.99 | 1.00 | 1.00 |
| 4–6 files | 0.95 | 0.94 | 0.97 |
| 7–10 files | 0.76 | 0.78 | 0.87 |
| 11+ files | **0.38** | **0.32** | 0.41 |

No obscure benign package under 7 files was flagged (0/142), so the model has not
learned "small means malicious." But a payload padded out with filler files escapes
both models most of the time. That is the main weakness of this detector.

**Name recognition helps popular packages.** For well-known projects the strongest
benign signal is often `pkg_typosquat_distance = 0` (the name *is* a popular name),
which flatters the `popular` slice. The `obscure` slice has no such help.

### What the chatbot shows: three tiers and a calibrated chance

Every analysis is scored twice, with no retraining involved:

1. **Classifier** — Model A's probability and SHAP evidence, as above.
2. **Code similarity** — the package is embedded with the same encoder as Model B
   and compared with all 9,656 embeddable training packages. The score is the
   distance-weighted share of malware among the 10 closest *distinct* families
   (near-copies at cosine ≥ 0.99 count once), and the UI shows the closest known
   malware file side by side with this package's most similar file. Collapsing
   duplicate families and weighting by distance cut false alarms at the 60% mark
   from 25 to 7 (of 1,448 held-out benign packages) at the same recall.

`ml/calibrate.py` fits a logistic calibrator on the held-out split and combines them:

| Tier | Rule | Held-out benign | Held-out malware |
|---|---|---|---|
| 🔴 Malicious | both methods flag it | **0** | 395 |
| 🟠 Suspicious — review the code | exactly one flags it, or together they reach ≥ 50% | 10 | 92 |
| 🟢 No threat found | neither | 1,438 | 34 |

The **chance it is really malware** is shown as a range, because the test set is
26% malware and real PyPI is not. The calibrated probability is re-weighted to a
1% prior (a package picked at random) and a 10% prior (one you already doubted).
Out of fold, predictions in the 80–95% band were malware 90.5% of the time.

**The calibration only covers malware that resembles the training data.** Of 7
packages reported to OSSF after the training snapshot and still on PyPI, 2 reach
the suspicious tier (`pullgetsage` via similarity, `websetup` via the classifier)
and 5 read as clean. The UI states this under every result.

### End-to-end smoke test

Run through the live API or the backend's own scorer:
- 12/12 benign packages cleared, including 4 random PyPI packages never seen in
  training; the closest call was `kerwin` at p=0.817 against a 0.85 threshold.
- 12/12 held-out malicious samples flagged (scored locally, since PyPI deletes
  malware once it's caught).
- The LLM explanation cited only features present in the evidence. It first misread
  absent features as present (a `0.0` next to "ships a README"). The prompt now
  states absence in words and labels close calls explicitly.

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
  features.py          static feature extractor  ← core
  behaviour.py         autorun reachability, data flow, tradecraft, non-Python payloads
  generalize.py        unseen-family experiments + monthly-retraining evaluation
  live.py              live PyPI packages outside the dataset, stored for re-scoring
  build_dataset.py     → features.parquet
  split.py             shared 70/15/15 grouped split + future holdout
  train_gbdt.py        Model A
  train_embed.py       Model B
  evaluate.py          head-to-head, three slices
  calibrate.py         tiers + calibrated chance (fit on validation) → models/calibration.json
  report.py            validation / test / future / live-PyPI performance report
  tests/
backend/
  fetch.py             PyPI resolution + safe unpack
  predict.py           scoring + SHAP evidence
  similarity.py        nearest known packages by code embedding
  assess.py            three-tier verdict + calibrated chance
  llm.py               Ollama; explains and answers follow-ups, never decides
  main.py              FastAPI: /analyze, /explain + /chat (SSE, POST), /health
  Dockerfile           multi-stage, CPU torch, encoder baked in, non-root
frontend/Dockerfile    node build → nginx-unprivileged (nginx/ holds the proxy + CSP)
ollama/                pinned Ollama image + pull-on-first-start entrypoint
docker-compose.yml     the three services; docker-compose.gpu.yml for NVIDIA
*.drawio               architecture, deployment, sequence, ML pipeline, routing
frontend/              React + Vite + TypeScript chat UI
  src/route.ts         package to analyse, or follow-up question?
```

## Limitations

- **Static analysis only.** No sandboxed execution, so behaviour that only manifests
  at runtime is invisible.
- **PyPI only.** The pipeline is structured to take npm next, but nothing here has
  been validated against JavaScript.
- **Novel obfuscation.** Model A can only see what someone wrote a feature for. A
  genuinely new packing scheme will evade it. Model B was meant to cover that gap but
  in practice misses the same padded packages Model A does.
- **Large packages are still harder.** Test recall is 99% for malware with ≤ 6 files, 80%
  for 11–30 files and 62% for 31+ (up from 38% for 11+ before the behaviour features).
- **New malware families: about 70% caught.** With monthly retraining, recall on
  never-seen families averages ~70% at 94.6% precision. Weak months are the ones dominated
  by compromised real libraries (Mar 2026: 47%) and by payload-less probes. Of 27 live
  OSSF-reported packages still on PyPI, 9 are flagged.
- **Retrain monthly.** These numbers assume it. A model frozen in June 2025 catches 66% of
  2026's new families; retrained monthly, 71% at the same threshold.
- **The explainer is a 4B model.** It stays within the evidence but can still word
  things clumsily (it once called a 2-file package's file count "unusually high").
- Trained on packages caught between roughly 2018 and 2026.
