# packeges-chatbot — Malicious PyPI Package Detector

A machine-learning classifier that decides whether a PyPI package is **malicious**, with a
confidence score, wrapped in a React chatbot where a **local LLM explains the model's
reasoning** in plain English.

> **The LLM never classifies.** The trained model produces the verdict and a ranked list
> of the features that drove it; the LLM's only job is to turn that structured evidence
> into readable prose. This is an ML project with an LLM presentation layer — not an
> "ask an LLM if it's malware" wrapper.

## Status

🚧 Under construction. See [the build plan](#roadmap).

## Architecture

```
ml/         data acquisition, feature engineering, model training & evaluation
backend/    FastAPI — resolves a package, scores it, streams an LLM explanation
frontend/   React + Vite + TypeScript chat UI
data/       downloaded packages (gitignored — no malware is ever committed)
```

## Data sources

| Source | Role | License |
|---|---|---|
| [DataDog/malicious-software-packages-dataset](https://github.com/DataDog/malicious-software-packages-dataset) | Malicious class — ~6.7k real PyPI samples caught in the wild, human-vetted | Apache-2.0 |
| [top-pypi-packages](https://hugovk.github.io/top-pypi-packages/) + PyPI JSON API | Benign class — popular packages | public data |
| PyPI simple index (random sample) | **Hard negatives** — small, obscure, amateur, but benign | public data |
| [ossf/malicious-packages](https://github.com/ossf/malicious-packages) | Cross-check, to keep known-bad packages out of the benign set | Apache-2.0 |

## Models

- **Model A — LightGBM over ~70 hand-engineered static-analysis features.** Production
  model. Fast on CPU, and SHAP gives the per-feature attribution the LLM narrates.
- **Model B — code embeddings + classifier head.** Benchmarked head-to-head against A.

See `ml/reports/comparison.md` once training has run.

## Safety

Packages are only ever unpacked and parsed as text/AST. `setup.py` is **never executed**.
Archive extraction is guarded against path traversal and capped on size and time.

## Roadmap

- [x] Phase 0 — scaffold, repo, local LLM
- [ ] Phase 1 — data acquisition
- [ ] Phase 2 — feature engineering & both models
- [ ] Phase 3 — FastAPI backend + Ollama explanations
- [ ] Phase 4 — React chat UI
- [ ] Phase 5 — evaluation write-up

## License

Apache-2.0
