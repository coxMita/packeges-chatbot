# syntax=docker/dockerfile:1.7
#
# API server: FastAPI + LightGBM + SHAP + the CodeBERTa encoder for similarity.
# Build from the repo root (it needs both backend/ and ml/):
#     docker build -f backend/Dockerfile -t packeges-chatbot-backend .
#
# Trained artifacts (ml/models) and the dataset (data/) are NOT baked in: they
# are produced locally, are gitignored, and data/ holds live malware samples.
# docker-compose.yml mounts both read-only.

ARG PYTHON_VERSION=3.12

# ---- build: virtualenv with CPU-only torch, plus the encoder weights ---------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch from its own index (the default PyPI wheel drags in ~2.5GB of CUDA),
# kept separate from the PyPI install so no package can be resolved from the
# wrong index.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY backend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Bake the code encoder into the image so the container runs offline and a
# first request never waits on a download. Pinned to the revision the
# similarity index was embedded with: another revision means another vector
# space. (This repo ships pytorch_model.bin only, no safetensors.)
ARG ENCODER_REVISION=e93b5898cff07f03f1c1c09cde284d1b85962363
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('huggingface/CodeBERTa-small-v1', revision='${ENCODER_REVISION}', \
allow_patterns=['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'pytorch_model.bin'])" \
 && mkdir -p /opt/hf/hub/models--huggingface--CodeBERTa-small-v1/refs \
 && printf '%s' "${ENCODER_REVISION}" > /opt/hf/hub/models--huggingface--CodeBERTa-small-v1/refs/main \
 && rm -rf /opt/hf/hub/.locks

# ---- runtime ------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# libgomp: OpenMP runtime LightGBM links against. curl-free on purpose: the
# healthcheck uses Python's stdlib.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home \
            --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

WORKDIR /app
COPY backend/*.py backend/
COPY ml/*.py ml/
RUN mkdir -p ml/models data \
 && python -m compileall -q backend ml

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=qwen3.5:4b

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8)"]

# One worker: the models are loaded per process and similarity search is
# CPU-bound and serialised anyway. Scale with replicas, not workers.
CMD ["uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
