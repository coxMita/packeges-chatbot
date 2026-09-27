# syntax=docker/dockerfile:1.7
#
# Local LLM that writes the explanations. The official image, pinned, run as a
# non-root user, with an entrypoint that pulls the configured model on first
# start (into the models volume) so `docker compose up` is the only step.

ARG OLLAMA_VERSION=0.34.2
FROM ollama/ollama:${OLLAMA_VERSION}

RUN groupadd --system --gid 10002 ollama \
 && useradd --system --uid 10002 --gid ollama --create-home --home-dir /home/ollama \
            --shell /usr/sbin/nologin ollama \
 && mkdir -p /models \
 && chown ollama:ollama /models

COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh

ENV HOME=/home/ollama \
    OLLAMA_MODELS=/models \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODEL=qwen3.5:4b \
    OLLAMA_KEEP_ALIVE=30m

USER ollama
EXPOSE 11434

# Healthy only once the model is pulled, so "healthy" means "can explain".
HEALTHCHECK --interval=15s --timeout=10s --start-period=30m --retries=5 \
    CMD ["/bin/sh", "-c", "test -f /tmp/model-ready && ollama list >/dev/null"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
