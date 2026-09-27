#!/bin/sh
# Start the Ollama server, pull $OLLAMA_MODEL if the volume does not have it
# yet, then hand the foreground to the server process.
set -eu

ollama serve &
server=$!
trap 'kill -TERM "$server" 2>/dev/null' TERM INT

i=0
until ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 60 ]; then
        echo "[entrypoint] ollama did not start within 60s" >&2
        exit 1
    fi
    sleep 1
done

if ollama show "$OLLAMA_MODEL" >/dev/null 2>&1; then
    echo "[entrypoint] $OLLAMA_MODEL already present"
else
    echo "[entrypoint] pulling $OLLAMA_MODEL (first start only; several GB)"
    ollama pull "$OLLAMA_MODEL"
fi
touch /tmp/model-ready

wait "$server"
