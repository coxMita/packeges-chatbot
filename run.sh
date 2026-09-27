#!/usr/bin/env bash
# Start the backend and frontend together.
#
#   ./run.sh          both (backend :8000, frontend :5173)
#   ./run.sh backend  backend only
#   ./run.sh frontend frontend only
set -euo pipefail
cd "$(dirname "$0")"

PY=./venv/bin/python

start_backend() {
  echo "==> backend on http://127.0.0.1:8000"
  "$PY" -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 "$@"
}

start_frontend() {
  echo "==> frontend on http://localhost:5173"
  (cd frontend && npm run dev)
}

case "${1:-both}" in
  backend)  start_backend ;;
  frontend) start_frontend ;;
  both)
    start_backend &
    BACKEND_PID=$!
    trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT INT TERM
    start_frontend
    ;;
  *) echo "usage: $0 [both|backend|frontend]" >&2; exit 1 ;;
esac
