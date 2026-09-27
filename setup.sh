#!/usr/bin/env bash
# One-shot environment setup. Requires `uv` (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Creating Python 3.12 venv (torch/lightgbm/shap have no 3.14 wheels yet)"
uv venv --python 3.12 venv

export VIRTUAL_ENV="$PWD/venv"

echo "==> Installing CPU-only torch from the PyTorch index"
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

echo "==> Installing the rest from PyPI"
uv pip install -r requirements.txt

echo "==> Done. Activate with: source venv/bin/activate"
