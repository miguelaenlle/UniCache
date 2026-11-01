#!/usr/bin/env bash
# Separate Python 3.12 environment; preserve the existing Python 3.10 setup.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
uv venv --python 3.12 --allow-existing .venv-sim312
uv pip sync --python .venv-sim312/bin/python requirements/simulator-local.lock
VLLM_TARGET_DEVICE=empty uv pip install --python .venv-sim312/bin/python \
  --no-deps --no-build-isolation -e .
echo 'Ready. Run bash scripts/run-hit-ratio-matrix.sh --jobs 2'
echo 'The matrix uses existing local model metadata and prepared datasets.'
