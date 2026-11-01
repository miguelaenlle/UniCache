#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 VLLM_USE_V1=0 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
exec "${SIMULATOR_PYTHON:-$PWD/.venv-sim312/bin/python}" scripts/cache_budget_sweep.py "$@"
