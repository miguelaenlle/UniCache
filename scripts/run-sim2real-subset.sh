#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 VLLM_USE_V1=0 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib" MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"
exec "${SIMULATOR_PYTHON:-$PWD/.venv-sim312/bin/python}" scripts/sim2real_subset.py "$@"
