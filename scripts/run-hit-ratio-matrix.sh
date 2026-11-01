#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export VLLM_CONFIG_ROOT="$PWD/.cache/vllm-config"
mkdir -p "$MPLCONFIGDIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"
exec .venv/bin/python scripts/hit_ratio_matrix.py "$@"
