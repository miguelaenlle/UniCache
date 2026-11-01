#!/usr/bin/env bash
# Compare policies on the bundled sample, or a supplied prepared task YAML.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run ./setup-simulator.sh first.' >&2
  exit 1
fi
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=0
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export VLLM_CONFIG_ROOT="$PWD/.cache/vllm-config"
export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$HF_HOME"
exec .venv/bin/python scripts/local_sim.py compare --blocks 2000 "$@"
