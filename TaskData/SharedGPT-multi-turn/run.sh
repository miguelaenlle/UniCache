#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=0
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export VLLM_CONFIG_ROOT="$PWD/.cache/vllm-config"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$MPLCONFIGDIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"
SIMULATOR_TTFT="${SIMULATOR_TTFT:-3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)}"
SIMULATOR_TPOT="${SIMULATOR_TPOT:-0.01}"

# Preserve the authors' original block size and simulated cache capacity.
.venv/bin/python run_simulation.py \
    --model "$PWD/models/Llama-3.2-1B-Instruct" \
    --max-num-seqs 32 \
    --max-model-len 131072   \
    --block-size 8 \
    --enable-prefix-caching \
    --simulator-enable-virtual-time \
    --simulator-ttft "$SIMULATOR_TTFT" \
    --simulator-tpot "$SIMULATOR_TPOT" \
    --device cpu \
    --simulator-num-gpu-blocks 50000  \
    --simulator-num-cpu-blocks 50000  \
    --eviction-policy "FIFO" \
    --simulator-output-dir  result/SharedGPT-multi-turn/ \
    --simulator-task-config TaskData/SharedGPT-multi-turn/tasks_config.yaml
