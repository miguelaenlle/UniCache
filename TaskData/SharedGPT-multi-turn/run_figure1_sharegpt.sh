#!/usr/bin/env bash
# ShareGPT-only comparison of the four policies in Figure 1.
# Based on this directory's upstream run.sh and tasks_config.yaml.
# The release does not specify the exact Figure 1 experiment configuration.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="$PWD/.venv/bin/python"
MODEL="$PWD/models/Llama-3.2-1B-Instruct"
CONFIG="$PWD/TaskData/SharedGPT-multi-turn/tasks_config.yaml"

# Upstream run.sh: 50,000 blocks x 8 tokens = 400,000-token capacity.
# Override GPU_BLOCKS to study another cache capacity; no figure fit is implied.
GPU_BLOCKS="${GPU_BLOCKS:-50000}"
SIMULATOR_TTFT="${SIMULATOR_TTFT:-3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)}"
SIMULATOR_TPOT="${SIMULATOR_TPOT:-0.01}"
export VLLM_USE_V1=0 PYTHONHASHSEED=0 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export VLLM_CONFIG_ROOT="$PWD/.cache/vllm-config"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p "$MPLCONFIGDIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"

[[ -x "$PYTHON" ]] || { echo 'Missing .venv; run setup-simulator.sh.' >&2; exit 1; }
[[ -f "$MODEL/config.json" && -f "$MODEL/tokenizer.json" ]] || {
  echo "Missing model configuration/tokenizer: $MODEL" >&2; exit 1;
}
"$PYTHON" - "$CONFIG" "$GPU_BLOCKS" <<'PY'
from pathlib import Path
import sys
import yaml
assert int(sys.argv[2]) > 0, 'GPU_BLOCKS must be positive'
config = yaml.safe_load(Path(sys.argv[1]).read_text())
for task in config['tasks']:
    if not Path(task['data_path']).is_file():
        sys.exit(f"Missing dataset: {task['data_path']}")
    print(f"Task: {task['name']}; session limit: {task['max_record']}")
PY

OUT="$PWD/result/figure1-sharegpt/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"
cp "$CONFIG" "$OUT/tasks_config.yaml"
{
  echo 'ShareGPT-only simulator comparison; exact Figure 1 configuration unverified.'
  echo "Upstream commit: $(git rev-parse HEAD)"
  echo "Model: $MODEL"
  echo "GPU blocks: $GPU_BLOCKS; block size: 8; token capacity: $((GPU_BLOCKS * 8))"
  echo 'Uses upstream ShareGPT task configuration unchanged (first 1,000 sessions).'
  echo "Prefill seconds: $SIMULATOR_TTFT; decode seconds/token: $SIMULATOR_TPOT"
  echo 'Max sequences: 32; max length: 131072.'
} > "$OUT/settings.txt"
echo "Results directory: $OUT"

for policy in LRU LFU FIFO WA; do
  mkdir -p "$OUT/$policy"
  echo "Running $policy ..."
  "$PYTHON" -u run_simulation.py \
    --model "$MODEL" \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --block-size 8 \
    --enable-prefix-caching \
    --simulator-enable-virtual-time \
    --simulator-ttft "$SIMULATOR_TTFT" \
    --simulator-tpot "$SIMULATOR_TPOT" \
    --device cpu \
    --simulator-num-gpu-blocks "$GPU_BLOCKS" \
    --simulator-num-cpu-blocks 50000 \
    --eviction-policy "$policy" \
    --simulator-output-dir "$OUT/$policy" \
    --simulator-task-config "$CONFIG" \
    --log-level WARNING \
    2>&1 | tee "$OUT/$policy/run.log"

  # Write the comparison after each successful policy, preserving partial work.
  "$PYTHON" - "$OUT" <<'PY'
import csv
import json
from pathlib import Path
import sys
out = Path(sys.argv[1])
rows = []
for policy in ['LRU', 'LFU', 'FIFO', 'WA']:
    for path in sorted((out / policy).glob('*/summary.json')):
        summary = json.loads(path.read_text())
        hit = summary['hit_rate_summary']
        rows.append({'policy': policy, 'hit_ratio': hit['global_hit_rate'],
                     'requests': hit['total_requests'],
                     'hit_tokens': hit['global_hit_tokens_num'],
                     'prompt_tokens': hit['global_tokens'],
                     'runtime_seconds': summary['simulation_time']})
with (out / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['policy', 'hit_ratio', 'requests',
        'hit_tokens', 'prompt_tokens', 'runtime_seconds'])
    writer.writeheader()
    writer.writerows(rows)
print('\nCompleted results:')
for row in rows:
    print(f"  {row['policy']:4s}  {row['hit_ratio']:.4%}")
PY
done
echo "Finished. Comparison CSV: $OUT/comparison.csv"
