#!/bin/bash
# LRU-only load sweep per Hybrid trace, to find each trace's knee k* (UniCache paper Table 5 repro).
# Usage: bash scripts/paper_repro/knee_sweep.sh gh200-1b|a100-3b "<loads>"
# Load k time-compresses the trace (session starts x k, think times / k); see run_table5.py --load.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=$1; loads=$2
case $cfg in
  gh200-1b) flags=(--model models/Llama-3.2-1B-Instruct --block-size 16 --blocks 33177
                   --timing aic --system h100_sxm --calibration calibrations/gh200-llama-3.2-1b-v8.json) ;;
  a100-3b)  flags=(--model models/Llama-3.2-3B-Instruct --block-size 8 --blocks 66355
                   --timing aic --system a100_sxm) ;;
  *) echo "unknown config $cfg"; exit 1 ;;
esac
for k in $loads; do
  out=result/paper-repro/a2/$cfg/knee/x$k
  [ -f "$out/table5.csv" ] && { echo "skip $out"; continue; }
  echo "$(date +%T) $cfg load x$k"
  .venv-sim312/bin/python scripts/paper_qttft/run_table5.py --out "$out" --load "$k" \
    --policies LRU --jobs 3 "${flags[@]}" > "$out.log" 2>&1 || { mkdir -p "$out"; echo "FAILED $out (see $out.log)"; }
  grep -E "^\| Trace#[0-9] \| LRU" "$out/table5.md" 2>/dev/null || true
done
