#!/bin/bash
# A2 main grid (UniCache paper Table 5 repro): all policies + NONE per Hybrid trace at a given load,
# then refine OPT (aic timing makes the offline map stale) and rebuild the table.
# Usage: bash scripts/paper_repro/a2_grid.sh gh200-1b|a100-3b Trace#N <load> [tag]
# Loads come from knee_sweep.sh: k* = smallest load where LRU QTTFT > 3x its lowest-load value.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=$1; trace=$2; load=$3; tag=${4:-x$load}
case $cfg in
  gh200-1b) flags=(--model models/Llama-3.2-1B-Instruct --block-size 16 --blocks 33177
                   --timing aic --system h100_sxm --calibration calibrations/gh200-llama-3.2-1b-v8.json) ;;
  a100-3b)  flags=(--model models/Llama-3.2-3B-Instruct --block-size 8 --blocks 66355
                   --timing aic --system a100_sxm) ;;
  *) echo "unknown config $cfg"; exit 1 ;;
esac
out=result/paper-repro/a2/$cfg/$tag
mkdir -p "$out"
log="$out/${trace#Trace}.log"
py=.venv-sim312/bin/python
echo "$(date +%T) $cfg $trace load x$load -> $out"
$py scripts/paper_qttft/run_table5.py --out "$out" --load "$load" --traces "$trace" --jobs 3 "${flags[@]}" > "$log" 2>&1
PYTHONHASHSEED=0 $py scripts/paper_qttft/refine_opt.py --opt-dir "$out/$trace/OPT" --iterations 6 >> "$log" 2>&1
$py scripts/paper_qttft/run_table5.py --out "$out" --load "$load" --traces "$trace" --table-only "${flags[@]}" >> "$log" 2>&1
n=${trace#Trace#}
cp "$out/table5.md" "$out/table5-T$n.md"; cp "$out/table5.csv" "$out/table5-T$n.csv"
grep -E "^\| $trace " "$out/table5-T$n.md" || true
