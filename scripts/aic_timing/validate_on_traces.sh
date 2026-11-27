#!/bin/bash
# Held-out validation of a timing calibration on the real-trace subsets (none used in a sweep fit).
# Runs the simulator one replay at a time (the policy cost uses the worker's CPU time).
#   bash scripts/aic_timing/validate_on_traces.sh calibrations/<cal>.json result/sim2real-transfer/check-<name>
set -euo pipefail
CAL=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
O=$2
cd "$(dirname "$0")/../.."
mkdir -p "$O"
export SIM2REAL_STEP_LOG=1
export SIM2REAL_EXTRA_FLAGS="--simulator-timing aic --simulator-aic-system h100_sxm --simulator-timing-calibration $CAL"
PY=".venv-sim312/bin/python scripts/sim2real/run_sim.py --mode sim --jobs 1 --block-size 16"
$PY --out "$O/light" --workloads alfred codeparrot@120 --blocks 512 --policies LRU Agent > "$O/light.log" 2>&1
$PY --out "$O/x1" --workloads codeparrot@600 --blocks 4096 --policies LRU > "$O/x1.log" 2>&1
for sc in 0.05 0.03; do
  SIM2REAL_ARRIVAL_SCALE=$sc $PY --out "$O/x$sc" --workloads codeparrot@600 --blocks 4096 --policies LRU UniCache > "$O/x$sc.log" 2>&1
done
echo "done: $O"
