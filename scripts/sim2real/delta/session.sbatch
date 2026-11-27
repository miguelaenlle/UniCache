#!/bin/bash
# Job 1 of the overnight run: smoke test, then (only if it passes) ALFRED, in one allocation.
# If the smoke test fails, hold the GPU for at most 20 min so a fix can be tried in this same
# allocation (srun --jobid=$SLURM_JOB_ID --overlap ...), then exit. Nothing else runs.
#   cd $K && sbatch -A <ACCOUNT> bundle/session.sbatch
#SBATCH --job-name=s2r-session
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
cd $K
echo "$(date '+%T') session start on $(hostname)"
bash $K/bundle/smoke.sbatch
CSV=$K/results/smoke-$SLURM_JOB_ID/gpu/runs.csv
ok=$(python3 - "$CSV" <<'EOF'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
except OSError:
    rows = []
good = [r for r in rows if r.get("status") == "0"]
lru = sorted(float(r["hit_rate"]) for r in good if r["policy"] == "LRU")
print("yes" if len(good) == 3 else "no", "LRU", lru, "Agent", [r["hit_rate"] for r in good if r["policy"] == "Agent"])
EOF
)
echo "$(date '+%T') SMOKE_RESULT $ok"
if [[ $ok == yes* ]]; then
  bash $K/bundle/run_trace.sbatch alfred 512 LRU,UniCache,LeCaR,Agent,LRUx3
  echo "$(date '+%T') session done"
else
  echo "$(date '+%T') SMOKE_FAILED: holding the GPU for up to 20 min for a fix"
  sleep 1200
fi
