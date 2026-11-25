#!/bin/bash
# One trace, all Tier-1 policies, on one GH200.
#   sbatch -A <ACCOUNT> --time=01:30:00 run_trace.sbatch <workload> <blocks>
#SBATCH --job-name=s2r-trace
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
W=$1; BLOCKS=$2
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
OUT=$K/results/gpu/$W
mkdir -p $OUT
cd $K
{ hostname; nvidia-smi; nvidia-smi topo -m; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; numactl -s; } > $OUT/hardware-$SLURM_JOB_ID.txt 2>&1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_AGENT_DIR=/bundle/agents
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results $K/sif/unicache.sif"
PY="cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 --out /results/gpu/$W --workloads $W --blocks $BLOCKS"
# Each run is skipped if already done, so a resubmitted job resumes.
for POL in LRU UniCache LeCaR Agent; do
  $RUN bash -c "$PY --policies $POL" || echo "FAILED $POL"
  tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/gpu-$W.tar.gz -C $K/results/gpu $W
done
$RUN bash -c "$PY --policies LRU --repeat 3" || echo "FAILED LRU repeats"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after-$SLURM_JOB_ID.txt
tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/gpu-$W.tar.gz -C $K/results/gpu $W
