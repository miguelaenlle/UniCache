#!/bin/bash
# TTFT/ITL baselines under load for the timing calibration (step 1 of the TTFT work).
# CodeParrot-600, 4,096 blocks, 16-token blocks; at 20x and 33x the recorded arrival rate:
# LRU x3 and UniCache x2 per load, all on this one node (10 replays, about 20 min).
#   cd $K && sbatch -A <ACCOUNT> bundle/ttft_baseline.sbatch
#SBATCH --job-name=s2r-ttft
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
cd $K
OUT=$K/results/gpu-load
mkdir -p $OUT
{ hostname; date; nvidia-smi; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; } > $OUT/hardware-$SLURM_JOB_ID.txt 2>&1
export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
export APPTAINERENV_UNICACHE_ALLOW_SMALL_KV_CACHE=1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_AGENT_DIR=/bundle/agents
# The image's scripts/sim2real predates SIM2REAL_ARRIVAL_SCALE; mount the updated copy over it.
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/sim2real:/opt/unicache/scripts/sim2real --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results $K/sif/unicache.sif"
for SC in 0.05 0.03; do
  export APPTAINERENV_SIM2REAL_ARRIVAL_SCALE=$SC
  PY="cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 --out /results/gpu-load/x$SC --workloads codeparrot@600 --blocks 4096"
  echo "$(date '+%T') load x$SC: LRU x3, UniCache x2"
  $RUN bash -c "$PY --policies LRU UniCache --repeat 2" || echo "FAILED x$SC repeat-2 set"
  $RUN bash -c "$PY --policies LRU --repeat 3" || echo "FAILED x$SC LRU rep 3"
done
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after-$SLURM_JOB_ID.txt
tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/gpu-load-$SLURM_JOB_ID.tar.gz -C $K/results gpu-load
echo "$(date '+%T') done"
