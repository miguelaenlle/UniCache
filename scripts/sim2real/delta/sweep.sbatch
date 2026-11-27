#!/bin/bash
# Synthetic calibration sweep on one GH200 (scripts/sim2real/make_sweep.py), one part per job so each
# fits in <= 15 min (short jobs start much sooner at our priority). The timing model is fitted on
# these runs only; the real traces are held-out validation.
#   cd $K && sbatch -A <ACCOUNT> bundle/sweep.sbatch <prefill|decode|load>
#SBATCH --job-name=s2r-sweep
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
cd $K
OUT=$K/results/gpu-sweep
mkdir -p $OUT
{ hostname; date; nvidia-smi; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; } > $OUT/hardware-$SLURM_JOB_ID.txt 2>&1
export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
export APPTAINERENV_UNICACHE_ALLOW_SMALL_KV_CACHE=1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_AGENT_DIR=/bundle/agents-sweep
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/sim2real:/opt/unicache/scripts/sim2real --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results $K/sif/unicache.sif"
PY="cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 --out /results/gpu-sweep"
run() { echo "$(date '+%T') $*"; $RUN bash -c "$PY $*" || echo "FAILED $*"; }
case "${1:?part: prefill|decode|load}" in
  prefill)
    run --blocks 16384 --policies LRU --workloads sweep-pre128-s0 sweep-pre512-s0 sweep-pre2048-s0 sweep-pre8192-s0 sweep-pre2048-s50 sweep-pre2048-s90
    run --blocks 512 --policies LRU Agent --workloads sweep-evict ;;
  decode)
    run --blocks 16384 --policies LRU --workloads sweep-dec1-c256 sweep-dec1-c2048 sweep-dec8-c256 sweep-dec8-c2048 sweep-dec16-c256 sweep-dec16-c2048 sweep-dec32-c256 sweep-dec32-c2048 ;;
  load)
    run --blocks 16384 --policies LRU --workloads sweep-rps2 sweep-rps5 sweep-rps10 sweep-rps20 sweep-rps40 ;;
esac
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after-$SLURM_JOB_ID.txt
tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/gpu-sweep-$1-$SLURM_JOB_ID.tar.gz -C $K/results gpu-sweep
echo "$(date '+%T') done"
