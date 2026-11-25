#!/bin/bash
# Downscoped first GPU job: smoke checks + one trace (ALFRED), one GH200, ~1 h.
#   sbatch -A <ACCOUNT> first_trace.sbatch <workload> <blocks>      e.g. alfred <C*>
#SBATCH --job-name=s2r-first
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
W=$1; BLOCKS=$2
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
OUT=$K/results/first-$SLURM_JOB_ID
mkdir -p $OUT
cd $K
{ hostname; date; nvidia-smi; nvidia-smi topo -m; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; numactl -s; } > $OUT/hardware.txt 2>&1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_AGENT_DIR=/bundle/agents
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $OUT:/out $K/sif/unicache.sif"
save() { tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/first-$SLURM_JOB_ID.tar.gz -C $K/results first-$SLURM_JOB_ID; }
gpu() { $RUN bash -c "cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 --out /out/gpu $*"; }

# 1. Smoke: imports, then a 2-min CodeParrot subset twice (determinism). Abort on failure.
$RUN bash -c "cd /opt/unicache && python -c 'import torch,vllm; print(torch.__version__, torch.cuda.get_device_name(0))'" | tee $OUT/import.txt
gpu --workloads codeparrot@120 --blocks 512 --policies LRU --repeat 2
save
# 2. The trace: each policy once (Agent exercises the sandbox), then LRU twice more for noise.
for POL in LRU UniCache LeCaR Agent; do
  gpu --workloads $W --blocks $BLOCKS --policies $POL || echo "FAILED $POL"
  save
done
gpu --workloads $W --blocks $BLOCKS --policies LRU --repeat 3 || echo "FAILED LRU repeats"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after.txt
save
echo DONE
