#!/bin/bash
# Step 3 GPU validation (eval slice): the TTFT agent's top 3 submissions vs the report's 5 heuristics, at 25x load
# (arrival scale 0.04), CodeParrot-600, 4,096 blocks, all on this one node.
#   bundle/agents-ttft/c1..c3/codeparrot.py  = top 3 submissions by tune TTFT (c1 = selected)
#   cd $K && sbatch -A <ACCOUNT> bundle/step3_validate.sbatch <A|B>   (two <= 15-min halves; each has LRU for
#   within-half normalization if the halves land on different nodes)
# Half A: LRU, c1, LFU, UniCache, c2, LRU.  Half B: FIFO, c1, LeCaR, c3, UniCache, LRU.
#SBATCH --job-name=s2r-step3
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
HALF=${1:?half A|B}
OUT=$K/results/gpu-step3
mkdir -p $OUT
{ hostname; date; nvidia-smi; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; } > $OUT/hardware-$HALF-$SLURM_JOB_ID.txt 2>&1
export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
export APPTAINERENV_UNICACHE_ALLOW_SMALL_KV_CACHE=1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_ARRIVAL_SCALE=0.04
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/sim2real:/opt/unicache/scripts/sim2real --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results $K/sif/unicache.sif"
replay() {  # replay <out-subdir> <policy> <repeat> [agent dir]
  local sub=$1 pol=$2 rep=$3
  export APPTAINERENV_SIM2REAL_AGENT_DIR=${4:-/bundle/agents}
  echo "$(date '+%T') $sub $pol rep<=$rep"
  $RUN bash -c "cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 --out /results/gpu-step3/$sub --workloads codeparrot@600 --blocks 4096 --policies $pol --repeat $rep" || echo "FAILED $sub $pol"
}
if [ "$HALF" = A ]; then
  replay A LRU 1
  replay A-c1 Agent 1 /bundle/agents-ttft/c1
  replay A LFU 1
  replay A UniCache 1
  replay A-c2 Agent 1 /bundle/agents-ttft/c2
  replay A LRU 2
else
  replay B FIFO 1
  replay B-c1 Agent 1 /bundle/agents-ttft/c1
  replay B LeCaR 1
  replay B-c3 Agent 1 /bundle/agents-ttft/c3
  replay B UniCache 1
  replay B LRU 1
fi
tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/gpu-step3-$HALF-$SLURM_JOB_ID.tar.gz -C $K/results gpu-step3
echo "$(date '+%T') done"
