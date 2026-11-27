#!/bin/bash
# One trace on one GH200: a list of policy runs, in order.
#   sbatch -A <ACCOUNT> --time=00:45:00 run_trace.sbatch <workload> <blocks> [runs]
# runs: comma-separated policies; "LRUx3" means LRU until 3 repeats exist.
#   default "LRU,UniCache,LeCaR,Agent,LRUx3" (6 runs). Split a trace into jobs of
#   <= 60 min, e.g. "LRU,UniCache,LeCaR" then "Agent,LRUx3": on DeltaAI's interactive
#   partition, our group's 1-GPU jobs of <= 60 min start in a median of 0 min (Nov 2026).
# Each run is skipped if already done, so a resubmitted job resumes.
#SBATCH --job-name=s2r-trace
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
W=$1; BLOCKS=$2; RUNS=${3:-LRU,UniCache,LeCaR,Agent,LRUx3}
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
OUT=$K/results/gpu/$W
mkdir -p $OUT
cd $K
{ hostname; date; nvidia-smi; nvidia-smi topo -m; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; numactl -s; } > $OUT/hardware-$SLURM_JOB_ID.txt 2>&1
# vLLM sets up a 1-process torch.distributed group over TCP on the node IP; on DeltaAI GH200
# nodes that connection times out (10 min per replay). Loopback works.
export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
# Keep --max-model-len 131072 with a small capped KV cache (see vllm/worker/worker.py).
export APPTAINERENV_UNICACHE_ALLOW_SMALL_KV_CACHE=1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_AGENT_DIR=/bundle/agents
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results $K/sif/unicache.sif"
PY="cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 --out /results/gpu/$W --workloads $W --blocks $BLOCKS"
save() { tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/gpu-$W.tar.gz -C $K/results/gpu $W; }
for R in ${RUNS//,/ }; do
  if [[ $R == *x* ]]; then POL=${R%x*}; REP=${R##*x}; else POL=$R; REP=1; fi
  echo "$(date '+%T') run $POL repeat=$REP"
  $RUN bash -c "$PY --policies $POL --repeat $REP" || echo "FAILED $R"
  save
done
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after-$SLURM_JOB_ID.txt
save
echo "$(date '+%T') done"
