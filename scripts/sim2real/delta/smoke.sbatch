#!/bin/bash
# Smoke test on one GH200 (<= 15 min). Submit from $K:  sbatch -A <ACCOUNT> smoke.sbatch
#SBATCH --job-name=s2r-smoke
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
OUT=$K/results/smoke-$SLURM_JOB_ID
mkdir -p $OUT
cd $K
{ hostname; nvidia-smi; nvidia-smi topo -m; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; numactl -s; } > $OUT/hardware.txt 2>&1
# vLLM sets up a 1-process torch.distributed group over TCP on the node IP; on DeltaAI GH200
# nodes that connection times out (10 min per replay). Loopback works.
export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
# Keep --max-model-len 131072 with a small capped KV cache (see vllm/worker/worker.py).
export APPTAINERENV_UNICACHE_ALLOW_SMALL_KV_CACHE=1
export APPTAINERENV_UNICACHE_POLICY_SANDBOX=${SANDBOX:-unshare}
export APPTAINERENV_SIM2REAL_AGENT_DIR=/bundle/agents
export APPTAINER_CACHEDIR=$K/apptainer-cache
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results $K/sif/unicache.sif"
$RUN bash -c "cd /opt/unicache && python -c 'import torch,vllm; print(torch.__version__, torch.cuda.get_device_name(0))'" | tee $OUT/import.txt
# 1) determinism: a ~2-min CodeParrot subset with LRU, twice; 2) an agent policy loads in the sandbox.
$RUN bash -c "cd /opt/unicache && python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 \
    --out /results/smoke-$SLURM_JOB_ID/gpu --workloads codeparrot@120 --blocks 512 --policies LRU --repeat 2 && \
  python scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 16 \
    --out /results/smoke-$SLURM_JOB_ID/gpu --workloads codeparrot@120 --blocks 512 --policies Agent"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after.txt
tar --exclude='*.pkl' --exclude='.cache' -czf $K/results/smoke-$SLURM_JOB_ID.tar.gz -C $K/results smoke-$SLURM_JOB_ID
