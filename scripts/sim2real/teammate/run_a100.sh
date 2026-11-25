#!/bin/bash
# Sim2real GPU run on one NVIDIA A100 (x86 Linux), ~1 hour of GPU time.
#
#   PY=/path/to/python  bash run_a100.sh            # from the unpacked bundle directory
#
# PY must be a Python whose environment has the UniCache vLLM fork's compiled
# libraries and torch 2.6 (the environment used for the earlier sim2real runs).
# See README-A100.md. Results land in ./results-a100.tar.gz; send that file back.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:?set PY to the python of the environment with the compiled UniCache fork}
GPU=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export SIM2REAL_AGENT_DIR=$PWD/agents
# Agent-written policies run in a separate process with no network.
export UNICACHE_POLICY_SANDBOX=${UNICACHE_POLICY_SANDBOX:-unshare}
OUT=$PWD/results-a100
mkdir -p $OUT
cd unicache
mkdir -p TaskData && ln -sfn ../../subsets TaskData/sim2real
{ hostname; date; nvidia-smi; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; lscpu | head -20; } > $OUT/hardware.txt 2>&1
$PY -c "import torch, vllm; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))" | tee $OUT/import.txt
$PY -c "from policy_agent.bridge import sandbox_command; sandbox_command('')" \
  || { echo "Policy sandbox unavailable: set UNICACHE_POLICY_SANDBOX=bwrap, or ask Miguel"; exit 1; }
save() { tar --exclude='*.pkl' --exclude='.cache' -czf ../results-a100.tar.gz -C .. results-a100; }
gpu() { $PY scripts/sim2real/run_sim.py --mode gpu --jobs 1 --block-size 8 --out $OUT/gpu "$@"; }

# 1. Smoke: a 2-minute CodeParrot subset with LRU, twice (must agree), ~6 min.
gpu --workloads codeparrot@120 --blocks 1024 --policies LRU --repeat 2
save
# 2. ALFRED at the pre-registered 8-token-block capacity: each policy once, then LRU x2 more.
for POL in LRU UniCache LeCaR Agent; do
  gpu --workloads alfred --blocks 1024 --policies $POL || echo "FAILED $POL"
  save
done
gpu --workloads alfred --blocks 1024 --policies LRU --repeat 3 || echo "FAILED LRU repeats"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $OUT/clocks-after.txt
save
echo "DONE: send $(cd .. && pwd)/results-a100.tar.gz to Miguel"
