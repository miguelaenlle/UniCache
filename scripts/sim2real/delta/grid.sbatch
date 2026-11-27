#!/bin/bash
# Controlled step-timing grid on one GH200 (prefill and decode, batch 1-32) for the timing
# calibration (scripts/aic_timing/fit_calibration.py). Dummy weights, V0 engine.
#   cd $K && sbatch -A <ACCOUNT> bundle/grid.sbatch
#SBATCH --job-name=s2r-grid
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
OUT=$K/results/grid-$SLURM_JOB_ID
mkdir -p $OUT
cd $K
{ hostname; date; nvidia-smi; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; } > $OUT/hardware.txt 2>&1
export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
RUN="apptainer exec --nv --bind $K/bundle:/bundle --bind $OUT:/out $K/sif/unicache.sif"
$RUN bash -c "cd /opt/unicache && PYTHONPATH=/opt/unicache python /bundle/profile_steps_grid.py --model /opt/unicache/models/Llama-3.2-1B-Instruct --load-format dummy --block-size 16 --max-num-seqs 32 --max-num-batched-tokens 131072 --out /out/grid_steps.csv"
