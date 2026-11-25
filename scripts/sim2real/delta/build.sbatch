#!/bin/bash
# Build the UniCache container on a GH200 node (login nodes cap users at 16 CPUs / 20 GiB).
# One GPU's share of a node: 72 of 288 cores, ~1/4 of memory. The GPU itself is idle.
#   cd $K/bundle && sbatch -A <ACCOUNT> build.sbatch
#SBATCH --job-name=s2r-build
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcjw/maenlle2/kvcal/logs/%x-%j.out
set -euo pipefail
K=/work/hdd/bcjw/maenlle2/kvcal
export APPTAINER_CACHEDIR=$K/apptainer-cache
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/s2r-build-$SLURM_JOB_ID
mkdir -p $APPTAINER_TMPDIR
cd $K/bundle
# ~3 GB peak per CUDA compile job: 24 jobs fit in 110 GB.
sed 's/MAX_JOBS=${MAX_JOBS:-16}/MAX_JOBS=24/' unicache.def > unicache-job.def
date; nproc; free -g | head -2
apptainer build --fakeroot $K/sif/unicache.sif.tmp unicache-job.def
mv $K/sif/unicache.sif.tmp $K/sif/unicache.sif
apptainer test $K/sif/unicache.sif
rm -rf $APPTAINER_TMPDIR
date; echo BUILD_OK
