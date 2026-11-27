#!/bin/bash
# Run on the DeltaAI login node (no Slurm job, no GPU charge):
#   setsid nohup bash build_lite_then_smoke.sh > $K/logs/build-lite.log 2>&1 < /dev/null &
# Builds the trimmed image with 4 compile jobs (login-node cap: 16 CPUs, 16 GiB soft memory),
# points sif/unicache.sif at it, then submits the 15-min smoke test as a small 1-GPU job.
set -uo pipefail
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
A=${ACCOUNT:-bcjw-dtai-gh}
log() { echo "$(date '+%F %T') $*"; }
export APPTAINER_CACHEDIR=$K/apptainer-cache
export APPTAINER_TMPDIR=/tmp/$USER-build-lite
mkdir -p "$APPTAINER_TMPDIR" "$K/sif"
cd "$K/bundle"
log "BUILD_START $(cat COMMIT 2>/dev/null)"
if MAX_JOBS=${MAX_JOBS:-4} apptainer build --force --fakeroot "$K/sif/unicache-lite.sif.tmp" unicache-lite.def \
   && apptainer test "$K/sif/unicache-lite.sif.tmp"; then
  mv "$K/sif/unicache-lite.sif.tmp" "$K/sif/unicache-lite.sif"
  ln -sfn unicache-lite.sif "$K/sif/unicache.sif"
  log "BUILD_OK $(du -h $K/sif/unicache-lite.sif | cut -f1)"
else
  log "BUILD_FAILED"; rm -rf "$APPTAINER_TMPDIR"; exit 1
fi
rm -rf "$APPTAINER_TMPDIR"
cd "$K"
SMOKE=$(sbatch --parsable -A $A -p ghx4-interactive -N1 -n1 --gpus-per-node=1 -c 8 --mem=32G -t 00:15:00 bundle/smoke.sbatch)
T0=$(date +%s)
log "SMOKE_SUBMITTED $SMOKE"
while squeue -h -j "$SMOKE" 2>/dev/null | grep -q .; do sleep 20; done
log "SMOKE_ENDED $SMOKE $(sacct -n -X -P -j $SMOKE -o State,Submit,Start,Elapsed,MaxRSS | head -1)"
