#!/bin/bash
# Driver v6 (synthetic sweep in three <= 15-min parts, then the remaining replays). Derived from the short-job version (run on the DeltaAI login node):
#   setsid nohup bash $K/bundle/overnight_short.sh > $K/logs/overnight.log 2>&1 < /dev/null &
# Why: on Nov 26 between 02:00 and 06:00, 1-GPU interactive jobs with limits <= 15 min started
# within 0-4 min, while 31-60 min jobs at our priority waited hours (ours: 6.5 h, never started).
# So every job is one replay (1 GPU, 8 CPUs, 32 GB, 15-20 min), submitted one at a time.
# A run that hits its limit is resubmitted (max 3 tries); a run that fails stops the pipeline.
set -uo pipefail
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
A=${ACCOUNT:-bcjw-dtai-gh}
cd $K
log() { echo "$(date '+%F %T') $*"; }
submit() { local t=$1; shift; sbatch --parsable -A $A -p ghx4-interactive -N1 -n1 --gpus-per-node=1 -c 8 --mem=32G -t "$t" "$@"; }
waitjob() {
  local j=$1 started=0
  while squeue -h -j $j 2>/dev/null | grep -q .; do
    if [ $started = 0 ] && [ "$(squeue -h -j $j -o %T)" = RUNNING ]; then
      started=1; log "START $j $(sacct -n -X -P -j $j -o Submit,Start,NodeList | head -1)"
    fi
    sleep 20
  done
  log "END $j $(sacct -n -X -P -j $j -o State,Submit,Start,Elapsed | head -1)"
}
done_runs() {
  local W=$1 B=$2 R
  for R in ${3//,/ }; do
    if [[ $R == *x* ]]; then
      local P=${R%x*} n=${R##*x} k
      for ((k = 0; k < n; k++)); do
        local d=$K/results/gpu/$W/$W/$B/$P; [ $k -gt 0 ] && d=$d/rep-$k
        ls $d/*/summary.json >/dev/null 2>&1 || return 1
      done
    else
      ls $K/results/gpu/$W/$W/$B/$R/*/summary.json >/dev/null 2>&1 || return 1
    fi
  done
}
part() {
  local W=$1 B=$2 R=$3 T=$4 try j st
  for try in 1 2 3; do
    done_runs $W $B $R && { log "PART_DONE $W $R"; return 0; }
    j=$(submit $T bundle/run_trace.sbatch $W $B $R); log "SUBMIT $j $W $R try $try limit $T"
    waitjob $j
    st=$(sacct -n -X -P -j $j -o State | head -1)
    done_runs $W $B $R && { log "PART_DONE $W $R $(grep -h "hit " logs/s2r-trace-$j.out 2>/dev/null | tail -1)"; return 0; }
    [[ $st == TIMEOUT* ]] || { log "PIPELINE_STOPPED $W $R: job $j ended $st with runs missing"; exit 1; }
  done
  log "PIPELINE_STOPPED $W $R: not done after 3 tries"; exit 1
}

log "DRIVER_V6_START"
# Wait for any of our jobs still queued or running (e.g. from the previous driver).
while squeue -h -u $USER 2>/dev/null | grep -q .; do sleep 20; done
# 1. Synthetic calibration sweep, three parts (each skipped once its last config has a result).
sweep_done() { ls $K/results/gpu-sweep/$1/*/summary.json >/dev/null 2>&1; }
for part in prefill:sweep-evict/512/Agent decode:sweep-dec32-c2048/16384/LRU load:sweep-rps40/16384/LRU; do
  name=${part%%:*}; marker=${part#*:}
  for try in 1 2; do
    sweep_done $marker && break
    j=$(submit 00:15:00 bundle/sweep.sbatch $name); log "SUBMIT $j sweep-$name try $try"
    waitjob $j
  done
  sweep_done $marker && log "SWEEP_PART_DONE $name" || log "SWEEP_PART_MISSING $name"
done
# 3. Remaining replays, one per job (finished runs are skipped).
for spec in "alfred 512 00:15:00" "codeparrot@600 4096 00:20:00" "sharegpt@100x2 1024 00:20:00"; do
  set -- $spec
  for R in LRU UniCache LeCaR Agent LRUx2 LRUx3; do part $1 $2 $R $3; done
done
log "OVERNIGHT_DONE"
