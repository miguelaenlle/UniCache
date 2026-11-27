#!/bin/bash
# Overnight driver, run on the DeltaAI login node (no GPU itself):
#   setsid nohup bash $K/bundle/overnight.sh > $K/logs/overnight.log 2>&1 < /dev/null &
# Submits the sim2real GPU jobs one at a time (the interactive partition allows one job per
# user). Each is 1 GPU, 8 CPUs, 32 GB, <= 60 min. A part that hits its time limit is
# resubmitted (runs resume); a part whose runs fail stops the pipeline so no GPU time is
# wasted. Stop it with:  pkill -f overnight.sh   (and scancel any queued job).
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
    sleep 30
  done
  log "END $j $(sacct -n -X -P -j $j -o State,Submit,Start,Elapsed | head -1)"
}
# done_runs <workload> <blocks> <runs>: all listed runs have a summary.json
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
# part <workload> <blocks> <runs> <limit>: submit until done, max 3 tries, stop on real failure
part() {
  local W=$1 B=$2 R=$3 T=$4 try j st
  for try in 1 2 3; do
    done_runs $W $B $R && { log "PART_DONE $W $R"; return 0; }
    j=$(submit $T bundle/run_trace.sbatch $W $B $R); log "SUBMIT $j part $W $R try $try limit $T"
    waitjob $j
    st=$(sacct -n -X -P -j $j -o State | head -1)
    done_runs $W $B $R && { log "PART_DONE $W $R"; return 0; }
    [[ $st == TIMEOUT* ]] || { log "PIPELINE_STOPPED $W $R: job $j ended $st with runs missing"; exit 1; }
  done
  log "PIPELINE_STOPPED $W $R: not done after 3 tries"; exit 1
}

log "OVERNIGHT_START"
if [ -n "${CANCEL:-}" ]; then scancel $CANCEL && log "cancelled $CANCEL"; sleep 5; fi
# Job 1: smoke test, then ALFRED.
if ! done_runs alfred 512 LRU,UniCache,LeCaR,Agent,LRUx3; then
  j=$(submit 01:00:00 bundle/session.sbatch); log "SUBMIT $j session (smoke + ALFRED)"
  waitjob $j
  grep -h "SMOKE_RESULT" logs/s2r-session-$j.out | sed "s/^/$(date '+%F %T') /"
  grep -q "SMOKE_RESULT yes" logs/s2r-session-$j.out || { log "PIPELINE_STOPPED: smoke test failed in $j"; exit 1; }
fi
part alfred 512 LRU,UniCache,LeCaR,Agent,LRUx3 00:45:00
part codeparrot@600 4096 LRU,UniCache,LeCaR 00:50:00
part codeparrot@600 4096 Agent,LRUx3 00:50:00
part sharegpt@100x2 1024 LRU,UniCache,LeCaR 00:50:00
part sharegpt@100x2 1024 Agent,LRUx3 00:50:00
log "OVERNIGHT_DONE"
