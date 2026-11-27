#!/bin/bash
# One-line-per-item status of the sim2real GPU pipeline on DeltaAI (read-only).
K=${K:-/work/hdd/bcjw/maenlle2/kvcal}
echo "now $(date '+%F %T')"
L=$K/logs/build-lite.log
grep -E "BUILD_|SMOKE_" $L 2>/dev/null | tail -4
p=$(pgrep -u $USER -f "cmake --build" | head -1)
if [ -n "$p" ] && [ -f /proc/$p/cwd/build.ninja ]; then
  d=/proc/$p/cwd
  done_c=$(find $d/CMakeFiles/_C.dir -name "*.o" 2>/dev/null | wc -l)
  done_fa=$(find $d/vllm-flash-attn -path "*_vllm_fa2_C.dir*" -name "*.o" 2>/dev/null | wc -l)
  tot_c=$(grep -cE "^build CMakeFiles/_C.dir/[^ ]+\.(cu|cpp|c)\.o:" $d/build.ninja)
  tot_fa=$(grep -cE "^build [^ ]*_vllm_fa2_C.dir/[^ ]+\.(cu|cpp|c)\.o:" $d/build.ninja)
  echo "compile: _C $done_c/$tot_c, fa2 $done_fa/$tot_fa, mem $(( $(cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current) / 1073741824 )) GiB"
elif pgrep -u $USER -f "mksquashfs" >/dev/null; then
  echo "packing image (mksquashfs)"
elif pgrep -u $USER -f "apptainer build" >/dev/null; then
  echo "build running: $(tail -1 $L | cut -c1-120)"
fi
squeue -u $USER -h -o "job %i %j %T limit=%l start=%S %R"
