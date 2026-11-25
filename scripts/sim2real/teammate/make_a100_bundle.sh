#!/bin/bash
# Build result/sim2real-transfer/sim2real-a100-bundle.tar.gz for a teammate's A100.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit first"; exit 1; }
OUT=result/sim2real-transfer; B=$OUT/sim2real-a100-bundle
rm -rf $B && mkdir -p $B/unicache $B/subsets $B/agents
git archive HEAD | tar -x -C $B/unicache
mkdir -p $B/unicache/models/Llama-3.2-1B-Instruct
cp models/Llama-3.2-1B-Instruct/*.json $B/unicache/models/Llama-3.2-1B-Instruct/
for w in alfred codeparrot@120; do cp -r TaskData/sim2real/$w $B/subsets/; done
cp TaskData/sim2real/subsets.json $B/subsets/
cp $OUT/bundle/agents/{alfred.py,codeparrot.py,SHA256SUMS} $B/agents/
cp scripts/sim2real/teammate/{run_a100.sh,overlay_compiled.sh,README-A100.md} $B/
git rev-parse HEAD > $B/COMMIT
tar -czf $OUT/sim2real-a100-bundle.tar.gz -C $OUT sim2real-a100-bundle
ls -la $OUT/sim2real-a100-bundle.tar.gz
