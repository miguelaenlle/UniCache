#!/bin/bash
# Reuse an already-built UniCache fork: copy its compiled CUDA libraries into this
# bundle's source tree, so nothing has to be recompiled. Our changes are Python-only.
#   bash overlay_compiled.sh /path/to/built/UniCache       (the checkout used for earlier sim2real runs)
set -euo pipefail
OLD=${1:?path to the previously built UniCache checkout}
cd "$(dirname "$0")/unicache"
n=0
for f in "$OLD"/vllm/*.so; do cp -v "$f" vllm/; n=$((n+1)); done
if [ -d "$OLD/vllm/vllm_flash_attn" ]; then
  mkdir -p vllm/vllm_flash_attn
  cp -rv "$OLD"/vllm/vllm_flash_attn/. vllm/vllm_flash_attn/
fi
echo "copied $n top-level .so files"; ls vllm/*.so vllm/vllm_flash_attn 2>/dev/null | head -20
