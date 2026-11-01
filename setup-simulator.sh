#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
uv venv --python 3.10 --allow-existing .venv
uv pip sync --python .venv/bin/python requirements/simulator-local.lock
VLLM_TARGET_DEVICE=empty uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -e .
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "unsloth/Llama-3.1-8B-Instruct",
    allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "generation_config.json"],
    local_dir="models/Llama-3.1-8B-Instruct",
)
PY
echo "Ready. Run ./sim, or ./sim --eviction-policy LRU"
