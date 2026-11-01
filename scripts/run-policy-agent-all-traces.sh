#!/usr/bin/env bash
# Sequential runs keep Claude costs and simulator resource use bounded.
set -euo pipefail
cd "$(dirname "$0")/.."
for trace in sharegpt mashqa agentbank-apps alfred codeparrot; do
  echo "Starting policy-agent workload: $trace (LRU; all comparisons; 30-attempt cap)"
  ./policy-agent run \
    --trace "$trace" \
    --serving_model llama-3.2-1b-instruct \
    --coding_model claude:opus \
    --start-policy LRU \
    --compare all \
    --require-baselines OPT \
    --max-submissions 30 "$@"
done
