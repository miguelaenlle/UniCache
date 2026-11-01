#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${SIMULATOR_PYTHON:-$PWD/.venv-sim312/bin/python}" scripts/evaluate_policy_v2.py "$@"
