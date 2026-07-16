#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-${ROOT}/configs/full_pipeline/gpt2_hybrid_3dcim.yaml}"

args=("${ROOT}/scripts/run_full_pipeline.py" --config "${CONFIG}")
[[ "${RUN_PHASE1:-1}" == "1" ]] || args+=(--skip-phase1)
[[ "${RUN_PHASE2:-1}" == "1" ]] || args+=(--skip-phase2)
[[ "${RUN_PHASE3:-1}" == "1" ]] || args+=(--skip-phase3)
[[ "${RUN_PHASE4:-1}" == "1" ]] || args+=(--skip-phase4)
[[ "${RUN_PHASE5:-1}" == "1" ]] || args+=(--skip-phase5)
[[ "${RUN_ADAPTIVE_QUALITY:-1}" == "1" ]] || args+=(--skip-adaptive-quality)

exec "${PYTHON_BIN}" "${args[@]}" "${@:2}"
