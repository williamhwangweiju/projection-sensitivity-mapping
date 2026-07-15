#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/full_pipeline/gpt2_3dcim.yaml}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"
RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_PHASE3="${RUN_PHASE3:-1}"
RUN_PHASE4="${RUN_PHASE4:-1}"

# ============================================================================
# Artifact filename patterns
# ============================================================================

# Phase 1
PHASE1_RESULTS_PATTERN="${PHASE1_RESULTS_PATTERN:-gpt2_aihwkit_calibrated_*.json}"

# Phase 2
PHASE2_TRACE_PATTERN="${PHASE2_TRACE_PATTERN:-trace.npz}"
PHASE2_METADATA_PATTERN="${PHASE2_METADATA_PATTERN:-metadata.json}"

# Phase 3
PHASE3_RANDOM_PLACEMENT_PATTERN="${PHASE3_RANDOM_PLACEMENT_PATTERN:-placement_random.csv}"
PHASE3_SEQUENTIAL_PLACEMENT_PATTERN="${PHASE3_SEQUENTIAL_PLACEMENT_PATTERN:-placement_sequential.csv}"
PHASE3_HARDWARE_PLACEMENT_PATTERN="${PHASE3_HARDWARE_PLACEMENT_PATTERN:-placement_hardware_only.csv}"
PHASE3_SENSITIVITY_PLACEMENT_PATTERN="${PHASE3_SENSITIVITY_PLACEMENT_PATTERN:-placement_static_sensitivity.csv}"

read_yaml_value() {
  local dotted_key="$1"
  CONFIG_PATH="$CONFIG" DOTTED_KEY="$dotted_key" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["CONFIG_PATH"]).read_text()) or {}
value = cfg
for key in os.environ["DOTTED_KEY"].split("."):
    value = value[key]
print(value)
PY
}

latest_file() {
  local directory="$1"
  local pattern="$2"
  local matches=()

  shopt -s nullglob
  matches=("${directory}"/${pattern})
  shopt -u nullglob

  if (( ${#matches[@]} == 0 )); then
    return 1
  fi

  ls -t "${matches[@]}" | head -n 1
}

require_file() {
  local path="$1"
  local description="$2"

  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "ERROR: missing ${description}: ${path:-<not found>}" >&2
    exit 1
  fi
}

PHASE1_NAME="$(read_yaml_value phase1.name)"
PHASE1_ROOT="$(read_yaml_value phase1.output_root)"
PHASE2_NAME="$(read_yaml_value phase2.name)"
PHASE2_ROOT="$(read_yaml_value phase2.output_root)"
PHASE3_NAME="$(read_yaml_value phase3.name)"
PHASE3_ROOT="$(read_yaml_value phase3.output_root)"
PHASE4_NAME="$(read_yaml_value phase4.name)"
PHASE4_ROOT="$(read_yaml_value phase4.output_root)"

PHASE2_OUTPUT_DIR="${PHASE2_ROOT}/${PHASE2_NAME}/seed_${SEED}"
PHASE3_OUTPUT_DIR="${PHASE3_ROOT}/${PHASE3_NAME}/seed_${SEED}"
PHASE4_OUTPUT_DIR="${PHASE4_ROOT}/${PHASE4_NAME}/seed_${SEED}"

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "========================================"
echo "Unified GPT-2 3D-CIM pipeline: Phases 1-4"
echo "Repository root: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "RUN_PHASE1: $RUN_PHASE1"
echo "RUN_PHASE2: $RUN_PHASE2"
echo "RUN_PHASE3: $RUN_PHASE3"
echo "RUN_PHASE4: $RUN_PHASE4"
echo "========================================"

if [[ "$RUN_PHASE1" == "1" ]]; then
  echo "[1/4] Phase 1: isolated AIHWKit DeltaPPL sensitivity and calibration"
  "$PYTHON_BIN" experiments/phase1_sensitivity/run_aihwkit_profiling.py \
    --config "$CONFIG"
fi

PHASE1_RESULTS_PATH="$(
  latest_file "$PHASE1_ROOT" "$PHASE1_RESULTS_PATTERN" || true
)"
require_file "$PHASE1_RESULTS_PATH" \
  "Phase-1 result matching ${PHASE1_ROOT}/${PHASE1_RESULTS_PATTERN}"

echo "Phase 1 result: $PHASE1_RESULTS_PATH"

if [[ "$RUN_PHASE2" == "1" ]]; then
  echo "[2/4] Phase 2: 72-tile fidelity trace"
  "$PYTHON_BIN" experiments/phase2_fidelity/run_fidelity_model.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output-dir "$PHASE2_OUTPUT_DIR" \
    "${OVERWRITE_FLAG[@]}"
fi

PHASE2_TRACE_PATH="$(
  latest_file "$PHASE2_OUTPUT_DIR" "$PHASE2_TRACE_PATTERN" || true
)"
PHASE2_METADATA_PATH="$(
  latest_file "$PHASE2_OUTPUT_DIR" "$PHASE2_METADATA_PATTERN" || true
)"

require_file "$PHASE2_TRACE_PATH" \
  "Phase-2 trace matching ${PHASE2_OUTPUT_DIR}/${PHASE2_TRACE_PATTERN}"
require_file "$PHASE2_METADATA_PATH" \
  "Phase-2 metadata matching ${PHASE2_OUTPUT_DIR}/${PHASE2_METADATA_PATTERN}"

echo "Phase 2 trace: $PHASE2_TRACE_PATH"
echo "Phase 2 metadata: $PHASE2_METADATA_PATH"

if [[ "$RUN_PHASE3" == "1" ]]; then
  echo "[3/4] Phase 3: static 3D-CIM placements"
  "$PYTHON_BIN" experiments/phase3_baselines/run_baseline_mappings.py \
    --config "$CONFIG" \
    --phase1-results "$PHASE1_RESULTS_PATH" \
    --phase2-trace "$PHASE2_TRACE_PATH" \
    --phase2-metadata "$PHASE2_METADATA_PATH" \
    --seed "$SEED" \
    --output-dir "$PHASE3_OUTPUT_DIR" \
    "${OVERWRITE_FLAG[@]}"
fi

PHASE3_RANDOM_PLACEMENT="$(
  latest_file "$PHASE3_OUTPUT_DIR" "$PHASE3_RANDOM_PLACEMENT_PATTERN" || true
)"
PHASE3_SEQUENTIAL_PLACEMENT="$(
  latest_file "$PHASE3_OUTPUT_DIR" "$PHASE3_SEQUENTIAL_PLACEMENT_PATTERN" || true
)"
PHASE3_HARDWARE_PLACEMENT="$(
  latest_file "$PHASE3_OUTPUT_DIR" "$PHASE3_HARDWARE_PLACEMENT_PATTERN" || true
)"
PHASE3_SENSITIVITY_PLACEMENT="$(
  latest_file "$PHASE3_OUTPUT_DIR" "$PHASE3_SENSITIVITY_PLACEMENT_PATTERN" || true
)"

require_file "$PHASE3_RANDOM_PLACEMENT" \
  "Phase-3 random placement matching ${PHASE3_OUTPUT_DIR}/${PHASE3_RANDOM_PLACEMENT_PATTERN}"
require_file "$PHASE3_SEQUENTIAL_PLACEMENT" \
  "Phase-3 sequential placement matching ${PHASE3_OUTPUT_DIR}/${PHASE3_SEQUENTIAL_PLACEMENT_PATTERN}"
require_file "$PHASE3_HARDWARE_PLACEMENT" \
  "Phase-3 hardware-only placement matching ${PHASE3_OUTPUT_DIR}/${PHASE3_HARDWARE_PLACEMENT_PATTERN}"
require_file "$PHASE3_SENSITIVITY_PLACEMENT" \
  "Phase-3 sensitivity placement matching ${PHASE3_OUTPUT_DIR}/${PHASE3_SENSITIVITY_PLACEMENT_PATTERN}"

"$PYTHON_BIN" scripts/validate_pipeline_contracts.py \
  --config "$CONFIG" \
  --phase1-results "$PHASE1_RESULTS_PATH" \
  --phase2-trace "$PHASE2_TRACE_PATH" \
  --phase3-dir "$PHASE3_OUTPUT_DIR"

if [[ "$RUN_PHASE4" == "1" ]]; then
  echo "[4/4] Phase 4: all-analog AIHWKit GPT-2 quality evaluation"
  "$PYTHON_BIN" experiments/phase4_quality/run_tile_noise_perplexity.py \
    --config "$CONFIG" \
    --phase1-results "$PHASE1_RESULTS_PATH" \
    --phase2-trace "$PHASE2_TRACE_PATH" \
    --phase2-metadata "$PHASE2_METADATA_PATH" \
    --phase3-dir "$PHASE3_OUTPUT_DIR" \
    --output-dir "$PHASE4_OUTPUT_DIR" \
    --seed "$SEED" \
    "${OVERWRITE_FLAG[@]}"
fi

echo "========================================"
echo "Pipeline complete"
echo "Phase 1: $PHASE1_RESULTS_PATH"
echo "Phase 2: $PHASE2_OUTPUT_DIR"
echo "Phase 3: $PHASE3_OUTPUT_DIR"
echo "Phase 4: $PHASE4_OUTPUT_DIR"
echo "========================================"