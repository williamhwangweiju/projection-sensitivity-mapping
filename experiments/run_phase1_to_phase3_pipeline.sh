#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"

PHASE1_CONFIG="${PHASE1_CONFIG:-configs/phase1_sensitivity/lammie_2026.yaml}"
PHASE2_CONFIG="${PHASE2_CONFIG:-configs/phase2_fidelity/mixed.yaml}"
PHASE3_CONFIG="${PHASE3_CONFIG:-configs/phase3_baselines/default.yaml}"
PHASE3_OUTPUT_DIR="${PHASE3_OUTPUT_DIR:-data/results/phase3_baselines/phase3_baselines/seed_${SEED}}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "========================================"
echo "Running full pipeline: Phase 1 -> 2 -> 3"
echo "Repository root: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Seed: $SEED"
echo "Phase 1 config: $PHASE1_CONFIG"
echo "Phase 2 config: $PHASE2_CONFIG"
echo "Phase 3 config: $PHASE3_CONFIG"
echo "Phase 3 output: $PHASE3_OUTPUT_DIR"
echo "========================================"

echo ""
echo "[1/3] Running Phase 1 sensitivity profiling..."
"$PYTHON_BIN" experiments/phase1_sensitivity/run_aihwkit_profiling.py \
  --config "$PHASE1_CONFIG"

PHASE1_RESULTS_PATH="$(
  ls -t data/results/phase1_sensitivity/*.json 2>/dev/null | head -n 1
)"
if [[ -z "$PHASE1_RESULTS_PATH" ]]; then
  echo "ERROR: Could not find Phase 1 output under data/results/phase1_sensitivity" >&2
  exit 1
fi
echo "Phase 1 output: $PHASE1_RESULTS_PATH"

echo ""
echo "[2/3] Running Phase 2 fidelity simulation..."
"$PYTHON_BIN" experiments/phase2_fidelity/run_fidelity_model.py \
  --config "$PHASE2_CONFIG" \
  --seed "$SEED" \
  "${OVERWRITE_FLAG[@]}"

PHASE2_EXPERIMENT_NAME="$(
  PHASE2_CONFIG_PATH="$PHASE2_CONFIG" "$PYTHON_BIN" - <<'PY'
import pathlib
import os
import yaml

cfg_path = pathlib.Path(os.environ["PHASE2_CONFIG_PATH"])
cfg = yaml.safe_load(cfg_path.read_text()) or {}
name = cfg.get("experiment", {}).get("name", "phase2_fidelity")
print(name)
PY
)"
PHASE2_OUTPUT_DIR="data/results/phase2_fidelity/fidelity_traces/${PHASE2_EXPERIMENT_NAME}/seed_${SEED}"
PHASE2_TRACE_PATH="${PHASE2_OUTPUT_DIR}/trace.npz"
PHASE2_METADATA_PATH="${PHASE2_OUTPUT_DIR}/metadata.json"

if [[ ! -f "$PHASE2_TRACE_PATH" ]]; then
  echo "ERROR: Phase 2 trace not found: $PHASE2_TRACE_PATH" >&2
  exit 1
fi
if [[ ! -f "$PHASE2_METADATA_PATH" ]]; then
  echo "ERROR: Phase 2 metadata not found: $PHASE2_METADATA_PATH" >&2
  exit 1
fi
echo "Phase 2 trace: $PHASE2_TRACE_PATH"
echo "Phase 2 metadata: $PHASE2_METADATA_PATH"

echo ""
echo "[3/3] Running Phase 3 baseline mappings (IBM 3D-CIM integration)..."
"$PYTHON_BIN" experiments/phase3_baselines/run_baseline_mappings.py \
  --config "$PHASE3_CONFIG" \
  --phase1-results "$PHASE1_RESULTS_PATH" \
  --phase2-trace "$PHASE2_TRACE_PATH" \
  --phase2-metadata "$PHASE2_METADATA_PATH" \
  --seed "$SEED" \
  --output-dir "$PHASE3_OUTPUT_DIR" \
  "${OVERWRITE_FLAG[@]}"

echo ""
echo "========================================"
echo "Pipeline complete."
echo "Phase 1: $PHASE1_RESULTS_PATH"
echo "Phase 2: $PHASE2_OUTPUT_DIR"
echo "Phase 3: $PHASE3_OUTPUT_DIR"
echo "========================================"
