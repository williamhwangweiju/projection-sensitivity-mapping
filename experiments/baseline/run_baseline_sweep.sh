#!/bin/bash

set -e

# Usage:
# ./scripts/run_baseline_sweep.sh toy
# ./scripts/run_baseline_sweep.sh gpt2_small

PRESET=${1:-toy}

OUT_DIR="results/baseline_${PRESET}_sweep"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Running IBM 3D-CIM baseline sweep"
echo "Preset: $PRESET"
echo "Output directory: $OUT_DIR"
echo "========================================"

echo "Running target length sweep..."

for LEN in 4 8 12 16 24
do
  python3 experiments/run_cim_baseline.py \
    --preset "$PRESET" \
    --target-len "$LEN" \
    --out "$OUT_DIR/${PRESET}_len${LEN}.json"
done

echo "Running layer depth sweep..."

for LAYERS in 1 3 6 12
do
  python3 experiments/run_cim_baseline.py \
    --preset "$PRESET" \
    --num-layers "$LAYERS" \
    --target-len 12 \
    --out "$OUT_DIR/${PRESET}_layers${LAYERS}_len12.json"
done

echo "Converting JSON results to CSV..."

python3 scripts/json_to_excel_csv.py \
  --input-dir "$OUT_DIR" \
  --output "$OUT_DIR/${PRESET}_summary.csv"

echo "========================================"
echo "Baseline sweep complete."
echo "Results saved to: $OUT_DIR"
echo "CSV summary saved to: $OUT_DIR/${PRESET}_summary.csv"
echo "========================================"