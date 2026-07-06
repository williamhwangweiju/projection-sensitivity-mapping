#!/usr/bin/env bash

set -euo pipefail

SCENARIOS=(
    static
    gradual_drift
    localized_fault
    thermal_variation
    mixed
)

SEEDS=(0 1 2 3 4)

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        python experiments/phase2_fidelity/run_fidelity_model.py \
            --config "configs/phase2_fidelity/${scenario}.yaml" \
            --seed "${seed}"
    done
done