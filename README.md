# Unified GPT-2 × AIHWKit × IBM 3D-CIM Pipeline

This overlay implements the finalized Phase 1–4 workflow with one shared YAML
configuration.

## Experiment definition

### Phase 1 — isolated projection sensitivity

Each of GPT-2 small's 48 transformer projections is converted to AIHWKit one at
a time while the other 47 remain digital.

The Phase-3 mapping score is:

```text
sensitivity_score_for_mapping = mean(PPL_one_projection_analog - PPL_digital)
sensitivity_score_unit = delta_ppl_total
```

Phase 1 also retains the empirical AIHWKit programming-noise calibration:

```text
noise_reference_scale = measured logical weight-noise std / 0.023
```

The calibration is used only to convert Phase-2 normalized tile noise into an
absolute GPT-2 weight-domain standard deviation in Phase 4.

### Phase 2 — time-varying tile fidelity

Phase 2 generates `noise_std[timestep, tile]` from gradual drift, correlated
thermal variation, and localized faults. A timestep is a frozen hardware-state
snapshot during one complete Phase-4 dataset evaluation.

### Phase 3 — physical IBM 3D-CIM placement

The IBM mapper produces 480 transformer-projection shard placements across 72
physical tiles × 8 tiers. Embeddings and `lm_head` remain digital.

The sensitivity-aware policy orders projections using Phase-1 total DeltaPPL.
Its reported proxy is a sensitivity-weighted variance heuristic, not a direct
prediction of Phase-4 perplexity.

### Phase 4 — all-projection AIHWKit forward

All 48 transformer projections are converted to `AnalogLinearMapped`
simultaneously. For each policy, timestep, and paired realization:

1. Read the Phase-2 noise for each physical tile.
2. Use the Phase-3 shard assignment to build a rectangular per-weight sigma map
   inside every full GPT-2 projection.
3. Materialize `W_noisy = W_reference + sigma_map * Z` once.
4. Load the exact noisy logical weights into AIHWKit with `force_exact=True`.
5. Keep the weights fixed for the complete WikiText pass.
6. Run the all-analog AIHWKit model and measure NLL/PPL.
7. Restore the exact all-analog reference weights.

AIHWKit internal programming, read, and drift noise are disabled in Phase 4 so
noise is not applied twice. AIHWKit still provides the mapped analog forward
path and configured DAC/ADC, bound-management, and noise-management behavior.

The main result is the paired comparison between blind/hardware-only placement
and sensitivity-aware placement under the same hardware snapshot and random
field. Phase-1 and Phase-4 absolute perplexities are not expected to match.

## Installation

Copy this overlay into the repository root while preserving directory paths.
It replaces the corresponding Phase 1–4 files and adds:

```text
src/evaluation/aihwkit_gpt2.py
```

## Run all four phases

```bash
bash scripts/run_full_pipeline.sh
```

Useful overrides:

```bash
SEED=42 OVERWRITE=1 bash scripts/run_full_pipeline.sh
RUN_PHASE1=0 bash scripts/run_full_pipeline.sh
RUN_PHASE4=0 bash scripts/run_full_pipeline.sh
```

All settings are in:

```text
configs/full_pipeline/gpt2_3dcim.yaml
```

For a smoke test, reduce `dataset.max_tokens`, use 512-token windows, and set
both Phase-1 seeds and Phase-4 realizations to 1. For final results, restore a
larger evaluation set and multiple seeds.

## Phase-4 outputs

Key files include:

```text
quality_by_policy.csv
quality_by_timestep.csv
paired_policy_differences.csv
paired_policy_summary.csv
projection_noise_assignments.csv
tile_noise_injection_records.csv
weight_checksums.csv
reference_analog_conversion.json
metadata.json
```

Positive paired differences mean the second policy in the pair has lower
DeltaNLL/DeltaPPL and therefore preserves quality better.

## Built-in contract checks

The workflow checks:

- 48 Phase-1 total-DeltaPPL sensitivity rows;
- 48 empirical noise-calibration rows;
- a `[T, 72]` Phase-2 trace in PCMLike programming-scale-equivalent units;
- 480 unique projection tiers for every Phase-3 policy;
- identical logical shard sets across policies;
- identical all-analog reference weights in the two Phase-4 models;
- identical NLL for independently converted all-analog reference models;
- policy invariance when all physical tiles have uniform noise;
- exact logical-weight restoration after every Phase-4 evaluation.

## Important modeling boundary

The IBM 3D-CIM placement remains the authoritative physical tile/tier mapping.
Phase 4 uses that placement to materialize the correct noise into each GPT-2
weight slice. AIHWKit may internally split a large mapped layer differently;
those internal tiles are used as the analog-forward backend and are not treated
as a second physical placement.
