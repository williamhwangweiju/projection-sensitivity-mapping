# Phase 1: AIHWKit-Native Projection Sensitivity Profiling

## Goal

Phase 1 establishes a projection-level sensitivity profile for GPT-2 under analog in-memory-computing nonidealities. The objective is to identify which transformer projections are most vulnerable to AIHWKit-simulated programming noise, so later phases can assign more sensitive projections to higher-fidelity hardware resources.

This phase corresponds to the sensitivity-profiling stage of the Lammie-style heterogeneous mapping workflow. The output of Phase 1 is a compact, mapper-ready JSON file containing clean perplexity, noisy perplexity, sensitivity scores, and sensitivity ranks for each selected GPT-2 projection.

## Role in the Full Project

The full project studies fidelity-aware adaptive mapping for GPT-2 on 3D analog CIM hardware.

Phase 1 provides the model-side sensitivity information:

- which GPT-2 projections are most affected by analog programming noise;
- how much each projection increases perplexity when converted to analog;
- a ranked sensitivity order for later hardware mapping.

Phase 2 provides the hardware-side fidelity model:

- tile-level noise/fidelity classes;
- time-varying degradation;
- thermal and localized fault effects.

Phase 3 combines both:

- sensitive projections from Phase 1 are preferentially mapped to high-fidelity tiles from Phase 2;
- alternative policies such as random, sequential, hardware-only, and sensitivity-aware mapping can be compared;
- IBM 3D-CIM is used for architectural time/energy evaluation while AIHWKit provides model-quality evaluation.

## Final Experiment Configuration

The final Phase 1 configuration uses GPT-2 on WikiText-103 with fixed-length causal language-model evaluation.

```yaml
model:
  name: gpt2
  device: cpu

dataset:
  name: Salesforce/wikitext
  config: wikitext-103-raw-v1
  split: test
  max_tokens: null
  sequence_length: 1024
  stride: 1024
  batch_size: 1
  document_separator: "\n\n"
  drop_incomplete_final_sequence: true

profiling:
  include_lm_head: false
  clip_sigma: 2.5
  tile_size: 512
  adc_dac_bits: 8
  programming_noise_scale: 1.0
  num_seeds: 10
  seed_stride: 1
  profile_blocks:
    - 0
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    - 7
    - 8
    - 9
    - 10
    - 11

experiment:
  seed: 42
  save_dir: ./data/results/phase1_sensitivity
  results_filename_prefix: programming_noise_sensitivity
```

## Model and Dataset

The evaluated model is Hugging Face GPT-2. The profiled projections are the four main linear projections in each transformer block:

- `attn.c_attn`
- `attn.c_proj`
- `mlp.c_fc`
- `mlp.c_proj`

The LM head is excluded in the final Phase 1 run because its vocabulary-sized output projection introduces additional tiling and clipping behavior that is not representative of the transformer-block projections used for adaptive mapping.

The dataset is `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, test split. Documents are concatenated using a double-newline separator to preserve document boundaries. The token stream is divided into fixed windows of 1024 tokens with stride 1024. Incomplete final windows are dropped so that all evaluated sequences have the same length.

## Analog Configuration

Each projection is converted to an AIHWKit analog module one at a time. All other model parameters remain digital FP32.

The analog setup is:

- Gaussian layer clipping with `WeightClipType.LAYER_GAUSSIAN`;
- clipping sigma `2.5`;
- mapped analog tiles with maximum input/output size `512 × 512`;
- approximately 8-bit DAC/ADC resolution;
- AIHWKit-native `PCMLikeNoiseModel`;
- programming noise scale `1.0`;
- read noise disabled;
- drift disabled.

The profiler explicitly triggers AIHWKit's configured clipping through the tile post-update hook before programming the analog weights. This keeps clipping inside AIHWKit rather than manually modifying tensors.

## Profiling Methodology

For each selected projection:

1. Resolve the GPT-2 projection path, such as `block_3/mlp.c_fc`.
2. Copy the projection into an `nn.Linear` module.
3. Convert only that projection to `AnalogLinearMapped` using the configured AIHWKit RPU setup.
4. Apply the configured AIHWKit clipping rule.
5. Program the analog weights using AIHWKit's native programming-noise model.
6. Temporarily replace the original GPT-2 projection with the analog projection.
7. Evaluate token-weighted perplexity on the prepared WikiText-103 batches.
8. Restore the original digital projection.
9. Repeat for the configured number of random seeds.
10. Compute mean and standard deviation of the perplexity increase.

The sensitivity score is:

```text
sensitivity = noisy_perplexity - clean_perplexity
```

Higher sensitivity means the projection is more vulnerable to analog programming noise.

## Saved Metrics

The final profiler saves only the metrics needed for Phase 1 analysis and Phase 3 mapping.

For each projection, the JSON includes:

- `block_id`
- `proj_name`
- `projection_label`
- `in_features`
- `out_features`
- `analog_tile_count`
- `ppl_clean`
- `ppl_noisy_mean`
- `ppl_noisy_std`
- `sensitivity_mean`
- `sensitivity_std`
- `sensitivity_per_seed`
- `realization_seeds`
- `sensitivity_rank`

The runner also saves metadata for reproducibility:

- model name and shape;
- dataset configuration;
- analog configuration;
- software versions;
- requested YAML configuration;
- timestamped output path.

## Output Schema

The runner writes a JSON file with the structure:

```json
{
  "metadata": {
    "paper": "Heterogeneous Mapping for Analog In-Memory Computing Accelerators: A Unified Workflow",
    "created_at_utc": "...",
    "software_versions": {...},
    "model": {...},
    "dataset": {...},
    "analog_configuration": {...}
  },
  "requested_config": {...},
  "results": {
    "digital_perplexity": 0.0,
    "projections": [
      {
        "block_id": "block_0",
        "proj_name": "attn.c_attn",
        "projection_label": "block_0/attn.c_attn",
        "in_features": 768,
        "out_features": 2304,
        "analog_tile_count": 10,
        "ppl_clean": 0.0,
        "ppl_noisy_mean": 0.0,
        "ppl_noisy_std": 0.0,
        "sensitivity_mean": 0.0,
        "sensitivity_std": 0.0,
        "sensitivity_per_seed": [],
        "realization_seeds": [],
        "sensitivity_rank": 1
      }
    ]
  }
}
```

## Analysis Outputs

The analyzer reads the runner JSON and generates:

1. a projection-sensitivity heatmap;
2. a sensitivity distribution plot by projection type;
3. a grouped per-block sensitivity plot.

It also prints:

- clean digital perplexity;
- overall sensitivity statistics;
- block-level average sensitivities;
- projection-type average sensitivities;
- top most sensitive projections;
- bottom least sensitive projections.

These plots and summaries are used to verify whether sensitivity patterns are structured rather than random, and to prepare the sensitivity ordering used by the Phase 3 mapper.

## Reproduction Commands

Run Phase 1 profiling:

```bash
python experiments/phase1_sensitivity/run_aihwkit_profiling.py \
  --config configs/lammie_2026.yaml
```

Analyze the latest Phase 1 result:

```bash
python experiments/phase1_sensitivity/analyze_phase1.py \
  --results-dir data/results/phase1_sensitivity
```

Analyze a specific result file:

```bash
python experiments/phase1_sensitivity/analyze_phase1.py \
  --results-file data/results/phase1_sensitivity/programming_noise_sensitivity_YYYYMMDD_HHMMSS.json
```

## Current Conclusion

Phase 1 produces a clean and reproducible GPT-2 projection-sensitivity profile under AIHWKit-native analog programming noise. The final implementation avoids unnecessary diagnostic metrics and keeps only the information required for analysis and downstream mapping.

The key outcome is a ranked list of projection sensitivities. This ranking becomes the model-side priority order for Phase 3, where sensitive projections can be assigned to higher-fidelity hardware tiles and compared against random, sequential, and hardware-only baselines.

Phase 1 is therefore complete when the runner successfully produces the sensitivity JSON and the analyzer confirms structured projection-level sensitivity trends across GPT-2 blocks and projection types.
