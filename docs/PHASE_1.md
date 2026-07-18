# Phase 1: Projection Sensitivity Profiling

Phase 1 measures how sensitive each profiled GPT-2 projection is to the repository's manual analog-weight noise model. Its primary artifact is the authoritative projection catalog used by Phase 1.5 for digital protection, by Phase 3 for physical sharding and importance, and by Phase 4 for cross-phase preprocessing checks.

The implementation profiles one projection as analog at a time while every other projection remains digital. The mapping score is a noise-only change in negative log-likelihood (NLL) relative to that projection's clipped nominal analog reference. It is **not** a change in perplexity relative to the untouched digital model.

## Entry points

- Profiler: `experiments/phase1_sensitivity/run_aihwkit_profiling.py`
- Ranking export: `experiments/phase1_sensitivity/analyze_results.py`
- Core profiler: `src/profilers/aihwkit_profiler.py`
- Projection discovery: `src/common/projections.py`
- Weight preprocessing and noise: `src/common/manual_weights.py`
- AIHWKit configuration and exact weight I/O: `src/common/analog.py`
- Canonical full configuration: `configs/full_pipeline/gpt2_hybrid_3dcim.yaml`
- Reduced smoke configuration: `configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml`

## What is profiled

Projection discovery is GPT-2-specific. For every selected transformer block, the profiler considers:

- `attn.c_attn`
- `attn.c_proj`
- `mlp.c_fc`
- `mlp.c_proj`

Projection IDs use the form `block_<index>/<role>`, for example `block_0/attn.c_attn`. When `profiling.include_lm_head` is true, `lm_head` is an additional candidate. GPT-2's Hugging Face `Conv1D` weights are transposed into canonical `[out_features, in_features]` form before profiling.

For GPT-2 Small, all 12 blocks produce 48 transformer candidates; including `lm_head` produces 49. `parameter_count` and `macs_per_token` are both recorded as `in_features * out_features`; bias parameters and bias additions are not included in these cost fields. The LM head records `tied_to_embedding: true` when it shares storage with the token embedding.

If `profile_blocks` is empty or omitted, all transformer blocks are profiled. If it is nonempty, it filters transformer blocks only. An enabled LM head remains included because it has no block index.

## Exact profiling method

### 1. Calibration data and clean baseline

The runner loads `model.name` with `AutoModelForCausalLM` and `AutoTokenizer`, converts the model to float32, disables the KV cache, and switches to evaluation mode. If the tokenizer has no pad token, its EOS token is used.

Dataset text is processed deterministically in source order:

1. Empty documents are skipped.
2. Each retained document is tokenized with `document_separator` appended and without added special tokens.
3. Tokens are concatenated and optionally truncated at `dataset.max_tokens`.
4. Fixed windows of `sequence_length` are created at `stride` intervals.
5. Labels belonging to context already scored by an overlapping previous window are masked with `-100`.
6. An incomplete final window is dropped by default.

NLL is token-weighted over all unmasked causal targets. Perplexity is `exp(NLL)`. The untouched digital model is evaluated once to obtain `clean_nll` and `clean_ppl`.

### 2. Projection preprocessing

For each projection with canonical weight matrix `W`, the population standard deviation is used:

```text
clip_threshold = analog.clip_sigma * std_population(W)
W_clipped      = clamp(W, -clip_threshold, +clip_threshold)
```

The programmed range is then computed from the clipped matrix:

```text
peak_to_peak: programmed_range = max(W_clipped) - min(W_clipped)
absmax:       programmed_range = max(abs(W_clipped))
```

Clipping happens once, before noise. The artifact records the original and clipped SHA-256 checksums, clipping threshold, programmed range, number and fraction of clipped weights, and the original population standard deviation.

### 3. Nominal analog reference

Only the current projection is converted to `AnalogLinearMapped`; all other modules stay digital. The clipped logical weights are installed with exact mapped-tile writes and verified by a logical readback. A nominal forward pass produces the projection-specific reference `nll_analog_reference` and `ppl_analog_reference`.

This nominal reference includes the configured AIHWKit mapped analog forward path, including input/output quantization and bound/noise management. It does not include the repository's manual Gaussian weight noise.

Internal AIHWKit programming noise, read noise, drift, and forward weight noise are disabled. The noise experiment below is materialized explicitly in logical-weight space.

### 4. Manual Gaussian realizations

For projection `p` and realization `r`:

```text
realization_seed   = experiment.seed + r * profiling.seed_stride
projection_seed    = SHA256-derived deterministic seed(realization_seed, projection_id)
Z[p,r]             ~ Normal(0, 1), with shape W_clipped
sigma_absolute     = analog.reference_noise_std * programmed_range
W_noisy[p,r,sign]  = W_clipped + sign * sigma_absolute * Z[p,r]
```

There is deliberately no clipping after noise. With `profiling.antithetic: true`, both signs `+1` and `-1` are evaluated from the same Gaussian field. Their NLL values are averaged into one realization:

```text
NLL_realization[p,r] = mean_sign NLL(W_noisy[p,r,sign])
PPL_realization[p,r] = exp(NLL_realization[p,r])
```

Consequently, antithetic PPL is the exponential of mean NLL, not the arithmetic mean of the two sign-level perplexities. Statistical counts equal `num_seeds`, not twice `num_seeds`.

### 5. Mapping sensitivity

Two deltas are retained for each realization:

```text
delta_nll_total = NLL_realization - clean_nll
delta_nll_noise = NLL_realization - nll_analog_reference
```

The mapping score is hardcoded as:

```text
sensitivity_score_for_mapping[p]
    = mean_r(delta_nll_noise[p,r])
```

The top-level unit is `delta_nll_noise`. A score can be negative because it is a finite Monte Carlo estimate; Phase 1 preserves that estimate. Phase 3 may apply `phase3.sensitivity_floor` when constructing physical shard importance.

For every summarized metric, the profiler records mean, sample standard deviation, standard error, normal-approximation 95% interval (`mean +/- 1.96 * SEM`), minimum, maximum, and count.

The original digital module and nominal weights are restored in a `finally` block before profiling the next projection.

## Canonical unified configuration

Use a configuration under `configs/full_pipeline/`. The effective Phase 1 portion of the primary configuration is:

```yaml
experiment:
  seed: 42

model:
  name: gpt2
  device: cpu

dataset:
  name: Salesforce/wikitext
  config: wikitext-103-raw-v1
  split: validation
  sequence_length: 1024
  stride: 1024
  batch_size: 1
  max_tokens: 65536
  document_separator: "\n\n"
  drop_incomplete_final_sequence: true

analog:
  clip_sigma: 2.5
  range_mode: peak_to_peak
  reference_noise_std: 0.023
  tile_size: 512
  adc_dac_bits: 8
  output_bound: null
  weight_scaling_omega: 1.0
  weight_scaling_columnwise: false

phase1:
  output_root: data/results/phase1_sensitivity
  results_filename_prefix: gpt2_hybrid_sensitivity

profiling:
  include_lm_head: true
  num_seeds: 5
  seed_stride: 1
  profile_blocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  antithetic: true
```

The relevant keys are:

| Key | Required | Meaning |
| --- | --- | --- |
| `model.name` | Yes | Hugging Face model identifier; current discovery requires GPT-2 structure. |
| `model.device` | Yes | PyTorch device, with `cpu` as the canonical configured default. |
| `experiment.seed` | Yes | Base seed for projection-specific Gaussian realizations. |
| `dataset.name`, `config`, `split` | Yes | Hugging Face dataset selection. |
| `dataset.sequence_length`, `stride` | Yes | Token-window geometry. |
| `dataset.batch_size` | No | Evaluation batch size; default `1`. |
| `dataset.max_tokens` | No | Token collection cap; `null` means no explicit cap. |
| `dataset.document_separator` | No | Text inserted after each retained document; default `"\n\n"`. |
| `dataset.drop_incomplete_final_sequence` | No | Drop a short final window; default `true`. |
| `analog.clip_sigma` | Yes | Symmetric clipping threshold in population-standard-deviation units. |
| `analog.range_mode` | No | `peak_to_peak` or `absmax`; default `peak_to_peak`. |
| `analog.reference_noise_std` | Yes | Noise standard deviation as a fraction of programmed range. |
| `analog.tile_size` | Yes | Maximum AIHWKit mapped input/output tile dimension. |
| `analog.adc_dac_bits` | Yes | Input and output quantization bit count; must be at least two. |
| `analog.output_bound` | No | Positive bound, or `null` to inherit the AIHWKit default. |
| `analog.weight_scaling_omega` | No | AIHWKit mapping scale; default `1.0`. |
| `analog.weight_scaling_columnwise` | No | Columnwise mapping-scale flag; default `false`. |
| `profiling.num_seeds` | Yes | Number of realization rows; must be positive. |
| `profiling.seed_stride` | No | Positive increment between realization seeds; default `1`. |
| `profiling.profile_blocks` | No | Transformer block subset; empty means all blocks. |
| `profiling.include_lm_head` | No | Include tied LM-head execution as a candidate; default `true`. |
| `profiling.antithetic` | No | Evaluate paired `+Z` and `-Z`; default `true`. |
| `phase1.output_root` | Yes | Artifact directory, resolved from the repository root when relative. |
| `phase1.results_filename_prefix` | No | Filename prefix; default `gpt2_hybrid_sensitivity`. |

> **No-op fields:** `profiling.sensitivity_field` and `profiling.sensitivity_unit` are present in the current full configurations but are not read by the implementation. The score and unit are hardcoded to `sensitivity_score_for_mapping` and `delta_nll_noise`. Changing those YAML values does not change profiling behavior.

> **Stale standalone configuration:** `configs/phase1_sensitivity/lammie_2026.yaml` uses an older schema. It lacks the required top-level `analog` and `phase1` mappings, places analog-like values under `profiling`, and defines an unused `programming_noise_scale`. It will fail with the current runner and its “AIHWKIT-NATIVE” description does not match the current manually materialized Gaussian-noise path. Do not use it for this pipeline.

## Running Phase 1

Run commands from the repository root. Install the declared dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

Before an expensive model run, verify the exact AIHWKit logical-weight write/noise/restore contract:

```bash
python3 scripts/smoke_aihwkit_contract.py --device cpu
```

That script exercises a small linear layer; it is not a GPT-2 end-to-end test.

### Reduced smoke profile

```bash
python3 -m experiments.phase1_sensitivity.run_aihwkit_profiling \
  --config configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml
```

The smoke configuration profiles only the four projections in block 0, omits the LM head, uses one unpaired realization, and caps calibration input at 1,024 collected tokens. It preserves the artifact contract but is not suitable for final research results.

### Full profile

```bash
python3 -m experiments.phase1_sensitivity.run_aihwkit_profiling \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

The runner prints the generated JSON path. With the primary configuration, the output resembles:

```text
data/results/phase1_sensitivity/
  gpt2_hybrid_sensitivity_YYYYmmdd_HHMMSS.json
```

The filename timestamp uses local time to second precision; `metadata.created_at_utc` is an ISO-8601 UTC timestamp.

### Export the ranking CSV

Direct profiling does not create a CSV automatically. Pass the JSON path to:

```bash
python3 -m experiments.phase1_sensitivity.analyze_results \
  --results-file data/results/phase1_sensitivity/gpt2_hybrid_sensitivity_YYYYmmdd_HHMMSS.json
```

Optional `--output-dir` changes the destination. Otherwise the analyzer writes `<profile>_ranking.csv` beside the JSON.

### Run as part of the complete pipeline

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

When Phase 1 is not skipped, the pipeline runs profiling, exports the ranking CSV, and then runs [Phase 1.5](PHASE_1_5.md). Use the direct Phase 1 commands when only the profile is wanted.

## JSON artifact contract

The top-level artifact has this shape:

```text
metadata
baseline
mapping_sensitivity_field
mapping_sensitivity_unit
projections[]
requested_config
```

### Top-level fields

| Field | Contents |
| --- | --- |
| `metadata` | Creation, repository, configuration, software, model, dataset, and analog provenance. |
| `baseline.clean_nll`, `baseline.clean_ppl` | Untouched digital-model calibration metrics. |
| `mapping_sensitivity_field` | Literal `sensitivity_score_for_mapping`. |
| `mapping_sensitivity_unit` | Literal `delta_nll_noise`. |
| `projections` | Ordered projection result rows. |
| `requested_config` | Complete parsed YAML mapping used by the run. |

`metadata` contains:

- `created_at_utc`, `repository_commit`, `config_path`, and `config_sha256`
- `software_versions.python`, `torch`, `transformers`, `datasets`, and `aihwkit`
- `model.name`, `n_layer`, `n_embd`, and `vocab_size`
- `dataset.name`, `config`, `split`, `sequence_length`, `stride`, `batch_size`, `max_tokens`, `collected_tokens`, `num_windows`, `num_batches`, and `predicted_tokens_per_pass`
- `analog_configuration`, including preprocessing version, clipping and range semantics, normalized-noise unit and distribution, tile/quantization settings, bounds, mapping settings, disabled internal-noise flags, profiling mode, seed count/stride, antithetic flag, and LM-head inclusion flag

### Projection row fields

Identity, shape, and logical cost:

```text
projection_id, module_path, role, block_index
in_features, out_features, parameter_count, macs_per_token
tied_to_embedding
```

Clean and nominal-reference metrics:

```text
nll_clean, ppl_clean
nll_analog_reference, ppl_analog_reference
delta_nll_analog_reference, delta_ppl_analog_reference
```

Preprocessing and run details:

```text
preprocessing
realizations[]
clip_value, programmed_range, clipped_fraction
reference_noise_std, realization_count, predicted_tokens
```

Mapping fields:

```text
sensitivity_score_for_mapping
sensitivity_score_unit
sensitivity_per_parameter
sensitivity_per_mac
```

`preprocessing` contains:

```text
original_std, clip_threshold, programmed_range, range_mode
num_weights, num_clipped, fraction_clipped
original_checksum, clipped_checksum
```

Each realization contains:

```text
realization, realization_seed, projection_noise_seed, antithetic_count
nll, ppl, delta_nll_total, delta_ppl_total
delta_nll_noise, delta_ppl_noise, noise_std_absolute
```

The following metric prefixes are also flattened into the projection row:

```text
nll, ppl, delta_nll_total, delta_ppl_total
delta_nll_noise, delta_ppl_noise, noise_std_absolute
```

Each prefix receives `_mean`, `_std`, `_sem`, `_ci95_low`, `_ci95_high`, `_minimum`, `_maximum`, and `_count`. Flattened counts are serialized as floating-point values by the current implementation.

## Ranking CSV contract

Rows are sorted by descending raw `sensitivity_score_for_mapping`. Exact columns are:

```text
rank
projection_id
role
sensitivity_score_for_mapping
delta_nll_analog_reference
parameter_count
macs_per_token
sensitivity_per_parameter
sensitivity_per_mac
clipped_fraction
tied_to_embedding
```

Uncertainty statistics remain available only in the JSON.

## Downstream contracts

- **Phase 1.5** requires each candidate's `projection_id`, `sensitivity_score_for_mapping`, `parameter_count`, `macs_per_token`, and optional `tied_to_embedding`; capacity calculation also needs `in_features` and `out_features`.
- **Phase 3** uses `projection_id`, dimensions, and mapping sensitivity to construct coordinate-preserving physical shards and shard importance.
- **Phase 4** treats `projections` as the authoritative analog/digital candidate universe. Projections omitted by a reduced profile remain digital rather than being silently analogized. It verifies original/clipped checksums, range mode, population standard deviation, clipping threshold, and programmed range against the current model and analog configuration.
- **Pipeline validation** requires unique projection IDs and `mapping_sensitivity_unit == "delta_nll_noise"`.

Use the same model and analog preprocessing configuration in downstream quality runs. A different model revision, clipping setting, or range mode can fail the Phase 1/Phase 4 preprocessing checks; other analog-setting changes can alter execution even when they are not part of that checksum validation.

## Cost and reproducibility

With the primary configuration, Phase 1 performs:

```text
1 clean pass
+ 49 candidates * (1 nominal reference pass + 5 seeds * 2 antithetic signs)
= 540 complete calibration-dataset passes
```

The smoke configuration performs nine passes. Runtime scales approximately with candidate count, realization count, antithetic sign count, calibration tokens, and model inference cost. All results stay in memory until the complete profile is saved.

Reproducibility support includes deterministic dataset construction, explicit projection-specific seeds, configuration contents and SHA-256, repository commit, software versions, model name, checksums, and dataset-window metadata. Remaining sources of variation include backend/device numerics, remotely resolved model or dataset revisions, and library versions. The artifact does not pin Hugging Face revisions.

## Validation and tests

Run the root project suite with the repository root on Python's module path:

```bash
python3 -m pytest -q tests
```

Relevant tests currently cover automatic-selection configuration and downstream contracts, but there is no direct unit or end-to-end test for `AIHWKITSensitivityProfiler`, the Phase 1 CLI, dataset window construction, analyzer output, or the complete JSON schema. `scripts/smoke_aihwkit_contract.py` is an executable simulator diagnostic, not a pytest.

Avoid unscoped repository-wide pytest collection: the repository vendors simulator source trees with their own test suites and native-extension expectations.

## Current limitations and failure modes

- The implementation assumes GPT-2 module structure and metadata fields; another causal LM is not supported merely because it loads through `AutoModelForCausalLM`.
- There is no checkpoint, resume, incremental artifact save, or per-projection recovery. An interrupted full run loses unsaved in-memory progress.
- Output filenames have second-resolution timestamps and no overwrite guard.
- There is no artifact schema-version field and only limited validation at phase boundaries.
- Block indices are not range-validated. A configuration that selects no transformer candidate and disables the LM head eventually fails when the runner accesses the first result.
- Raw Monte Carlo scores may be negative and should not be described as guaranteed nonnegative importance.
- The CSV omits the JSON uncertainty fields.
- The canonical configuration uses CPU. Phase 1's direct device conversion path has no dedicated CUDA test.
- `save_json` rejects non-finite values, but writes are not atomic. Numerical
  overflow or NaN can fail during serialization after the destination has been
  opened, leaving a truncated invalid artifact.
