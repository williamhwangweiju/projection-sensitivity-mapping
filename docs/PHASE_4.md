# Phase 4: Hybrid Language-Model Quality Evaluation

Phase 4 is the model-level evaluation stage. It combines automatically selected
digital projection sets, static Phase 3 placements, and time-varying Phase 2
tile noise, then measures GPT-2 negative log-likelihood (NLL) and perplexity
(PPL) through an AIHWKit analog-forward path.

The principal comparison is paired: for the same digital set, timestep, and
Gaussian realization, how much quality does `static_sensitivity` preserve
relative to random, sequential, and hardware-only placement?

## Inputs

The runner requires five inputs:

1. the unified YAML configuration;
2. a Phase 1 projection profile;
3. the Phase 1.5 operating-point artifact;
4. a Phase 2 `trace.npz`; and
5. the Phase 3 `phase3_manifest.json`.

The Phase 1 profile is also the authoritative projection candidate universe.
Projections absent from a reduced smoke profile remain digital and are not
silently converted to analog.

## Quick start

The full pipeline invokes Phase 4 after validating all upstream contracts:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

To evaluate existing artifacts directly:

```bash
python3 experiments/phase4_quality/run_hybrid_quality.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --operating-points data/results/phase1_5_digital_selection/digital_operating_points.json \
  --trace data/results/phase2_fidelity/fidelity_traces/mixed_96x8/seed_42/trace.npz \
  --phase3-manifest data/results/phase3_static_mapping/phase3_manifest.json
```

Repeat `--digital-set-id <id>` to request particular operating points. Requested
IDs must still be capacity-feasible, have Phase 3 placements, and match the
configured budget-type and selection-method filters.

Use `configs/full_pipeline/gpt2_hybrid_3dcim.yaml` or its smoke counterpart.
The older `configs/phase4_quality/default.yaml` uses a different schema
(`noise`, `evaluation`, and `output` sections) and is not a drop-in
configuration for the current runner.

## Evaluation design

### 1. Select the operating points

Phase 4 discards capacity-infeasible points, applies:

- `phase4.evaluate_budget_types`;
- `phase4.evaluate_selection_methods`; and
- any `--digital-set-id` constraints.

It orders the remaining points from the smallest digital projection count and
cost upward. If `phase4.max_operating_points` is greater than one, evenly
spaced indices are chosen with both ends of the available frontier included. A
limit of one selects only the cheapest point.

The primary configuration evaluates up to three points from the measured greedy
frontier. The smoke configuration evaluates up to two explicitly named points.

### 2. Build the evaluation corpus

The tokenizer and `AutoModelForCausalLM` are loaded from `model.name`. Phase 4
uses `evaluation_dataset` when present and otherwise falls back to `dataset`.
The shared batching code:

- concatenates non-empty documents with `document_separator`;
- tokenizes deterministically;
- optionally truncates to `max_tokens`;
- creates fixed windows of `sequence_length` with the configured `stride`;
- masks already-scored overlap and padding with `-100`; and
- reports token-weighted NLL and `exp(NLL)`.

The primary configuration deliberately separates validation data used by Phase
1 and Phase 1.5 from held-out WikiText test data used here.

### 3. Measure the digital reference

The untouched Hugging Face model is evaluated once:

```text
digital_nll
digital_ppl = exp(digital_nll)
```

These values are the common reference for all digital operating points and
placement policies.

### 4. Construct one nominal hybrid model per digital set

For each selected operating point, protected projections stay as their original
digital modules. Every remaining Phase 1 candidate is converted to
AIHWKit `AnalogLinearMapped`.

Before conversion, each canonical `[out, in]` projection weight is:

1. converted from GPT-2's stored orientation where necessary;
2. clipped once to `±clip_sigma × population_std`;
3. assigned a programmed range using `peak_to_peak` or `absmax`; and
4. written into AIHWKit while preserving its clean mapping scales.

For current Phase 1 artifacts, Phase 4 compares the resulting preprocessing
metadata and checksums with Phase 1. A mismatch in the original weight, clipped
weight, range mode, standard deviation, threshold, or programmed range stops
the run. Legacy profiles without a nested `preprocessing` mapping skip this
strict check.

The nominal hybrid evaluation contains clipping and the configured AIHWKit
analog I/O path, but no Phase 2 tile noise. It establishes:

```text
nominal_hybrid_nll
delta_nll_nominal_vs_digital = nominal_hybrid_nll - digital_nll
```

### 5. Materialize tile noise in logical weight coordinates

AIHWKit's internal programming, read, drift, and forward weight noise are
disabled. Phase 2 noise is added manually exactly once.

Let `W0,p` be the clipped nominal weight for projection `p`, `Rp` its
programmed range, and `Zp,r` an i.i.d. standard-normal field keyed by the
experiment seed, projection ID, and realization `r`. For the coordinates of
shard `s` assigned to tile `i`:

```text
Wnoisy,p[s] = W0,p[s] + sign × noise_std[t, i] × Rp × Zp,r[s]
```

There is no post-noise clipping.

The same coordinate field is reused across policies, timesteps, and operating
points wherever that projection remains analog. Only the tile-dependent scale
changes. This pairing reduces variance in policy comparisons.

If `phase4.antithetic` is true, the runner evaluates both `+Z` and `-Z` and
averages their NLL. Otherwise it evaluates only `+Z`.

### 6. Evaluate static placements over time

Phase 3 assignments remain fixed. At each requested timestep, Phase 4 replaces
the placement CSV's original tile-noise value with the current value from the
trace.

If a tile has become unavailable, the current implementation does not remap its
shards or route them digitally. It substitutes `phase4.unavailable_noise_std`,
falling back to the Phase 2 maximum if that field is absent. `faulted_shards`
and `unavailable_shards` are reported for interpretation.

For every digital set × timestep × realization × policy, Phase 4 reports:

```text
delta_nll_total = noisy_nll - digital_nll
delta_nll_tile  = noisy_nll - nominal_hybrid_nll
```

The first includes nominal hybrid conversion cost and tile noise. The second
isolates the additional degradation associated with the materialized tile
noise.

## Timesteps

When `phase4.timesteps` is a non-empty list, values are clamped to the trace
range, deduplicated, and sorted.

When it is null or empty, Phase 4 chooses:

- timestep 0;
- the midpoint;
- the final timestep; and
- the timestep immediately before and at the earliest scheduled fault onset,
  when a fault exists.

A timestep is a dimensionless, frozen hardware-state snapshot for one complete
dataset evaluation. It is not a wall-clock duration.

## Configuration

The current runner reads the following unified fields:

| Field | Meaning |
| --- | --- |
| `model.name` | Hugging Face model/tokenizer identifier |
| `model.device` | PyTorch device such as `cpu` or `cuda` |
| `evaluation_dataset` | Held-out dataset/windowing configuration; falls back to `dataset` |
| `analog.*` | Shared Phase 1/4 clipping, range, mapped-tile, ADC/DAC, bound, and scaling settings |
| `profiling.include_lm_head` | Whether the language-model head belongs to the candidate universe |
| `phase4.output_root` | Final artifact directory |
| `phase4.policies` | Placement policies to evaluate |
| `phase4.timesteps` | Explicit trace snapshots, or null for automatic selection |
| `phase4.num_realizations` | Gaussian fields per timestep and policy |
| `phase4.antithetic` | Evaluate paired `±Z` fields |
| `phase4.unavailable_noise_std` | Substitute scale for unavailable tiles |
| `phase4.evaluate_budget_types` | Allowed Phase 1.5 budget types |
| `phase4.evaluate_selection_methods` | Allowed Phase 1.5 selection methods |
| `phase4.max_operating_points` | Maximum evenly sampled frontier points; null evaluates all |

Example:

```yaml
model:
  name: gpt2
  device: cpu

phase4:
  output_root: data/results/phase4_hybrid_quality
  policies:
    - random
    - sequential
    - hardware_only
    - static_sensitivity
  timesteps: [0, 30, 60, 90, 119]
  num_realizations: 3
  antithetic: false
  unavailable_noise_std: 0.080
  evaluate_budget_types: [greedy_step]
  evaluate_selection_methods:
    - greedy_measured_gain_per_cost_per_macs_per_token
  max_operating_points: 3
```

The exact selection-method string depends on the Phase 1.5 objective and cost
field. It must match the value stored in the operating-point artifact.

## Artifacts

The default output directory contains:

| Artifact | Contents |
| --- | --- |
| `hybrid_quality_by_policy.csv` | Every digital-set/timestep/realization/policy evaluation |
| `nominal_hybrid_frontier.csv` | Digital and nominal-hybrid quality and cost for each selected point |
| `hybrid_quality_summary.csv` | Mean and standard deviation by digital set and policy, plus a bootstrap 95% interval for tile-noise ΔNLL |
| `paired_policy_summary.csv` | Paired `static_sensitivity` improvements over each baseline |
| `phase4_metadata.json` | Provenance, dataset metadata, analog settings, references, selected points, and artifact paths |

During a long run, `hybrid_quality_by_policy.partial.csv` is rewritten when the
runner leaves each operating-point evaluation, including a partially completed
point when rows were produced before an error. It is removed after the final
CSV is written successfully.

Important columns in `hybrid_quality_by_policy.csv` include:

- operating-point identity, selection method, budget, and digital cost;
- policy, timestep, and realization;
- noisy, digital, and nominal-hybrid NLL/PPL;
- total and tile-only quality deltas;
- the sensitivity-weighted placement proxy;
- materialized weight-noise RMS;
- faulted and unavailable shard counts; and
- predicted token count.

`ppl_from_mean_nll` is `exp(mean_nll)` and is the primary PPL representation.
When antithetic evaluation is enabled, `ppl_mean` is instead the arithmetic
mean of the per-sign perplexities.

The paired summary defines:

```text
NLL improvement = baseline delta_nll_tile - static_sensitivity delta_nll_tile
```

A positive value therefore means `static_sensitivity` produced lower NLL. The
bootstrap intervals resample paired timestep/realization differences.

## Sanity checks

After choosing a generated `digital_set_id`, run:

```bash
python3 experiments/phase4_quality/run_sanity_checks.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --operating-points data/results/phase1_5_digital_selection/digital_operating_points.json \
  --phase3-manifest data/results/phase3_static_mapping/phase3_manifest.json \
  --digital-set-id <digital_set_id>
```

This uses at most 4,096 tokens and verifies two invariants:

- zero tile noise reproduces nominal-hybrid NLL for every policy; and
- uniform tile noise makes NLL invariant to placement policy.

The default tolerance for each NLL check is `1e-6`. Optional unified config
fields `phase4.sanity_zero_nll_tolerance` and
`phase4.sanity_uniform_nll_tolerance` can override it.

The temporary digital-module swap machinery also has focused tests:

```bash
python3 -m pytest -q tests/test_hybrid_digital_swap.py
```

This test requires an importable AIHWKit installation.

## Cost and scaling

The number of noisy dataset passes is:

```text
operating_points × timesteps × realizations × policies × antithetic_signs
```

The primary configuration can perform 180 full noisy passes
(3 × 5 × 3 × 4), plus the digital and nominal-hybrid references. Each pass may
score up to 65,536 tokens. Start with
`configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml` and confirm the AIHWKit
contract before launching a paper-scale run.

## Interpretation and limitations

- Results measure NLL/PPL under this manually materialized Gaussian noise model;
  they are not measurements from physical hardware.
- Static placements are not adapted after drift or faults.
- Unavailability is approximated as a high noise scale, not zero output,
  infeasibility, or digital fallback.
- The main runner does not compute KL divergence, token agreement, latency,
  energy, communication cost, or migration cost.
- The placement proxy is diagnostic. Only the forward evaluation measures
  model-level quality and cross-projection interactions.
- The language-model head is weight-tied to the embedding in GPT-2. Phase 1.5
  reports its logical digital execution cost while excluding a duplicate tied
  copy from incremental storage cost.
- Model and dataset downloads must already be cached when running offline.
- No benchmark results are currently committed. Most generated result paths
  under `data/` are not ignored automatically, so inspect `git status` before
  committing.

See the [repository README](../README.md) for installation, smoke testing,
resume commands, and multi-seed execution.
