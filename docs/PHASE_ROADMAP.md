# Fidelity-Aware Projection Mapping for GPT-2 on 3D Analog CIM

> **Scope note:** This project concludes at Phase 4. Adaptive remapping and the
> end-to-end architectural / paper evaluation (formerly Phases 5 and 6) are out
> of scope and are not implemented in this repository.

## Project Objective

Develop and validate a fidelity-aware mapping workflow for GPT-2 inference on heterogeneous 3D analog compute-in-memory hardware.

The project combines:

- GPT-2 projection-sensitivity profiling,
- time-varying tile-fidelity simulation,
- static projection-to-tile mapping,
- tile-level weight-noise injection,
- language-model quality evaluation.

The central hypothesis is that protecting noise-sensitive GPT-2 projections with higher-fidelity hardware preserves language-model quality better than hardware-only or sensitivity-unaware mapping.

---

# Phase 0: Exploratory Semantic and Layer-Level Analysis

**Status:** Completed exploratory work

## Goal

Investigate whether semantic or attention-derived signals provide useful evidence about which GPT-2 layers or projections are most sensitive to hardware noise.

## Main Analyses

- Attention entropy by transformer layer.
- Attention received per token.
- Layer-level noise sensitivity.
- Relationship between attention behavior and measured perplexity degradation.

## Main Files

```text
experiments/semantic_profiler/run_layer_attention_entropy.py
experiments/semantic_profiler/run_layer_noise_sensitivity.py
experiments/semantic_profiler/run_token_attention_received.py
```

## Role in the Final Project

Phase 0 is exploratory and is not required by the final mapper. Its results may be used to motivate why different GPT-2 layers have different hardware-noise sensitivity.

## Completion Criteria

- [x] Attention entropy measured across GPT-2 layers.
- [x] Layer-level noise sensitivity evaluated.
- [x] Initial relationship between semantic behavior and sensitivity investigated.

---

# Phase 1: Projection-Sensitivity Profiling

**Status:** Implemented; paper-alignment validation ongoing

## Goal

Measure how strongly each GPT-2 projection affects language-model quality when exposed to controlled analog programming noise.

The primary projection identifiers are:

```text
block_i/attn.c_attn
block_i/attn.c_proj
block_i/mlp.c_fc
block_i/mlp.c_proj
```

for GPT-2 blocks `i = 0, ..., 11`.

## Method

For every projection:

1. Load clean pretrained GPT-2.
2. Apply the selected clipping and analog preprocessing.
3. Inject controlled programming noise into only that projection.
4. Run the same WikiText evaluation subset.
5. Measure perplexity and negative-log-likelihood degradation.
6. Repeat across multiple random noise realizations.
7. Save the projection sensitivity profile and all noise-scale metadata.

## Primary Metrics

Clean negative log-likelihood:

```text
NLL_c
```

Clean perplexity:

```text
PPL_c = exp(NLL_c)
```

Projection sensitivity in perplexity space:

```text
DeltaPPL_p = PPL_p - PPL_c
```

Preferred sensitivity in additive loss space:

```text
S_p = NLL_p - NLL_c
```

or equivalently:

```text
S_p = log(PPL_p / PPL_c)
```

## Required Noise Metadata

Phase 1 must save enough information for Phase 4 to reproduce the same noise units:

```text
projection_id
clip_method
clip_threshold
noise_reference_scale
noise_scale_definition
reference_sigma_normalized
tile_size
input_resolution
output_resolution
seed configuration
dataset configuration
```

Phase 4 must read this metadata instead of independently recomputing the clipping or weight scale.

## Main Files

```text
experiments/phase1_sensitivity/run_sensitivity_profile.py
experiments/phase1_sensitivity/analyze_phase1.py

src/profilers/aihwkit_profiler.py
src/profilers/sensitivity_profiler.py
```

## Expected Outputs

```text
data/results/phase1_sensitivity/<experiment>/
├── sensitivity_profile.json
├── projection_metrics.csv
├── projection_noise_metadata.json
├── config.yaml
└── metadata.json
```

## Completion Criteria

- [x] Clean GPT-2 perplexity is reproducible.
- [x] All 48 GPT-2 projections can be profiled.
- [x] Multiple random noise realizations are supported.
- [x] Projection-level perplexity degradation is recorded.
- [ ] Phase 1 clipping and noise scaling are fully documented.
- [ ] Projection noise-reference scales are saved explicitly.
- [ ] Relative projection-sensitivity trends are compared carefully with the Lammie workflow.
- [ ] Sensitivity is exported in both `DeltaPPL` and `DeltaNLL` form.

---

# Phase 2: Time-Varying Tile-Fidelity Simulation

**Status:** Implemented

## Goal

Model heterogeneous and time-varying reliability across 3D analog CIM tiles.

Each tile has a fidelity state that can change because of:

- initial manufacturing variation,
- gradual degradation,
- thermal fluctuation,
- localized faults,
- hard unavailability.

## Tile Noise Model

For tile `i` at timestep `t`:

```text
sigma_i,t
```

represents the tile's normalized effective programming-noise standard deviation.

A general decomposition is:

```text
sigma_i,t =
    sigma_initial_i
    + sigma_degradation_i,t
    + sigma_thermal_i,t
    + sigma_fault_i,t
```

The final implementation may combine components in variance space when that better matches the physical interpretation:

```text
sigma_i,t^2 =
    sigma_initial_i^2
    + sigma_degradation_i,t^2
    + sigma_thermal_i,t^2
    + sigma_fault_i,t^2
```

## Thermal Fluctuation

The thermal state can be modeled as an AR(1) process:

```text
theta_t = rho theta_t-1 + epsilon_t
epsilon_t ~ Normal(0, sigma_epsilon^2)
```

Its stationary variance is:

```text
Var(theta) = sigma_epsilon^2 / (1 - rho^2)
```

when `|rho| < 1`.

## Fault Semantics

Phase 2 must distinguish:

```text
degraded but operational tile
faulted but operational tile
unavailable tile
```

An unavailable tile must not be represented only as a very large Gaussian noise value. Hard failure behavior is handled explicitly in Phase 4.

## Main Files

```text
experiments/phase2_tile_fidelity/run_tile_fidelity_trace.py
experiments/phase2_tile_fidelity/analyze_tile_fidelity.py

src/hardware/tile_fidelity.py
src/hardware/thermal_model.py
src/hardware/fault_model.py
```

## Expected Outputs

```text
data/results/phase2_fidelity/<experiment>/
├── tile_fidelity_trace.npz
├── tile_fidelity_trace.csv
├── fault_events.csv
├── representative_timesteps.json
├── config.yaml
└── metadata.json
```

## Representative Timesteps

At minimum:

```text
initial
pre-fault
post-fault
final
```

Additional intermediate timesteps should be retained for correlation and adaptive-mapping experiments.

## Completion Criteria

- [x] Tiles have heterogeneous initial fidelity.
- [x] Gradual degradation is modeled.
- [x] Thermal fluctuation is modeled.
- [x] Localized faults are modeled.
- [x] Tile unavailability is represented separately.
- [x] Complete fidelity traces can be exported.
- [x] Representative timesteps can be selected automatically.
- [ ] Normalized tile-noise units are explicitly linked to the Phase 1 noise reference.

---

# Phase 3: Static Projection-to-Tile Mapping

**Status:** Implemented; integration validation ongoing

## Goal

Map GPT-2 projection shards to 3D analog CIM tiles using several static policies and compare their hardware-quality proxy values.

## Static Mapping Policies

### Random

Assign shards to valid physical locations using randomized tile ordering.

### Sequential

Assign shards using deterministic simulator or module order.

### Hardware-Only

Prioritize the lowest-noise tiles using the initial hardware state, without using projection sensitivity.

### Static Sensitivity-Aware

Prioritize sensitive GPT-2 projections and assign them to higher-fidelity tiles.

## Important Static-Mapping Rule

Every static policy creates one fixed placement:

```text
placement_policy = constant across timesteps
```

The hardware trace changes over time:

```text
sigma_i,t = time varying
```

The placement must not be regenerated at every timestep. Recomputing the placement using current tile health would already be adaptive mapping, which is out of scope for this project (which ends at Phase 4).

## Sharding

Each projection is divided into hardware-compatible shards according to:

```text
tier input capacity
tier output capacity
available tiers
tile and tier allocation constraints
```

Every shard record should retain:

```text
projection_id
sim_module_path
shard_id
input_start
input_end
output_start
output_end
tile_id
tier_id
number_of_weights
placement_policy
placement_seed
```

## Phase 3 Proxy

For projection `p`, let:

```text
S_p = Phase 1 projection sensitivity in DeltaNLL space
```

For shard `s` belonging to projection `p`, define:

```text
f_s =
    number_of_weights_in_shard_s
    / number_of_weights_in_projection_p
```

The preferred variance-weighted proxy is:

```text
Q(policy,t) =
    sum_p sum_s_in_p
    f_s S_p
    (sigma_s,t / sigma_reference)^2
```

A linear-sigma proxy may also be reported for comparison:

```text
Q_linear(policy,t) =
    sum_p sum_s_in_p
    f_s S_p sigma_s,t
```

The variance-weighted proxy should be the primary metric because the expected effect of zero-mean noise is more naturally associated with noise variance.

## Main Files

```text
experiments/phase3_baselines/run_baseline_mappings.py
experiments/phase3_baselines/analyze_baseline_mappings.py

src/mapping/placement.py
src/mapping/objective.py
src/mapping/policies.py
src/mapping/sharding.py
src/mapping/hardware_adapter.py
```

## Expected Outputs

```text
data/results/phase3_mapping/<experiment>/
├── placements/
│   ├── random.json
│   ├── sequential.json
│   ├── hardware_only.json
│   └── static_sensitivity.json
├── proxy_by_policy.csv
├── shard_assignments.csv
├── placement_validation.csv
├── config.yaml
└── metadata.json
```

## Completion Criteria

- [x] GPT-2 projections are converted into hardware-compatible shards.
- [x] Capacity constraints are enforced.
- [x] Tile and tier assignments are retained.
- [x] Random mapping is implemented.
- [x] Sequential mapping is implemented.
- [x] Hardware-only mapping is implemented.
- [x] Static sensitivity-aware mapping is implemented.
- [x] Static sensitivity-aware mapping reduces the Phase 3 proxy in heterogeneous cases.
- [ ] The proxy is expressed primarily in `DeltaNLL` and noise-variance space.
- [ ] Placement files contain enough coordinate metadata for exact Phase 4 reconstruction.
- [ ] Every static placement is verified to remain fixed across the complete trace.

---

# Phase 4: Tile-Level Language-Model Quality Validation

**Status:** Next phase

## Goal

Bridge Phase 3 placements back to the actual GPT-2 projection weights, apply tile-specific noise to the mapped weight slices, and measure real language-model quality.

Phase 3 establishes that sensitivity-aware mapping improves a proxy. Phase 4 determines whether this improvement corresponds to lower measured degradation in:

- negative log-likelihood,
- perplexity,
- KL divergence,
- next-token agreement.

The primary hypothesis is:

```text
E[DeltaNLL_static_sensitivity]
    <
E[DeltaNLL_hardware_only]
```

at heterogeneous degraded timesteps.

The expected perplexity trend is:

```text
PPL_clean
    <
PPL_static_sensitivity
    <
PPL_hardware_only
```

in expectation, but this strict ordering is not required for every individual random realization or homogeneous timestep.

---

## Phase 4A: Controlled Tile-Level Weight-Noise Materialization

### Goal

Apply the Phase 2 tile-noise field directly to the exact GPT-2 weight slices selected by Phase 3, then run ordinary Hugging Face GPT-2 inference.

This is the primary Phase 4 experiment because it is transparent, deterministic, and easy to validate.

### Interpretation

Phase 4A is:

```text
tile-level programmed-weight-noise materialization
followed by digital GPT-2 inference
```

It is not yet a full analog-forward AIHWKit simulation.

---

## Phase 4.1: Placement-to-GPT-2 Bridge

### Goal

Translate IBM 3D-CIM simulator module placements into exact Hugging Face GPT-2 weight coordinates.

### Module Mapping

```text
out_proj -> block_i/attn.c_proj
ffn1     -> block_i/mlp.c_fc
ffn2     -> block_i/mlp.c_proj
```

The Q/K/V bridge requires layer-aware handling:

```text
layer_0 q_proj_in -> block_0/attn.c_attn Q slice
layer_0 k_proj_in -> block_0/attn.c_attn K slice
layer_0 v_proj_in -> block_0/attn.c_attn V slice
```

When the simulator represents next-layer Q/K/V generation using `q_proj_out`, `k_proj_out`, and `v_proj_out`, these must be mapped to the next GPT-2 block:

```text
layer_i q_proj_out -> block_i+1/attn.c_attn Q slice
layer_i k_proj_out -> block_i+1/attn.c_attn K slice
layer_i v_proj_out -> block_i+1/attn.c_attn V slice
```

The exact naming and layer shift must be validated against the simulator version used by the project.

### Canonical Matrix Orientation

All evaluation code should use:

```text
W_canonical in R[out_features, in_features]
```

Hugging Face GPT-2 `Conv1D` stores weights as:

```text
[in_features, out_features]
```

Therefore:

```text
W_canonical = W_hf.T
W_hf = W_canonical.T
```

The bridge must convert simulator input ranges to canonical columns and simulator output ranges to canonical rows.

### GPT-2 Small Canonical Shapes

```text
attn.c_attn: [2304, 768]
attn.c_proj: [768, 768]
mlp.c_fc:    [3072, 768]
mlp.c_proj:  [768, 3072]
```

### Fused Q/K/V Coordinates

For hidden size `d`:

```text
Q rows: [0, d)
K rows: [d, 2d)
V rows: [2d, 3d)
```

For GPT-2 small:

```text
Q rows: [0, 768)
K rows: [768, 1536)
V rows: [1536, 2304)
```

### Assignment Record

Each mapped slice should contain:

```text
projection_id
hf_module_path
sim_module_path
sim_layer
qkv_component

canonical_row_start
canonical_row_end
canonical_col_start
canonical_col_end

tile_id
tier_id
policy
placement_seed
timestep

tile_noise_std_normalized
tile_noise_std_absolute
is_faulted
is_available
```

### Required Bridge Validation

For every projection:

```text
coverage[row_start:row_end, col_start:col_end] += 1
```

Require:

```text
coverage == 1 for every analog-mapped weight
coverage == 0 for every explicitly digital weight
```

Any value greater than one indicates overlapping shard assignments.

---

## Phase 4.2: Noise Materialization

### Normalized and Absolute Noise

Phase 2 tile noise should be interpreted as normalized noise:

```text
sigma_normalized_i,t
```

It must not be added directly to raw GPT-2 weights unless it is already expressed in absolute weight units.

For projection `p`, load the Phase 1 reference scale:

```text
r_p = projection-specific noise reference scale
```

Then:

```text
sigma_absolute_s,t =
    r_p sigma_normalized_tile(s),t
```

For shard `s`:

```text
W'_s =
    W_programmed_s
    + sigma_absolute_s,t Z_s
```

where:

```text
Z_s ~ Normal(0, 1)
```

### Shared Analog Preprocessing

Phase 1 and Phase 4 must use the same function for:

```text
weight clipping
weight normalization
deterministic programming preprocessing
noise reference scaling
```

The only intended difference is:

```text
Phase 1:
    one uniform sigma for the tested projection

Phase 4:
    a spatial sigma map determined by tile placement
```

### Paired Noise Realizations

For realization seed `k`, generate one standard-normal tensor per projection:

```text
Z_p,k ~ Normal(0, 1)
```

Use the same `Z_p,k` for all policies:

```text
epsilon_policy,p =
    sigma_map_policy,p elementwise-multiplied by Z_p,k
```

This reduces comparison variance and makes policy differences attributable to tile-noise assignment.

Keep these seeds separate:

```text
trace_seed
placement_seed
noise_realization_seed
dataset_seed
```

---

## Phase 4.3: Static Policy Evaluation Across Time

Load each static placement once:

```text
placement = load_static_placement(policy)
```

Then evaluate the changing Phase 2 trace:

```text
for timestep in selected_timesteps:
    sigma_map = apply_trace_to_fixed_placement(
        placement,
        tile_trace[timestep],
    )
```

Do not regenerate static placements at each timestep.

### Required Timesteps

```text
initial
pre-fault
post-fault
final
```

Additional intermediate timesteps should be included when measuring proxy-to-quality correlation.

### Expected Sanity Behavior

At a homogeneous initial state:

```text
E[PPL_random]
≈ E[PPL_sequential]
≈ E[PPL_hardware_only]
≈ E[PPL_static_sensitivity]
```

At heterogeneous degraded states:

```text
E[DeltaNLL_static_sensitivity]
<
E[DeltaNLL_hardware_only]
```

when the Phase 3 proxy is meaningful.

---

## Phase 4.4: Quality Evaluation Protocol

For every:

```text
policy
timestep
trace seed
placement seed
noise realization seed
```

perform:

1. Load or restore the clean GPT-2 checkpoint.
2. Load the fixed Phase 3 placement.
3. Load tile fidelity at the selected timestep.
4. Build the canonical per-weight sigma map.
5. Apply the same deterministic preprocessing used in Phase 1.
6. Materialize paired tile-level weight noise.
7. Run GPT-2 on the same token batches used by Phase 1.
8. Record model-quality metrics.
9. Restore all modified weights exactly.
10. Save provenance and checksums.

### Inference Configuration

```text
model.eval()
torch.no_grad()
use_cache = false
```

Use the same:

```text
dataset
tokenizer
sequence length
stride
batch size
document separator
drop-incomplete-sequence rule
```

as Phase 1 whenever possible.

---

## Primary Phase 4 Metrics

### Negative Log-Likelihood

```text
NLL =
    total valid-token cross-entropy
    / total valid-token count
```

Do not average batch perplexities.

### Perplexity

```text
PPL = exp(NLL)
```

### NLL Degradation

```text
DeltaNLL_policy =
    NLL_policy - NLL_clean
```

This is the primary quality-degradation metric.

### KL Divergence

Use:

```text
KL(clean || noisy)
```

averaged over the same valid next-token positions.

### Next-Token Agreement

```text
agreement =
    mean[
        argmax(logits_clean)
        ==
        argmax(logits_noisy)
    ]
```

### NLL Preservation Gain

Relative to hardware-only:

```text
gain_hardware =
    (DeltaNLL_hardware - DeltaNLL_static)
    / DeltaNLL_hardware
```

when `DeltaNLL_hardware` is sufficiently positive.

### Perplexity Preservation Ratio

```text
(PPL_hardware - PPL_static)
/
(PPL_hardware - PPL_clean)
```

This should be reported as a secondary metric because it can become unstable when the denominator is close to zero.

### Proxy Correlation

Measure:

```text
Phase 3 proxy vs DeltaNLL
Phase 3 proxy vs KL(clean || noisy)
```

Use:

```text
Spearman correlation as primary
Pearson correlation as secondary
bootstrap confidence intervals
```

---

## Statistical Evaluation

Use paired comparisons across identical noise realizations.

For each paired run:

```text
D_k =
    DeltaNLL_hardware,k
    - DeltaNLL_static,k
```

Report:

```text
mean(D)
standard deviation
95% confidence interval
fraction of realizations with D > 0
```

The primary success condition is:

```text
mean(D) > 0
```

with a confidence interval supporting a consistent advantage.

A strict ordering is not required for every individual seed.

---

## Hard Fault Evaluation

Hard unavailability must be evaluated separately from ordinary Gaussian degradation.

### Experiment A: Operational Degradation

```text
all mapped shards remain executable
tile fidelity changes through sigma values
```

This is the main static placement-quality experiment.

### Experiment B: Hard Failure

For an unavailable tile, choose and document one explicit behavior:

```text
zero the affected shard output
mark the placement infeasible
use a digital fallback
```

Adaptive remapping in response to failures is out of scope for this project (which ends at Phase 4).

Phase 4 should not silently convert unavailable tiles into a very large Gaussian sigma.

---

## Phase 4B: Optional AIHWKit-Calibrated Shard Validation

### Goal

Cross-check the controlled Phase 4A results using explicit AIHWKit analog shard modules.

Each mapped shard should be represented by an independently configured analog module so that it can receive its own tile-specific hardware parameters.

The forward pass reconstructs each projection from shard partial sums:

```text
y_output_block =
    sum over input shards
    AnalogShard(output_block, input_block)(x_input_block)
```

Bias is added once after shard accumulation.

### Purpose

Phase 4B evaluates whether the Phase 4A conclusions remain valid when additional analog-forward effects are included, such as:

```text
DAC quantization
ADC quantization
output bounds
analog forward noise
partial-sum behavior
device-specific programming behavior
```

Phase 4B should follow Phase 4A and should not block the primary bridge validation.

---

## Main Files

```text
experiments/phase4_quality/run_tile_noise_perplexity.py
experiments/phase4_quality/analyze_perplexity_results.py

src/evaluation/schemas.py
src/evaluation/placement_to_gpt2.py
src/evaluation/noise_materialization.py
src/evaluation/tile_noise_injection.py
src/evaluation/perplexity_evaluator.py

tests/evaluation/test_placement_to_gpt2.py
tests/evaluation/test_tile_noise_injection.py
tests/evaluation/test_perplexity_evaluator.py
```

## Expected Outputs

```text
data/results/phase4_quality/<experiment>/
├── seed_<seed>/
│   ├── quality_by_policy.csv
│   ├── quality_by_timestep.csv
│   ├── paired_policy_differences.csv
│   ├── projection_noise_assignments.csv
│   ├── tile_noise_injection_records.csv
│   ├── proxy_vs_quality_correlation.csv
│   ├── weight_checksums.csv
│   ├── config.yaml
│   └── metadata.json
└── aggregate/
    ├── aggregate_quality.csv
    ├── confidence_intervals.csv
    ├── ordering_success_rates.csv
    ├── proxy_correlations.csv
    └── figures/
```

---

## Required Phase 4 Tests

### Orientation Test

For every GPT-2 projection:

```text
Hugging Face Conv1D output
==
manual linear output using W_hf.T
```

within numerical tolerance.

### Q/K/V Reconstruction Test

Split and concatenate fused `attn.c_attn` canonical rows:

```text
cat(Q, K, V) == original canonical c_attn weight
```

### Coverage Test

Every analog-mapped weight must be covered exactly once.

### Zero-Noise Test

With all tile sigmas equal to zero:

```text
clean logits == injected-model logits
clean NLL == zero-noise NLL
```

within numerical tolerance.

### Uniform-Noise Mapping-Invariance Test

When every tile has the same sigma and paired projection-coordinate noise is used:

```text
all policies produce the same perturbed weights
```

assuming they map the same set of weights to analog hardware.

### Single-Tile Test

Set one tile to a large sigma and all other tiles to zero. Verify that only slices mapped to that tile change.

### Restoration Test

After evaluation, every modified GPT-2 tensor must exactly match its clean checkpoint value.

### Determinism Test

The same configuration and seeds must reproduce identical:

```text
assignment records
weight checksums
quality metrics
```

---

## Recommended Phase 4 Implementation Order

### Step 1

Bridge only:

```text
block_0/attn.c_proj
```

This avoids Q/K/V fusion and verifies basic orientation and slicing.

### Step 2

Bridge:

```text
block_0/attn.c_attn
```

Validate Q/K/V offsets and fused-matrix reconstruction.

### Step 3

Generate assignments for all 48 projections without modifying the model.

Run complete coverage and overlap checks.

### Step 4

Run a zero-noise full-model experiment and reproduce the Phase 1 clean perplexity.

### Step 5

Run a uniform-sigma experiment. All policies should become equivalent under paired projection-coordinate noise.

### Step 6

Run an artificial two-class fidelity experiment:

```text
good tiles: low sigma
bad tiles: high sigma
```

Use a large enough gap to make bridge errors easy to detect.

### Step 7

Run the real Phase 2 trace at representative timesteps.

### Step 8

Add optional AIHWKit shard-level validation.

---

## Phase 4 Completion Criteria

- [ ] Phase 3 placement files load automatically.
- [ ] Simulator modules resolve to the correct GPT-2 blocks and projections.
- [ ] Any `q_proj_out` next-layer offset is handled correctly.
- [ ] Simulator coordinates convert correctly into canonical GPT-2 coordinates.
- [ ] Fused Q/K/V slices reconstruct `attn.c_attn` exactly.
- [ ] `tile_id` and `tier_id` are retained.
- [ ] Every analog-mapped weight is covered exactly once.
- [ ] Normalized and absolute noise units are recorded.
- [ ] Phase 1 clipping and noise-reference logic are reused.
- [ ] Paired noise realizations are supported.
- [ ] Zero noise reproduces clean logits and perplexity.
- [ ] Uniform tile noise makes policy choice irrelevant.
- [ ] Static placements remain unchanged across timesteps.
- [ ] Clean, random, sequential, hardware-only, and static sensitivity-aware quality can be measured.
- [ ] Mean paired `DeltaNLL` differences include confidence intervals.
- [ ] Phase 3 proxy correlation is measured against `DeltaNLL` and KL divergence.
- [ ] Hard unavailability is handled explicitly.
- [ ] Phase 4 validates or falsifies the static sensitivity-aware quality hypothesis.

---

# End-to-End Experimental Flow

```text
Phase 1
GPT-2 projection sensitivity
        |
        v
Phase 2
Time-varying tile fidelity
        |
        v
Phase 3
Static projection-to-tile placements
        |
        v
Phase 4
Exact placement-to-weight bridge
and measured GPT-2 quality
```

---

# Central Validation Targets

## Static Mapping Target

At heterogeneous operational timesteps:

```text
E[DeltaNLL_static_sensitivity]
<
E[DeltaNLL_hardware_only]
```

## Homogeneous-State Sanity Target

When all tiles have equal fidelity:

```text
E[quality_random]
≈ E[quality_sequential]
≈ E[quality_hardware_only]
≈ E[quality_static_sensitivity]
```

## Proxy Validation Target

```text
Phase 3 proxy
positively correlates with
measured DeltaNLL and KL divergence
```
