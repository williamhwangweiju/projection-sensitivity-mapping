# Phase 2: Tile-Fidelity Trace Simulation

Phase 2 generates a deterministic, time-varying hardware trace for the
heterogeneous 3D-CIM substrate. Each trace value is a per-tile normalized
logical-weight noise standard deviation. Later phases use these values to place
projection shards and to scale manually materialized Gaussian weight noise.

Phase 2 does not load GPT-2, run AIHWKit inference, or simulate device
conductance directly. It is a lightweight NumPy simulation of tile-level noise,
availability, thermal grouping, and fault state.

> [!WARNING]
> The unified pipeline configurations are the canonical runnable inputs:
> [`configs/full_pipeline/gpt2_hybrid_3dcim.yaml`](../configs/full_pipeline/gpt2_hybrid_3dcim.yaml)
> and
> [`configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml`](../configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml).
> In `configs/phase2_fidelity/`, `static.yaml`, `gradual_drift.yaml`,
> `thermal_variation.yaml`, and `localized_fault.yaml` are empty placeholders.
> `mixed.yaml` uses a legacy, incompatible layout: it lacks the required
> `phase2` wrapper and places `degradation` outside `fidelity_model`. Do not pass
> those scenario files to the current runner unless they are first migrated to
> the unified schema shown below.

## Quick start

From the repository root, run the primary experiment:

```bash
python3 experiments/phase2_fidelity/run_fidelity_model.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

The script defaults to this same configuration. Override the configured trace
seed when needed:

```bash
python3 experiments/phase2_fidelity/run_fidelity_model.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --seed 7
```

For a four-timestep structural run, use the smoke configuration:

```bash
python3 experiments/phase2_fidelity/run_fidelity_model.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml
```

Only NumPy and PyYAML are needed by the Phase 2 runner. The full project
dependencies are listed in `requirements.txt`. The full-pipeline orchestrator
invokes the same runner automatically unless `--skip-phase2` is supplied.

## Model

Let `N` be the number of tiles, `T` the number of timesteps, and `R` the
configured `reference_noise_std`. All random draws use one
`numpy.random.default_rng(seed)` instance.

### Initial fidelity classes and baseline noise

For each class `c`, the simulator creates

```text
round(class_fraction[c] * N)
```

labels in YAML insertion order. It pads a short list with the literal class
name `medium`, truncates a long list to `N`, and then shuffles the labels. For a
tile `i` assigned to class `c`, baseline noise is

```text
b_i = R * Uniform(class_range[c].low, class_range[c].high)
```

Fractions should therefore sum to exactly `1.0`, and canonical configurations
should define `high`, `medium`, and `low`. The stored `class_code` is assigned by
alphabetically sorting class names. With the canonical names, `high = 0`,
`low = 1`, and `medium = 2`.

### Gradual drift

When drift is enabled, each tile receives a total fractional increase

```text
d_i ~ Uniform(total_increase_range.low, total_increase_range.high)
p_t = 0                              if T <= 1
p_t = t / (T - 1)                    otherwise
```

Drift contributes `d_i * p_t`; it is zero at timestep 0 and reaches `d_i` at
the final timestep.

### Thermal variation

Thermal-zone IDs are formed from `arange(N) % num_thermal_zones` and shuffled,
so zone populations differ by at most one tile. Tiles in the same zone share
the same fractional thermal state. Starting from `H_z,-1 = 0`, the state is
updated at every timestep, including timestep 0:

```text
epsilon_z,t ~ Normal(
  0,
  standard_deviation_fraction * sqrt(max(1 - correlation^2, 0))
)

H_z,t = correlation * H_z,t-1 + epsilon_z,t
```

Consequently, an enabled thermal process can perturb the initial trace row. An
enabled process with `abs(correlation) < 1` approaches the configured stationary
standard deviation but does not use a burn-in period.

### Localized faults

The model samples faulted tiles without replacement from `candidate_classes`.
If fewer candidates exist than requested, it silently uses all available
candidates. For each selected tile, onset `o_i` is sampled from the inclusive
integer range `onset_timestep_range`, and severity is sampled as

```text
g_i ~ Uniform(noise_increase_range.low, noise_increase_range.high)
```

At and after onset, the tile is marked faulted and its noise is multiplied by
`1 + g_i`. If `make_unavailable` is true, it is also unavailable from that
timestep onward. Faults are always permanent in the current implementation;
a `permanent` configuration field is not read.

### Effective noise and fidelity

Before clipping, the tile noise is

```text
q_i,t = b_i * (1 + d_i * p_t + H_zone(i),t)

raw_i,t = q_i,t                              before fault onset
raw_i,t = q_i,t * (1 + g_i)                  at/after fault onset

noise_i,t = clip(raw_i,t, min_noise_std, max_noise_std)
```

Disabled drift and thermal terms are zero; an unaffected tile has no fault
multiplier. The descriptive fidelity score is

```text
fidelity_i,t = 1 / (1 + noise_i,t / max(R, 1e-12))
```

Thus a tile at the reference noise has fidelity `0.5`, and lower noise produces
a higher score. Downstream evaluation consumes `noise_std`, not
`fidelity_score`.

The normalized noise unit means that Phase 4 materializes a projection-weight
perturbation as

```text
delta_weight = standard_normal * noise_i,t * programmed_projection_range
```

The trace therefore contains noise scales, not pre-sampled weight errors.
AIHWKit's internal programming, read, drift, and forward weight-noise paths are
disabled so this manual path is the single source of weight noise.

## Canonical configuration schema

`degradation` must be nested inside `phase2.fidelity_model`. A complete minimal
shape is:

```yaml
experiment:
  seed: 42

hardware:
  num_tiles: 96
  tiers_per_tile: 8
  tier_shape:
    rows: 512
    cols: 512
  num_thermal_zones: 8

phase2:
  name: mixed_96x8
  output_root: data/results/phase2_fidelity/fidelity_traces
  fidelity_model:
    num_timesteps: 120
    reference_noise_std: 0.023
    min_noise_std: 0.005
    max_noise_std: 0.080
    fidelity_classes:
      high:
        fraction: 0.25
        noise_multiplier_range: [0.65, 0.85]
      medium:
        fraction: 0.50
        noise_multiplier_range: [0.90, 1.15]
      low:
        fraction: 0.25
        noise_multiplier_range: [1.35, 1.70]
    degradation:
      gradual_drift:
        enabled: true
        total_increase_range: [0.12, 0.35]
      thermal_variation:
        enabled: true
        correlation: 0.94
        standard_deviation_fraction: 0.05
      localized_fault:
        enabled: true
        num_affected_tiles: 8
        onset_timestep_range: [35, 90]
        noise_increase_range: [0.25, 0.70]
        candidate_classes: [high, medium, low]
        make_unavailable: false
```

The fields used by Phase 2 are:

| Field | Requirement or default | Meaning |
| --- | --- | --- |
| `experiment.seed` | Required unless `--seed` is supplied | Seed for all trace randomness. |
| `hardware.num_tiles` | Required, positive | Number of independently modeled tile states. |
| `hardware.tiers_per_tile` | Required, positive | Physical tiers per tile; used downstream and recorded in metadata, but does not change trace generation. |
| `hardware.tier_shape.rows`, `.cols` | Required, positive | Tier dimensions used by later sharding; do not change trace generation. |
| `hardware.num_thermal_zones` | Default `1`, positive | Number of shared AR(1) thermal states. |
| `phase2.name` | Required | Output scenario directory name. `experiment.name` is ignored here. |
| `phase2.output_root` | Required | Output root; relative paths resolve from the repository root. |
| `num_timesteps` | Required | Number of rows in the trace. Use a positive integer. |
| `reference_noise_std` | Required | Baseline normalized noise scale and fidelity-score reference. |
| `min_noise_std`, `max_noise_std` | Required | Final clipping bounds. |
| `fidelity_classes` | Required | Class mappings containing `fraction` and `noise_multiplier_range`. |
| `gradual_drift.enabled` | Default `true` | Enables linear fractional drift. |
| `gradual_drift.total_increase_range` | Required when enabled | Per-tile total fractional increase at the final timestep. |
| `thermal_variation.enabled` | Default `true` | Enables shared zone-level AR(1) variation. |
| `thermal_variation.correlation` | Default `0.94` | AR(1) coefficient. Keep it within `[-1, 1]`. |
| `thermal_variation.standard_deviation_fraction` | Default `0.05` | Intended stationary fractional standard deviation. |
| `localized_fault.enabled` | Default `true` | Enables abrupt permanent faults. |
| `localized_fault.num_affected_tiles` | Default `0` | Requested number of fault candidates. |
| `localized_fault.candidate_classes` | Default: all classes | Classes eligible for fault selection. |
| `localized_fault.onset_timestep_range` | Default `[T, T]` | Inclusive onset range; the default schedules no active in-trace fault. |
| `localized_fault.noise_increase_range` | Default `[0.2, 0.6]` | Fractional multiplicative fault severity. |
| `localized_fault.make_unavailable` | Default `false` | Removes a faulted tile from available placement capacity. |

All three degradation mappings must currently exist even when their `enabled`
value is false. The implementation performs limited validation, so the config
author must ensure positive `num_timesteps` and reference noise, nonnegative
noise ranges, `min_noise_std <= max_noise_std`, valid range ordering, sensible
class fractions, and an AR(1) correlation in `[-1, 1]`.

## Artifacts and schemas

The output directory is

```text
<phase2.output_root>/<phase2.name>/seed_<effective_seed>/
├── trace.npz
├── metadata.json
└── timestep_summary.csv
```

Rerunning the same name and seed overwrites these files without an overwrite
prompt. No effective-config copy, per-tile CSV, or plot is generated.

### `trace.npz`

The compressed NumPy archive contains:

| Key | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `noise_std` | `float32` | `[T, N]` | Normalized per-tile noise standard deviation. |
| `fidelity_score` | `float32` | `[T, N]` | Descriptive score derived from `noise_std`. |
| `available` | `bool` | `[T, N]` | Whether each tile is available. |
| `faulted` | `bool` | `[T, N]` | Whether fault onset has occurred. |
| `class_code` | `int16` | `[N]` | Initial class code from alphabetically sorted class names. |
| `thermal_zone` | `int16` | `[N]` | Fixed thermal-zone assignment. |
| `fault_onset` | `int32` | `[N]` | Inclusive onset timestep; `-1` means no scheduled fault. |

The class-name/code mapping is not embedded in the archive. Retain the exact
source configuration whose SHA-256 is recorded, and reconstruct the mapping as
`enumerate(sorted(fidelity_classes))`.

### `metadata.json`

Metadata records `seed`, `repository_commit`, absolute `config_path`,
`config_sha256`, `noise_unit`, `num_tiles`, `tiers_per_tile`, `num_timesteps`,
`mean_noise_initial`, `mean_noise_final`, and `final_faulted_tiles`. The last
field counts tiles whose fault has activated by the final row; it is not an
unavailability count.

### `timestep_summary.csv`

There is one row per timestep with the columns `timestep`, `mean_noise_std`,
`mean_fidelity`, `faulted_tiles`, and `available_tiles`.

## Downstream contracts

Phase 3 loads `noise_std[phase3.mapping_timestep]` and the matching `available`
row. An unavailable tile contributes no slots; otherwise every tier on a tile
inherits the same tile noise value. The placement is static after it is built.

Phase 4 keeps those physical assignments and replaces each placement row's
noise with the current trace value at each evaluated timestep. If a tile becomes
unavailable after mapping, Phase 4 substitutes `phase4.unavailable_noise_std`
(falling back to the Phase 2 maximum) rather than remapping its shards.
`faulted` is used for reporting, and `fault_onset` can help select representative
timesteps when none are configured. `fidelity_score`, `class_code`, and
`thermal_zone` are currently informational.

The pipeline contract validator checks that the trace tile count matches
`hardware.num_tiles`, but it does not verify that the Phase 2 reference matches
`analog.reference_noise_std`, which Phase 1 uses to characterize projection
sensitivity. Keep them consistent so the characterized operating point matches
the tile-noise regime used for placement and Phase 4 evaluation.

## Reproducibility and tests

The same effective seed, configuration content and mapping order produce the
same trace. The runner records the Git commit and configuration SHA-256 in
metadata, and a CLI seed override is included in both metadata and the output
directory name.

One RNG stream supplies class shuffling, baseline values, drift, zone shuffling,
fault selection, and thermal innovations. Enabling or disabling an earlier
mechanism changes the later random draws, so separate scenario configurations
with the same seed are deterministic but not fully paired at every latent state.

Run the focused unit test with:

```bash
python3 -m pytest -q tests/test_fidelity.py
```

It checks deterministic generation, trace shape, clipping bounds, and final
fault count. It does not currently cover every disabled scenario, artifact
round-trip, availability transition, class allocation edge case, or invalid
configuration.

## Limitations

- Timesteps are dimensionless; the drift and thermal parameters are not
  calibrated physical temperature, time, conductance, or device-aging units.
- Fidelity is modeled per tile. There is no tier-, cell-, row-, or column-level
  variation, and all tiers on one tile share a noise scale.
- Thermal zones share a scalar AR(1) perturbation; there is no geometry-aware
  heat diffusion or inter-zone coupling.
- Drift is a linear fractional increase. Faults are permanent multiplicative
  jumps; transient faults and recovery are not implemented.
- Static Phase 3 assignments are not dynamically remapped after degradation or
  loss of availability.
- Hard clipping can hide additional drift or fault severity once a tile reaches
  a configured bound.
- Trace validation checks the main array shape and finite nonnegative noise, but
  many configuration invariants and auxiliary-array properties are not enforced.
- Because class-code labels are not stored with the trace, the exact source
  configuration is required for reliable decoding; the recorded hash detects
  changes but cannot reconstruct the file.
