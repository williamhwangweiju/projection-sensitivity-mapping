# Phase 3: Capacity-Aware Static Placement

Phase 3 turns each Phase 1.5 digital/analog operating point into a physical
placement of the remaining analog weight shards. It compares four deterministic
baseline policies on the same tile-fidelity snapshot and writes the placements
that Phase 4 later evaluates.

The current implementation uses an IBM-style 3D-CIM abstraction—tiles, tiers,
and rectangular crossbar capacity—implemented in `src/mapping/`. It does not
invoke the bundled `simulators/ibm_3d_cim` package, and it does not estimate
latency or energy.

## Position in the pipeline

Phase 3 consumes:

- the Phase 1 projection profile, including shape and sensitivity;
- the Phase 1.5 digital operating points;
- the Phase 2 tile-fidelity trace; and
- the shared hardware and placement configuration.

For every capacity-feasible operating point, it excludes protected digital
projections, shards the analog projections, and places the resulting shards
under every configured policy. The assignments are static: Phase 4 keeps each
`(tile_id, tier_id)` assignment fixed while changing the tile noise at later
timesteps.

## Quick start

The normal entry point is the full pipeline:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

To run Phase 3 from existing upstream artifacts:

```bash
python3 experiments/phase3_baselines/run_baseline_mappings.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --operating-points data/results/phase1_5_digital_selection/digital_operating_points.json \
  --trace data/results/phase2_fidelity/fidelity_traces/mixed_96x8/seed_42/trace.npz
```

All paths supplied on the command line must exist. Relative output paths in the
YAML are resolved from the repository root.

Use `configs/full_pipeline/gpt2_hybrid_3dcim.yaml` or its smoke counterpart as
the configuration source of truth. The older
`configs/phase3_baselines/default.yaml` does not contain the unified
`hardware`, `digital_selection`, or current `phase3` schema and is not a
drop-in configuration for this runner.

## Sharding model

Every physical tier holds at most one rectangular shard with dimensions no
larger than:

```text
hardware.tier_shape.rows × hardware.tier_shape.cols
```

For a projection with `R` output rows and `C` input columns, the ordinary shard
count is:

```text
ceil(R / tier_rows) × ceil(C / tier_cols)
```

GPT-2 stores Q, K, and V in one fused `attn.c_attn` matrix. Phase 3 first splits
that matrix into three semantic row regions and then shards each region. A shard
therefore never crosses a Q/K/V boundary.

For shard `s` of projection `p`:

```text
shard_weight(s) = weights_in_shard(s) / weights_in_projection(p)
sensitivity(p)  = max(phase1_sensitivity(p), phase3.sensitivity_floor)
importance(s)   = sensitivity(p) × shard_weight(s)
```

The default floor is zero. This preserves the raw Phase 1 estimate in the
profile while preventing a negative Monte Carlo sensitivity estimate from
reversing the placement objective.

With GPT-2 Small and 512 × 512 tiers, all 48 transformer projections produce
480 shards. Including the tied language-model head adds 198 shards. The primary
96-tile × 8-tier configuration therefore has 768 physical slots and can
represent the initial 678-shard all-analog candidate set.

Digital projections consume no analog tier and receive no tile-noise injection.
Phase 1.5 computes `analog_shard_count` and `capacity_feasible` for every
operating point before Phase 3 runs.

## Physical slots and capacity

Phase 3 reads `noise_std[phase3.mapping_timestep]` and
`available[phase3.mapping_timestep]` from the Phase 2 trace. Each available tile
contributes `hardware.tiers_per_tile` slots; every tier on one tile has the same
tile-level noise value. Unavailable tiles contribute no slots.

An operating point marked capacity-infeasible is skipped and recorded in
`phase3_manifest.json`. If the operating point is marked feasible but the
selected mapping timestep exposes fewer usable slots than required,
`place_shards` raises an error instead of silently dropping or duplicating
weights.

## Placement policies

| Policy | Slot order | Shard order | Intended baseline |
| --- | --- | --- | --- |
| `random` | Seeded shuffle of all usable slots | Catalog order | Blind random placement |
| `sequential` | `tile_id`, then `tier_id` | Catalog order | Deterministic in-order placement |
| `hardware_only` | Lowest tile noise first | Seeded permutation independent of sensitivity | Hardware-aware, workload-blind placement |
| `static_sensitivity` | Lowest tile noise first | Highest shard importance first | Sensitivity-aware static placement |

The `hardware_only` permutation is important: GPT-2 catalog order begins with
early blocks and can correlate with sensitivity. Permuting shards prevents that
baseline from accidentally becoming sensitivity-aware.

All policies place exactly the same analog shard set for a given digital
operating point. They differ only in the physical assignment.

## Placement proxy

Phase 3 reports two separable diagnostic objectives:

```text
proxy_variance = Σ importance(s) × tile_noise_std(s)²
proxy_noise    = Σ importance(s) × tile_noise_std(s)
```

`static_sensitivity` minimizes the variance form under this one-to-one
assignment model by pairing the most important shards with the quietest slots.
These values are placement heuristics, not predictions of NLL or perplexity.
Phase 4 provides the model-level quality measurement.

## Configuration

These unified fields define Phase 3 and its upstream capacity contract:

| Field | Meaning |
| --- | --- |
| `experiment.placement_seed` | Seed for `random` and `hardware_only`; defaults to 42 if absent |
| `hardware.num_tiles` | Nominal tile count used by Phase 1.5 capacity accounting and cross-phase validation; the placement runner gets its actual tile count from the trace |
| `hardware.tiers_per_tile` | Available tier slots per tile |
| `hardware.tier_shape.rows` | Maximum shard output rows |
| `hardware.tier_shape.cols` | Maximum shard input columns |
| `phase3.output_root` | Root directory for all placements |
| `phase3.mapping_timestep` | Phase 2 snapshot used to construct the static placement |
| `phase3.sensitivity_floor` | Lower bound applied only to placement sensitivity |
| `phase3.policies` | Policies to materialize |

Example:

```yaml
experiment:
  placement_seed: 42

hardware:
  num_tiles: 96
  tiers_per_tile: 8
  tier_shape:
    rows: 512
    cols: 512

phase3:
  output_root: data/results/phase3_static_mapping
  mapping_timestep: 0
  sensitivity_floor: 0.0
  policies:
    - random
    - sequential
    - hardware_only
    - static_sensitivity
```

Only those four policy names are implemented.

## Artifacts

The default output layout is:

```text
data/results/phase3_static_mapping/
├── phase3_manifest.json
├── phase3_summary.csv
└── <digital_set_id>/
    ├── digital_operating_point.json
    ├── placement_random.csv
    ├── placement_sequential.csv
    ├── placement_hardware_only.csv
    └── placement_static_sensitivity.csv
```

Each placement CSV contains one row per analog shard:

| Field group | Columns |
| --- | --- |
| Identity | `policy`, `timestep`, `shard_id`, `projection_id`, `shard_index` |
| Logical coordinates | `row_start`, `row_end`, `col_start`, `col_end` |
| Weighting | `weight_count`, `shard_weight`, `sensitivity`, `importance` |
| Physical assignment | `tile_id`, `tier_id`, `tile_noise_std` |

`phase3_summary.csv` has one row per digital-set/policy pair and records the
placement path, analog shard count, and both proxy values.

`phase3_manifest.json` also records the repository commit, configuration hash,
all upstream artifact paths, every generated placement, and any operating
points skipped for capacity.

## Contract validation

The full pipeline automatically validates the cross-phase contracts before
Phase 4. To validate existing artifacts directly:

```bash
python3 scripts/validate_pipeline_contracts.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --operating-points data/results/phase1_5_digital_selection/digital_operating_points.json \
  --trace data/results/phase2_fidelity/fidelity_traces/mixed_96x8/seed_42/trace.npz \
  --phase3-manifest data/results/phase3_static_mapping/phase3_manifest.json
```

The validator checks:

- the digital and analog projection sets form an exact partition;
- the Phase 2 tile count matches the hardware configuration;
- capacity flags agree with shard counts;
- protected digital projections never appear in a placement;
- no shard or physical tier is reused;
- every policy covers the same analog shard set; and
- every configured policy is present for each placed operating point.

Run the focused unit tests with:

```bash
python3 -m pytest -q tests/test_sharding_and_mapping.py
```

## Interpretation and limitations

- Placements are static. Later drift or faults change the noise attached to the
  assigned tiles but do not trigger migration or remapping.
- Capacity is one shard per tier. The model does not reserve cells, stack
  multiple shards in a tier, or model routing congestion.
- Tile fidelity is shared by all tiers on a tile; there is no tier-level
  variation.
- The objective uses Phase 1 `delta_nll_noise` as an additive importance
  heuristic. Interactions among simultaneously noisy projections are measured
  only in Phase 4.
- The main path does not call the IBM 3D-SiM mapper and does not produce
  latency, energy, communication, or thermal-physics estimates.
- The runner expects at least one analog shard in each materialized operating
  point. The default search stops well before an all-digital point.

Continue with [Phase 4](PHASE_4.md) after the manifest and placement CSVs have
been generated and validated.
