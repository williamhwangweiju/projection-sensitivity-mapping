# Phase 3: Sharding and Static Placement

Phase 3 turns the all-analog projection set into physical shard placements on
the tile-and-tier substrate, one placement per configured policy, all at the
same mapping timestep of the Phase-2 trace.

## Entry points

- Runner: `experiments/phase3_baselines/run_baseline_mappings.py`
- Sharding: `src/mapping/sharding.py`
- Placement policies: `src/mapping/placement.py`
- Diagnostic objectives: `src/mapping/objective.py`
- Canonical configuration: `configs/full_pipeline/gpt2_hybrid_3dcim.yaml`

## Inputs

1. the unified YAML configuration;
2. the Phase 1 projection profile (authoritative candidate universe and
   measured sensitivity);
3. the Phase 2 `trace.npz` (tile noise at the mapping timestep).

## Quick start

```bash
python3 experiments/phase3_baselines/run_baseline_mappings.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --trace data/results/phase2_fidelity/fidelity_traces/mixed_96x8/seed_42/trace.npz
```

## Sharding model

Every profiled projection deploys analog. Each canonical `[out, in]` weight
matrix is tiled into `tier_rows × tier_cols` crossbar shards; fused
`attn.c_attn` matrices are first split into semantic Q/K/V row regions that a
shard may not cross. Each shard carries:

```text
importance = max(sensitivity, phase3.sensitivity_floor) × shard_weight
```

where `shard_weight` is the shard's fraction of the projection's weights.
The runner verifies up front that the required shard count fits the
substrate (`num_tiles × tiers_per_tile`) and fails otherwise.


## Placement policies

| Policy | Slot order | Shard order | Intended baseline |
| --- | --- | --- | --- |
| `random` | Seeded shuffle of all usable slots | Catalog order | Blind random placement |
| `sequential` | `tile_id`, then `tier_id` | Catalog order | Deterministic in-order placement |
| `hardware_only` | Lowest tile noise first | Seeded permutation independent of sensitivity | Hardware-aware, workload-blind placement |
| `static_sensitivity` | Lowest tile noise first | Highest measured importance first | Sensitivity-aware static placement |

The `hardware_only` permutation matters: GPT-2 catalog order begins with
early blocks and correlates with sensitivity; permuting shards keeps that
baseline honestly workload-blind. All policies place exactly the same shard
set and differ only in the physical assignment.

## Placement proxy

Phase 3 reports two separable diagnostic objectives per placement:

```text
proxy_variance = Σ importance(s) × tile_noise_std(s)²
proxy_noise    = Σ importance(s) × tile_noise_std(s)
```

`static_sensitivity` minimizes the variance form under this one-to-one
assignment model. These are placement heuristics, not predictions of NLL;
Phase 4 provides the model-level measurement.

## Configuration

| Field | Meaning |
| --- | --- |
| `phase3.output_root` | Artifact directory |
| `phase3.mapping_timestep` | Trace timestep whose tile noise informs the mapping |
| `phase3.policies` | Placement policies to generate |
| `phase3.sensitivity_floor` | Negative Monte-Carlo sensitivities are floored to this value for importance |
| `hardware.*` | Tile/tier geometry and counts |
| `experiment.placement_seed` | Seed for the stochastic policies |

## Artifacts

| Artifact | Contents |
| --- | --- |
| `placement_<policy>.csv` | One row per shard: identity, coordinates, importance, assigned tile/tier, tile noise at mapping time |
| `phase3_manifest.json` | Provenance, upstream paths, capacity accounting, per-policy placement paths and proxy objectives |
| `phase3_summary.csv` | One row per policy |

## Contract validation

`scripts/validate_pipeline_contracts.py` (run automatically by the pipeline
driver) checks unique projection IDs, the sensitivity unit, trace/hardware
tile agreement, capacity, exact projection coverage, no duplicate shard or
reused physical tier, valid tile references, and identical shard sets across
all policies.

## Interpretation and limitations

- Placement is static: Phase 4 re-reads tile noise over time but never
  remaps shards.
- Capacity is tier-granular; communication, latency, and energy are not
  modeled.
- Shard importance assumes uniform sensitivity within a projection
  (`importance = sensitivity × shard_weight`); finer-grained structure
  (rows, heads, Q/K/V asymmetry) is not distinguished.
