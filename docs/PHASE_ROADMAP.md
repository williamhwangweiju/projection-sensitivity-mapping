# Project Phase Roadmap

## Project Objective

Develop and evaluate a fidelity-aware mapping framework for GPT-2 inference on heterogeneous, time-varying 3D analog compute-in-memory hardware.

The framework combines:

1. **Projection sensitivity** from GPT-2.
2. **Tile fidelity** from the hardware model.
3. **Capacity-aware placement** on IBM 3D-CIM resources.
4. **Migration-aware adaptive remapping** as hardware quality changes.

The central research question is:

> Can projection-level sensitivity be used to preserve language-model quality by placing important weight projections on higher-fidelity compute resources, while limiting the latency and energy overhead of remapping?

---

## System Architecture

```text
Phase 1: GPT-2 sensitivity profile
              │
              ▼
     Projection catalog and scores
              │
              ├──────────────┐
              │              │
              ▼              ▼
Phase 2: Tile-fidelity   IBM 3D-CIM adapter
trace over time         geometry and cost model
              │              │
              └──────┬───────┘
                     ▼
Phase 3: Static capacity-aware mappings
                     │
                     ▼
Phase 4: Adaptive migration-aware mapping
                     │
                     ▼
Phase 5: End-to-end evaluation
```

### Scope

The initial study focuses on the 48 transformer projections in GPT-2 Small:

- `attn.c_attn`
- `attn.c_proj`
- `mlp.c_fc`
- `mlp.c_proj`

The embedding and language-model head can be added later as an extension. This keeps the initial mapping problem aligned with the Phase 1 sensitivity profile.

---

# Phase 1: Projection-Sensitivity Profiling

**Status:** Completed / validation in progress

## Goal

Establish projection-level sensitivity to analog hardware noise for GPT-2 Small.

## Deliverables

- Sensitivity score for each projection in every transformer block.
- Perplexity, negative log-likelihood, and KL-divergence measurements.
- Results across multiple programmed-noise realizations.
- Reusable sensitivity profile for later mapping phases.
- Validation against the target AIHWKit and paper methodology.

## Main Files

```text
experiments/phase1_sensitivity/run_sensitivity_profile.py
src/profilers/sensitivity_profiler.py
src/models/gpt2_model.py
```

## Key Steps

1. Load the pretrained GPT-2 Small model.
2. Load and tokenize the evaluation dataset consistently.
3. Measure clean digital perplexity and negative log-likelihood.
4. Replace one projection at a time with its analog/noisy equivalent.
5. Evaluate multiple independent programmed-noise realizations.
6. Measure:
   - Perplexity increase.
   - Negative-log-likelihood increase.
   - KL divergence from the clean model.
7. Aggregate the realizations into a normalized sensitivity score.
8. Save the complete profile and experiment metadata.

## Expected Outputs

```text
data/profiles/phase1_sensitivity/
├── sensitivity_profile.json
├── sensitivity_profile.csv
├── realization_metrics.csv
├── config.yaml
└── metadata.json
```

The primary profile should follow a structure similar to:

```json
{
  "block_0/attn.c_attn": {
    "sensitivity_score": 0.82,
    "mean_perplexity_delta": 4.21,
    "mean_kl_divergence": 0.014
  }
}
```

## Completion Criteria

- [x] Every selected GPT-2 projection can be profiled independently.
- [x] Results are reproducible for fixed seeds.
- [x] Sensitivity differs meaningfully across projections.
- [ ] Final methodology and results are validated against the target reference.
- [ ] The saved profile has a stable schema that Phase 3 can load directly.

---

# Phase 2: Heterogeneous Tile-Fidelity Model

**Status:** Completed / ready for Phase 3 integration

## Goal

Model a collection of IBM 3D-CIM-style hardware tiles with heterogeneous and time-varying fidelity.

## Deliverables

- Hardware configuration and validated tile-state representation.
- High-, medium-, and low-fidelity tile initialization.
- Continuous per-tile noise values.
- Gradual degradation, thermal variation, and localized faults.
- Reproducible fidelity traces over time.
- Per-tile and per-timestep summary artifacts.

## Main Files

```text
experiments/phase2_fidelity/run_fidelity_model.py
src/simulators/tile_fidelity.py
src/simulators/hardware.py
```

## Key Steps

1. Initialize the configured hardware resources.
2. Assign each tile:
   - Nominal fidelity class.
   - Base noise standard deviation.
   - Thermal zone.
   - Tile-specific drift rate.
3. Simulate time-varying degradation:
   - **Gradual drift:** progressive noise increase.
   - **Thermal variation:** temporally correlated fluctuation.
   - **Localized degradation:** permanent sudden noise increase or tile unavailability.
4. Track the current state of every tile at every timestep.
5. Save the complete trace and summary files.
6. Validate reproducibility and degradation behavior.

## Authoritative Hardware-Quality Signal

The mapper should use:

```text
current_noise_std
```

or the equivalent trace entry:

```text
noise_std[timestep, tile_id]
```

The high-, medium-, and low-fidelity labels are descriptive categories, not the primary optimization variable.

## Expected Outputs

```text
data/results/phase2_fidelity/fidelity_traces/<experiment>/seed_<seed>/
├── trace.npz
├── tile_summary.csv
├── timestep_summary.csv
├── config.yaml
└── metadata.json
```

The trace should expose:

```text
noise_std[timestep, tile]
fidelity_score[timestep, tile]
available[timestep, tile]
faulted[timestep, tile]
```

## Completion Criteria

- [x] Tiles have heterogeneous initial fidelity.
- [x] Fidelity evolves over at least 100 timesteps.
- [x] Gradual, thermal, and localized degradation are supported.
- [x] Fixed seeds reproduce the same trace.
- [x] Tile availability and faults are tracked.
- [ ] Add a stable trace-loading interface for later phases.
- [ ] Add a helper for extracting a hardware snapshot at one timestep.

---

# Phase 3: IBM 3D-CIM Integration and Static Mapping Baselines

**Status:** Next phase

## Goal

Connect the software mapping framework to IBM 3D-CIM hardware geometry and cost information, then establish capacity-aware static mapping baselines.

Phase 3 does **not** perform runtime remapping. Every policy creates one placement that remains fixed while the Phase 2 hardware trace evolves.

## Deliverables

- IBM 3D-CIM adapter.
- GPT-2 projection catalog with hardware dimensions.
- Projection-to-crossbar sharding.
- Capacity-aware placement representation.
- Random mapping baseline.
- Sequential mapping baseline.
- Hardware-only fidelity mapping baseline.
- Static sensitivity-aware mapping baseline.
- Evaluation of every static mapping across the complete fidelity trace.

## Main Files

```text
experiments/phase3_baselines/run_baseline_mappings.py

src/integrations/threedsim_adapter.py

src/mappers/base_mapper.py
src/mappers/static_mapper.py
src/mappers/random_mapper.py
src/mappers/sequential_mapper.py
src/mappers/hardware_only_mapper.py

src/mapping/projection_catalog.py
src/mapping/sharding.py
src/mapping/placement.py
src/mapping/objective.py
```

## Phase 3.1: IBM 3D-CIM Adapter

Create a small adapter around the IBM 3D-CIM simulator rather than placing all mapping logic inside the simulator.

The adapter should expose:

- Number of tiles.
- Tiers available per tile.
- Tier dimensions.
- Usable capacity and reserved resources.
- Model mapping/resource requirements.
- Baseline inference latency.
- Baseline inference energy.
- Data-movement or communication information when available.

Example interface:

```python
class ThreeDSimAdapter:
    def get_hardware_config(self) -> HardwareConfig:
        ...

    def get_projection_requirements(
        self,
        projection: ProjectionSpec,
    ) -> ProjectionResourceRequirement:
        ...

    def get_baseline_metrics(self) -> HardwareMetrics:
        ...
```

The IBM simulator provides hardware structure and cost information. The project mapper remains responsible for assigning projection shards to physical or logical tile resources.

## Phase 3.2: Projection Catalog

Load the Phase 1 profile and combine it with each GPT-2 projection's dimensions.

```python
@dataclass(frozen=True)
class ProjectionSpec:
    projection_id: str
    block_id: str
    projection_name: str

    out_features: int
    in_features: int
    num_weights: int

    sensitivity_score: float
```

GPT-2 `Conv1D` weights must be converted into a canonical:

```text
[out_features, in_features]
```

orientation before calculating resource requirements.

## Phase 3.3: Projection Sharding

Divide each projection into crossbar-compatible shards.

For a tier with `R` rows and `C` columns:

```text
row_shards = ceil(out_features / R)
column_shards = ceil(in_features / C)
num_shards = row_shards × column_shards
```

For a `512 × 512` tier, the approximate GPT-2 Small requirements are:

| Projection | Canonical shape | Approximate tiers per block |
|---|---:|---:|
| `attn.c_attn` | `2304 × 768` | 10 |
| `attn.c_proj` | `768 × 768` | 4 |
| `mlp.c_fc` | `3072 × 768` | 12 |
| `mlp.c_proj` | `768 × 3072` | 12 |

This gives approximately:

```text
38 tiers per transformer block
456 tiers for 12 transformer blocks
```

The exact values must be confirmed through the IBM 3D-CIM adapter because the simulator may reserve capacity or use a different physical encoding.

## Phase 3.4: Capacity Validation

Before implementing the policies, verify the meaning of:

```yaml
tiers: 1024
tier_shape: [512, 512]
```

If `tiers: 1024` means 1,024 fully usable tiers per tile, the complete transformer could fit on one tile under a simple capacity model. That would make heterogeneous mapping trivial.

The integration must therefore determine whether:

- `tiers` represents usable compute tiers per tile.
- Some tiers are reserved.
- Weight encoding reduces usable capacity.
- Compute parallelism imposes placement constraints.
- The simulator distributes layers across tiles for scheduling reasons.
- Logical tile groups should be used as the fidelity unit.

Phase 3 should not proceed to policy comparison until the placement problem is meaningfully capacity constrained.

## Phase 3.5: Placement Representation

```python
@dataclass(frozen=True)
class ShardPlacement:
    shard_id: str
    tile_id: int
    tier_start: int
    tiers_used: int


@dataclass
class Placement:
    assignments: dict[str, ShardPlacement]
    used_tiers_by_tile: dict[int, int]
```

Every mapper must satisfy:

```text
used_tiers_by_tile[tile_id] <= usable_tiers_per_tile[tile_id]
```

Unavailable tiles cannot receive new assignments.

## Phase 3.6: Static Baselines

### Random Mapper

- Randomly order projection shards.
- Randomly choose feasible tiles.
- Respect tile capacity and availability.
- Use fixed seeds for reproducibility.

### Sequential Mapper

- Process projections in model execution order.
- Fill resources sequentially.
- Ignore both sensitivity and tile fidelity.

### Hardware-Only Mapper

- Rank available tiles by current fidelity at timestep 0.
- Ignore projection sensitivity.
- Place shards using only hardware quality and capacity.

This baseline determines whether hardware awareness alone provides an advantage.

### Static Sensitivity-Aware Mapper

- Rank projections by decreasing Phase 1 sensitivity.
- Rank tiles by increasing noise at timestep 0.
- Greedily assign the most sensitive projections to the best available resources.
- Keep the placement fixed for the complete hardware trace.

This baseline determines how well sensitivity-aware placement works without runtime adaptation.

## Initial Mapping Objective

Use the continuous tile noise rather than only fidelity classes.

A first-order sensitivity-weighted hardware-error proxy is:

```text
J_t(M) =
    Σ_s shard_weight_s
        × projection_sensitivity_s
        × (tile_noise_std_t / reference_noise_std)^2
```

where:

```text
shard_weight_s =
    weights_in_shard_s / weights_in_parent_projection_s
```

This metric can be evaluated cheaply at every timestep.

Later phases can replace the quadratic approximation with measured sensitivity curves at multiple noise levels.

## Phase 3 Evaluation

Evaluate each static placement across all Phase 2 timesteps.

Metrics:

- Sensitivity-weighted tile error.
- Mean assigned noise.
- Sensitivity-weighted mean assigned noise.
- Capacity utilization.
- Number of projections or shards on faulted tiles.
- Number of projections or shards on unavailable tiles.
- Estimated perplexity degradation.
- Measured perplexity and KL divergence at selected checkpoints.
- IBM 3D-CIM baseline latency and energy.
- Policy generation time.

Static policies should report:

```text
remapping_events = 0
weight_data_moved_after_initialization = 0
```

## Expected Outputs

```text
data/results/phase3_baselines/<experiment>/seed_<seed>/
├── projection_catalog.csv
├── projection_shards.csv
├── placement_random.csv
├── placement_sequential.csv
├── placement_hardware_only.csv
├── placement_static_sensitivity.csv
├── timestep_metrics.csv
├── policy_summary.csv
├── config.yaml
└── metadata.json
```

## Completion Criteria

- [ ] Phase 1 sensitivity profiles load automatically.
- [ ] Phase 2 fidelity traces load automatically.
- [ ] IBM 3D-CIM geometry and usable capacity are extracted.
- [ ] GPT-2 projection dimensions are canonicalized correctly.
- [ ] Projections are divided into valid crossbar shards.
- [ ] All placements satisfy hardware capacity.
- [ ] Random mapping is reproducible.
- [ ] Sequential mapping follows execution order.
- [ ] Hardware-only mapping ignores sensitivity.
- [ ] Static sensitivity-aware mapping uses Phase 1 scores.
- [ ] Every policy can be evaluated across every Phase 2 timestep.
- [ ] Static mappings become suboptimal after selected drift or fault events.
- [ ] Results establish meaningful lower and upper baseline bounds.

---

# Phase 4: Adaptive Mapping Algorithm

**Status:** Planned after Phase 3

## Goal

Implement a migration-aware adaptive mapper that changes projection placement when hardware fidelity changes enough to justify remapping overhead.

## Deliverables

- Adaptive sensitivity-aware mapper.
- Naive cost-unaware adaptive baseline.
- Migration-cost model.
- Threshold-based remapping decision.
- Cooldown and hysteresis controls.
- Capacity-aware remapping.
- Detailed migration-event records.

## Main Files

```text
experiments/phase4_adaptive/run_adaptive_mapping.py

src/mappers/adaptive_mapper.py
src/mappers/migration_cost.py
src/mappers/remapping_policy.py
```

## Key Steps

1. Load the initial placement and Phase 2 hardware trace.
2. At each decision timestep:
   - Read the current tile snapshot.
   - Evaluate the current placement.
   - Generate a candidate sensitivity-aware placement.
   - Estimate the expected quality improvement.
   - Estimate migration energy and latency.
3. Remap only when the predicted benefit exceeds the configured threshold.
4. Add cooldown and hysteresis to prevent oscillation.
5. Record every moved shard and migration event.
6. Compare against a naive policy that remaps whenever a better placement exists.

## Remapping Decision

A candidate placement should be accepted when:

```text
current_error
- candidate_error
> migration_weight × migration_cost
+ remapping_threshold
```

The decision may also require:

- A minimum number of timesteps since the last remap.
- A minimum relative improvement.
- A persistent improvement across multiple observations.
- No violation of capacity or availability constraints.

## Migration Cost

Estimate migration cost using:

- Total bytes of weights moved.
- Source and destination tiles.
- Communication distance when available.
- IBM 3D-CIM data-movement latency.
- IBM 3D-CIM data-movement energy.
- Reprogramming overhead.
- Temporary service interruption if modeled.

## Policies to Compare

1. Random static.
2. Sequential static.
3. Hardware-only static.
4. Sensitivity-aware static.
5. Naive adaptive sensitivity-aware.
6. Migration-aware adaptive sensitivity-aware.

## Expected Outputs

```text
data/results/phase4_adaptive/<experiment>/seed_<seed>/
├── initial_placement.csv
├── final_placement.csv
├── timestep_metrics.csv
├── remapping_events.csv
├── moved_shards.csv
├── threshold_sweep.csv
├── config.yaml
└── metadata.json
```

## Completion Criteria

- [ ] The mapper detects when the current placement has degraded.
- [ ] Candidate placements satisfy all capacity constraints.
- [ ] The naive adaptive policy responds to fidelity changes.
- [ ] The migration-aware policy suppresses low-value remaps.
- [ ] Cooldown and hysteresis prevent repeated oscillation.
- [ ] Migration bytes, latency, and energy are tracked.
- [ ] Threshold sweeps produce a clear quality-overhead trade-off.
- [ ] Adaptive mapping improves quality over static mapping under changing hardware.
- [ ] Migration-aware adaptation uses less overhead than naive adaptation.

---

# Phase 5: Comprehensive Evaluation

**Status:** Planned

## Goal

Evaluate all mapping strategies under consistent hardware, model, and degradation conditions and produce publication-ready results.

## Deliverables

- Complete comparison of all six mapping strategies.
- Evaluation across multiple random seeds.
- Evaluation across multiple degradation scenarios.
- Quality-versus-overhead trade-off analysis.
- IBM 3D-CIM latency and energy integration.
- Statistical summaries and confidence intervals.
- Publication-ready figures and tables.
- Final deployment recommendations.

## Main Files

```text
experiments/phase5_evaluation/run_full_evaluation.py
experiments/phase5_evaluation/analyze_results.py
experiments/phase5_evaluation/generate_figures.py
```

## Evaluation Scenarios

At minimum:

1. **Static heterogeneous hardware**
   - Different initial tile fidelities.
   - No temporal degradation.

2. **Gradual drift**
   - Slowly increasing noise over time.

3. **Thermal variation**
   - Temporally correlated fluctuations.

4. **Localized degradation**
   - Sudden permanent quality loss in selected tiles.

5. **Tile failure**
   - Selected tiles become unavailable.

6. **Combined degradation**
   - Drift, thermal variation, and faults together.

Each scenario should run across multiple seeds.

## Primary Metrics

### Model Quality

- Perplexity.
- Negative log-likelihood.
- KL divergence from the clean model.
- Next-token agreement with the clean model.
- Sensitivity-weighted tile error.

### Hardware and Performance

- Inference latency.
- Inference energy.
- Communication latency.
- Communication energy.
- Tile-capacity utilization.
- Tile-fidelity utilization.

### Adaptation Overhead

- Number of remapping events.
- Number of moved shards.
- Total weight bytes moved.
- Migration latency.
- Migration energy.
- Percentage of time spent remapping.

### Trade-Off Metrics

- Quality improvement per megabyte moved.
- Quality improvement per unit of migration energy.
- Quality improvement per unit of migration latency.
- Cumulative quality loss over time.
- Cumulative total cost over time.

## Evaluation Protocol

For each combination of:

```text
mapping policy
× degradation scenario
× random seed
× remapping threshold
```

record all metrics using the same:

- GPT-2 checkpoint.
- Dataset subset.
- Tokenization configuration.
- Projection scope.
- Hardware configuration.
- Noise reference.
- Number of timesteps.
- Evaluation checkpoints.

Use lightweight proxy metrics at every timestep and run full GPT-2 perplexity and KL evaluation at selected representative timesteps.

## Expected Outputs

```text
data/results/phase5_evaluation/
├── all_runs.csv
├── aggregate_metrics.csv
├── statistical_tests.csv
├── figures/
│   ├── quality_over_time.pdf
│   ├── quality_vs_migration_cost.pdf
│   ├── remapping_events.pdf
│   ├── energy_latency_tradeoff.pdf
│   └── policy_comparison.pdf
└── tables/
    ├── main_results.csv
    ├── ablation_results.csv
    └── scenario_results.csv
```

## Completion Criteria

- [ ] All six strategies are evaluated under identical conditions.
- [ ] Results include multiple seeds and degradation scenarios.
- [ ] Adaptive mapping improves cumulative quality over static baselines.
- [ ] Migration-aware adaptation reduces overhead relative to naive adaptation.
- [ ] Sensitivity-aware policies outperform comparable sensitivity-unaware policies.
- [ ] Results include measured language-model quality, not only proxy metrics.
- [ ] Latency and energy results are connected to IBM 3D-CIM.
- [ ] Conclusions are supported by statistical analysis.
- [ ] Figures and tables are publication ready.

---

# Cross-Phase Interfaces

To avoid coupling phases together, use stable data interfaces.

## Phase 1 to Phase 3

```text
SensitivityProfile
projection_id -> sensitivity_score and quality metrics
```

## Phase 2 to Phases 3 and 4

```text
TileFidelityTrace
timestep × tile -> noise, fidelity, availability, fault state
```

Recommended helper:

```python
snapshot = trace.get_snapshot(timestep)
```

## IBM 3D-CIM to Phases 3–5

```text
HardwareConfig
ProjectionResourceRequirement
HardwareMetrics
MigrationCostEstimate
```

## Phase 3 to Phase 4

```text
Placement
policy metadata
capacity utilization
initial objective value
```

## Phases 3 and 4 to Phase 5

Use the same timestep-level and policy-level metric schemas so all strategies can be compared without custom analysis code.

---

# Testing Strategy

## Unit Tests

```text
tests/phase1/
tests/phase2/
tests/phase3/
tests/phase4/
```

Important tests include:

- Projection dimension canonicalization.
- Crossbar sharding.
- Capacity accounting.
- Placement validity.
- Mapper reproducibility.
- Sensitivity ordering.
- Hardware-fidelity ordering.
- Trace snapshot extraction.
- Migration-byte calculation.
- Remapping threshold behavior.
- Cooldown and hysteresis.
- Faulted and unavailable tile handling.

## Integration Tests

- Load a Phase 1 profile into Phase 3.
- Load a Phase 2 trace into Phase 3.
- Extract IBM 3D-CIM geometry.
- Generate every static placement.
- Run an adaptive policy through a complete trace.
- Evaluate a small GPT-2 dataset checkpoint under a selected placement.

---

# Overall Success Criteria

## Phase 1

- [ ] Sensitivity profiles show clear and reproducible differentiation between projections.

## Phase 2

- [ ] The fidelity model produces realistic and validated heterogeneous degradation patterns.

## Phase 3

- [ ] Static baselines establish meaningful capacity-aware performance bounds.

## Phase 4

- [ ] Migration-aware adaptation preserves quality while reducing remapping overhead relative to naive adaptation.

## Phase 5

- [ ] Results provide clear evidence that projection sensitivity improves mapping decisions on heterogeneous and changing 3D-CIM hardware.

## Research Success

The project is successful if it demonstrates that:

1. GPT-2 projections have meaningfully different sensitivity to analog hardware noise.
2. Hardware fidelity changes can make an initially good static placement become suboptimal.
3. Sensitivity-aware placement preserves model quality better than sensitivity-unaware placement.
4. Runtime adaptation can recover quality after degradation.
5. Migration-aware decision logic achieves a better quality-overhead trade-off than remapping whenever the hardware ranking changes.

---

# Updated Timeline

Assuming part-time work and that Phases 1 and 2 are substantially complete:

| Phase | Estimated effort | Main dependency |
|---|---:|---|
| Phase 1 final validation | 3–5 days | AIHWKit methodology validation |
| Phase 2 interface cleanup | 1–2 days | Trace loader and snapshot API |
| Phase 3 IBM integration and baselines | 1–2 weeks | IBM tier and capacity semantics |
| Phase 4 adaptive mapper | 2 weeks | Stable Phase 3 placement model |
| Phase 5 evaluation and analysis | 2–3 weeks | Reproducible end-to-end pipeline |
| Writing and figure refinement | 1–2 weeks | Final experimental results |

The most important immediate milestone is:

> Verify IBM 3D-CIM resource semantics, create the projection-sharding and placement abstractions, and implement the four static Phase 3 baselines.
