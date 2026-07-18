# Phase 1.5: Digital Protection Selection

Phase 1.5 converts the [Phase 1](PHASE_1.md) projection profile into a frontier of hybrid digital/analog operating points. Each point partitions the profiled projection candidate universe into:

- projections protected by digital execution; and
- projections left for analog execution, sharding, placement, and hardware-noise evaluation.

The phase supports two distinct selection paths:

1. **Profile-score selectors** cheaply rank Phase 1 sensitivity under count, parameter, or MAC targets and accept explicitly named sets.
2. **Measured greedy selection** repeatedly promotes the projection with the best measured nominal-hybrid calibration-NLL improvement, optionally normalized by digital cost.

The canonical primary pipeline uses the measured greedy path. The reduced smoke pipeline disables it and uses cheap and explicit operating points to preserve cross-phase structure at low cost.

## Entry points

- Cheap and explicit point generation: `experiments/phase1_5_digital_selection/select_digital_operating_points.py`
- Measured greedy search: `experiments/phase1_5_digital_selection/select_greedy_marginal.py`
- Candidate, selector, and cost accounting: `src/mapping/digital_selection.py`
- Capacity calculation and QKV-aware sharding: `src/mapping/sharding.py`
- Temporary analog/digital module swaps: `src/evaluation/aihwkit_gpt2.py`
- Pipeline orchestration: `scripts/run_full_pipeline.py`
- Canonical full configuration: `configs/full_pipeline/gpt2_hybrid_3dcim.yaml`
- Reduced smoke configuration: `configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml`

## Input contract and candidate universe

The Phase 1 JSON must contain a top-level `projections` list and, for the cheap selector, top-level `mapping_sensitivity_unit`. Every candidate row needs:

```text
projection_id
sensitivity_score_for_mapping
parameter_count
macs_per_token
in_features
out_features
```

`tied_to_embedding` is optional and defaults to false. Measured greedy conversion also uses persisted Phase 1 preprocessing metadata when present to verify that model weights, clipping, programmed range, and checksums still agree.

The Phase 1 projection list is authoritative. All operating-point counts and fractions are relative to this profiled candidate universe, not the entire GPT-2 model. If a reduced Phase 1 profile omitted transformer blocks or the LM head, downstream hybrid conversion leaves those unprofiled modules digital and excludes them from Phase 1.5 cost and capacity totals.

Use the same model and analog preprocessing configuration that created the Phase 1 artifact. In particular, a profile containing `lm_head` requires `profiling.include_lm_head: true` when measured greedy conversion reconstructs the candidate model.

## Path A: profile-score operating points

The cheap selector reads Phase 1 rows into immutable candidates with four cost/ranking attributes:

```text
projection_id
sensitivity = sensitivity_score_for_mapping
parameter_count
macs_per_token
```

### Ranking methods

For projection `p`, the supported methods are exactly:

```text
sensitivity_rank:
    score(p) = sensitivity(p)

sensitivity_per_parameter:
    score(p) = sensitivity(p) / max(parameter_count(p), 1)

sensitivity_per_mac:
    score(p) = sensitivity(p) / max(macs_per_token(p), 1)
```

Candidates are sorted by descending score and then ascending projection ID. Negative Phase 1 scores are not filtered; sufficiently large targets can select them.

### Projection-count targets

For `budgets.projection_counts`, forced projections are inserted first. Ranked candidates are then added until the requested count is reached or candidates are exhausted.

The requested count is not validated against `[0, candidate_count]`. If the forced set is larger than the requested count, all forced projections remain selected and the achieved count exceeds the request.

### Parameter- and MAC-fraction targets

For a cost field `cost`, the selector computes:

```text
target_cost = requested_fraction * sum_all_candidates cost(p)
```

Forced projections are inserted first, then whole ranked projections are added while accumulated selected cost is below the target.

These fractions are **target thresholds, not maximum budgets**. Because projections are indivisible, the achieved fraction can overshoot the requested value. Forced projections can also exceed the requested target before ranked selection begins. Requested fractions are validated to lie in `[0, 1]`.

### Forced and explicit sets

`digital_selection.forced_digital` applies to count and fraction selectors. Every forced ID must be present in the Phase 1 candidate set.

`digital_selection.explicit_sets` behaves differently: each named list is used exactly as written. The global forced set is not automatically unioned into explicit sets. An explicit point records:

```text
selection_method = explicit:<name>
budget_type      = explicit
budget_value     = number of supplied IDs
```

Explicit IDs are not validated by the point generator before accounting. A misspelled or unknown ID can therefore survive into an invalid point and fail a later pipeline contract check.

### Duplicate-set handling

After all explicit and ranked records are built, records with the same sorted `digital_projection_ids` tuple are collapsed. The first record wins. Explicit sets are generated before automatic methods, so an explicit name is preserved when a later selector yields the same set. YAML insertion order determines which of two duplicate explicit sets survives.

This deduplication compares projection sets, not `digital_set_id` values.

## Cost accounting

Let `C` be the profiled candidate universe and `D` the selected digital set. Logical projection-parameter and MAC fractions are:

```text
digital_parameter_fraction
    = sum[p in D] parameter_count(p) / sum[p in C] parameter_count(p)

digital_mac_fraction
    = sum[p in D] macs_per_token(p) / sum[p in C] macs_per_token(p)
```

Analog fractions are one minus the corresponding digital fraction.

GPT-2's LM head may share its weight tensor with the already-digital token embedding. The logical digital parameter and MAC fields still include the LM head when selected because they describe its execution size. Incremental digital-storage accounting excludes every candidate with `tied_to_embedding: true` from both numerator and denominator:

```text
digital_incremental_storage_fraction
    = sum[p in D and not tied(p)] parameter_count(p)
      / sum[p in C and not tied(p)] parameter_count(p)
```

These fields are logical research proxies. They do not model end-to-end latency, energy, activation traffic, digital accelerator utilization, or the parameter cost of unprofiled/nonprojection modules.

## Physical capacity annotation

Every operating point receives:

```text
analog_shard_count
available_physical_tiers
capacity_feasible
```

Each analog matrix is divided according to `hardware.tier_shape`. A physical tier holds one shard. For an ordinary projection region:

```text
shards = ceil(region_out_features / tier_rows)
         * ceil(in_features / tier_cols)
```

GPT-2's fused `attn.c_attn` matrix is first split into semantic Q, K, and V output regions so that no shard crosses a Q/K/V boundary. Available slots are:

```text
available_physical_tiers = hardware.num_tiles * hardware.tiers_per_tile
capacity_feasible        = analog_shard_count <= available_physical_tiers
```

The selector annotates feasibility; it does not automatically promote projections to make a cheap point fit. Phase 3 later skips points marked infeasible.

Under the canonical 512-by-512 tier geometry, full GPT-2 Small uses 480 transformer shards plus 198 LM-head shards. The primary `96 * 8 = 768`-tier substrate can therefore hold the initial 678-shard all-profiled-candidates-analog point.

## Path B: measured greedy marginal selection

Measured greedy search optimizes quality directly on a calibration subset. It is a greedy search, not an exhaustive search over all `2^N` digital/analog partitions.

### 1. Reference and nominal hybrid

The search loads a fresh pretrained model and tokenizer. It uses top-level `digital_selection_dataset` when present; otherwise it uses `dataset`. The untouched digital model is evaluated to obtain:

```text
digital_reference_nll
digital_reference_ppl
```

All profiled candidates except permanently forced digital projections are then converted to clipped nominal analog modules exactly once. Phase 1 preprocessing metadata is checked during conversion. No manual Phase 1 or Phase 2 weight noise is applied during this search; “nominal” includes clipping and the configured AIHWKit forward behavior.

Trials temporarily swap selected candidates' original digital modules into the forward graph. The converted analog modules and their nominal weights remain available and are restored after each trial. Results for an already measured digital set are cached in memory for the process lifetime.

### 2. Greedy utility

Let `D_k` be the current digital set and `L(D_k)` its measured nominal-hybrid NLL. For every eligible candidate `p` not already selected:

```text
gain(p | D_k) = L(D_k) - L(D_k union {p})
```

With `objective: gain_per_cost`:

```text
utility(p | D_k) = gain(p | D_k) / max(cost(p), 1)
```

With `objective: nll_gain`:

```text
utility(p | D_k) = gain(p | D_k)
```

`cost_field` is either `macs_per_token` or `parameter_count`. The primary configuration uses marginal NLL gain per MAC.

The winning trial maximizes, in order:

1. utility;
2. marginal NLL gain;
3. lower candidate cost; and
4. lexicographically greater projection ID for an otherwise exact tie.

The selected sets are nested: every successful step adds one projection and never removes an earlier promotion.

### 3. Candidate pool and trial constraints

The pool is ranked once by descending raw Phase 1 sensitivity and ascending projection ID. `candidate_pool_size: null` or a value less than or equal to zero evaluates every candidate; a positive value restricts every step to the highest-ranked prefix. This truncation is a speed ablation and can exclude a projection that would have had high measured marginal gain.

A trial is skipped before model evaluation if its achieved logical cost would exceed either:

```text
greedy_marginal.max_digital_mac_fraction
greedy_marginal.max_digital_parameter_fraction
```

Forced projections are included in those achieved fractions. The initial forced set is still recorded even if it already exceeds a configured maximum; maximum checks gate new promotions.

### 4. Records and stopping

Step zero records the forced digital set. `max_promotions` counts later greedy promotions, not the total number of digital projections including forced ones.

Search stops when any of these conditions holds:

- the current point is capacity-feasible and `delta_nll_nominal_vs_digital <= target_delta_nll_vs_digital`;
- no remaining candidate satisfies both digital-fraction constraints;
- the best gain is less than or equal to `minimum_marginal_nll_gain`; or
- `max_promotions` has been reached.

The delta used for the quality target is:

```text
delta_nll_nominal_vs_digital = measured_nominal_nll - digital_reference_nll
```

Capacity is not a trial filter. An initially infeasible point can remain in the recorded frontier while promotions reduce its analog shard count; target-based stopping waits until a recorded point is also capacity-feasible.

### 5. Recommendation

The standalone greedy artifact recommends:

1. among feasible points meeting the target, the point with minimum digital MAC fraction and then minimum digital parameter fraction;
2. otherwise, among feasible points, the point with lowest measured calibration delta NLL and then lowest digital MAC fraction; or
3. if no searched point is feasible, the final searched point.

This recommendation is based on nominal calibration data. Phase 4 held-out and hardware-noise results remain necessary to assess generalization and deployed quality.

## Canonical unified configuration

Measured search must use the same `model`, `analog`, and LM-head inclusion settings as Phase 1. The primary selection-specific configuration is:

```yaml
model:
  name: gpt2
  device: cpu

digital_selection_dataset:
  name: Salesforce/wikitext
  config: wikitext-103-raw-v1
  split: validation
  sequence_length: 1024
  stride: 1024
  batch_size: 1
  max_tokens: 16384
  document_separator: "\n\n"
  drop_incomplete_final_sequence: true

hardware:
  num_tiles: 96
  tiers_per_tile: 8
  tier_shape:
    rows: 512
    cols: 512
  num_thermal_zones: 8

profiling:
  include_lm_head: true

analog:
  clip_sigma: 2.5
  range_mode: peak_to_peak
  reference_noise_std: 0.023
  tile_size: 512
  adc_dac_bits: 8
  output_bound: null
  weight_scaling_omega: 1.0
  weight_scaling_columnwise: false

digital_selection:
  output_root: data/results/phase1_5_digital_selection
  forced_digital: []
  methods: []
  budgets:
    projection_counts: []
    parameter_fractions: []
    mac_fractions: []
  explicit_sets: {}

  greedy_marginal:
    enabled: true
    forced_digital: []
    candidate_pool_size: null
    max_promotions: 12
    objective: gain_per_cost
    cost_field: macs_per_token
    target_delta_nll_vs_digital: 0.10
    minimum_marginal_nll_gain: 0.0
    max_digital_mac_fraction: 0.50
    max_digital_parameter_fraction: 0.50
```

The full file also supplies the required `analog` settings documented in [Phase 1](PHASE_1.md#canonical-unified-configuration).

The primary `methods`, budget lists, and `explicit_sets` are intentionally empty. Cheap selection therefore creates an empty base artifact before measured greedy search appends its frontier. In contrast, the smoke configuration uses:

```yaml
digital_selection:
  greedy_marginal:
    enabled: false
  methods: [sensitivity_rank]
  budgets:
    projection_counts: [0, 1]
    parameter_fractions: []
    mac_fractions: []
  explicit_sets:
    smoke_all_analog: []
    smoke_protect_top: [block_0/attn.c_proj]
```

> **Standalone enabled warning:** `greedy_marginal.enabled` is checked by `scripts/run_full_pipeline.py`, not by `select_greedy_marginal.py`. Invoking the greedy script directly always runs the search, even when the YAML says `enabled: false`.

> **Phase 1 configuration warning:** Do not use the stale `configs/phase1_sensitivity/lammie_2026.yaml` artifact/configuration path with this phase. Use one unified `configs/full_pipeline/*.yaml` file so the profile, candidate universe, analog preprocessing, hardware capacity, and selection settings agree. The Phase 1 `profiling.sensitivity_field` and `profiling.sensitivity_unit` keys are no-ops; Phase 1.5 reads the artifact's hardcoded `sensitivity_score_for_mapping` field.

## Running Phase 1.5

Run commands from the repository root. Set the exact Phase 1 artifact once for the commands below:

```bash
PHASE1_PROFILE=data/results/phase1_sensitivity/gpt2_hybrid_sensitivity_YYYYmmdd_HHMMSS.json
```

### Cheap/explicit operating points

```bash
python3 -m experiments.phase1_5_digital_selection.select_digital_operating_points \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 "$PHASE1_PROFILE"
```

`--config` defaults to the primary full-pipeline configuration; `--phase1` is required. This writes:

```text
data/results/phase1_5_digital_selection/digital_operating_points.json
data/results/phase1_5_digital_selection/digital_operating_points.csv
```

With the primary configuration these files initially contain no operating-point rows because all cheap methods and explicit sets are empty.

### Measured greedy frontier

Run the greedy search after the base JSON exists:

```bash
python3 -m experiments.phase1_5_digital_selection.select_greedy_marginal \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 "$PHASE1_PROFILE" \
  --output data/results/phase1_5_digital_selection/greedy_marginal_points.json \
  --append-to data/results/phase1_5_digital_selection/digital_operating_points.json
```

Both `--config` and `--phase1` are required. `--output` defaults to the path shown above. `--append-to` is optional, but the supplied file must already exist; a missing path is silently not appended.

The search writes both JSON and CSV for its standalone frontier, then merges new projection sets into the base artifact and rewrites the base CSV with the extended measured-point columns.

### Complete pipeline

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

The pipeline runs the cheap generator first, checks `greedy_marginal.enabled`, writes `greedy_marginal_points.json`, and appends the frontier into `digital_operating_points.json`. Downstream phases consume the combined base path, not the standalone greedy path.

To regenerate selection from an existing profile while continuing the rest of the pipeline:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --skip-phase1 \
  --phase1-artifact "$PHASE1_PROFILE" \
  --reselect-digital
```

Without `--reselect-digital`, skipping Phase 1 requires `--operating-points-artifact` and reuses Phase 1.5 rather than regenerating it.

## Common operating-point JSON schema

Every operating point contains:

| Field | Meaning |
| --- | --- |
| `digital_set_id` | Deterministic `digital_<12 hex>` identifier. |
| `selection_method` | Explicit, score-based, or measured-greedy method label. |
| `budget_type`, `budget_value` | Requested target metadata; counts are serialized as floating-point values. |
| `digital_projection_ids`, `analog_projection_ids` | Sorted partition of the profiled candidate universe. |
| `digital_projection_count`, `analog_projection_count` | Projection counts. |
| `digital_parameter_count`, `total_parameter_count` | Logical candidate projection parameters. |
| `digital_parameter_fraction`, `analog_parameter_fraction` | Logical candidate parameter fractions. |
| `digital_incremental_storage_parameter_count` | Selected untied parameter count. |
| `total_incremental_storage_parameter_count` | Total untied candidate parameter count. |
| `digital_incremental_storage_fraction` | Selected share of untied candidate storage. |
| `digital_macs_per_token`, `total_macs_per_token` | Logical candidate MAC proxies. |
| `digital_mac_fraction`, `analog_mac_fraction` | Logical candidate MAC fractions. |
| `analog_shard_count` | Required analog physical tiers after QKV-aware sharding. |
| `available_physical_tiers` | `num_tiles * tiers_per_tile`. |
| `capacity_feasible` | Whether nominal total tier count covers the analog shards. |

`digital_set_id` hashes `selection_method`, `budget_type`, requested `budget_value`, and sorted digital IDs. Hardware geometry, configuration hash, measured quality, and source artifact are not part of the ID. The same projection set can therefore have different IDs before set-level deduplication, while changing hardware alone leaves an ID unchanged.

## Base operating-point artifacts

`digital_operating_points.json` initially contains:

```text
phase1_path
profile_mapping_unit
operating_points[]
```

Before measured append, `digital_operating_points.csv` has exact columns:

```text
digital_set_id
selection_method
budget_type
budget_value
digital_projection_count
digital_parameter_fraction
digital_incremental_storage_fraction
digital_mac_fraction
analog_shard_count
available_physical_tiers
capacity_feasible
digital_projection_ids
```

Projection IDs are semicolon-separated in CSV. The JSON remains the authoritative full schema because the CSV omits analog IDs and most absolute counts.

## Measured greedy artifacts

Every measured greedy point adds:

```text
measured_nominal_nll
measured_nominal_ppl
digital_reference_nll
digital_reference_ppl
delta_nll_nominal_vs_digital
marginal_nll_gain
marginal_gain_per_cost
promoted_projection_id
evaluated_candidate_promotions
selection_objective
selection_cost_field
```

At step zero, `promoted_projection_id` is null and both marginal values are zero. For `objective: nll_gain`, the current `marginal_gain_per_cost` field stores the raw objective utility rather than a cost-normalized value despite its name.

The standalone `greedy_marginal_points.json` contains:

```text
phase1_path
dataset
predicted_tokens
digital_reference_nll
digital_reference_ppl
selection_method
forced_digital_projection_ids
candidate_projection_ids
candidate_pool_projection_ids
operating_points[]
recommended_operating_point
```

The exact primary method string is:

```text
greedy_measured_gain_per_cost_per_macs_per_token
```

The standalone greedy CSV contains:

```text
digital_set_id
selection_method
budget_type
budget_value
promoted_projection_id
digital_projection_count
digital_parameter_fraction
digital_incremental_storage_fraction
digital_mac_fraction
measured_nominal_nll
measured_nominal_ppl
delta_nll_nominal_vs_digital
marginal_nll_gain
marginal_gain_per_cost
analog_shard_count
available_physical_tiers
capacity_feasible
digital_projection_ids
```

When `--append-to` succeeds, the base JSON retains its original top-level fields, merges unique sets, and adds:

```text
measured_greedy_source
recommended_digital_set_id
recommended_digital_projection_ids
recommendation_reason
```

The combined base CSV is rewritten with the standalone greedy CSV columns. Cheap rows have blank measured-only cells.

## Downstream contracts

- **Phase 3** creates placements only for points with `capacity_feasible: true`. It must shard exactly `analog_projection_ids` and exclude every protected digital projection.
- **Pipeline validation** requires each digital/analog pair to be a disjoint, complete partition of the Phase 1 IDs. It checks unique `digital_set_id` values and recomputes the capacity flag.
- **Phase 4** filters points by `phase4.evaluate_budget_types`, `phase4.evaluate_selection_methods`, and optionally explicit requested IDs. The primary configuration selects `greedy_step` points with method `greedy_measured_gain_per_cost_per_macs_per_token`, then samples the nested frontier with `phase4.max_operating_points`.
- **Multi-seed runs** intentionally reuse Phase 1 and Phase 1.5 while regenerating Phases 2 through 4 for new hardware trace seeds.

An infeasible point can remain in the Phase 1.5 artifact for frontier completeness, but it will have no Phase 3 placement and cannot be evaluated by Phase 4.

## Cost and reproducibility

Cheap score-based selection performs no model inference and is fast relative to profiling or quality evaluation.

For the primary measured search with 49 candidates, no forced projections, and 12 allowed promotions, the worst-case number of trial passes is:

```text
49 + 48 + ... + 38 = 522 trial calibration passes
```

Including one full-digital reference and the initial nominal-hybrid point gives up to 524 complete passes over the 16,384-token selection calibration subset. Quality targets, budget constraints, or nonpositive marginal gains can stop the run earlier. A restricted candidate pool reduces this cost but changes the search space.

Nominal measurements are deterministic for a fixed environment and set, and repeated sets are memoized only in process memory. The search does not persist its cache or partial frontier, so interruption loses all unsaved progress. The model and dataset are resolved through Hugging Face without pinned revisions.

The cheap artifact and measured artifact record the Phase 1 path, but they do not record a repository commit, configuration hash, artifact schema version, or source Phase 1 hash. Preserve the exact configuration and Phase 1 JSON alongside reported results.

## Validation and tests

Run the root project tests with:

```bash
python3 -m pytest -q tests
```

Relevant coverage includes:

- forced count selection;
- cost-normalized ranking;
- operating-point cost fractions;
- monotonic fraction-target sets;
- primary configuration assertions that no projection is hardcoded digital;
- primary all-analog nominal capacity;
- Phase 4's expected automatic method filter; and
- temporary digital-module swap and exception restoration when Torch and AIHWKit are available.

There is currently no direct test for the measured greedy loop, candidate-pool truncation, stopping/recommendation logic, CLI artifact generation, explicit-ID validation, capacity annotation in the selector, append/deduplication behavior, or complete JSON/CSV schemas.

## Current limitations and failure modes

- Fraction targets can overshoot; they are not strict maximum budgets.
- Forced projections can exceed requested cheap targets and do not apply automatically to explicit sets.
- Explicit sets do not validate unknown IDs at generation time.
- Duplicate projection IDs in a Phase 1 artifact are not rejected by Phase 1.5 before accounting.
- Although `candidates_from_profile` has a fallback for nested `results.projections`, both workflow scripts later access top-level `profile["projections"]`; a top-level list is therefore required in practice.
- Count targets, `max_promotions`, candidate-pool size, and greedy maximum fractions do not receive comprehensive range validation.
- Standalone greedy ignores `greedy_marginal.enabled`.
- A missing `--append-to` file causes no append and no error.
- If a greedy projection set already exists under a different selector record, append deduplication keeps the old record. Recommendation metadata can consequently name a greedy `digital_set_id` that is absent from the combined `operating_points` list.
- For `objective: nll_gain`, `marginal_gain_per_cost` is a misleading field name because it contains raw gain.
- Capacity considers nominal total physical slots only. It does not incorporate Phase 2 time-dependent availability, placement constraints beyond one shard per tier, communication, or thermal topology.
- Selection cost fractions cover profiled projection weights/MACs only, not the entire model or measured system cost.
- Measured search evaluates nominal calibration quality, not Phase 2 hardware noise or held-out test quality.
- There is no checkpoint/resume, persisted measurement cache, incremental frontier save, output overwrite guard, provenance hash, or artifact schema version.
- The primary cheap stage is intentionally empty. If greedy search fails before append, the remaining base artifact contains no usable point.
