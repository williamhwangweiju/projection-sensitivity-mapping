# Phase 4: All-Analog Quality Evaluation

Phase 4 is the model-level evaluation stage. It converts every profiled
projection to AIHWKit analog execution, materializes time-varying Phase-2
tile noise under each static Phase-3 placement, and measures GPT-2 negative
log-likelihood (NLL) and perplexity (PPL) on held-out data.

The principal comparison is paired: for the same timestep and Gaussian
realization, how much quality does each method policy
(`static_sensitivity`) preserve relative to the `random`,
`sequential`, and `hardware_only` baselines?

## Inputs

1. the unified YAML configuration;
2. the Phase 1 projection profile (candidate universe + preprocessing
   checksums);
3. the Phase 2 `trace.npz`; and
4. the Phase 3 `phase3_manifest.json`.

## Quick start

The pipeline driver invokes Phase 4 after contract validation:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

To evaluate existing artifacts directly:

```bash
python3 experiments/phase4_quality/run_hybrid_quality.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --trace data/results/phase2_fidelity/fidelity_traces/mixed_96x8/seed_42/trace.npz \
  --phase3-manifest data/results/phase3_static_mapping/phase3_manifest.json
```

## Evaluation design

### 1. References

The untouched model (loaded from `model.checkpoint`, i.e. the Phase-0 HWA
weights) is evaluated once on the held-out corpus:

```text
digital_nll, digital_ppl = exp(digital_nll)
```

Every projection is then converted to `AnalogLinearMapped`: canonical
weights are clipped once at `analog.clip_sigma` population standard
deviations, written with exact mapped-tile writes, and verified against the
Phase-1 preprocessing checksums. The zero-tile-noise evaluation of this
model gives the nominal reference:

```text
nominal_hybrid_nll
delta_nll_nominal_vs_digital = nominal_hybrid_nll - digital_nll
```

This delta is the deployment (clipping + ADC/DAC) cost of the all-analog
conversion — the number that answers "why is no digital protection needed?"

### 2. Materialize tile noise in logical weight coordinates

AIHWKit's internal programming, read, drift, and forward weight noise are
disabled; Phase-2 noise is added manually exactly once per evaluation. For
the coordinates of shard `s` assigned to tile `i` at timestep `t`:

```text
Wnoisy,p[s] = W0,p[s] + noise_std[t, i] × Rp × Zp,r[s]
```

with `Rp` the projection's programmed range and `Zp,r` an i.i.d.
standard-normal field keyed by the experiment seed, projection ID, and
realization `r`. There is no post-noise clipping. The same coordinate field
is reused across policies and timesteps; only the tile-dependent scale
changes, so policy comparisons are paired.

### 3. Evaluate static placements over time

At each requested timestep the placement CSV's tile-noise values are
replaced with the current trace values; unavailable tiles receive
`phase4.unavailable_noise_std`. For every timestep × realization × policy:

```text
delta_nll_total = noisy_nll - digital_nll
delta_nll_tile  = noisy_nll - nominal_hybrid_nll
```

The first includes the nominal conversion cost; the second isolates the
degradation attributable to tile noise — the quantity placement policies
compete over.

## Timesteps

`phase4.timesteps` values are clamped to the trace range, deduplicated, and
sorted. When null, Phase 4 chooses timestep 0, the midpoint, the final
timestep, and the timesteps around the earliest fault onset.

## Checkpointing and resume

`hybrid_quality_by_policy.partial.csv` is rewritten after every completed
timestep block. On restart, completed (policy, timestep, realization)
evaluations are loaded and skipped — noise fields are order-independent, so
a preempted session repeats at most one timestep. The partial file is
removed once the final CSV is written.

## Configuration

| Field | Meaning |
| --- | --- |
| `model.checkpoint` | Phase-0 HWA weights ([Phase 0](PHASE_0.md)); null = pretrained (PTQ contrast) |
| `evaluation_dataset` | Held-out dataset/windowing; falls back to `dataset` |
| `analog.*` | Shared Phase 1/4 clipping, range, tile, ADC/DAC, and scaling settings |
| `phase4.output_root` | Artifact directory |
| `phase4.policies` | Placement policies to evaluate |
| `phase4.method_policies` / `phase4.baseline_policies` | Paired-summary comparison sets |
| `phase4.timesteps` | Explicit trace snapshots, or null for automatic selection |
| `phase4.num_realizations` | Gaussian fields per timestep and policy |
| `phase4.antithetic` | Evaluate paired ±Z fields |
| `phase4.unavailable_noise_std` | Substitute scale for unavailable tiles |

## Artifacts

| Artifact | Contents |
| --- | --- |
| `hybrid_quality_by_policy.csv` | Every timestep/realization/policy evaluation |
| `nominal_reference.json` | Digital and nominal all-analog quality and their delta |
| `hybrid_quality_summary.csv` | Per-policy means/stds, descriptive bootstrap interval, mean degraded NLL/PPL |
| `paired_policy_summary.csv` | Paired improvements of every method policy over every baseline |
| `phase4_metadata.json` | Provenance, model source, dataset metadata, analog settings, artifact paths |

The paired summary defines, for every configured method/baseline pair:

```text
NLL improvement = baseline delta_nll_tile - method delta_nll_tile
```

A positive value means the method policy produced lower NLL.

> [!WARNING]
> Within one run, the timestep × realization samples share a single
> correlated hardware trace, one model, and paired noise fields. The
> bootstrap intervals here are **descriptive within-trace spreads, not
> inferential 95% confidence intervals**. For paper claims, run several
> trace seeds (`scripts/run_multiseed_pipeline.py`) and aggregate with
> `scripts/aggregate_multiseed.py`, which treats each independent trace as
> one statistical unit and reports across-trace t-based intervals.

## Sanity checks

```bash
python3 experiments/phase4_quality/run_sanity_checks.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 data/results/phase1_sensitivity/<profile>.json \
  --phase3-manifest data/results/phase3_static_mapping/phase3_manifest.json
```

This uses at most 4,096 tokens and verifies two invariants: zero tile noise
reproduces the nominal NLL for every policy, and uniform tile noise makes
NLL invariant to placement policy (default tolerance `1e-6`;
`phase4.sanity_zero_nll_tolerance` / `phase4.sanity_uniform_nll_tolerance`
override).

## Cost and scaling

The number of noisy dataset passes is `timesteps × realizations × policies`
— 75 for the primary configuration (5 × 3 × 5), plus the digital and
nominal references. Each pass scores the full held-out corpus through the
AIHWKit analog forward path.

## Interpretation and limitations

- Results measure NLL/PPL under the manually materialized Gaussian noise
  model; they are not measurements from physical hardware.
- Static placements are not adapted after drift or faults; unavailability is
  approximated as a high noise scale.
- AIHWKit partitions each projection internally for I/O quantization
  (`tile_size`), which does not coincide with the Phase-3 Q/K/V-preserving
  physical shard boundaries; the partition is identical across placement
  policies, so paired comparisons are unaffected.
- The runner does not compute KL divergence, token agreement, latency,
  energy, communication, or migration cost.
- The placement proxy column is diagnostic; only the forward evaluation
  measures model-level quality.
