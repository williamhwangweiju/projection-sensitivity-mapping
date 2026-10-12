# Sensitivity-Aware Placement for All-Analog GPT-2 on Aging In-Memory Tiles

Code for the IEEE ISCAS 2027 paper *"Sensitivity-Aware Placement Sustains
All-Analog GPT-2 Inference on Aging In-Memory Tiles"*. GPT-2 Small is fine-tuned under a
documented analog noise contract (hardware-aware training), each of its 49
weight projections is profiled for noise sensitivity, and the resulting shard
placement onto a heterogeneous, aging tile-and-tier substrate is compared
against workload-blind placements on the full WikiText-103 test set through
IBM's AIHWKit.

Claim under study: for a hardware-aware-trained GPT-2 with every weight
projection on analog tiles, sensitivity-aware placement holds language-model
quality better than random, sequential, and hardware-only placement as
heterogeneous tiles degrade.

This repository contains the pipeline, its configurations, tests, and the
Colab workflow; experiment outputs are not committed.

## Pipeline

```mermaid
flowchart LR
    W[WikiText train] --> P0[Phase 0<br/>HWA fine-tune]
    P0 --> P1[Phase 1<br/>projection sensitivity]
    C[WikiText validation] --> P1
    H[Hardware and fault config] --> P2[Phase 2<br/>tile-fidelity trace]
    P1 --> P3[Phase 3<br/>sharding and placement]
    P2 --> P3
    T[WikiText test] --> P4[Phase 4<br/>all-analog NLL/PPL]
    P1 --> P4
    P2 --> P4
    P3 --> P4
```

| Stage | Question | Entry point | Guide |
| --- | --- | --- | --- |
| Phase 0 | Make the all-analog deployment functional (noise-injection fine-tuning) | `experiments/phase0_hwa_training/run_hwa_training.py` | [docs/PHASE_0.md](docs/PHASE_0.md) |
| Phase 1 | Which projections are most sensitive to analog weight noise? | `experiments/phase1_sensitivity/run_aihwkit_profiling.py` | [docs/PHASE_1.md](docs/PHASE_1.md) |
| Phase 1 check | Is the one-at-a-time profile rank-stable with all 49 projections noisy? | `experiments/phase1_sensitivity/run_leave_one_out.py` | [docs/PHASE_1.md](docs/PHASE_1.md#leave-one-out-check-rank-stability-of-the-additive-profile) |
| Fisher-proxy comparison | How does placement from calibration gradient-square scores compare with measured sensitivity? | `scripts/run_fisher_proxy_analysis.py` | [Running](#running) |
| Phase 2 | How does tile noise evolve (heterogeneity, drift, thermal variation, faults)? | `experiments/phase2_fidelity/run_fidelity_model.py` | [docs/PHASE_2.md](docs/PHASE_2.md) |
| Phase 3 | How are analog shards assigned to tile/tier slots under each policy? | `experiments/phase3_baselines/run_baseline_mappings.py` | [docs/PHASE_3.md](docs/PHASE_3.md) |
| Phase 4 | How does each placement age on the held-out test set? | `experiments/phase4_quality/run_hybrid_quality.py` | [docs/PHASE_4.md](docs/PHASE_4.md) |
| Hybrid comparator | LM head / top-2 projections digital, remainder analog | `experiments/phase4_quality/run_hybrid_baselines.py` | [docs/PHASE_4.md](docs/PHASE_4.md#digitalanalog-hybrid-comparator) |
| Robustness checks | How robust is placement to tile-map error, and how much would adaptive re-mapping gain? | `scripts/robustness_analyses.py` | [Running](#running) |

## Experiment contract

The canonical configuration is `configs/full_pipeline/gpt2_hybrid_3dcim.yaml`
(`gpt2_hybrid_3dcim_smoke.yaml` is its fast structural counterpart). Across
hardware traces, the Phase-0 checkpoint and Phase-1 profile are reused;
`experiment.seed` changes, and `--vary-placement-seed` also changes
`experiment.placement_seed` for the random and hardware-only baselines.
Shared rules:

- GPT-2 `Conv1D` weights are handled in canonical `[out, in]` coordinates.
- Each projection is clipped once at `analog.clip_sigma` population standard
  deviations; tile `noise_std` is the logical weight-noise standard deviation
  as a fraction of the projection's programmed (peak-to-peak) range.
- Phase 0 trains through the same clip-and-noise model (perturb-forward /
  clean-update); Phases 1 and 4 use identical preprocessing and noise
  materialization with AIHWKit's internal noise sources disabled.
- Phase 3 places every profiled projection on analog tiers, one shard per tier,
  preserving fused Q/K/V row boundaries.
- Phase 4 keeps placements fixed over time and uses paired coordinate-level
  Gaussian fields, so policies differ only through their assigned tile scales.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 scripts/smoke_aihwkit_contract.py --device cpu   # checks the AIHWKit native extension
```

Requirements pin `aihwkit==1.1.0` and require `transformers>=4.30,<5`;
models and datasets are fetched from the Hugging Face Hub on first use.

The configurations default to CPU. For full experiments, set `model.device`
to `cuda` on a machine with a CUDA-enabled AIHWKit installation, or use the
[Colab workflow](docs/COLAB.md). Full training, profiling, and evaluation runs
are intended for a GPU.

## Running

Single trace (smoke first, then the paper configuration):

```bash
python3 scripts/run_full_pipeline.py --config configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml
python3 scripts/run_full_pipeline.py --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

Phase 0 resumes from its newest step checkpoint and Phase 4 from its
per-timestep partial CSV. Resume with the same configuration and inputs; use
a new output directory when changing them. Phase 1 profiling restarts from
the beginning if interrupted; the leave-one-out check supports resume.
`--skip-phase*` and artifact-path flags reuse existing artifacts (see
`scripts/run_full_pipeline.py --help`).

Independent hardware traces reuse the Phase-0 checkpoint and Phase-1 profile,
and are aggregated with each trace as one statistical unit:

```bash
python3 scripts/run_multiseed_pipeline.py --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 /path/to/profile.json --trace-seeds 41 42 43 44 45 \
  --vary-placement-seed --output-root data/results/multiseed
python3 scripts/aggregate_multiseed.py \
  --manifest data/results/multiseed/multiseed_run_manifest.yaml \
  --output data/results/multiseed/cross_trace_paired_summary.csv
```

Replace `/path/to/profile.json` with the Phase-1 artifact from the paper
configuration, and keep `model.checkpoint` pointed at its shared Phase-0
checkpoint. The command above runs all five traces into one campaign
directory; the Colab notebook can instead reuse the original seed-42 run.

The additive experiments of the paper — placement bounds
(`configs/full_pipeline/gpt2_allanalog_bounds.yaml`), the one-factor Phase-2
sweep (`configs/phase2_sweep/`), the leave-one-out check, and the hybrid
comparator (`scripts/run_hybrid_baselines_multiseed.py`) — are driven by the
Colab notebook `notebooks/projection_sensitivity_mapping.ipynb`
([docs/COLAB.md](docs/COLAB.md)).

The Fisher-proxy comparison has a separate driver. For a Phase-2 trace
generated with seed 42:

```bash
python3 scripts/run_fisher_proxy_analysis.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 /path/to/profile.json --trace /path/to/trace.npz \
  --max-batches 32 --trace-seed 42 \
  --output-root data/results/fisher_proxy/trace_seed_42
```

Replace the artifact paths with the matching Phase-1 profile and Phase-2
trace. The driver computes proxy scores, builds `static_fisher` placement,
and evaluates it against `static_sensitivity`. Use `--proxy-artifact` to
reuse computed scores across traces, and `--phase3-manifest` to reuse base
placements from the same trace.

The map-error and adaptive re-mapping checks of the paper run on CPU in
seconds from the Phase-1 profile alone:

```bash
python3 scripts/robustness_analyses.py --phase1 /path/to/profile.json
```

## Tests and validation

```bash
python3 -m pytest -q tests
python3 scripts/validate_pipeline_contracts.py --help   # cross-phase contract checks (run automatically after Phase 3)
```

## Repository layout

```text
configs/          canonical, smoke, bounds, GPT-2 Medium and Phase-2 sweep configurations
docs/             phase guides, Colab runbook, shared references
experiments/      executable phase entry points
notebooks/        Colab phase-selector notebook
scripts/          pipeline drivers, multiseed/aggregation, Fisher and hybrid campaigns, robustness checks, validation
src/analysis      rank-correlation and Pareto helpers
src/common        dataset windows, metrics, projection handles, analog settings, loaders
src/training      Phase-0 noise-injection wrappers
src/profilers     Phase-1 sensitivity and leave-one-out profilers
src/simulators    tile-fidelity trace model
src/mapping       sharding, placement policies, proxy objectives
src/evaluation    hybrid analog conversion, paired noise materialization, hybrid designs
tests/            structural and contract tests
```

## Modeling boundaries

- The tile-fidelity trace is phenomenological: literature-motivated scenario
  values, not device-calibrated quantities ([Phase 2 provenance](docs/PHASE_2.md#parameter-provenance)).
- All tiers of one tile share one noise scale; placements are static (no
  remapping after drift or faults).
- Capacity is tier-granular; communication, latency, and energy are not modeled.
- The analog LM-head copy is noisy while the tied embedding lookup stays exact,
  in training and deployment alike.
