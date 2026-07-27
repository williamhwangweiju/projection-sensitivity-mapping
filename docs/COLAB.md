# Running the pipeline on Google Colab

The development machine cannot run AIHWKit, so Phases 1 and 4 (and full
pipeline runs) execute on Colab. The repository is cloned inside a mounted
Google Drive folder so artifacts survive session preemption.

## One-time setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content/drive/MyDrive
git clone <your-repo-url> projection-sensitivity-mapping-hybrid-auto
cd projection-sensitivity-mapping-hybrid-auto
```

Install pinned dependencies. Keep torch matched to the Colab CUDA image and
respect the repository pins (`transformers<5`, `datasets<4`,
`aihwkit==1.1.0`):

```bash
pip install -r requirements.txt
python3 scripts/smoke_aihwkit_contract.py --device cpu
```

If the `aihwkit` wheel fails to build, install the CPU wheel from PyPI first
(`pip install aihwkit==1.1.0`) — this pipeline disables AIHWKit's internal
noise and does not require the CUDA-enabled build; the GPU is used by
PyTorch for model forwards and Phase 0 training.

Edit the target configuration before launching:

- `model.device: cuda`
- `hwa_training.precision: bf16` (A100/L4; use `fp32` on T4)

## Session recipe

Smoke first, then the full run:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim_smoke.yaml
```

```bash
RUN_PHASE0=1 bash scripts/run_full_pipeline.sh \
  configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

The driver order is: Phase 0 (HWA fine-tune, producing
`model.checkpoint`) → Phase 1 → proxy scores → Phase 1.5 → Phase 2 →
Phase 3 → contract validation → Phase 4 → energy analysis.

For the generalization run:

```bash
RUN_PHASE0=1 bash scripts/run_full_pipeline.sh \
  configs/full_pipeline/gpt2_medium_hybrid_3dcim.yaml
```

### PTQ contrast run

The vanilla (no-HWA) contrast uses the same configuration with
`hwa_training.enabled: false` and `model.checkpoint: null` — or reuse
existing vanilla artifacts with the skip flags below. Keep the two artifact
trees separate.

## Resuming after preemption

- **Phase 0** resumes automatically from the newest
  `checkpoints/step_XXXXXX/` directory; pass `--no-resume` to the Phase 0
  runner to restart. `keep_last_checkpoints` bounds Drive usage
  (a gpt2-small checkpoint is ~0.5 GB; gpt2-medium ~1.4 GB).
- **Later phases** resume through artifact injection:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --skip-phase0 \
  --skip-phase1 \
  --phase1-artifact  data/results/phase1_sensitivity/<profile>.json \
  --proxy-artifact   data/results/phase1_sensitivity/proxy_sensitivity_<ts>.json \
  --operating-points-artifact data/results/phase1_5_digital_selection/digital_operating_points.json \
  --skip-phase2 \
  --trace-artifact   data/results/phase2_fidelity/fidelity_traces/mixed_96x8/seed_42/trace.npz \
  --skip-phase3 \
  --phase3-manifest  data/results/phase3_static_mapping/phase3_manifest.json
```

Phase 4 has no within-run resume, but
`hybrid_quality_by_policy.partial.csv` is written after each operating
point; a rerun with the flags above repeats only Phase 4.

## Expected runtimes (order of magnitude)

| Stage | gpt2 (small) | gpt2-medium |
| --- | --- | --- |
| Phase 0 (2000 steps) | ~1–2 h on L4, less on A100 | ~4–6 h on L4 |
| Phase 1 (540 passes over 64k tokens) | dominant profiling cost; several hours on GPU | ~3× small |
| Proxy scores (32 gradient batches) | minutes | minutes |
| Phase 1.5 measured greedy | up to ~524 passes; hours | ~3× small |
| Phase 2 + Phase 3 | seconds–minutes (CPU) | seconds–minutes |
| Phase 4 (3 points × 5 timesteps × 3 realizations × 5 policies) | 225 noisy passes + LAMBADA at realization 0 | ~3× small |

Budget sessions accordingly: Phase 0 and Phase 1 belong in separate Colab
sessions with artifacts on Drive, then a final session runs Phases 2–4 from
the saved artifacts.

## Multi-seed hardware traces

```bash
python3 scripts/run_multiseed_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
  --phase1 <profile>.json \
  --operating-points <digital_operating_points.json> \
  --proxy <proxy_sensitivity.json> \
  --trace-seeds 41 42 43 \
  --output-root data/results/multiseed
```

`--proxy` reuses one proxy sidecar across seeds (required when the policy
list includes `static_fisher`). Phase 0 is skipped and its checkpoint is
reused; the script fails fast if `model.checkpoint` does not exist.

Then produce the inferential paper table (traces as statistical units —
the within-trace bootstrap intervals are descriptive only):

```bash
python3 scripts/aggregate_multiseed.py \
  --manifest data/results/multiseed/multiseed_run_manifest.yaml
```

## HWA versus PTQ artifact trees

The notebook separates variants as `results/<mode>/hwa/seed_<seed>` and
`results/<mode>/ptq/seed_<seed>`. If you have artifacts from before this
separation (stored directly under `results/<mode>/seed_<seed>`), move the
Phase 0 output into the `hwa` subtree and the vanilla artifacts into the
`ptq` subtree once, then never mix them again.
