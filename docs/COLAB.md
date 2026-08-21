# Running the pipeline on Google Colab

Phases 1 and 4 require AIHWKit and are executed on Colab. The repository is
cloned inside a mounted Google Drive folder so artifacts survive session
preemption; the committed notebook
(`notebooks/projection_sensitivity_mapping.ipynb`) drives everything.

## One-time setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content/drive/MyDrive
git clone <your-repo-url> projection-sensitivity-mapping-hybrid-auto
```

Open the notebook (from GitHub via
`colab.research.google.com/github/...` or a Drive copy) and run Cells 1–7.
Cell 3 installs pinned dependencies (`transformers<5`, `datasets<4`,
`aihwkit==1.1.0`); Cell 4 probes the RPUCuda **mapped-tile** path with the
repository RPU configuration — if the probe fails, the session's GPU cannot
run the pipeline (switch runtime type, e.g. to a T4, and rerun from Cell 2).
Hugging Face caches live on local disk, not on the Drive mount (heavy cache
I/O through the FUSE mount has caused mid-run mount failures).

Set in Cell 1: `USE_HWA` (hardware-aware deployment vs the PTQ contrast) and
the `RUN_PHASE...` flags. `hwa_training.precision: bf16` requires an
L4/A100; use `fp32` on a T4.

## Session recipe

1. Smoke run (`RUN_MODE = "smoke"`) to validate contracts.
2. Phase 0 (`RUN_PHASE0 = 1`, everything else 0): ~1–2 h on L4, T4 slower.
   Auto-resumes from step checkpoints after preemption.
3. Phase 1 (`RUN_PHASE1 = 1`): the dominant profiling cost, several
   hours on GPU (540 calibration passes).
4. Phases 2–4 (`RUN_PHASE2 = RUN_PHASE3 = RUN_PHASE4 = 1`): Phases 2–3 take
   seconds; Phase 4 performs 75 noisy passes over the full held-out test set
   (~6–10 h on a T4) and checkpoints after every timestep block, so a
   preempted session repeats at most one timestep — rerun Cell 8 with the
   same flags to resume.

HWA and PTQ artifacts live in separate Drive trees
(`results/<mode>/hwa/seed_<seed>` vs `results/<mode>/ptq/seed_<seed>`) and
can never overwrite each other.

## Multi-seed hardware traces

Phases 0–1 are reused; each trace seed gets an isolated Phase 2–4 tree:

```bash
python scripts/run_multiseed_pipeline.py \
  --config <generated Drive config> \
  --phase1 <profile>.json \
  --trace-seeds 41 43 44 45 \
  --vary-placement-seed \
  --output-root <drive>/results/full/hwa/multiseed
```

The script fails fast if `model.checkpoint` does not exist. A completed
single-seed run can join the aggregate by appending a
`{trace_seed, output_root}` entry to `multiseed_run_manifest.yaml`.

Then produce the inferential paper table (traces as statistical units — the
within-trace bootstrap intervals are descriptive only):

```bash
python scripts/aggregate_multiseed.py \
  --manifest <drive>/results/full/hwa/multiseed/multiseed_run_manifest.yaml
```

## Leave-one-out profile check (Cell 8h)

The reviewer-requested rank-stability check of the additive Phase-1 profile
(all 49 projections analog, one restored at a time; see
[docs/PHASE_1.md](PHASE_1.md#leave-one-out-check-rank-stability-of-the-additive-profile)).
It reuses the Phase-0 checkpoint and the Phase-1 JSON, costs about one Phase-1
run (501 calibration passes), and writes `leave_one_out/` next to the Phase-1
artifact:

```bash
python experiments/phase1_sensitivity/run_leave_one_out.py \
  --config <generated Drive config> \
  --phase1 <drive>/results/full/hwa/seed_42/phase1/<profile>.json \
  --restore-mode nominal
```

The printed `rank_agreement` block (Spearman rho, Kendall tau, top-k overlap)
is the number the paper's limitation paragraph asks for.
