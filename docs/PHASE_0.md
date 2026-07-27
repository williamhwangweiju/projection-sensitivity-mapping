# Phase 0: Hardware-Aware Noise-Injection Fine-Tuning

Phase 0 fine-tunes the base language model under the same analog weight-noise
model that the rest of the pipeline deploys, producing the checkpoint that
Phases 1, 1.5, and 4 load through `model.checkpoint`. The motivation is the
hardware-aware-training literature (Joshi et al. 2020; Rasch et al. 2023) and
our own preliminary vanilla runs, where post-training deployment left
low-digital-budget operating points with severely degraded perplexity
(artifacts not committed to this repository); noise-aware fine-tuning is
expected to move the digital-budget frontier into a usable regime so
placement effects can be measured on a deployable model.

## Entry points

- Runner: `experiments/phase0_hwa_training/run_hwa_training.py`
- Noise wrappers: `src/training/hwa.py`
- Shared loader consumed downstream: `src/common/model_loading.py`
- Canonical configuration: `configs/full_pipeline/gpt2_hybrid_3dcim.yaml`
  (`hwa_training` section; smoke counterpart is disabled by default)

## Training scheme

The scheme is perturb-forward / clean-update, following hardware-aware
training practice for analog inference accelerators (Joshi et al., Nat.
Commun. 2020; see [REFERENCES](REFERENCES.md)):

1. Every analog-candidate projection (all 48 transformer projections and the
   tied LM head under the primary configuration) is wrapped by a
   `NoisyProjection` module. The wrapped module remains a submodule, so
   parameter identity and optimizer state are unaffected.
2. Each forward pass computes the projection output from a temporary copy of
   the live weight that is:
   - clipped at `analog.clip_sigma` population standard deviations with a
     straight-through estimator (`W + (clamp(W) - W).detach()`), so clipped
     coordinates keep receiving gradient; and
   - perturbed with additive i.i.d. Gaussian noise of standard deviation
     `sigma_norm × programmed_range`, where `sigma_norm` is drawn per
     projection per micro-batch from `Uniform(hwa_training.noise.noise_std_range)`.
3. Gradients are therefore evaluated at the perturbed point while optimizer
   updates apply to the clean parameters. The parameters are never mutated by
   the noise path.

The default `noise_std_range` `[0.0, 0.046]` spans zero to twice the
deployment reference (`analog.reference_noise_std = 0.023`), covering the
Phase-2 tile multiplier range (0.65–1.70×) while the zero end anchors clean
quality.

**Tied LM head.** GPT-2 ties `lm_head.weight` to the token embedding. The
wrapper perturbs only the LM-head matmul; the embedding lookup reads the
clean parameter. This matches deployment, where the analog LM-head copy is
noisy while the digital embedding is exact. Weight updates still flow to the
shared tensor from both paths.

## Quick start

```bash
python3 experiments/phase0_hwa_training/run_hwa_training.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

Phase 0 always starts from the pretrained `model.name` weights (never from
`model.checkpoint`, which is Phase 0's own output). Interrupted runs resume
automatically from the newest step checkpoint; pass `--no-resume` to restart.

As part of the full pipeline:

```bash
python3 scripts/run_full_pipeline.py \
  --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml
```

The driver runs Phase 0 when `hwa_training.enabled` is true and verifies that
the produced directory equals `model.checkpoint`. `RUN_PHASE0=0` (shell
wrapper) or `--skip-phase0` skips it; a skipped Phase 0 requires the
checkpoint to already exist when `model.checkpoint` is set.

## Configuration

| Field | Meaning |
| --- | --- |
| `hwa_training.enabled` | Gate for the pipeline driver. |
| `hwa_training.output_root` | Root for checkpoints and metadata. |
| `hwa_training.dataset` | Training-corpus windowing (same schema as `dataset`). |
| `hwa_training.max_steps` | Optimizer steps. |
| `hwa_training.gradient_accumulation_steps` | Micro-batches per optimizer step. |
| `hwa_training.learning_rate`, `weight_decay`, `warmup_steps`, `max_grad_norm` | AdamW + cosine schedule hyperparameters. |
| `hwa_training.precision` | `bf16` (CUDA autocast) or `fp32`; CPU always runs fp32. |
| `hwa_training.seed` | Shuffling and per-projection noise seeds. |
| `hwa_training.noise.noise_std_range` | Uniform sampling range for `sigma_norm`. |
| `hwa_training.noise.clip_in_forward` | Apply the STE clip during training forwards. |
| `hwa_training.noise.include_lm_head` | Wrap the tied LM head. |
| `hwa_training.noise.exclude_projection_ids` | Candidates to leave un-noised. |
| `hwa_training.checkpoint_every_steps`, `keep_last_checkpoints` | Step-checkpoint cadence and retention. |
| `hwa_training.eval_every_steps`, `eval_max_tokens` | Periodic clean/noisy NLL probe on `dataset` (validation) tokens; `0` disables. |

## Artifacts

| Artifact | Contents |
| --- | --- |
| `checkpoints/step_XXXXXX/` | `save_pretrained` model + tokenizer, `training_state.pt` (optimizer, scheduler, step, RNG and noise-generator states, batch order), `trainer_state.json` |
| `checkpoint_final/` | Unwrapped fine-tuned model + tokenizer, weight tie preserved |
| `hwa_metadata.json` | Provenance (commit, config hash, versions), recipe, noise settings, training-corpus metadata, loss and eval history |

Downstream phases consume `checkpoint_final` via `model.checkpoint`; every
runner records the resolved source in its own metadata (`model_source`).

## Modeling boundaries

- Noise is sampled per projection per micro-batch, not per token or per tile;
  Phase-2 spatial structure is not simulated during training.
- The training forward path is plain `F.linear` — no ADC/DAC quantization or
  bound management. Those effects enter in Phases 1 and 4 through AIHWKit.
- The clip threshold and programmed range are recomputed from the live
  weights each forward, matching how deployment preprocessing would treat the
  final weights, but the Phase-1/4 checksum contract applies to the final
  checkpoint only.
- A resumed run reproduces the saved batch order and noise-generator streams,
  but bitwise equality with an uninterrupted run is not guaranteed across
  device or library changes.
