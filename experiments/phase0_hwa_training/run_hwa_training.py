#!/usr/bin/env python3
"""Phase 0: hardware-aware noise-injection fine-tuning of the base model."""
from __future__ import annotations

import argparse
import math
import platform
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_yaml, resolve_path, save_json


def _latest_checkpoint(checkpoint_root: Path) -> Path | None:
    if not checkpoint_root.is_dir():
        return None
    steps = sorted(checkpoint_root.glob("step_*"), key=lambda path: path.name)
    for candidate in reversed(steps):
        if (candidate / "training_state.pt").is_file():
            return candidate
    return None


def _prune_checkpoints(checkpoint_root: Path, keep: int) -> None:
    steps = sorted(checkpoint_root.glob("step_*"), key=lambda path: path.name)
    for stale in steps[: max(0, len(steps) - keep)]:
        shutil.rmtree(stale, ignore_errors=True)


def _cosine_lr_lambda(warmup_steps: int, max_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def main(config_path: Path, resume: bool = True) -> Path:
    # Heavy dependencies are imported lazily so --help stays dependency-light.
    import numpy as np
    import torch
    import transformers

    from src.common.dataset import build_causal_lm_batches
    from src.common.metrics import evaluate_nll_ppl
    from src.common.model_loading import load_model_and_tokenizer
    from src.training.hwa import (
        HWANoiseSettings,
        set_noise_enabled,
        unwrap_analog_candidates,
        wrap_analog_candidates,
    )

    config = load_yaml(config_path)
    hwa_cfg = config["hwa_training"]
    output_root = resolve_path(hwa_cfg["output_root"])
    checkpoint_root = output_root / "checkpoints"
    final_dir = output_root / "checkpoint_final"

    max_steps = int(hwa_cfg["max_steps"])
    accumulation = int(hwa_cfg.get("gradient_accumulation_steps", 1))
    learning_rate = float(hwa_cfg["learning_rate"])
    weight_decay = float(hwa_cfg.get("weight_decay", 0.01))
    warmup_steps = int(hwa_cfg.get("warmup_steps", 0))
    max_grad_norm = float(hwa_cfg.get("max_grad_norm", 1.0))
    precision = str(hwa_cfg.get("precision", "fp32"))
    seed = int(hwa_cfg.get("seed", config["experiment"]["seed"]))
    checkpoint_every = int(hwa_cfg.get("checkpoint_every_steps", 250))
    keep_last = int(hwa_cfg.get("keep_last_checkpoints", 2))
    eval_every = int(hwa_cfg.get("eval_every_steps", 0))

    device = torch.device(str(config["model"]["device"]))
    use_bf16 = precision == "bf16" and device.type == "cuda"

    # Phase 0 always starts from the pretrained base model, never from its own
    # previous output; model.checkpoint is what downstream phases consume.
    base_config = deepcopy(dict(config))
    if base_config["model"].get("checkpoint") is not None:
        print(
            "Phase 0 ignores model.checkpoint and fine-tunes from "
            f"model.name={base_config['model']['name']!r}."
        )
        base_config["model"] = {**base_config["model"], "checkpoint": None}

    resume_dir = _latest_checkpoint(checkpoint_root) if resume else None
    if resume_dir is not None:
        print(f"Resuming Phase 0 from {resume_dir}")
        base_config["model"] = {**base_config["model"], "checkpoint": str(resume_dir)}

    model, tokenizer, model_source = load_model_and_tokenizer(base_config, device=device)
    model.train()

    settings = HWANoiseSettings.from_config(config)
    settings.validate()
    wrapped = wrap_analog_candidates(model, settings, seed=seed)
    print(f"Wrapped {len(wrapped)} analog-candidate projections for noise injection.")

    train_config = {"dataset": hwa_cfg["dataset"]}
    batches, train_metadata = build_causal_lm_batches(train_config, tokenizer)
    print(
        f"Training corpus: {train_metadata['collected_tokens']} tokens in "
        f"{train_metadata['num_batches']} micro-batches."
    )

    eval_batches = None
    if eval_every > 0:
        eval_config = deepcopy(dict(config))
        eval_config["dataset"] = deepcopy(config["dataset"])
        eval_config["dataset"]["max_tokens"] = int(hwa_cfg.get("eval_max_tokens", 65536))
        eval_batches, _ = build_causal_lm_batches(eval_config, tokenizer)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine_lr_lambda(warmup_steps, max_steps)
    )

    start_step = 0
    rng = np.random.default_rng(seed)
    saved_order: list[int] | None = None
    saved_cursor = 0
    if resume_dir is not None:
        state = torch.load(
            resume_dir / "training_state.pt", map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["numpy_rng_state"]
        if len(state.get("batch_order", [])) == len(batches):
            saved_order = [int(value) for value in state["batch_order"]]
            saved_cursor = int(state.get("batch_cursor", 0))
        for projection_id, generator_state in state.get("noise_generator_states", {}).items():
            if projection_id in wrapped and generator_state is not None:
                wrapped[projection_id].set_generator_state(generator_state, device)

    def save_checkpoint(step: int) -> Path:
        step_dir = checkpoint_root / f"step_{step:06d}"
        unwrap_analog_candidates(model, wrapped)
        try:
            model.save_pretrained(step_dir)
            tokenizer.save_pretrained(step_dir)
        finally:
            for projection_id, wrapper in wrapped.items():
                setattr(wrapper.unwrap_parent, wrapper.unwrap_attribute, wrapper)
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "batch_order": [int(value) for value in order],
                "batch_cursor": int(cursor),
                "noise_generator_states": {
                    projection_id: wrapper.generator_state()
                    for projection_id, wrapper in wrapped.items()
                },
            },
            step_dir / "training_state.pt",
        )
        save_json(
            step_dir / "trainer_state.json",
            {"step": step, "max_steps": max_steps, "learning_rate": scheduler.get_last_lr()[0]},
        )
        _prune_checkpoints(checkpoint_root, keep_last)
        return step_dir

    def run_eval(step: int) -> dict[str, float]:
        model.eval()
        set_noise_enabled(wrapped, False)
        clean_nll, clean_ppl, _ = evaluate_nll_ppl(model, eval_batches, device)
        set_noise_enabled(wrapped, True)
        noisy_nll, noisy_ppl, _ = evaluate_nll_ppl(model, eval_batches, device)
        model.train()
        record = {
            "step": step,
            "clean_nll": float(clean_nll),
            "clean_ppl": float(clean_ppl),
            "noisy_nll": float(noisy_nll),
            "noisy_ppl": float(noisy_ppl),
        }
        print(
            f"eval step={step} clean_nll={clean_nll:.4f} clean_ppl={clean_ppl:.2f} "
            f"noisy_nll={noisy_nll:.4f} noisy_ppl={noisy_ppl:.2f}"
        )
        return record

    eval_history: list[dict[str, float]] = []
    loss_history: list[float] = []
    if saved_order is not None:
        order = np.asarray(saved_order)
        cursor = saved_cursor
    else:
        order = rng.permutation(len(batches))
        cursor = 0
    autocast_dtype = torch.bfloat16 if use_bf16 else None
    micro_losses: list[float] = []

    for step in range(start_step, max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses.clear()
        for _ in range(accumulation):
            if cursor >= len(order):
                order = rng.permutation(len(batches))
                cursor = 0
            batch = batches[int(order[cursor])]
            cursor += 1
            moved = {key: value.to(device) for key, value in batch.items()}
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss = model(**moved).loss
            else:
                loss = model(**moved).loss
            (loss / accumulation).backward()
            micro_losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        step_loss = sum(micro_losses) / len(micro_losses)
        loss_history.append(step_loss)
        if (step + 1) % 25 == 0 or step == start_step:
            print(
                f"step={step + 1}/{max_steps} loss={step_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        completed = step + 1
        if eval_every > 0 and eval_batches is not None and completed % eval_every == 0:
            eval_history.append(run_eval(completed))
        if completed % checkpoint_every == 0 and completed < max_steps:
            save_checkpoint(completed)

    if eval_every > 0 and eval_batches is not None and (
        not eval_history or eval_history[-1]["step"] != max_steps
    ):
        eval_history.append(run_eval(max_steps))

    unwrap_analog_candidates(model, wrapped)
    model.eval()
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metadata_path = output_root / "hwa_metadata.json"
    save_json(
        metadata_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "software_versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "model_source": model_source,
            "device": str(device),
            "precision": "bf16" if use_bf16 else "fp32",
            "seed": seed,
            "max_steps": max_steps,
            "completed_steps": max_steps,
            "gradient_accumulation_steps": accumulation,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "noise_settings": {
                "clip_sigma": settings.clip_sigma,
                "range_mode": settings.range_mode,
                "noise_std_range": list(settings.noise_std_range),
                "clip_in_forward": settings.clip_in_forward,
                "include_lm_head": settings.include_lm_head,
                "exclude_projection_ids": sorted(settings.exclude_projection_ids),
                "wrapped_projection_count": len(wrapped),
                "scheme": "perturb_forward_clean_update",
            },
            "train_dataset": train_metadata,
            "final_loss": loss_history[-1] if loss_history else None,
            "eval_history": eval_history,
            "checkpoint_final": str(final_dir),
        },
    )
    print(f"Phase 0 complete: {final_dir}")
    return final_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing step checkpoints and restart from the pretrained model.",
    )
    args = parser.parse_args()
    main(args.config, resume=not args.no_resume)
