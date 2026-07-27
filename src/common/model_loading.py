"""Shared causal-LM/tokenizer loading with optional local checkpoint override.

Every runner that previously repeated the from_pretrained idiom loads through
this module so the Phase 0 hardware-aware checkpoint substitutes uniformly:
when ``model.checkpoint`` is set, weights come from that directory while
``model.name`` continues to identify the architecture in metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import resolve_path


def resolve_model_source(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the from_pretrained source and provenance metadata.

    ``model.checkpoint`` (a local ``save_pretrained`` directory, resolved
    relative to the repository root) takes precedence over ``model.name``.
    A configured but missing checkpoint is an error rather than a silent
    fallback: profiling the wrong weights would poison every downstream phase.
    """
    model_cfg = config["model"]
    model_name = str(model_cfg["name"])
    checkpoint = model_cfg.get("checkpoint")
    if checkpoint is None:
        return model_name, {
            "model_name": model_name,
            "checkpoint": None,
            "loaded_from": model_name,
        }
    checkpoint_path = resolve_path(checkpoint)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"model.checkpoint does not exist: {checkpoint_path}. "
            "Run Phase 0 (experiments/phase0_hwa_training) first, or set "
            "model.checkpoint to null to load pretrained weights."
        )
    return str(checkpoint_path), {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "loaded_from": str(checkpoint_path),
    }


def load_model_and_tokenizer(
    config: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the configured model/tokenizer with the canonical evaluation idiom.

    The model is returned as float32, padded with the EOS token, cache
    disabled, and in eval mode. When ``device`` is None the model is left on
    its default device (Phase 1 relies on the profiler's own device handling);
    otherwise it is moved to the requested device.
    """
    source, metadata = resolve_model_source(config)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if device is not None:
        model.to(torch.device(str(device)))
    model.eval()
    return model, tokenizer, metadata
