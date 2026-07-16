"""Quality and statistics utilities."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor


def evaluate_nll_ppl(
    model: Any,
    batches: Iterable[Mapping[str, Tensor]],
    device: torch.device,
) -> tuple[float, float, int]:
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels")
            labels = input_ids.clone() if labels is None else labels.to(device).clone()
            if attention_mask is not None:
                labels.masked_fill_(attention_mask == 0, -100)
            token_count = int((labels[..., 1:] != -100).sum().item())
            if token_count == 0:
                continue
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            total_nll += float(output.loss.item()) * token_count
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("No valid causal-LM prediction tokens were evaluated.")
    nll = total_nll / total_tokens
    return float(nll), float(math.exp(nll)), total_tokens


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence.")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    sem = std / math.sqrt(array.size) if array.size > 1 else 0.0
    half_width = 1.96 * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }
