"""Token-weighted language-model metrics and statistical summaries."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def evaluate_nll_ppl(
    model: Any,
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, int]:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            moved = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**moved)
            valid = int((moved["labels"][:, 1:] != -100).sum().item())
            total_loss += float(outputs.loss.item()) * valid
            total_tokens += valid
    if total_tokens <= 0:
        raise ValueError("No predicted tokens were evaluated.")
    nll = total_loss / total_tokens
    return nll, math.exp(nll), total_tokens


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list.")
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "count": int(len(array)),
    }
