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


def evaluate_lambada_accuracy(
    model: Any,
    examples: Iterable[Mapping[str, Any]],
    device: torch.device,
    batch_size: int = 8,
    pad_token_id: int | None = None,
) -> tuple[float, int]:
    """Greedy final-word accuracy: every target token must be the argmax.

    Examples follow the ``build_lambada_examples`` contract. Sequences are
    right-padded within each batch; causal attention plus the attention mask
    keeps padding from influencing the scored positions.
    """
    example_list = list(examples)
    if not example_list:
        raise ValueError("No LAMBADA examples to evaluate.")
    if pad_token_id is None:
        pad_token_id = int(getattr(model.config, "pad_token_id", 0) or 0)
    correct = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(example_list), batch_size):
            chunk = example_list[start : start + batch_size]
            lengths = [int(example["input_ids"].shape[0]) for example in chunk]
            width = max(lengths)
            input_ids = torch.full(
                (len(chunk), width), pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, (example, length) in enumerate(zip(chunk, lengths)):
                input_ids[row, :length] = example["input_ids"]
                attention_mask[row, :length] = 1
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            predictions = logits.argmax(dim=-1)
            for row, (example, length) in enumerate(zip(chunk, lengths)):
                target_len = int(example["target_len"])
                targets = example["input_ids"][length - target_len : length].to(device)
                # Token at index i is predicted by logits at index i - 1.
                predicted = predictions[row, length - target_len - 1 : length - 1]
                if bool(torch.equal(predicted, targets)):
                    correct += 1
    return correct / len(example_list), len(example_list)


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
