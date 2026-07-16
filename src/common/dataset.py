"""WikiText preprocessing shared verbatim by Phase 1 and Phase 4."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from datasets import load_dataset


def _make_window(
    token_ids: Sequence[int],
    start: int,
    sequence_length: int,
    previous_end: int,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    end = min(start + sequence_length, len(token_ids))
    tokens = list(token_ids[start:end])
    target_length = min(end - previous_end, len(tokens))
    padding = sequence_length - len(tokens)

    input_ids = tokens + [pad_token_id] * padding
    attention_mask = [1] * len(tokens) + [0] * padding
    labels = list(input_ids)

    for index in range(len(tokens) - target_length):
        labels[index] = -100
    for index in range(len(tokens), sequence_length):
        labels[index] = -100

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    valid_targets = int((batch["labels"][1:] != -100).sum().item())
    return batch, end, valid_targets


def build_lm_batches(
    config: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    dataset_cfg = config["dataset"]
    name = str(dataset_cfg["name"])
    dataset_config = str(dataset_cfg["config"])
    split = str(dataset_cfg["split"])
    sequence_length = int(dataset_cfg["sequence_length"])
    stride = int(dataset_cfg["stride"])
    batch_size = int(dataset_cfg["batch_size"])
    max_tokens: Optional[int] = (
        None if dataset_cfg.get("max_tokens") is None else int(dataset_cfg["max_tokens"])
    )
    separator = str(dataset_cfg.get("document_separator", "\n\n"))
    drop_incomplete = bool(dataset_cfg.get("drop_incomplete_final_sequence", True))

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if stride <= 0 or batch_size <= 0:
        raise ValueError("stride and batch_size must be positive.")

    raw_dataset = load_dataset(name, dataset_config, split=split)
    token_ids: list[int] = []
    for sample in raw_dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text + separator, add_special_tokens=False))
        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break

    windows: list[dict[str, torch.Tensor]] = []
    predicted_tokens = 0
    previous_end = 0
    start = 0
    while start < len(token_ids):
        remaining = len(token_ids) - start
        if remaining < sequence_length and drop_incomplete:
            break
        if remaining < 2:
            break
        window, end, valid_targets = _make_window(
            token_ids,
            start,
            sequence_length,
            previous_end,
            int(tokenizer.pad_token_id),
        )
        windows.append(window)
        predicted_tokens += valid_targets
        previous_end = end
        if end >= len(token_ids):
            break
        start += stride

    if not windows:
        raise ValueError("Dataset preprocessing produced no evaluation windows.")

    batches = [
        {
            key: torch.stack([window[key] for window in windows[index : index + batch_size]])
            for key in ("input_ids", "attention_mask", "labels")
        }
        for index in range(0, len(windows), batch_size)
    ]

    metadata = {
        "name": name,
        "config": dataset_config,
        "split": split,
        "sequence_length": sequence_length,
        "stride": stride,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "document_separator": separator,
        "drop_incomplete_final_sequence": drop_incomplete,
        "collected_tokens": len(token_ids),
        "num_windows": len(windows),
        "num_batches": len(batches),
        "predicted_tokens_per_pass": predicted_tokens,
    }
    return batches, metadata
