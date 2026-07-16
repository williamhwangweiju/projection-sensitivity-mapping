"""Deterministic fixed-window causal-language-model dataset construction."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from datasets import load_dataset


def build_causal_lm_batches(
    config: Mapping[str, Any], tokenizer: Any
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    cfg = config["dataset"]
    dataset = load_dataset(str(cfg["name"]), str(cfg["config"]), split=str(cfg["split"]))
    separator = str(cfg.get("document_separator", "\n\n"))
    max_tokens = cfg.get("max_tokens")
    token_ids: list[int] = []
    for sample in dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text + separator, add_special_tokens=False))
        if max_tokens is not None and len(token_ids) >= int(max_tokens):
            token_ids = token_ids[: int(max_tokens)]
            break

    sequence_length = int(cfg["sequence_length"])
    stride = int(cfg["stride"])
    batch_size = int(cfg.get("batch_size", 1))
    drop_incomplete = bool(cfg.get("drop_incomplete_final_sequence", True))
    if sequence_length < 2 or stride <= 0 or batch_size <= 0:
        raise ValueError("Invalid sequence_length, stride, or batch_size.")

    pad_id = int(tokenizer.pad_token_id)
    windows: list[dict[str, torch.Tensor]] = []
    predicted_tokens = 0
    previous_end = 0
    start = 0
    while start < len(token_ids):
        end = min(start + sequence_length, len(token_ids))
        tokens = token_ids[start:end]
        if len(tokens) < sequence_length and drop_incomplete:
            break
        if len(tokens) < 2:
            break
        target_length = min(end - previous_end, len(tokens))
        padding = sequence_length - len(tokens)
        input_ids = tokens + [pad_id] * padding
        attention_mask = [1] * len(tokens) + [0] * padding
        labels = list(input_ids)
        for index in range(len(tokens) - target_length):
            labels[index] = -100
        for index in range(len(tokens), sequence_length):
            labels[index] = -100
        window = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        predicted_tokens += int((window["labels"][1:] != -100).sum().item())
        windows.append(window)
        previous_end = end
        if end >= len(token_ids):
            break
        start += stride

    if not windows:
        raise ValueError("Dataset preprocessing produced no evaluation windows.")
    batches = [
        {
            key: torch.stack([w[key] for w in windows[i : i + batch_size]])
            for key in ("input_ids", "attention_mask", "labels")
        }
        for i in range(0, len(windows), batch_size)
    ]
    metadata = {
        "name": str(cfg["name"]),
        "config": str(cfg["config"]),
        "split": str(cfg["split"]),
        "sequence_length": sequence_length,
        "stride": stride,
        "batch_size": batch_size,
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "collected_tokens": len(token_ids),
        "num_windows": len(windows),
        "num_batches": len(batches),
        "predicted_tokens_per_pass": predicted_tokens,
    }
    return batches, metadata
