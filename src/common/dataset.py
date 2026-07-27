"""Deterministic fixed-window causal-language-model dataset construction."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from datasets import load_dataset


def lambada_target_split(text: str) -> tuple[str, str]:
    """Split a LAMBADA passage into (context, target).

    The benchmark asks the model to predict the final word. The target keeps
    its leading space so tokenization matches how the word appears mid-text.
    A passage without any interior whitespace has no valid context and
    returns an empty context; callers should skip those examples.
    """
    stripped = text.rstrip()
    context, separator, last_word = stripped.rpartition(" ")
    if not separator or not context or not last_word:
        return "", stripped
    return context, " " + last_word


def build_lambada_examples(
    cfg: Mapping[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build final-word-prediction examples from a LAMBADA-style dataset.

    Each example is ``{"input_ids": LongTensor[seq], "target_len": int}``
    where the last ``target_len`` tokens are the target word. Examples are
    taken in source order up to ``max_examples``.
    """
    name = str(cfg.get("name", "EleutherAI/lambada_openai"))
    subset = cfg.get("config", "en")
    split = str(cfg.get("split", "test"))
    max_examples = cfg.get("max_examples")
    dataset = (
        load_dataset(name, str(subset), split=split)
        if subset is not None
        else load_dataset(name, split=split)
    )
    examples: list[dict[str, Any]] = []
    skipped = 0
    for sample in dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        context, target = lambada_target_split(text)
        if not context:
            skipped += 1
            continue
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not context_ids or not target_ids:
            skipped += 1
            continue
        examples.append(
            {
                "input_ids": torch.tensor(context_ids + target_ids, dtype=torch.long),
                "target_len": len(target_ids),
            }
        )
        if max_examples is not None and len(examples) >= int(max_examples):
            break
    if not examples:
        raise ValueError("LAMBADA preprocessing produced no examples.")
    metadata = {
        "name": name,
        "config": None if subset is None else str(subset),
        "split": split,
        "max_examples": None if max_examples is None else int(max_examples),
        "num_examples": len(examples),
        "skipped_examples": skipped,
        "mean_target_tokens": float(
            sum(example["target_len"] for example in examples) / len(examples)
        ),
    }
    return examples, metadata


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
