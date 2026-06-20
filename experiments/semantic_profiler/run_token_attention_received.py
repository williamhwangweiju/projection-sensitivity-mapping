"""
Phase 0 experiment:
Build a token attention-received importance map for future JIT use.

For every WikiText-2 sample, this script:
1. Runs pretrained GPT-2 with output_attentions=True.
2. Measures how much attention each token receives from later tokens.
3. Normalizes the token scores to [0, 1] within each prompt.
4. Saves only the information needed by the future JIT policy.

Outputs:
    results/phase0_token_attention_received.csv
    results/phase0_token_attention_received.json

The JIT-facing signal is:
    attention_received_score
"""

import csv
import json
from pathlib import Path
from typing import Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gpt2_with_attentions(model_name: str):
    """Load GPT-2 using eager attention when supported."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
        )

    model.eval()
    return model


def compute_attention_received_score(
    attentions: Sequence[torch.Tensor],
) -> list[float]:
    """
    Compute one normalized attention-received score per token.

    For token j:

        avg_received(j)
          = sum attention[layer, head, q, j] for q > j
            -------------------------------------------------
            num_layers * num_heads * number_of_future_queries

    Self-attention is excluded. The average is min-max normalized within the
    prompt to produce attention_received_score in [0, 1].
    """
    if attentions is None or len(attentions) == 0:
        raise ValueError(
            "No attention tensors were returned. Enable output_attentions=True "
            "and eager attention."
        )

    first_attention = attentions[0]

    if first_attention.ndim != 4:
        raise ValueError(
            "Expected attention shape [batch, heads, query, key], "
            f"but received {tuple(first_attention.shape)}."
        )

    batch_size, num_heads, num_queries, num_keys = first_attention.shape

    if batch_size != 1:
        raise ValueError(
            f"This script expects batch_size=1, but received {batch_size}."
        )

    if num_queries != num_keys:
        raise ValueError(
            "Expected square causal self-attention matrices, but received "
            f"{num_queries} queries and {num_keys} keys."
        )

    seq_len = num_queries
    num_layers = len(attentions)

    # Keep only entries where a later query token q attends to an earlier key j.
    later_query_mask = torch.tril(
        torch.ones(
            (seq_len, seq_len),
            device=first_attention.device,
            dtype=torch.bool,
        ),
        diagonal=-1,
    )

    # For sequence length 5: [4, 3, 2, 1, 0].
    future_query_count = torch.arange(
        seq_len - 1,
        -1,
        -1,
        device=first_attention.device,
        dtype=torch.float64,
    )

    total_received = torch.zeros(
        seq_len,
        device=first_attention.device,
        dtype=torch.float64,
    )

    for layer_idx, attention in enumerate(attentions):
        if attention.ndim != 4:
            raise ValueError(
                f"Layer {layer_idx} has invalid shape {tuple(attention.shape)}."
            )

        if attention.shape[0] != 1:
            raise ValueError(
                f"Layer {layer_idx} has batch size {attention.shape[0]}; expected 1."
            )

        if attention.shape[-2:] != (seq_len, seq_len):
            raise ValueError(
                f"Layer {layer_idx} has inconsistent sequence dimensions "
                f"{tuple(attention.shape[-2:])}."
            )

        if attention.shape[1] != num_heads:
            raise ValueError(
                f"Layer {layer_idx} has {attention.shape[1]} heads, "
                f"but layer 0 has {num_heads}."
            )

        # [1, heads, query, key] -> [heads, query, key]
        layer_attention = attention[0]
        external_attention = (
            layer_attention * later_query_mask.unsqueeze(0)
        )

        # Sum over heads and query positions; keep one value per key token.
        total_received += external_attention.sum(
            dim=(0, 1),
            dtype=torch.float64,
        )

    denominator = future_query_count * num_heads * num_layers

    avg_received = torch.where(
        denominator > 0,
        total_received / denominator,
        torch.zeros_like(total_received),
    )

    # Min-max normalize valid tokens to [0, 1] for the JIT importance map.
    valid_mask = denominator > 0
    scores = torch.zeros_like(avg_received)

    if valid_mask.any():
        valid_values = avg_received[valid_mask]
        min_value = valid_values.min()
        max_value = valid_values.max()

        if max_value > min_value:
            scores[valid_mask] = (
                valid_values - min_value
            ) / (max_value - min_value)
        else:
            scores[valid_mask] = 1.0

    return scores.cpu().tolist()


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    sample_id: int,
) -> dict:
    """Run one prompt and return a token importance map."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    scores = compute_attention_received_score(outputs.attentions)
    token_ids = inputs["input_ids"][0].tolist()
    token_texts = [
        tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda token_idx: scores[token_idx],
        reverse=True,
    )

    ranks = [0 for _ in scores]
    for rank, token_idx in enumerate(ranked_indices, start=1):
        ranks[token_idx] = rank

    tokens = [
        {
            "token_index": token_idx,
            "token_id": token_id,
            "token_text": token_texts[token_idx],
            "attention_received_score": scores[token_idx],
            "attention_received_rank": ranks[token_idx],
        }
        for token_idx, token_id in enumerate(token_ids)
    ]

    return {
        "sample_id": sample_id,
        "num_tokens": len(token_ids),
        "tokens": tokens,
    }


def save_csv(results: list[dict], output_path: Path) -> None:
    """Save only fields needed for plotting and future JIT integration."""
    fieldnames = [
        "sample_id",
        "num_tokens",
        "token_index",
        "token_id",
        "token_text",
        "attention_received_score",
        "attention_received_rank",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in results:
            for token in sample["tokens"]:
                writer.writerow(
                    {
                        "sample_id": sample["sample_id"],
                        "num_tokens": sample["num_tokens"],
                        "token_index": token["token_index"],
                        "token_id": token["token_id"],
                        "token_text": token["token_text"],
                        "attention_received_score": token[
                            "attention_received_score"
                        ],
                        "attention_received_rank": token[
                            "attention_received_rank"
                        ],
                    }
                )


def save_json(
    results: list[dict],
    output_path: Path,
    metadata: dict,
) -> None:
    """Save a JIT-readable token importance map."""
    output = {
        "metadata": metadata,
        "importance_maps": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


def main() -> None:
    model_name = "gpt2"
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-raw-v1"
    split = "test"

    num_samples = 1000
    max_length = 1024
    minimum_words = 8

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_gpt2_with_attentions(model_name)

    print(
        f"Loading dataset: {dataset_name}, "
        f"config={dataset_config}, split={split}"
    )

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
    )

    results: list[dict] = []

    for row in dataset:
        prompt = row["text"].strip()

        if not prompt or len(prompt.split()) < minimum_words:
            continue

        sample_id = len(results)

        print(f"\nRunning sample {sample_id + 1}/{num_samples}")
        print(f"Prompt preview: {prompt[:100]!r}")

        result = run_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            sample_id=sample_id,
        )

        results.append(result)

        top_tokens = sorted(
            result["tokens"],
            key=lambda token: token["attention_received_score"],
            reverse=True,
        )[:5]

        print(f"Number of tokens: {result['num_tokens']}")
        print("Top attention-received tokens:")

        for token in top_tokens:
            print(
                f"  rank={token['attention_received_rank']:>2} "
                f"index={token['token_index']:>4} "
                f"token={token['token_text']!r} "
                f"score={token['attention_received_score']:.6f}"
            )

        if len(results) >= num_samples:
            break

    if not results:
        raise RuntimeError("No valid WikiText-2 samples were processed.")

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "phase0_token_attention_received.csv"
    json_path = results_dir / "phase0_token_attention_received.json"

    metadata = {
        "experiment": "token_attention_received",
        "model_name": model_name,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "num_samples": len(results),
        "max_length": max_length,
        "primary_signal": "attention_received_score",
        "score_range": "[0, 1] within each prompt",
    }

    save_csv(results, csv_path)
    save_json(results, json_path, metadata)

    print("\nExperiment complete.")
    print(f"Processed samples: {len(results)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
