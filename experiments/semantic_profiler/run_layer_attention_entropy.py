"""
Phase 0 experiment:
Run attention entropy analysis over multiple WikiText-2 prompts.

This script:
1. Loads pretrained Hugging Face GPT-2.
2. Runs multiple WikiText-2 samples through the model.
3. Extracts the attention tensors from every transformer layer.
4. Computes raw and normalized attention entropy per layer.
5. Saves:
   - detailed per-prompt results as JSON
   - an aggregated per-layer summary as JSON
   - an Excel-friendly per-layer summary as CSV
"""

import csv
import json
from pathlib import Path
from typing import Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_gpt2_with_attentions(model_name: str):
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

def compute_attention_entropy(
    attentions: Sequence[torch.Tensor],
) -> tuple[list[float], list[float]]:
    """
    Compute average raw and normalized attention entropy for every layer.

    Each attention tensor normally has shape:
        [batch_size, num_heads, num_query_tokens, num_key_tokens]

    For every query token and attention head, raw entropy is:

        H(p) = -sum_i p_i log(p_i)

    Because GPT-2 uses causal attention, query token q can attend to only
    q + 1 key tokens. The maximum possible entropy at that position is:

        H_max(q) = log(q + 1)

    The normalized entropy is therefore:

        H_normalized(q) = H(q) / log(q + 1)

    Token position 0 is excluded from the layer average because it can attend
    only to itself, so its maximum entropy is log(1) = 0.
    """
    layer_entropies: list[float] = []
    layer_normalized_entropies: list[float] = []

    if attentions is None:
        raise ValueError(
            "The model did not return attention tensors. "
            "Make sure output_attentions=True is enabled."
        )

    for layer_idx, attn in enumerate(attentions):
        if attn.ndim != 4:
            raise ValueError(
                f"Layer {layer_idx} attention tensor should have 4 dimensions, "
                f"but received shape {tuple(attn.shape)}."
            )

        # attn shape:
        # [batch_size, num_heads, num_query_tokens, num_key_tokens]
        num_query_tokens = attn.shape[-2]

        # Avoid log(0). The original attention tensor is still used in the
        # multiplication, so zero-probability entries contribute zero entropy.
        attn_safe = attn.clamp(min=1e-9)

        # Sum over key-token probabilities.
        # Result shape:
        # [batch_size, num_heads, num_query_tokens]
        entropy = -(attn * torch.log(attn_safe)).sum(dim=-1)

        # For causal attention:
        # query token 0 has 1 available key
        # query token 1 has 2 available keys
        # ...
        available_tokens = torch.arange(
            1,
            num_query_tokens + 1,
            device=attn.device,
            dtype=attn.dtype,
        )

        # Maximum entropy for a uniform distribution over the available keys.
        max_entropy = torch.log(available_tokens)

        # Shape [num_query_tokens] -> [1, 1, num_query_tokens]
        # so it broadcasts across batches and attention heads.
        max_entropy = max_entropy.view(1, 1, num_query_tokens)

        # Token 0 has max entropy 0, so exclude it before dividing.
        if num_query_tokens > 1:
            valid_entropy = entropy[:, :, 1:]
            valid_max_entropy = max_entropy[:, :, 1:]
            normalized_entropy = valid_entropy / valid_max_entropy

            # Average across batch, heads, and valid query-token positions.
            avg_entropy = valid_entropy.mean().item()
            avg_normalized_entropy = normalized_entropy.mean().item()
        else:
            # A one-token sequence has no meaningful normalized entropy.
            avg_entropy = 0.0
            avg_normalized_entropy = 0.0

        layer_entropies.append(avg_entropy)
        layer_normalized_entropies.append(avg_normalized_entropy)

    return layer_entropies, layer_normalized_entropies

def run_single_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int,
) -> dict:
    """
    Run one prompt through GPT-2 and return per-layer entropy measurements.
    """
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
        )

    logits = outputs.logits
    attentions = outputs.attentions

    layer_entropies, layer_normalized_entropies = (
        compute_attention_entropy(attentions)
    )

    # Predict the next token from the final prompt position.
    next_token_id = torch.argmax(logits[0, -1, :]).item()
    next_token = tokenizer.decode([next_token_id])

    return {
        "prompt": prompt,
        "token_ids": inputs["input_ids"][0].tolist(),
        "num_tokens": int(inputs["input_ids"].shape[1]),
        "layer_entropies": layer_entropies,
        "layer_normalized_entropies": layer_normalized_entropies,
        "avg_entropy_all_layers": (
            sum(layer_entropies) / len(layer_entropies)
        ),
        "avg_normalized_entropy_all_layers": (
            sum(layer_normalized_entropies)
            / len(layer_normalized_entropies)
        ),
        "next_token_id": next_token_id,
        "next_token": next_token,
    }

def build_layer_summary(results: list[dict]) -> tuple[list[float], list[float]]:
    """
    Average raw and normalized entropy across all prompts for each layer.

    Every prompt receives equal weight.
    """
    if not results:
        raise ValueError("Cannot build a summary because no results were collected.")

    num_layers = len(results[0]["layer_normalized_entropies"])

    layer_entropy_sums = [0.0 for _ in range(num_layers)]
    layer_normalized_entropy_sums = [0.0 for _ in range(num_layers)]

    for sample_idx, sample in enumerate(results):
        raw_values = sample["layer_entropies"]
        normalized_values = sample["layer_normalized_entropies"]

        if len(raw_values) != num_layers or len(normalized_values) != num_layers:
            raise ValueError(
                f"Sample {sample_idx} has an inconsistent number of layers."
            )

        for layer_idx in range(num_layers):
            layer_entropy_sums[layer_idx] += raw_values[layer_idx]
            layer_normalized_entropy_sums[layer_idx] += (
                normalized_values[layer_idx]
            )

    num_samples = len(results)

    avg_entropy_per_layer = [
        value / num_samples for value in layer_entropy_sums
    ]

    avg_normalized_entropy_per_layer = [
        value / num_samples
        for value in layer_normalized_entropy_sums
    ]

    return avg_entropy_per_layer, avg_normalized_entropy_per_layer

def save_outputs(
    results: list[dict],
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_length: int,
    results_dir: Path,
) -> None:
    """
    Save detailed results and aggregated per-layer summaries.
    """
    avg_entropy_per_layer, avg_normalized_entropy_per_layer = (
        build_layer_summary(results)
    )

    detailed_json_path = results_dir / "phase0_wikitext_entropy.json"
    summary_json_path = results_dir / "phase0_wikitext_entropy_summary.json"
    summary_csv_path = results_dir / "phase0_wikitext_entropy_summary.csv"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Detailed per-prompt output.
    with detailed_json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "num_samples": len(results),
        "max_length": max_length,
        "avg_entropy_per_layer": avg_entropy_per_layer,
        "avg_normalized_entropy_per_layer": (
            avg_normalized_entropy_per_layer
        ),
    }

    # Compact JSON summary.
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # Excel-friendly CSV summary.
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "layer_number",
                "avg_entropy_per_layer",
                "avg_normalized_entropy_per_layer",
            ],
        )
        writer.writeheader()

        for layer_idx in range(len(avg_entropy_per_layer)):
            writer.writerow(
                {
                    "layer_number": layer_idx,
                    "avg_entropy_per_layer": (
                        avg_entropy_per_layer[layer_idx]
                    ),
                    "avg_normalized_entropy_per_layer": (
                        avg_normalized_entropy_per_layer[layer_idx]
                    ),
                }
            )

    print("\nAverage attention entropy per layer:")
    print(
        f"{'Layer':<8}"
        f"{'Raw entropy':<20}"
        f"{'Normalized entropy':<22}"
    )

    for layer_idx in range(len(avg_entropy_per_layer)):
        print(
            f"{layer_idx:<8}"
            f"{avg_entropy_per_layer[layer_idx]:<20.10f}"
            f"{avg_normalized_entropy_per_layer[layer_idx]:<22.10f}"
        )

    print(f"\nProcessed samples: {len(results)}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Detailed JSON: {detailed_json_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Excel CSV: {summary_csv_path}")

def main() -> None:
    model_name = "gpt2"
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-raw-v1"
    split = "test"

    num_samples = 1000
    max_length = 1024

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
    sample_count = 0

    for row in dataset:
        prompt = row["text"].strip()

        # Skip empty lines.
        if not prompt:
            continue

        # Skip very short lines because attention entropy is less meaningful.
        if len(prompt.split()) < 8:
            continue

        print(f"\nRunning sample {sample_count + 1}/{num_samples}")
        print(f"Prompt preview: {prompt[:100]!r}")

        result = run_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
        )

        results.append(result)

        print(f"Number of tokens: {result['num_tokens']}")
        print(
            "Average entropy across layers: "
            f"{result['avg_entropy_all_layers']:.4f}"
        )
        print(
            "Average normalized entropy across layers: "
            f"{result['avg_normalized_entropy_all_layers']:.4f}"
        )
        print(f"Next token: {result['next_token']!r}")

        sample_count += 1

        if sample_count >= num_samples:
            break

    if not results:
        raise RuntimeError(
            "No valid WikiText-2 samples were processed. "
            "Check the dataset and filtering conditions."
        )

    # This assumes the script is located two directories below the repository
    # root, matching your existing project layout.
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"

    save_outputs(
        results=results,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_length=max_length,
        results_dir=results_dir,
    )

if __name__ == "__main__":
    main()
