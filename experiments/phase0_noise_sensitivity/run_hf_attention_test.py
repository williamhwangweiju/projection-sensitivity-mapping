"""
Phase 0 experiment:
Run attention entropy analysis over multiple prompts.

This script:
1. Loads a small Hugging Face causal language model.
2. Runs multiple prompts(WikiText-2) through the model.
3. Extracts attention tensors.
4. Computes average attention entropy per layer.
5. Saves results to JSON for later plotting.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_attention_entropy(attentions):
    layer_entropies = []
    layer_normalized_entropies = []

    for layer_idx, attn in enumerate(attentions):
        # attn shape:
        # [batch_size, num_heads, num_query_tokens, num_key_tokens]

        # Avoid log(0)
        attn_safe = attn.clamp(min=1e-9)

        # Raw entropy: -sum(p * logp)
        entropy = -(attn * torch.log(attn_safe)).sum(dim=-1)

        # Number of query tokens in this layer
        seq_len = attn.shape[-2]

        # For causal GPT-style attention:
        # token 0 can attend to 1 token
        # token 1 can attend to 2 tokens
        # token 2 can attend to 3 tokens
        # ...
        available_tokens = torch.arange(
            1,
            seq_len + 1,
            device=attn.device,
            dtype=attn.dtype,
        )

        # Maximum possible entropy for each query token is log(number of available tokens)
        max_entropy = torch.log(available_tokens)

        # Avoid division by zero for the first token, where log(1) = 0
        max_entropy = max_entropy.clamp(min=1e-9)

        # Shape max_entropy from [seq_len] to [1, 1, seq_len]
        # so it broadcasts across batch and heads
        max_entropy = max_entropy.view(1, 1, seq_len)

        # Normalized entropy should be roughly between 0 and 1
        normalized_entropy = entropy / max_entropy

        # For the first token, set it to 0
        normalized_entropy[:, :, 0] = 0.0

        # Average over batch, heads, and query tokens
        avg_entropy = entropy.mean().item()
        avg_normalized_entropy = normalized_entropy.mean().item()

        layer_entropies.append(avg_entropy)
        layer_normalized_entropies.append(avg_normalized_entropy)

    return layer_entropies, layer_normalized_entropies


def run_single_prompt(model, tokenizer, prompt, max_length=64):
    # Tokenize the prompt
    # return_tensors="pt" returns PyTorch tensors
    # truncation=True cuts long text to max_length tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        # Unpack dictionary to function args
        # Run the model
        outputs = model(**inputs)

    # logits shape: [batch_size, seq_len, vocab_size]
    logits = outputs.logits

    # attentions is a tuple with one tensor per layer
    # each attention tensor shape: [batch_size, num_heads, num_query_tokens, num_key_tokens]
    attentions = outputs.attentions

    layer_entropies, layer_normalized_entropies = compute_attention_entropy(attentions)

    # Predict next token from the last position
    next_token_id = torch.argmax(logits[0, -1, :]).item()
    next_token = tokenizer.decode([next_token_id])

    result = {
        "prompt": prompt,
        "token_ids": inputs["input_ids"][0].tolist(),
        "num_tokens": inputs["input_ids"].shape[1],
        "layer_entropies": layer_entropies,
        "layer_normalized_entropies": layer_normalized_entropies,
        "avg_entropy_all_layers": sum(layer_entropies) / len(layer_entropies),
        "avg_normalized_entropy_all_layers": sum(layer_normalized_entropies) / len(layer_normalized_entropies),
        "next_token_id": next_token_id,
        "next_token": next_token,
    }

    return result


def main():
    model_name = "gpt2"
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-raw-v1"
    split = "test"

    # Keep this small at first so the experiment runs quickly
    num_samples = 1000
    max_length = 1024

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
    )

    model.eval()

    print(f"Loading dataset: {dataset_name}, config={dataset_config}, split={split}")

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
    )

    results = []
    sample_count = 0

    for row in dataset:
        prompt = row["text"].strip()

        # Skip empty lines
        if not prompt:
            continue

        # Skip very short lines because attention entropy is less meaningful
        if len(prompt.split()) < 8:
            continue

        print(f"\nRunning sample {sample_count + 1}/{num_samples}")
        print(f"Prompt preview: {repr(prompt[:100])}")

        result = run_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
        )

        results.append(result)

        print(f"Number of tokens: {result['num_tokens']}")
        print(f"Average entropy across layers: {result['avg_entropy_all_layers']:.4f}")
        print(f"Average normalized entropy across layers: {result['avg_normalized_entropy_all_layers']:.4f}")
        print(f"Next token: {repr(result['next_token'])}")

        sample_count += 1

        if sample_count >= num_samples:
            break

    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "phase0_wikitext_entropy.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()