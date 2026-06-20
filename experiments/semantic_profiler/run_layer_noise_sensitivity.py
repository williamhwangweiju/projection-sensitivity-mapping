"""
Phase 0 experiment:
Measure noise sensitivity by Transformer layer.

For each GPT-2 layer:
1. Run clean inference.
2. Inject Gaussian noise into that layer's output hidden states.
3. Run noisy inference.
4. Compare clean logits vs. noisy logits using KL divergence.
5. Track whether the predicted next token changes.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_clean_logits(model, inputs):
    """
    Run the model normally and return clean logits.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits shape: [batch_size, seq_len, vocab_size]
    return outputs.logits

def make_noise_hook(noise_std):
    """
    Create a forward hook that adds Gaussian noise to a layer's output hidden states.

    In GPT-2, each transformer block usually returns either:
    - a tuple, where output[0] is hidden_states
    - or directly a tensor, depending on version/configuration

    hidden_states shape:
        [batch_size, seq_len, hidden_dim]
    """

    def hook(module, input, output):
        # Case 1: output is a tuple, common for GPT-2 blocks
        if isinstance(output, tuple):
            hidden_states = output[0]

            activation_std = hidden_states.std()
            noise = torch.randn_like(hidden_states) * activation_std * noise_std
            noisy_hidden_states = hidden_states + noise

            # Convert output[1:] explicitly to tuple to avoid tuple/list mismatch
            return (noisy_hidden_states,) + tuple(output[1:])

        # Case 2: output is directly a tensor
        elif torch.is_tensor(output):
            hidden_states = output

            activation_std = hidden_states.std()
            noise = torch.randn_like(hidden_states) * activation_std * noise_std
            noisy_hidden_states = hidden_states + noise

            return noisy_hidden_states

        # Case 3: unexpected output type
        else:
            raise TypeError(f"Unexpected layer output type: {type(output)}")

    return hook

def run_noisy_logits(model, inputs, layer_idx, noise_std):
    """
    Inject noise into one specific transformer layer and return noisy logits.
    """

    # GPT-2 transformer layers are stored here:
    # model.transformer.h[0], model.transformer.h[1], ..., model.transformer.h[11]
    target_layer = model.transformer.h[layer_idx]

    # Register hook on the selected layer
    hook_handle = target_layer.register_forward_hook(make_noise_hook(noise_std))

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        noisy_logits = outputs.logits

    finally:
        # Remove the hook after this run.
        # Otherwise the noise would stay active for future runs.
        hook_handle.remove()

    return noisy_logits

def compute_kl_divergence(clean_logits, noisy_logits):
    """
    Compare clean and noisy next-token distributions.

    We use only the final token position:
        logits[0, -1, :]

    clean_probs:
        probability distribution from clean model

    noisy_log_probs:
        log probability distribution from noisy model

    KL(clean || noisy):
        measures how much the noisy prediction distribution changed.
    """

    clean_last_logits = clean_logits[0, -1, :]
    noisy_last_logits = noisy_logits[0, -1, :]

    clean_probs = F.softmax(clean_last_logits, dim=-1)
    noisy_log_probs = F.log_softmax(noisy_last_logits, dim=-1)

    # First argument expects log probabilities
    kl = F.kl_div(
        noisy_log_probs,
        clean_probs,
        reduction="sum",
    )

    return kl.item()

def did_next_token_change(clean_logits, noisy_logits):
    """
    Check whether the top-1 predicted next token changed.
    """

    clean_next_token = torch.argmax(clean_logits[0, -1, :]).item()
    noisy_next_token = torch.argmax(noisy_logits[0, -1, :]).item()

    return clean_next_token != noisy_next_token

def main():
    model_name = "gpt2"
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-raw-v1"
    split = "test"

    num_samples = 1000
    max_length = 1024

    # Noise strength.
    noise_std = 0.02

    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "phase0_layer_noise_sensitivity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    num_layers = len(model.transformer.h)
    print(f"Number of transformer layers: {num_layers}")

    print(f"Loading dataset: {dataset_name}, config={dataset_config}, split={split}")

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
    )

    # Store results per layer
    layer_kl_sums = [0.0 for _ in range(num_layers)]
    layer_token_change_counts = [0 for _ in range(num_layers)]

    used_samples = 0

    for row in dataset:
        prompt = row["text"].strip()

        # Skip empty or very short samples
        if not prompt:
            continue

        if len(prompt.split()) < 8:
            continue

        print(f"\nRunning sample {used_samples + 1}/{num_samples}")
        print(f"Prompt preview: {repr(prompt[:100])}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        # Run clean inference once
        clean_logits = get_clean_logits(model, inputs)

        # Inject noise into each layer separately
        for layer_idx in range(num_layers):
            noisy_logits = run_noisy_logits(
                model=model,
                inputs=inputs,
                layer_idx=layer_idx,
                noise_std=noise_std,
            )

            kl = compute_kl_divergence(clean_logits, noisy_logits)
            changed = did_next_token_change(clean_logits, noisy_logits)

            layer_kl_sums[layer_idx] += kl

            if changed:
                layer_token_change_counts[layer_idx] += 1

        used_samples += 1

        if used_samples >= num_samples:
            break

    # Average metrics across samples
    avg_kl_per_layer = [
        kl_sum / used_samples
        for kl_sum in layer_kl_sums
    ]

    token_change_rate_per_layer = [
        count / used_samples
        for count in layer_token_change_counts
    ]

    results = {
        "model_name": model_name,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "num_samples": used_samples,
        "max_length": max_length,
        "noise_std": noise_std,
        "avg_kl_per_layer": avg_kl_per_layer,
        "token_change_rate_per_layer": token_change_rate_per_layer,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nAverage KL divergence per layer:")
    for layer_idx, kl in enumerate(avg_kl_per_layer):
        print(f"Layer {layer_idx:02d}: KL = {kl:.6f}")

    print("\nNext-token change rate per layer:")
    for layer_idx, rate in enumerate(token_change_rate_per_layer):
        print(f"Layer {layer_idx:02d}: change rate = {rate:.4f}")

    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    main()