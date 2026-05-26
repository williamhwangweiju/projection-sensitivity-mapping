"""
Phase 0 analysis:
Load saved attention entropy results, compute average normalized entropy
per layer across all dataset samples, and plot the result.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Find repo root based on this script location
    repo_root = Path(__file__).resolve().parents[2]

    # Input JSON from your dataset experiment
    input_path = repo_root / "results" / "phase0_wikitext_entropy.json"

    # Output folder for figures
    figures_dir = repo_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Output figure path
    output_plot_path = figures_dir / "phase0_avg_normalized_entropy_by_layer.png"

    # Load JSON results
    with open(input_path, "r") as f:
        results = json.load(f)

    # If the JSON contains one object instead of a list, convert it to a list
    if isinstance(results, dict):
        results = [results]

    # Extract normalized entropy values
    # Shape after this: [num_samples, num_layers]
    normalized_entropy_matrix = np.array([
        sample["layer_normalized_entropies"]
        for sample in results
    ])

    # Compute average entropy per layer across samples
    # axis=0 means average over samples
    avg_entropy_per_layer = normalized_entropy_matrix.mean(axis=0)

    # Compute standard deviation per layer across samples
    std_entropy_per_layer = normalized_entropy_matrix.std(axis=0)

    num_layers = avg_entropy_per_layer.shape[0]
    layer_ids = np.arange(num_layers)

    print("\nAverage normalized entropy per layer:")
    for layer_idx, avg_entropy in enumerate(avg_entropy_per_layer):
        print(f"Layer {layer_idx:02d}: avg normalized entropy = {avg_entropy:.4f}")

    # Plot average normalized entropy per layer
    plt.figure(figsize=(8, 5))
    plt.plot(layer_ids, avg_entropy_per_layer, marker="o")

    # Add error bars showing variation across samples
    plt.fill_between(
        layer_ids,
        avg_entropy_per_layer - std_entropy_per_layer,
        avg_entropy_per_layer + std_entropy_per_layer,
        alpha=0.2,
    )

    plt.xlabel("Transformer Layer")
    plt.ylabel("Average Normalized Attention Entropy")
    plt.title("Average Normalized Attention Entropy per Layer")
    plt.xticks(layer_ids)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_plot_path, dpi=300)
    print(f"\nSaved plot to: {output_plot_path}")


if __name__ == "__main__":
    main()