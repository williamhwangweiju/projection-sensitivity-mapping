"""
Plot layer-wise noise sensitivity.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    repo_root = Path(__file__).resolve().parents[2]

    input_path = repo_root / "results" / "phase0_layer_noise_sensitivity.json"
    figures_dir = repo_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_plot_path = figures_dir / "phase0_layer_noise_sensitivity.png"

    with open(input_path, "r") as f:
        results = json.load(f)

    avg_kl_per_layer = np.array(results["avg_kl_per_layer"])
    token_change_rate = np.array(results["token_change_rate_per_layer"])

    num_layers = len(avg_kl_per_layer)
    layer_ids = np.arange(num_layers)

    print("\nLayer noise sensitivity:")
    for layer_idx, kl in enumerate(avg_kl_per_layer):
        print(f"Layer {layer_idx:02d}: KL = {kl:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(layer_ids, avg_kl_per_layer, marker="o")

    plt.xlabel("Transformer Layer")
    plt.ylabel("Average KL Divergence")
    plt.title("Output Degradation Under Layer Noise")
    plt.xticks(layer_ids)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_plot_path, dpi=300)

    print(f"\nSaved plot to: {output_plot_path}")


if __name__ == "__main__":
    main()