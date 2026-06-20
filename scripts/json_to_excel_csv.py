import json
import csv
import argparse
from pathlib import Path


def safe_get(d, keys, default=None):
    """
    Safely get nested dictionary values.
    Example: safe_get(data, ["results", "energy_nj"])
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def extract_row(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)

    energy_nj = safe_get(data, ["results", "energy_nj"], 0)
    passive_nj = safe_get(data, ["results", "energy_breakdown", "passive"], 0)
    tier_linear_nj = safe_get(data, ["results", "energy_breakdown", "tier_linear"], 0)
    communication_nj = safe_get(data, ["results", "energy_breakdown", "communication"], 0)

    return {
        "file": json_path.name,

        # Experiment info
        "experiment": data.get("experiment"),
        "preset": data.get("preset"),

        # Model config
        "num_layers": safe_get(data, ["model", "num_layers"]),
        "d_model": safe_get(data, ["model", "d_model"]),
        "nhead": safe_get(data, ["model", "nhead"]),
        "d_ff": safe_get(data, ["model", "d_ff"]),
        "vocab_size": safe_get(data, ["model", "vocab_size"]),

        # Inference config
        "batch_size": safe_get(data, ["inference", "batch_size"]),
        "start_len": safe_get(data, ["inference", "start_len"]),
        "target_len": safe_get(data, ["inference", "target_len"]),
        "kv_caching": safe_get(data, ["inference", "kv_caching"]),

        # Accelerator config
        "tiles": safe_get(data, ["accelerator", "tiles"]),
        "tiers": safe_get(data, ["accelerator", "tiers"]),
        "tier_rows": safe_get(data, ["accelerator", "tier_shape"], [None, None])[0],
        "tier_cols": safe_get(data, ["accelerator", "tier_shape"], [None, None])[1],

        # Mapping config
        "mapping_strategy": safe_get(data, ["mapping", "strategy"]),
        "split_ffn": safe_get(data, ["mapping", "split_ffn"]),
        "stack_embedding": safe_get(data, ["mapping", "stack_embedding"]),

        # Main results
        "execution_time_ns": safe_get(data, ["results", "execution_time_ns"]),
        "execution_time_us": safe_get(data, ["results", "execution_time_ns"], 0) / 1000,
        "energy_nj": energy_nj,
        "energy_uj": energy_nj / 1000,
        "peak_memory_bytes": safe_get(data, ["results", "peak_memory_bytes"]),
        "peak_memory_kb": safe_get(data, ["results", "peak_memory_bytes"], 0) / 1024,
        "flops": safe_get(data, ["results", "flops"]),

        # Energy breakdown
        "passive_energy_nj": passive_nj,
        "tier_linear_energy_nj": tier_linear_nj,
        "communication_energy_nj": communication_nj,
        "mha_dram_energy_nj": safe_get(data, ["results", "energy_breakdown", "mha", "dram"], 0),
        "mha_comp_energy_nj": safe_get(data, ["results", "energy_breakdown", "mha", "comp"], 0),
        "layer_norm_energy_nj": safe_get(data, ["results", "energy_breakdown", "layer_norm"], 0),
        "digital_relu_energy_nj": safe_get(data, ["results", "energy_breakdown", "digital_relu"], 0),
        "digital_add_energy_nj": safe_get(data, ["results", "energy_breakdown", "digital_add"], 0),

        # Energy percentages
        "passive_energy_percent": (passive_nj / energy_nj * 100) if energy_nj else 0,
        "tier_linear_energy_percent": (tier_linear_nj / energy_nj * 100) if energy_nj else 0,
        "communication_energy_percent": (communication_nj / energy_nj * 100) if energy_nj else 0,

        # Latency breakdown
        "tier_linear_latency_ns": safe_get(data, ["results", "latency_breakdown", "tier_linear"], 0),
        "communication_latency_ns": safe_get(data, ["results", "latency_breakdown", "communication"], 0),
        "mha_dram_latency_ns": safe_get(data, ["results", "latency_breakdown", "mha", "dram"], 0),
        "mha_comp_latency_ns": safe_get(data, ["results", "latency_breakdown", "mha", "comp"], 0),
        "layer_norm_latency_ns": safe_get(data, ["results", "latency_breakdown", "layer_norm"], 0),
        "digital_relu_latency_ns": safe_get(data, ["results", "latency_breakdown", "digital_relu"], 0),
        "digital_add_latency_ns": safe_get(data, ["results", "latency_breakdown", "digital_add"], 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert IBM 3D-CIM JSON results into a CSV file for Excel."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/step1_baseline_sweep",
        help="Folder containing JSON result files.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/cim_baseline_summary.csv",
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    rows = [extract_row(path) for path in json_files]

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} JSON files.")
    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()