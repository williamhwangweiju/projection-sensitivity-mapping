#!/usr/bin/env python3
"""Analyze Phase 1 AIHWKit projection-sensitivity results.

The analyzer consumes the JSON produced by ``run_aihwkit_profiling.py`` and
creates summary statistics plus three plots used to inspect the sensitivity
profile before Phase 2/3 mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULT_FILE_PATTERN = "programming_noise_sensitivity_*.json"

PROJECTION_ORDER = [
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "lm_head",
]

PROJECTION_COLORS = {
    "attn.c_attn": "red",
    "attn.c_proj": "orange",
    "mlp.c_fc": "blue",
    "mlp.c_proj": "lightblue",
    "lm_head": "darkred",
}

SensitivityTable = Dict[str, Dict[str, float]]


def find_repo_root(script_path: Path) -> Path:
    """Find the repository root from this script location."""
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "src" / "profilers" / "aihwkit_profiler.py").is_file():
            return candidate
    return resolved.parent


REPO_ROOT = find_repo_root(Path(__file__))


def block_sort_key(block_id: str) -> Tuple[int, int, str]:
    """Sort GPT-2 transformer blocks numerically and place the LM head last."""
    if block_id == "head":
        return (1, 0, block_id)
    if block_id.startswith("block_"):
        try:
            return (0, int(block_id.split("_", maxsplit=1)[1]), block_id)
        except (IndexError, ValueError):
            pass
    return (2, 0, block_id)


def find_latest_results_file(
    results_dir: Optional[Path] = None,
    pattern: str = RESULT_FILE_PATTERN,
) -> Path:
    """Return the newest Phase 1 JSON result matching ``pattern``."""
    root = results_dir.expanduser().resolve() if results_dir else REPO_ROOT / "data" / "results"
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    candidates = [path.resolve() for path in root.rglob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No result file matching {pattern!r} was found under {root}. "
            "Pass --results-file to select a file explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_results_payload(results_file: Path) -> Dict[str, Any]:
    """Load and validate the runner output JSON."""
    if not results_file.is_file():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with results_file.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise TypeError("Result JSON root must be an object.")

    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("Result JSON must contain a 'results' object.")

    projections = results.get("projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError("results.projections must be a non-empty list.")

    digital_perplexity = float(results["digital_perplexity"])
    if not np.isfinite(digital_perplexity):
        raise ValueError("results.digital_perplexity must be finite.")

    return payload


def extract_sensitivity_tables(
    payload: Mapping[str, Any],
) -> Tuple[SensitivityTable, SensitivityTable, float]:
    """Convert projection records into block/projection lookup tables."""
    results = payload["results"]
    projections = results["projections"]

    sensitivities: SensitivityTable = {}
    sensitivity_stds: SensitivityTable = {}

    required_fields = {"block_id", "proj_name", "sensitivity_mean", "sensitivity_std"}
    for index, record in enumerate(projections):
        if not isinstance(record, Mapping):
            raise TypeError(f"Projection record {index} is not an object.")

        missing = required_fields.difference(record)
        if missing:
            raise ValueError(f"Projection record {index} is missing {sorted(missing)}.")

        block_id = str(record["block_id"])
        proj_name = str(record["proj_name"])
        mean_value = float(record["sensitivity_mean"])
        std_value = float(record["sensitivity_std"])

        if proj_name not in PROJECTION_ORDER:
            raise ValueError(f"Unknown projection name: {proj_name!r}")
        if not np.isfinite(mean_value) or not np.isfinite(std_value):
            raise ValueError(f"Non-finite sensitivity for {block_id}/{proj_name}.")

        block_values = sensitivities.setdefault(block_id, {})
        if proj_name in block_values:
            raise ValueError(f"Duplicate projection result: {block_id}/{proj_name}")

        block_values[proj_name] = mean_value
        sensitivity_stds.setdefault(block_id, {})[proj_name] = std_value

    return sensitivities, sensitivity_stds, float(results["digital_perplexity"])


def configured_projection_order(sensitivities: Mapping[str, Mapping[str, float]]) -> List[str]:
    """Return profiler projection names that are present in this run."""
    present = {
        proj_name
        for block_values in sensitivities.values()
        for proj_name in block_values
    }
    return [proj_name for proj_name in PROJECTION_ORDER if proj_name in present]


def finite_values(sensitivities: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """Return all finite mean sensitivity values."""
    values = [
        float(value)
        for block_values in sensitivities.values()
        for value in block_values.values()
        if np.isfinite(value)
    ]
    if not values:
        raise ValueError("No finite sensitivity values were found.")
    return np.asarray(values, dtype=np.float64)


def projection_value_or_nan(
    block_values: Mapping[str, float],
    proj_name: str,
) -> float:
    """Return a projection sensitivity or NaN when the projection is absent."""
    return float(block_values.get(proj_name, np.nan))


def configure_delta_ppl_axis(ax: Any, values: Sequence[float]) -> None:
    """Use log scale for positive deltas and symlog when signs are mixed."""
    values_array = np.asarray(values, dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return

    if np.all(values_array > 0.0):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        return

    absolute_values = np.abs(values_array)
    nonzero_values = absolute_values[absolute_values > 0.0]
    linthresh = max(float(nonzero_values.min()), float(absolute_values.max()) * 1e-4, 1e-12)
    ax.set_yscale("symlog", linthresh=linthresh, base=10)


def summarize_sensitivities(
    sensitivities: Mapping[str, Mapping[str, float]],
    sensitivity_stds: Mapping[str, Mapping[str, float]],
    digital_perplexity: float,
) -> None:
    """Print compact sensitivity statistics and rankings."""
    all_values = finite_values(sensitivities)
    projection_names = configured_projection_order(sensitivities)

    print("\n" + "=" * 72)
    print("PHASE 1 AIHWKIT PROJECTION-SENSITIVITY ANALYSIS")
    print("=" * 72)
    print(f"\nDigital FP32 perplexity: {digital_perplexity:.6f}")
    print(f"Profiled projections: {all_values.size}")

    print("\nOverall delta-perplexity statistics:")
    print(f"  Mean: {all_values.mean():.6f}")
    print(f"  Std:  {all_values.std(ddof=0):.6f}")
    print(f"  Min:  {all_values.min():.6f}")
    print(f"  Max:  {all_values.max():.6f}")

    print("\nBlock-average sensitivity:")
    for block_id in sorted(sensitivities, key=block_sort_key):
        values = np.asarray(list(sensitivities[block_id].values()), dtype=np.float64)
        print(f"  {block_id:8s}: {values.mean():.6f}")

    print("\nProjection-type sensitivity:")
    for proj_name in projection_names:
        values = np.asarray(
            [
                block_values[proj_name]
                for block_values in sensitivities.values()
                if proj_name in block_values
            ],
            dtype=np.float64,
        )
        print(
            f"  {proj_name:12s}: mean={values.mean():.6f}, "
            f"std={values.std(ddof=0):.6f}, "
            f"range=[{values.min():.6f}, {values.max():.6f}]"
        )

    ranked: List[Tuple[float, float, str]] = []
    for block_id, block_values in sensitivities.items():
        for proj_name, mean_value in block_values.items():
            ranked.append(
                (
                    float(mean_value),
                    float(sensitivity_stds[block_id][proj_name]),
                    f"{block_id}/{proj_name}",
                )
            )
    ranked.sort(key=lambda item: item[0], reverse=True)
    top_count = min(5, len(ranked))

    print(f"\nTop {top_count} most sensitive projections:")
    for rank, (mean_value, std_value, label) in enumerate(ranked[:top_count], start=1):
        print(f"  {rank}. {label:28s}: {mean_value:.6f} +/- {std_value:.6f}")

    print(f"\nBottom {top_count} least sensitive projections:")
    for rank, (mean_value, std_value, label) in enumerate(ranked[-top_count:][::-1], start=1):
        print(f"  {rank}. {label:28s}: {mean_value:.6f} +/- {std_value:.6f}")


def plot_sensitivities(
    sensitivities: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> List[Path]:
    """Create heatmap, projection-distribution, and per-block plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(sensitivities, key=block_sort_key)
    projection_names = configured_projection_order(sensitivities)
    all_values = finite_values(sensitivities)
    output_files: List[Path] = []

    heatmap_data = np.full((len(blocks), len(projection_names)), np.nan, dtype=np.float64)
    for row, block_id in enumerate(blocks):
        for column, proj_name in enumerate(projection_names):
            heatmap_data[row, column] = projection_value_or_nan(
                sensitivities[block_id],
                proj_name,
            )

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.45 * len(blocks) + 1.5)))
    image = ax.imshow(np.ma.masked_invalid(heatmap_data), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(projection_names)))
    ax.set_yticks(range(len(blocks)))
    ax.set_xticklabels(projection_names, rotation=20, ha="right")
    ax.set_yticklabels(blocks)
    ax.set_xlabel("Projection")
    ax.set_ylabel("GPT-2 block")
    ax.set_title("Phase 1 Projection Sensitivity")
    plt.colorbar(image, ax=ax, label="Mean Delta Perplexity")

    for row in range(len(blocks)):
        for column in range(len(projection_names)):
            value = heatmap_data[row, column]
            ax.text(
                column,
                row,
                "-" if np.isnan(value) else f"{value:.5g}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    heatmap_path = output_dir / "phase1_sensitivity_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    output_files.append(heatmap_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    boxplot_data = [
        [
            sensitivities[block_id][proj_name]
            for block_id in blocks
            if proj_name in sensitivities[block_id]
        ]
        for proj_name in projection_names
    ]
    box = ax.boxplot(boxplot_data, patch_artist=True)

    for patch, proj_name in zip(box["boxes"], projection_names):
        patch.set_facecolor(PROJECTION_COLORS.get(proj_name, "darkred"))

    ax.boxplot(boxplot_data)
    ax.set_xticks(range(1, len(projection_names) + 1))
    ax.set_xticklabels(projection_names, rotation=20, ha="right")
    configure_delta_ppl_axis(ax, all_values)
    ax.set_ylabel("Mean Delta Perplexity")
    ax.set_title("Sensitivity Distribution by Projection")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    distribution_path = output_dir / "phase1_sensitivity_distribution.png"
    plt.savefig(distribution_path, dpi=150)
    plt.close(fig)
    output_files.append(distribution_path)

    fig, ax = plt.subplots(figsize=(max(10.0, 0.8 * len(blocks)), 6))
    x_positions = np.arange(len(blocks))
    width = min(0.8 / max(len(projection_names), 1), 0.18)
    center = (len(projection_names) - 1) / 2.0

    for index, proj_name in enumerate(projection_names):
        values = [projection_value_or_nan(sensitivities[block_id], proj_name) for block_id in blocks]
        ax.bar(
            x_positions + (index - center) * width,
            values,
            width=width,
            label=proj_name,
            color=PROJECTION_COLORS.get(proj_name, "darkred"),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(blocks, rotation=30, ha="right")
    configure_delta_ppl_axis(ax, all_values)
    ax.set_ylabel("Mean Delta Perplexity")
    ax.set_title("Per-Block Sensitivity by Projection")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    block_path = output_dir / "phase1_sensitivity_by_block.png"
    plt.savefig(block_path, dpi=150)
    plt.close(fig)
    output_files.append(block_path)

    return output_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and plot Phase 1 AIHWKit sensitivity results."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--results-file", type=Path, help="Specific Phase 1 JSON result.")
    selection.add_argument("--results-dir", type=Path, help="Directory containing Phase 1 JSON results.")
    parser.add_argument(
        "--pattern",
        default=RESULT_FILE_PATTERN,
        help="Filename pattern used when selecting the newest result file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plots. Defaults to the selected result file directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Load one Phase 1 result file, summarize it, and generate plots."""
    args = parse_args()

    if args.results_file is not None:
        results_file = args.results_file.expanduser().resolve()
    else:
        results_file = find_latest_results_file(
            results_dir=args.results_dir,
            pattern=args.pattern,
        )

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else results_file.parent

    print("\nLoading Phase 1 results:")
    print(f"  Results: {results_file}")
    print(f"  Plots:   {output_dir}")

    payload = load_results_payload(results_file)
    sensitivities, sensitivity_stds, digital_perplexity = extract_sensitivity_tables(payload)
    summarize_sensitivities(sensitivities, sensitivity_stds, digital_perplexity)
    output_files = plot_sensitivities(sensitivities, output_dir)

    print("\nSaved plots:")
    for path in output_files:
        print(f"  {path}")
    print("\nPhase 1 analysis complete.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
