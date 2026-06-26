#!/usr/bin/env python3
"""Analyze and visualize the Lammie 2026 AIHWKIT sensitivity results.

Reads the single JSON file produced by:

    run_lammie_2026_aihwkit.py

Expected top-level structure:

    {
        "metadata": {...},
        "resolved_config": {...},
        "results": {
            "digital_perplexity": ...,
            "projections": [...],
            "block_averages": {...},
            "sensitivity_ranking": [...]
        }
    }

The script preserves the original three Phase 1 plots:

1. Projection sensitivity heatmap
2. Sensitivity distribution by projection type
3. Per-block sensitivity grouped by projection type

Examples:

    python3 experiments/phase1_sensitivity/analyze_phase1.py

    python3 experiments/phase1_sensitivity/analyze_phase1.py \
        --results-file results/lammie_2026/lammie_2026_aihwkit_stage1_2_20260624_120000.json

    python3 experiments/phase1_sensitivity/analyze_phase1.py \
        --results-dir results/lammie_2026
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


RESULT_FILE_PATTERN = "lammie_2026_aihwkit_stage1_2_*.json"

# Canonical names emitted by AIHWKITLammieSensitivityProfiler.
PROJECTION_ORDER = [
    "c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "lm_head",
]

# Accept older aliases if an earlier result file used them.
PROJECTION_ALIASES = {
    "q_proj": "c_attn",
    "c_attn": "c_attn",
    "out_proj": "attn.c_proj",
    "attn.c_proj": "attn.c_proj",
    "fc1": "mlp.c_fc",
    "mlp.c_fc": "mlp.c_fc",
    "fc2": "mlp.c_proj",
    "mlp.c_proj": "mlp.c_proj",
    "lm_head": "lm_head",
}


def find_repo_root(script_path: Path) -> Path:
    """Find the repository root from the analyzer's location."""
    resolved = script_path.resolve()

    for candidate in (resolved.parent, *resolved.parents):
        if (
            candidate / "src" / "profilers" / "aihwkit_profiler.py"
        ).is_file():
            return candidate

    # Keep the analyzer usable even when copied outside the repository.
    return resolved.parent


REPO_ROOT = find_repo_root(Path(__file__))


def block_sort_key(block_id: str) -> Tuple[int, int, str]:
    """Sort block_0 ... block_11 numerically and the LM head last."""
    if block_id in {"head", "lm_head"}:
        return (1, 0, block_id)

    if block_id.startswith("block_"):
        try:
            return (
                0,
                int(block_id.split("_", maxsplit=1)[1]),
                block_id,
            )
        except (ValueError, IndexError):
            pass

    return (2, 0, block_id)


def normalize_block_id(block_id: str) -> str:
    """Use lm_head as the display row for the output-head projection."""
    return "lm_head" if block_id == "head" else block_id


def normalize_projection_name(proj_name: str) -> str:
    """Normalize old aliases to the profiler's canonical names."""
    try:
        return PROJECTION_ALIASES[proj_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown projection name in result file: {proj_name!r}"
        ) from exc


def find_latest_results_file(
    explicit_results_dir: Optional[Path] = None,
) -> Path:
    """Find the newest AIHWKIT result JSON."""
    if explicit_results_dir is not None:
        search_dirs = [explicit_results_dir]
    else:
        search_dirs = [
            REPO_ROOT / "data" / "results",
            REPO_ROOT / "results" / "lammie_2026",
            REPO_ROOT / "results",
        ]

    candidates: List[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            candidates.extend(directory.glob(RESULT_FILE_PATTERN))

    if not candidates:
        searched = "\n".join(f"  - {path}" for path in search_dirs)
        raise FileNotFoundError(
            "No AIHWKIT profiling result file was found.\n"
            f"Expected files matching {RESULT_FILE_PATTERN!r} in:\n"
            f"{searched}\n"
            "Pass --results-file to select a file explicitly."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_results_payload(results_file: Path) -> Dict[str, Any]:
    """Load and minimally validate the run script's JSON output."""
    if not results_file.is_file():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with results_file.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise TypeError("The result JSON root must be an object.")

    # The current driver wraps values under "results". Supporting a direct
    # result object makes the analyzer tolerant of manually extracted files.
    results = payload.get("results", payload)
    if not isinstance(results, dict):
        raise TypeError("The JSON 'results' value must be an object.")

    projections = results.get("projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError(
            "The JSON does not contain results.projections as a non-empty list."
        )

    return payload


def extract_sensitivity_tables(
    payload: Mapping[str, Any],
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    float,
]:
    """Convert projection records into the original nested plot structure.

    Returns:
        sensitivities:
            block -> projection -> mean delta perplexity
        sensitivity_stds:
            block -> projection -> standard deviation across noise seeds
        digital_perplexity:
            clean FP32 perplexity
    """
    results = payload.get("results", payload)
    projections = results["projections"]

    sensitivities: Dict[str, Dict[str, float]] = {}
    sensitivity_stds: Dict[str, Dict[str, float]] = {}

    for index, record in enumerate(projections):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Projection result at index {index} is not an object."
            )

        required = {
            "block_id",
            "proj_name",
            "sensitivity_mean",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"Projection result at index {index} is missing: "
                f"{sorted(missing)}"
            )

        block_id = normalize_block_id(str(record["block_id"]))
        proj_name = normalize_projection_name(str(record["proj_name"]))
        mean_value = float(record["sensitivity_mean"])
        std_value = float(record.get("sensitivity_std", np.nan))

        if not np.isfinite(mean_value):
            raise ValueError(
                f"Non-finite sensitivity for {block_id}/{proj_name}: "
                f"{mean_value}"
            )

        if proj_name in sensitivities.setdefault(block_id, {}):
            raise ValueError(
                f"Duplicate projection result: {block_id}/{proj_name}"
            )

        sensitivities[block_id][proj_name] = mean_value
        sensitivity_stds.setdefault(block_id, {})[proj_name] = std_value

    if len(projections) != 49:
        print(
            f"Warning: expected 49 projection records, found "
            f"{len(projections)}."
        )

    digital_perplexity = float(results.get("digital_perplexity", np.nan))
    if not np.isfinite(digital_perplexity):
        # Older result files may only store it inside every projection record.
        clean_values = {
            float(record["ppl_clean"])
            for record in projections
            if "ppl_clean" in record
        }
        if len(clean_values) == 1:
            digital_perplexity = next(iter(clean_values))

    return sensitivities, sensitivity_stds, digital_perplexity


def projection_value_or_nan(
    block_sensitivities: Mapping[str, float],
    proj_name: str,
) -> float:
    """Return a projection value or NaN when absent from that row."""
    return float(block_sensitivities.get(proj_name, np.nan))


def finite_values(
    sensitivities: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Return all finite mean sensitivity values."""
    values = [
        float(value)
        for block in sensitivities.values()
        for value in block.values()
        if np.isfinite(value)
    ]
    if not values:
        raise ValueError("No finite sensitivity values were found.")
    return np.asarray(values, dtype=np.float64)


def configure_delta_ppl_axis(
    ax,
    values: Sequence[float],
) -> str:
    """Use log scale when valid; use symmetric-log if values are non-positive.

    Delta perplexity is not mathematically guaranteed to remain positive for
    every finite-sample noise realization. A normal logarithmic axis cannot
    display zero or negative means, so symmetric-log is used only when needed.
    """
    values_array = np.asarray(values, dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]

    if values_array.size == 0:
        return "linear"

    if np.all(values_array > 0.0):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0)
        )
        ax.yaxis.set_major_formatter(
            mticker.LogFormatterMathtext(base=10.0)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(
                base=10.0,
                subs=np.arange(2, 10) * 0.1,
            )
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        return "log"

    maximum = float(np.max(np.abs(values_array)))
    positive_nonzero = np.abs(
        values_array[np.nonzero(values_array)]
    )
    if positive_nonzero.size:
        linthresh = max(
            float(np.min(positive_nonzero)),
            maximum * 1e-4,
            1e-12,
        )
    else:
        linthresh = 1e-12

    ax.set_yscale(
        "symlog",
        linthresh=linthresh,
        base=10,
    )
    return "symlog"


def analyze_sensitivities(
    sensitivities: Mapping[str, Mapping[str, float]],
    sensitivity_stds: Mapping[str, Mapping[str, float]],
    digital_perplexity: float,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Print the sensitivity distribution and ranking."""
    all_values = finite_values(sensitivities)
    block_avgs: Dict[str, float] = {}
    proj_values: Dict[str, List[float]] = {}

    for block_id in sorted(sensitivities, key=block_sort_key):
        values = np.asarray(
            list(sensitivities[block_id].values()),
            dtype=np.float64,
        )
        block_avgs[block_id] = float(np.mean(values))

        for proj_name, value in sensitivities[block_id].items():
            proj_values.setdefault(proj_name, []).append(float(value))

    print("\n" + "=" * 70)
    print("PHASE 1: AIHWKIT SENSITIVITY ANALYSIS")
    print("=" * 70)

    if np.isfinite(digital_perplexity):
        print(
            f"\nDigital FP32 perplexity: "
            f"{digital_perplexity:.6f}"
        )

    print("\nOverall delta-perplexity statistics:")
    print(f"  Mean sensitivity: {np.mean(all_values):.6f}")
    print(f"  Std sensitivity:  {np.std(all_values):.6f}")
    print(f"  Min sensitivity:  {np.min(all_values):.6f}")
    print(f"  Max sensitivity:  {np.max(all_values):.6f}")

    print("\nBlock-level average sensitivities:")
    for block_id in sorted(block_avgs, key=block_sort_key):
        print(f"  {block_id:8s}: {block_avgs[block_id]:.6f}")

    print("\nProjection-type average sensitivities:")
    for proj_name in PROJECTION_ORDER:
        if proj_name not in proj_values:
            continue
        values = np.asarray(
            proj_values[proj_name],
            dtype=np.float64,
        )
        print(
            f"  {proj_name:12s}: "
            f"mean={np.mean(values):.6f}, "
            f"std={np.std(values):.6f}, "
            f"range=[{np.min(values):.6f}, "
            f"{np.max(values):.6f}]"
        )

    ranked: List[Tuple[float, float, str]] = []
    for block_id, projections in sensitivities.items():
        for proj_name, mean_value in projections.items():
            std_value = sensitivity_stds.get(
                block_id,
                {},
            ).get(proj_name, np.nan)
            ranked.append(
                (
                    float(mean_value),
                    float(std_value),
                    f"{block_id}/{proj_name}",
                )
            )

    ranked.sort(key=lambda item: item[0], reverse=True)

    print("\nTop 5 most sensitive projections:")
    for rank, (mean_value, std_value, name) in enumerate(
        ranked[:5],
        start=1,
    ):
        if np.isfinite(std_value):
            print(
                f"  {rank}. {name:28s}: "
                f"{mean_value:.6f} +/- {std_value:.6f}"
            )
        else:
            print(
                f"  {rank}. {name:28s}: {mean_value:.6f}"
            )

    print("\nBottom 5 least sensitive projections:")
    for rank, (mean_value, std_value, name) in enumerate(
        reversed(ranked[-5:]),
        start=1,
    ):
        if np.isfinite(std_value):
            print(
                f"  {rank}. {name:28s}: "
                f"{mean_value:.6f} +/- {std_value:.6f}"
            )
        else:
            print(
                f"  {rank}. {name:28s}: {mean_value:.6f}"
            )

    return block_avgs, proj_values


def plot_sensitivities(
    sensitivities: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> List[Path]:
    """Generate the original three plots from AIHWKIT delta-PPL results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(sensitivities, key=block_sort_key)
    proj_names = PROJECTION_ORDER
    all_values = finite_values(sensitivities)

    # blocks = [
    #     block_id
    #     for block_id in sorted(sensitivities, key=block_sort_key)
    #     if block_id != "lm_head"
    # ]

    # proj_names = [
    #     proj_name
    #     for proj_name in PROJECTION_ORDER
    #     if proj_name != "lm_head"
    # ]

    # plot_only_sensitivities = {
    #     block_id: {
    #         proj_name: value
    #         for proj_name, value in sensitivities[block_id].items()
    #         if proj_name in proj_names
    #     }
    #     for block_id in blocks
    # }

    # all_values = finite_values(plot_only_sensitivities)
    output_files: List[Path] = []

    # ------------------------------------------------------------------
    # Plot 1: heatmap of mean delta perplexity
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    data = np.full(
        (len(blocks), len(proj_names)),
        np.nan,
        dtype=np.float64,
    )
    for row, block_id in enumerate(blocks):
        for column, proj_name in enumerate(proj_names):
            data[row, column] = projection_value_or_nan(
                sensitivities[block_id],
                proj_name,
            )

    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(
        masked_data,
        cmap="YlOrRd",
        aspect="auto",
    )

    ax.set_xticks(range(len(proj_names)))
    ax.set_yticks(range(len(blocks)))
    ax.set_xticklabels(proj_names)
    ax.set_yticklabels(blocks)
    ax.set_ylabel("Transformer Block")
    ax.set_xlabel("Projection Type")
    ax.set_title("Projection Noise Sensitivity Heatmap")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Delta Perplexity")

    for row in range(len(blocks)):
        for column in range(len(proj_names)):
            value = data[row, column]
            if np.isnan(value):
                ax.text(
                    column,
                    row,
                    "-",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
            else:
                ax.text(
                    column,
                    row,
                    f"{value:.5f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    plt.tight_layout()
    heatmap_file = output_dir / "phase1_sensitivity_heatmap.png"
    plt.savefig(heatmap_file, dpi=150)
    plt.close(fig)
    output_files.append(heatmap_file)
    print(f"\n✓ Saved: {heatmap_file.name}")

    # ------------------------------------------------------------------
    # Plot 2: distribution by projection type
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    proj_data: Dict[str, List[float]] = {
        projection: [] for projection in proj_names
    }
    for block_id in blocks:
        for proj_name in proj_names:
            value = projection_value_or_nan(
                sensitivities[block_id],
                proj_name,
            )
            if np.isfinite(value):
                proj_data[proj_name].append(value)

    boxplot_data = [
        proj_data[projection]
        if proj_data[projection]
        else [np.nan]
        for projection in proj_names
    ]
    ax.boxplot(boxplot_data)
    ax.set_xticks(range(1, len(proj_names) + 1))
    ax.set_xticklabels(proj_names)

    configure_delta_ppl_axis(ax, all_values)
    ax.set_ylabel("Mean Delta Perplexity")
    ax.set_title("Sensitivity Distribution by Projection Type")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    distribution_file = (
        output_dir / "phase1_sensitivity_distribution.png"
    )
    plt.savefig(distribution_file, dpi=150)
    plt.close(fig)
    output_files.append(distribution_file)
    print(f"✓ Saved: {distribution_file.name}")

    # ------------------------------------------------------------------
    # Plot 3: grouped per-block sensitivity
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(blocks))
    width = 0.16
    center = (len(proj_names) - 1) / 2.0

    for index, proj_name in enumerate(proj_names):
        values = [
            projection_value_or_nan(
                sensitivities[block_id],
                proj_name,
            )
            for block_id in blocks
        ]
        ax.bar(
            x + (index - center) * width,
            values,
            width=width,
            label=proj_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    configure_delta_ppl_axis(ax, all_values)
    ax.set_ylabel("Mean Delta Perplexity")
    ax.set_title("Per-Block Sensitivity by Projection Type")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    block_file = output_dir / "phase1_sensitivity_by_block.png"
    plt.savefig(block_file, dpi=150)
    plt.close(fig)
    output_files.append(block_file)
    print(f"✓ Saved: {block_file.name}")

    return output_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and plot the Lammie 2026 AIHWKIT sensitivity "
            "profiling output."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--results-file",
        type=Path,
        help="Specific run_lammie_2026_aihwkit.py JSON output.",
    )
    selection.add_argument(
        "--results-dir",
        type=Path,
        help=(
            "Directory containing "
            f"{RESULT_FILE_PATTERN}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for generated plots. Defaults to the selected "
            "result file's directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load one AIHWKIT run, analyze it, and create the plots."""
    args = parse_args()

    if args.results_file is not None:
        results_file = args.results_file.resolve()
    else:
        results_dir = (
            args.results_dir.resolve()
            if args.results_dir is not None
            else None
        )
        results_file = find_latest_results_file(results_dir)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_file.parent
    )

    print("\nLoading AIHWKIT results from:")
    print(f"  Results: {results_file}")
    print(f"  Plots:   {output_dir}")

    payload = load_results_payload(results_file)
    (
        sensitivities,
        sensitivity_stds,
        digital_perplexity,
    ) = extract_sensitivity_tables(payload)

    analyze_sensitivities(
        sensitivities=sensitivities,
        sensitivity_stds=sensitivity_stds,
        digital_perplexity=digital_perplexity,
    )

    plot_sensitivities(
        sensitivities=sensitivities,
        output_dir=output_dir,
    )

    print("\n" + "=" * 70)
    print("✓ Phase 1 AIHWKIT analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
