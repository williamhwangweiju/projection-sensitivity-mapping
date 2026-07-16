#!/usr/bin/env python3
"""Aggregate Phase-4 results hierarchically across independent trace seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_json, save_json
from src.common.tabular import write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def bootstrap_trace_mean_ci(
    values: list[float], *, seed: int, samples: int = 10000
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(
    metadata_paths: Iterable[Path], output_root: Path, bootstrap_seed: int
) -> Path:
    metadata_paths = [path.resolve() for path in metadata_paths]
    if len(metadata_paths) < 2:
        raise ValueError("Use at least two independent Phase-4 trace runs.")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for trace_index, metadata_path in enumerate(metadata_paths):
        metadata = load_json(metadata_path)
        trace_seed = int(metadata.get("experiment_seed", trace_index))
        for row in read_csv(Path(metadata["artifacts"]["quality"])):
            all_rows.append({"trace_seed": trace_seed, **row})
    merged_path = write_csv(output_root / "multiseed_phase4_quality.csv", all_rows)

    # First average paired realization/timestep differences within each trace.
    keyed: dict[
        tuple[int, str, int, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in all_rows:
        key = (
            int(row["trace_seed"]),
            str(row["digital_set_id"]),
            int(row["timestep"]),
            int(row["realization"]),
        )
        keyed[key][str(row["policy"])] = row

    comparisons = (
        ("hardware_only", "static_sensitivity"),
        ("sequential", "static_sensitivity"),
        ("random", "static_sensitivity"),
    )
    within_trace: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for (trace_seed, digital_set_id, _, _), policies in keyed.items():
        for baseline, method in comparisons:
            if baseline not in policies or method not in policies:
                continue
            difference = float(policies[baseline]["delta_nll_tile"]) - float(
                policies[method]["delta_nll_tile"]
            )
            within_trace[(trace_seed, digital_set_id, baseline, method)].append(
                difference
            )

    trace_rows: list[dict[str, Any]] = []
    for (trace_seed, digital_set_id, baseline, method), values in sorted(
        within_trace.items()
    ):
        trace_rows.append(
            {
                "trace_seed": trace_seed,
                "digital_set_id": digital_set_id,
                "baseline_policy": baseline,
                "method_policy": method,
                "paired_evaluations_within_trace": len(values),
                "mean_nll_improvement_within_trace": float(np.mean(values)),
                "median_nll_improvement_within_trace": float(np.median(values)),
                "win_fraction_within_trace": float(
                    np.mean(np.asarray(values) > 0.0)
                ),
            }
        )
    trace_path = write_csv(output_root / "trace_level_policy_differences.csv", trace_rows)

    # Then bootstrap over independent trace means, not correlated timesteps.
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in trace_rows:
        grouped[
            (
                str(row["digital_set_id"]),
                str(row["baseline_policy"]),
                str(row["method_policy"]),
            )
        ].append(float(row["mean_nll_improvement_within_trace"]))

    summary_rows: list[dict[str, Any]] = []
    for (digital_set_id, baseline, method), values in sorted(grouped.items()):
        low, high = bootstrap_trace_mean_ci(
            values, seed=bootstrap_seed, samples=10000
        )
        summary_rows.append(
            {
                "digital_set_id": digital_set_id,
                "baseline_policy": baseline,
                "method_policy": method,
                "independent_trace_count": len(values),
                "mean_trace_level_nll_improvement": float(np.mean(values)),
                "median_trace_level_nll_improvement": float(np.median(values)),
                "std_across_traces": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "bootstrap_trace_ci95_low": low,
                "bootstrap_trace_ci95_high": high,
                "trace_win_fraction": float(np.mean(np.asarray(values) > 0.0)),
            }
        )
    summary_path = write_csv(output_root / "multiseed_paired_summary.csv", summary_rows)

    manifest_path = output_root / "multiseed_analysis_manifest.json"
    save_json(
        manifest_path,
        {
            "phase4_metadata_paths": [str(path) for path in metadata_paths],
            "bootstrap_seed": int(bootstrap_seed),
            "hierarchy": (
                "paired differences averaged within each hardware trace, then "
                "confidence intervals bootstrapped across independent traces"
            ),
            "artifacts": {
                "merged_quality": str(merged_path),
                "trace_level_differences": str(trace_path),
                "paired_summary": str(summary_path),
            },
        },
    )
    print(f"Multi-seed analysis: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-metadata", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/results/paper_artifacts/multiseed",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    args = parser.parse_args()
    main(args.phase4_metadata, args.output_root, args.bootstrap_seed)
