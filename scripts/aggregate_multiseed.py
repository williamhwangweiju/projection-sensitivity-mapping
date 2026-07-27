#!/usr/bin/env python3
"""Aggregate Phase-4 paired policy comparisons across hardware-trace seeds.

Within one trace, the 15 timestep/realization evaluations are correlated
(shared trace, shared model, paired noise); their bootstrap intervals are
descriptive only. This script treats each independent hardware-trace seed as
the statistical unit: it computes the mean paired NLL improvement per trace,
then reports the across-trace mean, standard deviation, a t-based 95%
confidence interval over traces, and the fraction of traces the method policy
wins. These are the inferentially meaningful paper numbers.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.tabular import write_csv

# Two-sided 97.5% Student-t quantiles for tiny trace counts (df = n - 1).
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t_quantile(df: int) -> float:
    if df <= 0:
        return math.nan
    if df in T_975:
        return T_975[df]
    return 1.96 + 2.4 / df  # adequate approximation beyond df=9


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def per_trace_paired_means(
    rows: list[dict[str, str]],
    method_policies: list[str],
    baseline_policies: list[str],
) -> dict[tuple[str, str, str], float]:
    """Mean paired delta_nll_tile improvement per (set, baseline, method)."""
    keyed: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["digital_set_id"]), int(row["timestep"]), int(row["realization"]))
        keyed[key][str(row["policy"])] = float(row["delta_nll_tile"])
    sums: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (digital_set_id, _, _), policies in keyed.items():
        for method in method_policies:
            for baseline in baseline_policies:
                if method in policies and baseline in policies and method != baseline:
                    sums[(digital_set_id, baseline, method)].append(
                        policies[baseline] - policies[method]
                    )
    return {key: sum(values) / len(values) for key, values in sums.items() if values}


def main(
    manifest_path: Path,
    output_path: Path,
    method_policies: list[str],
    baseline_policies: list[str],
) -> Path:
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"No runs listed in {manifest_path}.")

    per_trace: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for run in runs:
        trace_seed = int(run["trace_seed"])
        quality_csv = Path(run["output_root"]) / "phase4" / "hybrid_quality_by_policy.csv"
        if not quality_csv.is_file():
            raise FileNotFoundError(
                f"Missing Phase-4 rows for trace seed {trace_seed}: {quality_csv}"
            )
        rows = read_rows(quality_csv)
        for key, value in per_trace_paired_means(
            rows, method_policies, baseline_policies
        ).items():
            per_trace[key][trace_seed] = value

    output_rows: list[dict[str, Any]] = []
    for (digital_set_id, baseline, method), by_seed in sorted(per_trace.items()):
        values = [by_seed[seed] for seed in sorted(by_seed)]
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else math.nan
        margin = t_quantile(n - 1) * sem if n > 1 else math.nan
        output_rows.append(
            {
                "digital_set_id": digital_set_id,
                "baseline_policy": baseline,
                "method_policy": method,
                "trace_seeds": ";".join(str(seed) for seed in sorted(by_seed)),
                "n_traces": n,
                "mean_nll_improvement": mean,
                "std_across_traces": std,
                "t_ci95_low": mean - margin if n > 1 else math.nan,
                "t_ci95_high": mean + margin if n > 1 else math.nan,
                "trace_win_fraction": sum(1 for v in values if v > 0) / n,
                "min_trace_improvement": min(values),
                "max_trace_improvement": max(values),
            }
        )
    if not output_rows:
        raise ValueError("No paired comparisons found across the listed runs.")
    write_csv(output_path, output_rows)
    print(f"Cross-trace paired summary ({len(runs)} traces): {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/results/multiseed/multiseed_run_manifest.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/results/multiseed/cross_trace_paired_summary.csv",
    )
    parser.add_argument(
        "--method-policies",
        nargs="+",
        default=["static_sensitivity", "static_fisher"],
    )
    parser.add_argument(
        "--baseline-policies",
        nargs="+",
        default=["random", "sequential", "hardware_only"],
    )
    args = parser.parse_args()
    main(args.manifest, args.output, args.method_policies, args.baseline_policies)
