#!/usr/bin/env python3
"""Join the Phase-4 quality frontier with the analytical energy model.

Produces one row per (operating point, policy) with the per-token energy
breakdown next to nominal and degraded quality, a per-policy Pareto flag on
(total energy, mean degraded NLL), and a quality-versus-energy figure.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.pareto import pareto_frontier
from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.tabular import write_csv
from src.cost.energy_model import (
    EnergyModelParams,
    all_digital_energy_pj,
    operating_point_energy,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    policies = sorted({str(row["policy"]) for row in rows})
    for policy in policies:
        group = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: float(row["total_energy_pj_per_token"]),
        )
        axis.plot(
            [float(row["total_energy_pj_per_token"]) / 1e6 for row in group],
            [float(row["mean_degraded_ppl_from_nll"]) for row in group],
            marker="o",
            label=policy,
        )
    axis.set_xlabel("Energy per token (uJ)")
    axis.set_ylabel("Degraded perplexity (from mean NLL)")
    axis.set_yscale("log")
    axis.set_title("Quality versus energy across digital-protection budgets")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(
    config_path: Path,
    operating_points_path: Path,
    phase3_manifest_path: Path,
    frontier_csv_path: Path,
) -> Path:
    config = load_yaml(config_path)
    params = EnergyModelParams.from_config(config)
    params.validate()

    points = {
        str(point["digital_set_id"]): point
        for point in load_json(operating_points_path)["operating_points"]
    }
    manifest = load_json(phase3_manifest_path)["placements"]
    placement_paths = {
        (str(row["digital_set_id"]), str(row["policy"])): Path(row["placement_path"])
        for row in manifest
    }
    frontier_rows = read_csv_rows(frontier_csv_path)
    if not frontier_rows:
        raise ValueError(f"No rows in {frontier_csv_path}.")

    energy_cache: dict[tuple[str, str], dict[str, float]] = {}
    output_rows: list[dict[str, Any]] = []
    for row in frontier_rows:
        digital_set_id = str(row["digital_set_id"])
        policy = str(row["policy"])
        point = points.get(digital_set_id)
        placement_path = placement_paths.get((digital_set_id, policy))
        if point is None or placement_path is None:
            print(f"Skipping {digital_set_id}/{policy}: missing point or placement.")
            continue
        key = (digital_set_id, policy)
        if key not in energy_cache:
            placement_rows = read_csv_rows(placement_path)
            energy_cache[key] = operating_point_energy(point, placement_rows, params)
        energy = energy_cache[key]
        output_rows.append(
            {
                **row,
                **energy,
                "all_digital_energy_pj_per_token": all_digital_energy_pj(point, params),
            }
        )
    if not output_rows:
        raise ValueError("No frontier rows could be joined with placements.")

    # Per-policy Pareto flag over (energy, degraded NLL).
    for policy in {str(row["policy"]) for row in output_rows}:
        group = [row for row in output_rows if row["policy"] == policy]
        optimal = pareto_frontier(
            group,
            cost_field="total_energy_pj_per_token",
            quality_field="mean_degraded_nll",
        )
        optimal_ids = {str(row["digital_set_id"]) for row in optimal}
        for row in group:
            row["pareto_optimal"] = str(row["digital_set_id"]) in optimal_ids

    output_root = resolve_path(config["phase4"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(output_root / "energy_quality_frontier.csv", output_rows)
    figure_path = output_root / "energy_quality_pareto.png"
    plot_frontier(output_rows, figure_path)
    metadata_path = output_root / "energy_metadata.json"
    save_json(
        metadata_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "operating_points_path": str(operating_points_path),
            "phase3_manifest_path": str(phase3_manifest_path),
            "frontier_csv_path": str(frontier_csv_path),
            "energy_model": {
                "e_mac_digital_pj": params.e_mac_digital_pj,
                "e_analog_mac_pj": params.e_analog_mac_pj,
                "e_adc_pj": params.e_adc_pj,
                "e_dac_pj": params.e_dac_pj,
                "scope": "first_order_mac_adc_dac_only",
            },
            "artifacts": {
                "frontier": str(csv_path),
                "figure": str(figure_path),
            },
        },
    )
    print(f"Energy/quality frontier saved to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml",
    )
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    parser.add_argument("--frontier-csv", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.operating_points, args.phase3_manifest, args.frontier_csv)
