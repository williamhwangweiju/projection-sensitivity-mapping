#!/usr/bin/env python3
"""Build paper-ready tables from raw hybrid mapping artifacts."""
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

from src.analysis.pareto import pareto_frontier, spearman_correlation
from src.common.config import load_json, save_json
from src.common.tabular import write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(
    operating_points_path: Path,
    phase4_metadata_path: Path,
    phase5_manifest_path: Path | None,
    phase5_quality_metadata_path: Path | None,
    output_root: Path,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    points = {
        row["digital_set_id"]: row
        for row in load_json(operating_points_path)["operating_points"]
    }
    phase4 = load_json(phase4_metadata_path)
    nominal = read_csv(Path(phase4["artifacts"]["nominal_frontier"]))
    summary = read_csv(Path(phase4["artifacts"]["summary"]))
    quality = read_csv(Path(phase4["artifacts"]["quality"]))

    frontier_rows: list[dict[str, Any]] = []
    for row in nominal:
        point = points[row["digital_set_id"]]
        frontier_rows.append({
            "digital_set_id": row["digital_set_id"],
            "selection_method": row["selection_method"],
            "digital_projection_ids": row["digital_projection_ids"],
            "digital_projection_count": int(row["digital_projection_count"]),
            "digital_parameter_fraction": float(row["digital_parameter_fraction"]),
            "digital_incremental_storage_fraction": float(row.get("digital_incremental_storage_fraction", row["digital_parameter_fraction"])),
            "digital_mac_fraction": float(row["digital_mac_fraction"]),
            "analog_shard_count": int(point["analog_shard_count"]),
            "nominal_hybrid_nll": float(row["nominal_hybrid_nll"]),
            "nominal_hybrid_ppl": float(row["nominal_hybrid_ppl"]),
            "delta_nll_nominal_vs_digital": float(row["delta_nll_nominal_vs_digital"]),
        })
    frontier_path = write_csv(output_root / "digital_protection_frontier.csv", frontier_rows)
    pareto_mac = pareto_frontier(
        frontier_rows,
        cost_field="digital_mac_fraction",
        quality_field="delta_nll_nominal_vs_digital",
    )
    pareto_storage = pareto_frontier(
        frontier_rows,
        cost_field="digital_incremental_storage_fraction",
        quality_field="delta_nll_nominal_vs_digital",
    )
    pareto_mac_path = write_csv(output_root / "pareto_digital_mac_vs_nominal_nll.csv", pareto_mac)
    pareto_storage_path = write_csv(output_root / "pareto_storage_vs_nominal_nll.csv", pareto_storage)

    placement_path = write_csv(output_root / "static_placement_quality_summary.csv", summary)
    proxy_corr_rows: list[dict[str, Any]] = []
    for point_id in sorted({row["digital_set_id"] for row in quality}):
        subset = [row for row in quality if row["digital_set_id"] == point_id]
        proxy_corr_rows.append({
            "digital_set_id": point_id,
            "samples": len(subset),
            "spearman_proxy_vs_delta_nll_tile": spearman_correlation(
                [float(row["proxy_variance"]) for row in subset],
                [float(row["delta_nll_tile"]) for row in subset],
            ),
        })
    proxy_path = write_csv(output_root / "proxy_quality_correlation.csv", proxy_corr_rows)

    adaptive_artifacts: dict[str, str | None] = {
        "mapping_manifest": None,
        "quality_summary": None,
        "event_summary": None,
    }
    if phase5_manifest_path is not None and phase5_manifest_path.is_file():
        phase5 = load_json(phase5_manifest_path)
        adaptive_artifacts["mapping_manifest"] = str(phase5_manifest_path)
        event_paths = [Path(run["adaptive_events_path"]) for run in phase5["adaptive_runs"]]
        event_rows: list[dict[str, str]] = []
        for path in event_paths:
            event_rows.extend(read_csv(path))
        if event_rows:
            adaptive_artifacts["event_summary"] = str(
                write_csv(output_root / "adaptive_remapping_events.csv", event_rows)
            )
    if phase5_quality_metadata_path is not None and phase5_quality_metadata_path.is_file():
        phase5_quality = load_json(phase5_quality_metadata_path)
        adaptive_summary = Path(phase5_quality["artifacts"]["summary"])
        adaptive_artifacts["quality_summary"] = str(
            write_csv(output_root / "adaptive_quality_summary.csv", read_csv(adaptive_summary))
        )

    manifest_path = output_root / "paper_artifact_manifest.json"
    save_json(manifest_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operating_points_path": str(operating_points_path),
        "phase4_metadata_path": str(phase4_metadata_path),
        "phase5_manifest_path": None if phase5_manifest_path is None else str(phase5_manifest_path),
        "phase5_quality_metadata_path": None if phase5_quality_metadata_path is None else str(phase5_quality_metadata_path),
        "artifacts": {
            "digital_protection_frontier": str(frontier_path),
            "pareto_mac": str(pareto_mac_path),
            "pareto_incremental_storage": str(pareto_storage_path),
            "static_placement_summary": str(placement_path),
            "proxy_quality_correlation": str(proxy_path),
            "adaptive": adaptive_artifacts,
        },
        "primary_metrics": {
            "absolute_deployment": "delta_nll_total = noisy_hybrid_nll - digital_nll",
            "placement_increment": "delta_nll_tile = noisy_hybrid_nll - nominal_hybrid_nll",
            "digital_cost": [
                "digital_mac_fraction",
                "digital_parameter_fraction",
                "digital_incremental_storage_fraction",
            ],
        },
    })
    print(f"Paper artifacts saved to: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--phase4-metadata", type=Path, required=True)
    parser.add_argument("--phase5-manifest", type=Path)
    parser.add_argument("--phase5-quality-metadata", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/results/paper_artifacts")
    args = parser.parse_args()
    main(
        args.operating_points,
        args.phase4_metadata,
        args.phase5_manifest,
        args.phase5_quality_metadata,
        args.output_root,
    )
