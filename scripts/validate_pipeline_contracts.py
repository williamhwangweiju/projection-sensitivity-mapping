#!/usr/bin/env python3
"""Validate digital-set, sharding, capacity, and cross-phase artifact contracts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_json, load_yaml
from src.evaluation.noise_materialization import read_placement_csv
from src.simulators.tile_fidelity import load_trace


def validate_pipeline(
    config_path: Path,
    phase1_path: Path,
    operating_points_path: Path,
    trace_path: Path,
    phase3_manifest_path: Path,
) -> None:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    points_payload = load_json(operating_points_path)
    phase3 = load_json(phase3_manifest_path)
    trace = load_trace(str(trace_path))

    projection_rows = profile["projections"]
    projection_ids = [str(row["projection_id"]) for row in projection_rows]
    if len(projection_ids) != len(set(projection_ids)):
        raise ValueError("Phase 1 contains duplicate projection IDs.")
    if profile["mapping_sensitivity_unit"] != "delta_nll_noise":
        raise ValueError("Hybrid pipeline requires Phase-1 delta_nll_noise sensitivity.")
    if trace.noise_std.shape[1] != int(config["hardware"]["num_tiles"]):
        raise ValueError("Phase-2 tile count disagrees with the hardware config.")

    point_by_id: dict[str, dict[str, Any]] = {}
    full_set = set(projection_ids)
    for point in points_payload["operating_points"]:
        point_id = str(point["digital_set_id"])
        if point_id in point_by_id:
            raise ValueError(f"Duplicate digital_set_id: {point_id}")
        point_by_id[point_id] = point
        digital = set(point["digital_projection_ids"])
        analog = set(point["analog_projection_ids"])
        if digital & analog or digital | analog != full_set:
            raise ValueError(f"Digital/analog partition is invalid for {point_id}.")
        feasible = int(point["analog_shard_count"]) <= int(point["available_physical_tiers"])
        if feasible != bool(point["capacity_feasible"]):
            raise ValueError(f"Capacity flag mismatch for {point_id}.")

    grouped: dict[str, dict[str, set[str]]] = {}
    for item in phase3["placements"]:
        point_id = str(item["digital_set_id"])
        policy = str(item["policy"])
        point = point_by_id[point_id]
        if not bool(point["capacity_feasible"]):
            raise ValueError(f"Phase 3 generated a placement for infeasible {point_id}.")
        rows = read_placement_csv(item["placement_path"])
        shard_ids = {str(row["shard_id"]) for row in rows}
        if len(shard_ids) != len(rows):
            raise ValueError(f"Duplicate shard in {point_id}/{policy}.")
        slots = {(int(row["tile_id"]), int(row["tier_id"])) for row in rows}
        if len(slots) != len(rows):
            raise ValueError(f"Reused physical tier in {point_id}/{policy}.")
        digital = set(point["digital_projection_ids"])
        if any(str(row["projection_id"]) in digital for row in rows):
            raise ValueError(f"A protected digital projection was sharded in {point_id}/{policy}.")
        if len(rows) != int(point["analog_shard_count"]):
            raise ValueError(f"Analog shard count mismatch in {point_id}/{policy}.")
        grouped.setdefault(point_id, {})[policy] = shard_ids

    expected_policies = set(str(value) for value in config["phase3"]["policies"])
    for point_id, policy_sets in grouped.items():
        if set(policy_sets) != expected_policies:
            raise ValueError(f"Missing policies for {point_id}: {expected_policies - set(policy_sets)}")
        reference = next(iter(policy_sets.values()))
        if any(shards != reference for shards in policy_sets.values()):
            raise ValueError(f"Policies do not place the same analog shard set for {point_id}.")

    print("Pipeline contracts validated successfully.")
    print(f"  {len(projection_ids)} profiled digital/analog candidates")
    print(f"  {len(point_by_id)} digital operating points")
    print(f"  {len(grouped)} capacity-feasible operating points with placements")
    print("  protected projections excluded from analog capacity/noise")
    print("  identical analog shard sets across static policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    args = parser.parse_args()
    validate_pipeline(args.config, args.phase1, args.operating_points, args.trace, args.phase3_manifest)
