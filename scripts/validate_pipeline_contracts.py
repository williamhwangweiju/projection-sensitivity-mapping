#!/usr/bin/env python3
"""Validate sharding, capacity, and cross-phase artifact contracts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_json, load_yaml
from src.evaluation.noise_materialization import read_placement_csv
from src.mapping.sharding import count_projection_shards
from src.simulators.tile_fidelity import load_trace


def validate_pipeline(
    config_path: Path,
    phase1_path: Path,
    trace_path: Path,
    phase3_manifest_path: Path,
) -> None:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    phase3 = load_json(phase3_manifest_path)
    trace = load_trace(str(trace_path))

    projection_rows = profile["projections"]
    projection_ids = [str(row["projection_id"]) for row in projection_rows]
    if len(projection_ids) != len(set(projection_ids)):
        raise ValueError("Phase 1 contains duplicate projection IDs.")
    if profile["mapping_sensitivity_unit"] != "delta_nll_noise":
        raise ValueError("Pipeline requires Phase-1 delta_nll_noise sensitivity.")
    if trace.noise_std.shape[1] != int(config["hardware"]["num_tiles"]):
        raise ValueError("Phase-2 tile count disagrees with the hardware config.")

    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    total_tiers = int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    expected_shards = sum(
        count_projection_shards(
            str(row["projection_id"]),
            int(row["out_features"]),
            int(row["in_features"]),
            tier_rows,
            tier_cols,
        )
        for row in projection_rows
    )
    if expected_shards > total_tiers:
        raise ValueError(
            f"All-analog deployment needs {expected_shards} tiers but the "
            f"substrate provides {total_tiers}."
        )

    all_projection_ids = set(projection_ids)
    policy_shards: dict[str, set[str]] = {}
    for item in phase3["placements"]:
        policy = str(item["policy"])
        rows = read_placement_csv(item["placement_path"])
        shard_ids = {str(row["shard_id"]) for row in rows}
        if len(shard_ids) != len(rows):
            raise ValueError(f"Duplicate shard in placement {policy}.")
        slots = {(int(row["tile_id"]), int(row["tier_id"])) for row in rows}
        if len(slots) != len(rows):
            raise ValueError(f"Reused physical tier in placement {policy}.")
        placed_projections = {str(row["projection_id"]) for row in rows}
        if placed_projections != all_projection_ids:
            raise ValueError(
                f"Placement {policy} does not cover exactly the profiled "
                "projection set."
            )
        if len(rows) != expected_shards:
            raise ValueError(
                f"Placement {policy} has {len(rows)} shards; expected {expected_shards}."
            )
        if any(int(row["tile_id"]) >= trace.noise_std.shape[1] for row in rows):
            raise ValueError(f"Placement {policy} references a tile outside the trace.")
        policy_shards[policy] = shard_ids

    expected_policies = set(str(value) for value in config["phase3"]["policies"])
    if set(policy_shards) != expected_policies:
        raise ValueError(
            f"Missing placement policies: {sorted(expected_policies - set(policy_shards))}"
        )
    reference = next(iter(policy_shards.values()))
    if any(shards != reference for shards in policy_shards.values()):
        raise ValueError("Policies do not place the same analog shard set.")

    print("Pipeline contracts validated successfully.")
    print(f"  {len(projection_ids)} profiled projections, all analog")
    print(f"  {expected_shards} shards on {total_tiers} available tiers")
    print(f"  identical shard sets across {len(policy_shards)} policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    args = parser.parse_args()
    validate_pipeline(args.config, args.phase1, args.trace, args.phase3_manifest)
