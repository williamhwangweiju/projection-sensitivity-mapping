"""Phase 3 objective and evaluation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from src.simulators.tile_fidelity import TileFidelityTrace

from .placement import Placement
from .sharding import ProjectionShard


def evaluate_placement_over_trace(
    *,
    placement: Placement,
    shards: list[ProjectionShard],
    trace: TileFidelityTrace,
    reference_noise_std: float,
) -> list[dict[str, Any]]:
    """Evaluate one static placement over all timesteps in a fidelity trace."""
    if reference_noise_std <= 0.0:
        raise ValueError("reference_noise_std must be positive.")

    shard_by_id = {
        shard.shard_id: shard
        for shard in shards
    }

    rows: list[dict[str, Any]] = []
    for timestep in range(trace.num_timesteps):
        objective_sum = 0.0
        noise_sum = 0.0
        weighted_noise_sum = 0.0
        weight_sum = 0.0
        shards_on_faulted_tiles = 0
        shards_on_unavailable_tiles = 0

        projections_on_faulted_tiles: set[str] = set()
        projections_on_unavailable_tiles: set[str] = set()

        for shard_id, assignment in placement.assignments.items():
            shard = shard_by_id[shard_id]
            tile_id = assignment.tile_id
            tile_noise = float(trace.noise_std[timestep, tile_id])
            tile_faulted = bool(trace.faulted[timestep, tile_id])
            tile_available = bool(trace.available[timestep, tile_id])

            shard_weight = shard.shard_weight
            projection_sensitivity = shard.sensitivity_score
            shard_importance = shard_weight * projection_sensitivity

            objective_contribution = (
                shard_importance
                * (tile_noise / reference_noise_std) ** 2
            )

            objective_sum += objective_contribution
            noise_sum += tile_noise
            weighted_noise_sum += shard_importance * tile_noise
            weight_sum += shard_importance

            if tile_faulted:
                shards_on_faulted_tiles += 1
                projections_on_faulted_tiles.add(shard.projection_id)
            if not tile_available:
                shards_on_unavailable_tiles += 1
                projections_on_unavailable_tiles.add(shard.projection_id)

        shard_count = len(placement.assignments)
        mean_noise = noise_sum / shard_count if shard_count else 0.0
        weighted_mean_noise = (
            weighted_noise_sum / weight_sum
            if weight_sum > 0.0
            else 0.0
        )

        placement_feasible = shards_on_unavailable_tiles == 0

        row = {
            "policy": placement.policy_name,
            "timestep": timestep,
            "sensitivity_weighted_tile_error": objective_sum,
            "mean_assigned_noise_std": mean_noise,
            "sensitivity_weighted_mean_assigned_noise_std": weighted_mean_noise,
            "capacity_utilization": placement.capacity_utilization,
            "shards_on_faulted_tiles": shards_on_faulted_tiles,
            "shards_on_unavailable_tiles": shards_on_unavailable_tiles,
            "projections_on_faulted_tiles": len(projections_on_faulted_tiles),
            "projections_on_unavailable_tiles": len(projections_on_unavailable_tiles),
            "placement_feasible": placement_feasible,
            "service_failure": not placement_feasible,
        }

        if hasattr(placement, "available_capacity_utilization"):
            row["available_capacity_utilization"] = (
                placement.available_capacity_utilization
            )

        rows.append(row)

    return rows


def build_policy_summary(
    *,
    timestep_rows: list[dict[str, Any]],
    placements: Mapping[str, Placement],
) -> list[dict[str, Any]]:
    """Build one summary row per static policy."""
    rows_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timestep_rows:
        rows_by_policy[str(row["policy"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for policy_name, rows in sorted(rows_by_policy.items()):
        if not rows:
            continue

        rows = sorted(rows, key=lambda row: int(row["timestep"]))

        objective_values = np.asarray(
            [float(row["sensitivity_weighted_tile_error"]) for row in rows],
            dtype=np.float64,
        )
        mean_noise_values = np.asarray(
            [float(row["mean_assigned_noise_std"]) for row in rows],
            dtype=np.float64,
        )
        weighted_noise_values = np.asarray(
            [float(row["sensitivity_weighted_mean_assigned_noise_std"]) for row in rows],
            dtype=np.float64,
        )
        faulted_counts = np.asarray(
            [int(row["shards_on_faulted_tiles"]) for row in rows],
            dtype=np.int64,
        )
        unavailable_counts = np.asarray(
            [int(row["shards_on_unavailable_tiles"]) for row in rows],
            dtype=np.int64,
        )
        service_failure_counts = np.asarray(
            [int(bool(row.get("service_failure", False))) for row in rows],
            dtype=np.int64,
        )

        placement = placements[policy_name]
        summary_row = {
            "policy": policy_name,
            "timesteps_evaluated": len(rows),
            "mean_sensitivity_weighted_tile_error": float(np.mean(objective_values)),
            "final_sensitivity_weighted_tile_error": float(objective_values[-1]),
            "peak_sensitivity_weighted_tile_error": float(np.max(objective_values)),
            "mean_assigned_noise_std": float(np.mean(mean_noise_values)),
            "mean_sensitivity_weighted_assigned_noise_std": float(
                np.mean(weighted_noise_values)
            ),
            "max_shards_on_faulted_tiles": int(np.max(faulted_counts)),
            "max_shards_on_unavailable_tiles": int(np.max(unavailable_counts)),
            "timesteps_with_service_failure": int(np.sum(service_failure_counts)),
            "capacity_utilization": placement.capacity_utilization,
            "remapping_events": 0,
            "weight_data_moved_after_initialization": 0,
        }

        if hasattr(placement, "available_capacity_utilization"):
            summary_row["available_capacity_utilization"] = (
                placement.available_capacity_utilization
            )

        summary_rows.append(summary_row)

    return summary_rows
