"""Phase-3 sensitivity-weighted placement proxy.

The proxy combines Phase-1 total DeltaPPL sensitivity with the squared ratio
of each mapped tile's current noise to the Phase-1 reference noise.  It is a
placement heuristic, not a prediction of full-model Phase-4 DeltaPPL.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .placement import Placement
from .sharding import ProjectionShard


def _trace_arrays(trace: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noise = np.asarray(trace.noise_std, dtype=np.float64)
    faulted = np.asarray(trace.faulted, dtype=bool)
    available = np.asarray(trace.available, dtype=bool)
    if noise.ndim != 2 or faulted.shape != noise.shape or available.shape != noise.shape:
        raise ValueError("Phase-2 trace arrays must all have shape [T, num_tiles].")
    return noise, faulted, available


def evaluate_placement_over_trace(
    *,
    placement: Placement,
    shards: Sequence[ProjectionShard],
    trace: Any,
    reference_noise_std: float,
) -> list[dict[str, Any]]:
    """Evaluate a static placement using a sensitivity-weighted proxy.

    For shard ``s`` from projection ``p`` on tile ``i`` at timestep ``t``::

        proxy_s,t = shard_weight_s * DeltaPPL_p,reference
                    * (sigma_i,t / sigma_reference)^2

    The score is used only to compare placements under a shared trace.
    """

    if reference_noise_std <= 0.0:
        raise ValueError("reference_noise_std must be positive.")
    noise, faulted, available = _trace_arrays(trace)

    rows: list[dict[str, Any]] = []
    for timestep in range(noise.shape[0]):
        weighted_proxy = 0.0
        weighted_noise_sum = 0.0
        total_shard_weight = 0.0
        shards_on_faulted = 0
        shards_on_unavailable = 0
        projections_on_faulted: set[str] = set()
        projections_on_unavailable: set[str] = set()

        for shard in shards:
            assignment = placement.assignments[shard.shard_id]
            tile_id = assignment.tile_id
            sigma = float(noise[timestep, tile_id])
            variance_ratio = (sigma / reference_noise_std) ** 2
            weighted_proxy += (
                shard.shard_weight
                * shard.sensitivity_score
                * variance_ratio
            )
            weighted_noise_sum += shard.shard_weight * sigma
            total_shard_weight += shard.shard_weight

            if bool(faulted[timestep, tile_id]):
                shards_on_faulted += 1
                projections_on_faulted.add(shard.projection_id)
            if not bool(available[timestep, tile_id]):
                shards_on_unavailable += 1
                projections_on_unavailable.add(shard.projection_id)

        rows.append(
            {
                "policy": placement.policy_name,
                "timestep": timestep,
                "sensitivity_weighted_variance_proxy": float(weighted_proxy),
                # Backward-compatible aliases.
                "quality_proxy_delta_nll": float(weighted_proxy),
                # Backward-compatible alias.
                "weighted_noise_sum": float(weighted_proxy),
                "mean_weighted_tile_noise": (
                    weighted_noise_sum / total_shard_weight
                    if total_shard_weight > 0.0
                    else 0.0
                ),
                "reference_noise_std": float(reference_noise_std),
                "proxy_noise_scaling": "variance_ratio_squared",
                "shard_weight": float(total_shard_weight),
                "shards_on_faulted": shards_on_faulted,
                "shards_on_unavailable": shards_on_unavailable,
                "projections_on_faulted": len(projections_on_faulted),
                "projections_on_unavailable": len(projections_on_unavailable),
                "available_tiles": int(np.count_nonzero(available[timestep])),
                "faulted_tiles": int(np.count_nonzero(faulted[timestep])),
            }
        )
    return rows


def build_policy_summary(
    *,
    timestep_rows: Sequence[Mapping[str, Any]],
    placements: Mapping[str, Placement],
) -> list[dict[str, Any]]:
    """Aggregate Phase-3 proxy metrics by policy."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in timestep_rows:
        grouped[str(row["policy"])].append(row)

    summaries: list[dict[str, Any]] = []
    for policy, rows in sorted(grouped.items()):
        proxy = np.asarray(
            [float(row["sensitivity_weighted_variance_proxy"]) for row in rows],
            dtype=np.float64,
        )
        placement = placements[policy]
        summaries.append(
            {
                "policy": policy,
                "num_timesteps": len(rows),
                "sensitivity_weighted_variance_proxy_mean": float(proxy.mean()),
                "quality_proxy_delta_nll_mean": float(proxy.mean()),
                "sensitivity_weighted_variance_proxy_std": float(proxy.std(ddof=0)),
                "quality_proxy_delta_nll_std": float(proxy.std(ddof=0)),
                "sensitivity_weighted_variance_proxy_initial": float(proxy[0]),
                "quality_proxy_delta_nll_initial": float(proxy[0]),
                "sensitivity_weighted_variance_proxy_final": float(proxy[-1]),
                "quality_proxy_delta_nll_final": float(proxy[-1]),
                # Backward-compatible alias.
                "mean_sensitivity_weighted_tile_error": float(proxy.mean()),
                "max_shards_on_faulted": max(
                    int(row["shards_on_faulted"]) for row in rows
                ),
                "max_shards_on_unavailable": max(
                    int(row["shards_on_unavailable"]) for row in rows
                ),
                "capacity_utilization": placement.capacity_utilization,
                "available_capacity_utilization": (
                    placement.available_capacity_utilization
                ),
            }
        )
    return summaries
