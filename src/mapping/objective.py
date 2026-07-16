"""Phase-3 sensitivity-weighted variance proxy."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from src.mapping.placement import Assignment


def evaluate_placement(
    assignments: Iterable[Assignment],
    tile_noise_std: np.ndarray,
    *,
    reference_noise_std: float,
    faulted: np.ndarray | None = None,
    available: np.ndarray | None = None,
) -> dict[str, float | int]:
    if tile_noise_std.ndim != 1:
        raise ValueError("tile_noise_std must be one-dimensional.")
    if reference_noise_std <= 0:
        raise ValueError("reference_noise_std must be positive.")
    weighted_variance = 0.0
    weighted_noise = 0.0
    total_shard_weight = 0.0
    shards_on_faulted = 0
    shards_on_unavailable = 0
    projection_ids_on_faulted: set[str] = set()
    for assignment in assignments:
        tile_id = assignment.tile_id
        normalized = float(tile_noise_std[tile_id]) / reference_noise_std
        importance = assignment.shard.importance
        weighted_variance += importance * normalized**2
        weighted_noise += importance * normalized
        total_shard_weight += assignment.shard.shard_weight
        if faulted is not None and bool(faulted[tile_id]):
            shards_on_faulted += 1
            projection_ids_on_faulted.add(assignment.shard.projection_id)
        if available is not None and not bool(available[tile_id]):
            shards_on_unavailable += 1
    return {
        "sensitivity_weighted_variance_proxy": float(weighted_variance),
        "sensitivity_weighted_noise_proxy": float(weighted_noise),
        "total_shard_weight": float(total_shard_weight),
        "shards_on_faulted_tiles": int(shards_on_faulted),
        "shards_on_unavailable_tiles": int(shards_on_unavailable),
        "projections_on_faulted_tiles": int(len(projection_ids_on_faulted)),
    }
