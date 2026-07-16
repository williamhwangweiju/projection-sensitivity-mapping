"""Placement quality proxies and migration costs."""
from __future__ import annotations

from typing import Iterable, Mapping, Any

from src.mapping.placement import PlacementRecord


def placement_proxy(records: Iterable[PlacementRecord], *, variance: bool = True) -> float:
    total = 0.0
    for row in records:
        noise_term = row.tile_noise_std ** 2 if variance else row.tile_noise_std
        total += row.importance * noise_term
    return total


def migration_cost(
    previous: Iterable[PlacementRecord], current: Iterable[PlacementRecord]
) -> dict[str, float]:
    old = {row.shard_id: row for row in previous}
    new = {row.shard_id: row for row in current}
    if old.keys() != new.keys():
        raise ValueError("Migration comparisons require identical analog shard sets.")
    moved = [
        shard_id for shard_id in old
        if (old[shard_id].tile_id, old[shard_id].tier_id) != (new[shard_id].tile_id, new[shard_id].tier_id)
    ]
    moved_weights = sum(new[shard_id].weight_count for shard_id in moved)
    return {
        "moved_shards": float(len(moved)),
        "moved_weights": float(moved_weights),
        "moved_bytes_fp32": float(4 * moved_weights),
    }


def records_from_dicts(rows: Iterable[Mapping[str, Any]]) -> list[PlacementRecord]:
    return [PlacementRecord(**{field: row[field] for field in PlacementRecord.__dataclass_fields__}) for row in rows]
