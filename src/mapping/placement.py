"""Placement representations for Phase 3 mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .sharding import MappedShardRecord


@dataclass(frozen=True, slots=True)
class ShardPlacement:
    """Concrete tile/tier assignment for one projection shard."""

    shard_id: str
    tile_id: int
    tier_start: int
    tiers_used: int


@dataclass(slots=True)
class Placement:
    """Full static placement assignment."""

    policy_name: str
    assignments: dict[str, ShardPlacement]
    used_tiers_by_tile: dict[int, int]
    usable_tiers_by_tile: dict[int, int]
    available_tiles: frozenset[int] | None = None

    @property
    def capacity_utilization(self) -> float:
        total_used = sum(self.used_tiers_by_tile.values())
        total_usable = sum(self.usable_tiers_by_tile.values())
        if total_usable == 0:
            return 0.0
        return total_used / total_usable

    @property
    def available_capacity_utilization(self) -> float:
        """Capacity utilization relative only to tiles available at placement time."""
        if self.available_tiles is None:
            return self.capacity_utilization

        total_used = sum(self.used_tiers_by_tile.values())
        total_available = sum(
            self.usable_tiers_by_tile[tile_id]
            for tile_id in self.available_tiles
        )
        if total_available == 0:
            return 0.0
        return total_used / total_available


def build_placement_from_mapped_shards(
    *,
    policy_name: str,
    records: Sequence[MappedShardRecord],
    tiers_per_tile: int,
    num_tiles: int,
    available_tiles: set[int],
) -> Placement:
    """Build a Placement from the actual 3D-CIM mapping records."""
    assignments: dict[str, ShardPlacement] = {}
    used_tier_slots_by_tile: dict[int, set[int]] = {
        tile_id: set()
        for tile_id in range(num_tiles)
    }

    for record in records:
        shard_id = record.shard.shard_id
        assignments[shard_id] = ShardPlacement(
            shard_id=shard_id,
            tile_id=record.tile_id,
            tier_start=record.tier_start,
            tiers_used=record.tiers_used,
        )
        used_tier_slots_by_tile.setdefault(record.tile_id, set()).add(
            record.tier_start
        )

    usable_tiers_by_tile = {
        tile_id: tiers_per_tile
        for tile_id in range(num_tiles)
    }
    used_tiers_by_tile = {
        tile_id: len(used_tier_slots_by_tile.get(tile_id, set()))
        for tile_id in range(num_tiles)
    }

    return Placement(
        policy_name=policy_name,
        assignments=assignments,
        used_tiers_by_tile=used_tiers_by_tile,
        usable_tiers_by_tile=usable_tiers_by_tile,
        available_tiles=frozenset(available_tiles),
    )


def mapped_shards_with_placement_to_rows(
    *,
    policy_name: str,
    records: Sequence[MappedShardRecord],
    placement: Placement,
) -> list[dict[str, Any]]:
    """Build the existing combined projection_shards.csv rows."""
    rows: list[dict[str, Any]] = []
    for record in records:
        shard = record.shard
        assignment = placement.assignments[shard.shard_id]
        rows.append(
            {
                "policy": policy_name,
                "shard_id": shard.shard_id,
                "projection_id": shard.projection_id,
                "block_id": shard.block_id,
                "projection_name": shard.projection_name,
                "execution_index": shard.execution_index,
                "row_shard_index": shard.row_shard_index,
                "col_shard_index": shard.col_shard_index,
                "tile_id": assignment.tile_id,
                "tier_start": assignment.tier_start,
                "tiers_used": assignment.tiers_used,
                "weights_in_shard": shard.weights_in_shard,
                "weights_in_projection": shard.weights_in_projection,
                "shard_weight": shard.shard_weight,
                "sensitivity_score": shard.sensitivity_score,
                "shard_importance": shard.shard_weight * shard.sensitivity_score,
            }
        )
    return rows
