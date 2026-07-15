"""Placement representations for Phase 3 mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .sharding import MappedShardRecord


@dataclass(frozen=True, slots=True)
class ShardPlacement:
    """Concrete physical tile/tier assignment for one projection shard."""

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
        return 0.0 if total_usable == 0 else total_used / total_usable

    @property
    def available_capacity_utilization(self) -> float:
        if self.available_tiles is None:
            return self.capacity_utilization
        total_used = sum(self.used_tiers_by_tile.values())
        total_available = sum(
            self.usable_tiers_by_tile[tile_id]
            for tile_id in self.available_tiles
        )
        return 0.0 if total_available == 0 else total_used / total_available


def build_placement_from_mapped_shards(
    *,
    policy_name: str,
    records: Sequence[MappedShardRecord],
    tiers_per_tile: int,
    num_tiles: int,
    available_tiles: set[int],
) -> Placement:
    """Build a validated Placement from authoritative mapping records."""

    assignments: dict[str, ShardPlacement] = {}
    used_slots: dict[int, set[int]] = {
        tile_id: set() for tile_id in range(num_tiles)
    }

    for record in records:
        if record.shard.shard_id in assignments:
            raise ValueError(f"Duplicate shard ID: {record.shard.shard_id}")
        if not 0 <= record.tile_id < num_tiles:
            raise ValueError(f"Invalid tile ID {record.tile_id}.")
        if not 0 <= record.tier_start < tiers_per_tile:
            raise ValueError(f"Invalid tier ID {record.tier_start}.")
        if record.tiers_used != 1:
            raise ValueError("Each extracted crossbar shard must use one tier.")
        if record.tier_start in used_slots[record.tile_id]:
            raise ValueError(
                f"Tile {record.tile_id}, tier {record.tier_start} is reused."
            )

        assignments[record.shard.shard_id] = ShardPlacement(
            shard_id=record.shard.shard_id,
            tile_id=record.tile_id,
            tier_start=record.tier_start,
            tiers_used=record.tiers_used,
        )
        used_slots[record.tile_id].add(record.tier_start)

    usable = {tile_id: tiers_per_tile for tile_id in range(num_tiles)}
    used = {tile_id: len(used_slots[tile_id]) for tile_id in range(num_tiles)}
    return Placement(
        policy_name=policy_name,
        assignments=assignments,
        used_tiers_by_tile=used,
        usable_tiers_by_tile=usable,
        available_tiles=frozenset(available_tiles),
    )


def mapped_shards_with_placement_to_rows(
    *,
    policy_name: str,
    records: Sequence[MappedShardRecord],
    placement: Placement,
) -> list[dict[str, Any]]:
    """Build the combined projection_shards.csv rows."""

    rows: list[dict[str, Any]] = []
    for record in records:
        shard = record.shard
        assignment = placement.assignments[shard.shard_id]
        rows.append(
            {
                "policy": policy_name,
                "module_name": record.module_name,
                "shard_id": shard.shard_id,
                "projection_id": shard.projection_id,
                "block_id": shard.block_id,
                "projection_name": shard.projection_name,
                "execution_index": shard.execution_index,
                "row_shard_index": shard.row_shard_index,
                "col_shard_index": shard.col_shard_index,
                "sim_input_start": shard.sim_input_start,
                "sim_input_end": shard.sim_input_end,
                "sim_output_start": shard.sim_output_start,
                "sim_output_end": shard.sim_output_end,
                "tile_id": assignment.tile_id,
                "tier_start": assignment.tier_start,
                "tiers_used": assignment.tiers_used,
                "used_rows": record.used_rows,
                "used_cols": record.used_cols,
                "weights_in_shard": shard.weights_in_shard,
                "weights_in_projection": shard.weights_in_projection,
                "shard_weight": shard.shard_weight,
                "sensitivity_score": shard.sensitivity_score,
                "sensitivity_score_unit": shard.sensitivity_score_unit,
                "sensitivity_reference_noise_std": (
                    shard.sensitivity_reference_noise_std
                ),
                "shard_importance": (
                    shard.shard_weight * shard.sensitivity_score
                ),
            }
        )
    return rows
