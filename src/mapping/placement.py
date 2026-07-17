"""Static and dynamic capacity-aware shard placement policies."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

import numpy as np

from src.mapping.sharding import ProjectionShard


@dataclass(frozen=True)
class PhysicalSlot:
    tile_id: int
    tier_id: int
    noise_std: float
    available: bool


@dataclass(frozen=True)
class PlacementRecord:
    policy: str
    timestep: int
    shard_id: str
    projection_id: str
    shard_index: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    weight_count: int
    shard_weight: float
    sensitivity: float
    importance: float
    tile_id: int
    tier_id: int
    tile_noise_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_slots(noise: Sequence[float], available: Sequence[bool], tiers_per_tile: int) -> list[PhysicalSlot]:
    slots: list[PhysicalSlot] = []
    for tile_id, (tile_noise, is_available) in enumerate(zip(noise, available)):
        if not bool(is_available):
            continue
        for tier_id in range(tiers_per_tile):
            slots.append(PhysicalSlot(tile_id, tier_id, float(tile_noise), True))
    return slots


def place_shards(
    shards: Iterable[ProjectionShard],
    *,
    noise: Sequence[float],
    available: Sequence[bool],
    tiers_per_tile: int,
    policy: str,
    timestep: int,
    seed: int,
) -> list[PlacementRecord]:
    shard_list = list(shards)
    slots = build_slots(noise, available, tiers_per_tile)
    if len(shard_list) > len(slots):
        raise ValueError(f"Analog set requires {len(shard_list)} tiers but only {len(slots)} are available.")
    if policy == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(slots)
        ordered_shards = list(shard_list)
    elif policy == "sequential":
        slots.sort(key=lambda slot: (slot.tile_id, slot.tier_id))
        ordered_shards = list(shard_list)
    elif policy == "hardware_only":
        slots.sort(key=lambda slot: (slot.noise_std, slot.tile_id, slot.tier_id))
        # The catalog order enumerates block 0 first, which correlates with
        # sensitivity and would make this baseline accidentally
        # sensitivity-aware. A seeded permutation keeps the assignment order
        # independent of shard importance while remaining reproducible.
        permutation = np.random.default_rng(seed).permutation(len(shard_list))
        ordered_shards = [shard_list[index] for index in permutation]
    elif policy == "static_sensitivity":
        slots.sort(key=lambda slot: (slot.noise_std, slot.tile_id, slot.tier_id))
        ordered_shards = sorted(shard_list, key=lambda shard: (-shard.importance, shard.shard_id))
    else:
        raise ValueError(f"Unknown placement policy: {policy}")
    records: list[PlacementRecord] = []
    for shard, slot in zip(ordered_shards, slots):
        records.append(
            PlacementRecord(
                policy=policy,
                timestep=int(timestep),
                shard_id=shard.shard_id,
                projection_id=shard.projection_id,
                shard_index=shard.shard_index,
                row_start=shard.row_start,
                row_end=shard.row_end,
                col_start=shard.col_start,
                col_end=shard.col_end,
                weight_count=shard.weight_count,
                shard_weight=shard.shard_weight,
                sensitivity=shard.sensitivity,
                importance=shard.importance,
                tile_id=slot.tile_id,
                tier_id=slot.tier_id,
                tile_noise_std=slot.noise_std,
            )
        )
    validate_placement(records, len(shard_list))
    return records


def validate_placement(records: Iterable[PlacementRecord], expected_shards: int | None = None) -> None:
    rows = list(records)
    if expected_shards is not None and len(rows) != expected_shards:
        raise ValueError("Placement shard count mismatch.")
    if len({row.shard_id for row in rows}) != len(rows):
        raise ValueError("A shard was placed more than once.")
    if len({(row.tile_id, row.tier_id) for row in rows}) != len(rows):
        raise ValueError("A physical tier was reused.")
