"""Physical 3D-CIM slot assignment policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from src.mapping.sharding import ProjectionShard


@dataclass(frozen=True)
class PhysicalSlot:
    tile_id: int
    tier_id: int
    noise_std: float
    available: bool


@dataclass(frozen=True)
class Assignment:
    policy: str
    shard: ProjectionShard
    tile_id: int
    tier_id: int
    mapping_timestep: int
    tile_noise_std_at_mapping: float

    def to_dict(self) -> dict[str, Any]:
        row = self.shard.to_dict()
        row.update(
            {
                "policy": self.policy,
                "tile_id": self.tile_id,
                "tier_id": self.tier_id,
                "mapping_timestep": self.mapping_timestep,
                "tile_noise_std_at_mapping": self.tile_noise_std_at_mapping,
            }
        )
        return row


def build_slots(
    noise_std: np.ndarray,
    available: np.ndarray,
    tiers_per_tile: int,
) -> list[PhysicalSlot]:
    if noise_std.ndim != 1 or available.ndim != 1 or noise_std.shape != available.shape:
        raise ValueError("noise_std and available must be equal-length 1-D arrays.")
    if tiers_per_tile <= 0:
        raise ValueError("tiers_per_tile must be positive.")
    slots = []
    for tile_id in range(noise_std.size):
        for tier_id in range(tiers_per_tile):
            slots.append(
                PhysicalSlot(
                    tile_id=tile_id,
                    tier_id=tier_id,
                    noise_std=float(noise_std[tile_id]),
                    available=bool(available[tile_id]),
                )
            )
    return slots


def _canonical_shards(shards: Iterable[ProjectionShard]) -> list[ProjectionShard]:
    return sorted(
        shards,
        key=lambda shard: (
            shard.block_index,
            shard.projection_id,
            shard.out_start,
            shard.in_start,
            shard.shard_id,
        ),
    )


def _available_slots(slots: Iterable[PhysicalSlot]) -> list[PhysicalSlot]:
    return [slot for slot in slots if slot.available]


def assign_policy(
    policy: str,
    shards: Iterable[ProjectionShard],
    slots: Iterable[PhysicalSlot],
    *,
    seed: int,
    mapping_timestep: int,
) -> list[Assignment]:
    shard_list = _canonical_shards(shards)
    slot_list = _available_slots(slots)
    if len(slot_list) < len(shard_list):
        raise RuntimeError(
            f"Only {len(slot_list)} available physical tiers for "
            f"{len(shard_list)} shards."
        )

    canonical_slots = sorted(slot_list, key=lambda slot: (slot.tile_id, slot.tier_id))
    if policy == "sequential":
        ordered_shards = shard_list
        ordered_slots = canonical_slots
    elif policy == "random":
        rng = np.random.default_rng(int(seed))
        ordered_shards = shard_list
        indices = rng.permutation(len(canonical_slots))
        ordered_slots = [canonical_slots[int(index)] for index in indices]
    elif policy == "hardware_only":
        ordered_shards = shard_list
        ordered_slots = sorted(
            canonical_slots,
            key=lambda slot: (slot.noise_std**2, slot.tile_id, slot.tier_id),
        )
    elif policy == "static_sensitivity":
        # Rearrangement-optimal assignment for the separable Phase-3 proxy:
        # descending shard importance paired with ascending tile variance.
        ordered_shards = sorted(
            shard_list,
            key=lambda shard: (
                -shard.importance,
                -shard.sensitivity,
                shard.block_index,
                shard.projection_id,
                shard.out_start,
                shard.in_start,
            ),
        )
        ordered_slots = sorted(
            canonical_slots,
            key=lambda slot: (slot.noise_std**2, slot.tile_id, slot.tier_id),
        )
    else:
        raise ValueError(f"Unsupported placement policy: {policy!r}")

    assignments = [
        Assignment(
            policy=policy,
            shard=shard,
            tile_id=slot.tile_id,
            tier_id=slot.tier_id,
            mapping_timestep=int(mapping_timestep),
            tile_noise_std_at_mapping=slot.noise_std,
        )
        for shard, slot in zip(ordered_shards, ordered_slots)
    ]
    validate_assignments(assignments, shard_list)
    return assignments


def validate_assignments(
    assignments: Iterable[Assignment],
    expected_shards: Iterable[ProjectionShard],
) -> None:
    rows = list(assignments)
    expected_ids = {shard.shard_id for shard in expected_shards}
    actual_ids = [row.shard.shard_id for row in rows]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise RuntimeError("Placement does not cover every shard exactly once.")
    slots = [(row.tile_id, row.tier_id) for row in rows]
    if len(slots) != len(set(slots)):
        raise RuntimeError("Placement assigns more than one shard to a physical tier.")
