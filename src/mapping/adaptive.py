"""Migration-aware adaptive placement with fixed offline projection sensitivities."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from src.mapping.objective import migration_cost, placement_proxy
from src.mapping.placement import PlacementRecord, place_shards
from src.mapping.sharding import ProjectionShard


def refresh_placement_state(
    records: Iterable[PlacementRecord],
    *,
    timestep: int,
    noise: Sequence[float],
    available: Sequence[bool],
    unavailable_noise_std: float,
    policy: str = "adaptive_sensitivity",
) -> list[PlacementRecord]:
    """Refresh tile-state values without changing physical assignments."""
    output: list[PlacementRecord] = []
    for row in records:
        tile_id = int(row.tile_id)
        current_noise = (
            float(noise[tile_id]) if bool(available[tile_id]) else float(unavailable_noise_std)
        )
        output.append(
            replace(
                row,
                policy=policy,
                timestep=int(timestep),
                tile_noise_std=current_noise,
            )
        )
    return output


def adaptive_step(
    shards: Iterable[ProjectionShard],
    current: Iterable[PlacementRecord],
    *,
    timestep: int,
    noise: Sequence[float],
    available: Sequence[bool],
    tiers_per_tile: int,
    seed: int,
    unavailable_noise_std: float,
    minimum_relative_improvement: float,
    migration_penalty_per_moved_weight_fraction: float,
    max_moved_weight_fraction: float,
    cooldown_satisfied: bool,
) -> tuple[list[PlacementRecord], dict[str, float | bool | str]]:
    """Evaluate one full-remap candidate and accept it only when worthwhile."""
    shard_list = list(shards)
    current_refreshed = refresh_placement_state(
        current,
        timestep=timestep,
        noise=noise,
        available=available,
        unavailable_noise_std=unavailable_noise_std,
    )
    candidate = place_shards(
        shard_list,
        noise=noise,
        available=available,
        tiers_per_tile=tiers_per_tile,
        policy="adaptive_sensitivity",
        timestep=timestep,
        seed=seed,
    )
    current_proxy = placement_proxy(current_refreshed, variance=True)
    candidate_proxy = placement_proxy(candidate, variance=True)
    improvement = current_proxy - candidate_proxy
    relative_improvement = improvement / max(abs(current_proxy), 1e-30)
    migration = migration_cost(current_refreshed, candidate)
    total_weights = sum(shard.weight_count for shard in shard_list)
    moved_fraction = float(migration["moved_weights"]) / max(total_weights, 1)
    penalized_gain = relative_improvement - (
        migration_penalty_per_moved_weight_fraction * moved_fraction
    )
    accept = (
        cooldown_satisfied
        and relative_improvement >= minimum_relative_improvement
        and moved_fraction <= max_moved_weight_fraction
        and penalized_gain > 0.0
    )
    selected = candidate if accept else current_refreshed
    diagnostics: dict[str, float | bool | str] = {
        "accepted": accept,
        "reason": (
            "accepted" if accept else
            "cooldown" if not cooldown_satisfied else
            "movement_cap" if moved_fraction > max_moved_weight_fraction else
            "insufficient_penalized_gain"
        ),
        "current_proxy_variance": current_proxy,
        "candidate_proxy_variance": candidate_proxy,
        "absolute_proxy_improvement": improvement,
        "relative_proxy_improvement": relative_improvement,
        "moved_shards": migration["moved_shards"],
        "moved_weights": migration["moved_weights"],
        "moved_bytes_fp32": migration["moved_bytes_fp32"],
        "moved_weight_fraction": moved_fraction,
        "migration_penalty": migration_penalty_per_moved_weight_fraction * moved_fraction,
        "penalized_gain": penalized_gain,
    }
    return selected, diagnostics
