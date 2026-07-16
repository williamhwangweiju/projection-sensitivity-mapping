from __future__ import annotations

import numpy as np

from src.mapping.objective import evaluate_placement
from src.mapping.placement import assign_policy, build_slots
from src.mapping.sharding import ProjectionSpec, build_gpt2_shards


def specs() -> list[ProjectionSpec]:
    result = []
    shapes = {
        "attn.c_attn": (2304, 768),
        "attn.c_proj": (768, 768),
        "mlp.c_fc": (3072, 768),
        "mlp.c_proj": (768, 3072),
    }
    for block in range(12):
        for index, (name, shape) in enumerate(shapes.items()):
            result.append(
                ProjectionSpec(
                    projection_id=f"block_{block}/{name}",
                    block_index=block,
                    projection_name=name,
                    hf_module_path=f"transformer.h.{block}.{name}",
                    out_features=shape[0],
                    in_features=shape[1],
                    sensitivity=float(48 - (block * 4 + index)),
                )
            )
    return result


def test_gpt2_small_has_480_shards_and_exact_projection_coverage() -> None:
    shards = build_gpt2_shards(specs(), 512)
    assert len(shards) == 480
    for spec in specs():
        selected = [shard for shard in shards if shard.projection_id == spec.projection_id]
        assert sum(shard.weight_count for shard in selected) == (
            spec.out_features * spec.in_features
        )
        assert abs(sum(shard.shard_weight for shard in selected) - 1.0) < 1e-12


def test_static_sensitivity_minimizes_separable_proxy() -> None:
    shards = build_gpt2_shards(specs(), 512)
    tile_noise = np.linspace(0.005, 0.060, 72)
    slots = build_slots(tile_noise, np.ones(72, dtype=bool), 8)
    values = {}
    for policy in ("random", "sequential", "hardware_only", "static_sensitivity"):
        assignments = assign_policy(
            policy, shards, slots, seed=42, mapping_timestep=0
        )
        values[policy] = evaluate_placement(
            assignments,
            tile_noise,
            reference_noise_std=0.023,
        )["sensitivity_weighted_variance_proxy"]
    assert values["static_sensitivity"] <= values["random"]
    assert values["static_sensitivity"] <= values["sequential"]
    assert values["static_sensitivity"] <= values["hardware_only"]
