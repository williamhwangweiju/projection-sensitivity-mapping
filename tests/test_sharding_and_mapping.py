from src.mapping.objective import placement_proxy
from src.mapping.placement import place_shards
from src.mapping.sharding import build_shards


def profile_rows():
    return [
        {
            "projection_id": "p_sensitive",
            "out_features": 6,
            "in_features": 6,
            "sensitivity_score_for_mapping": 10.0,
        },
        {
            "projection_id": "p_robust",
            "out_features": 6,
            "in_features": 6,
            "sensitivity_score_for_mapping": 1.0,
        },
        {
            "projection_id": "p_digital",
            "out_features": 6,
            "in_features": 6,
            "sensitivity_score_for_mapping": 100.0,
        },
    ]


def test_digital_projection_is_excluded_from_shards():
    shards = build_shards(profile_rows(), digital_projection_ids=["p_digital"], tier_rows=4, tier_cols=4)
    assert {shard.projection_id for shard in shards} == {"p_sensitive", "p_robust"}
    assert len(shards) == 8


def test_static_sensitivity_minimizes_separable_proxy():
    shards = build_shards(profile_rows(), digital_projection_ids=["p_digital"], tier_rows=4, tier_cols=4)
    noise = [0.01, 0.02, 0.04, 0.08]
    available = [True] * 4
    hardware = place_shards(shards, noise=noise, available=available, tiers_per_tile=2, policy="hardware_only", timestep=0, seed=42)
    sensitivity = place_shards(shards, noise=noise, available=available, tiers_per_tile=2, policy="static_sensitivity", timestep=0, seed=42)
    assert placement_proxy(sensitivity, variance=True) <= placement_proxy(hardware, variance=True)


def test_all_policies_cover_same_shards():
    shards = build_shards(profile_rows(), digital_projection_ids=["p_digital"], tier_rows=4, tier_cols=4)
    expected = {shard.shard_id for shard in shards}
    for policy in ("random", "sequential", "hardware_only", "static_sensitivity"):
        records = place_shards(shards, noise=[0.01, 0.02, 0.03, 0.04], available=[True] * 4, tiers_per_tile=2, policy=policy, timestep=0, seed=7)
        assert {record.shard_id for record in records} == expected


def test_hardware_only_is_reproducible_and_seed_dependent():
    shards = build_shards(profile_rows(), digital_projection_ids=["p_digital"], tier_rows=4, tier_cols=4)
    noise = [0.01, 0.02, 0.04, 0.08]
    available = [True] * 4
    kwargs = dict(noise=noise, available=available, tiers_per_tile=2, policy="hardware_only", timestep=0)
    first = place_shards(shards, seed=7, **kwargs)
    second = place_shards(shards, seed=7, **kwargs)
    assert [(r.shard_id, r.tile_id, r.tier_id) for r in first] == [
        (r.shard_id, r.tile_id, r.tier_id) for r in second
    ]
    mappings = {
        tuple((r.shard_id, r.tile_id, r.tier_id) for r in place_shards(shards, seed=s, **kwargs))
        for s in range(6)
    }
    assert len(mappings) > 1


def test_hardware_only_assignment_order_ignores_catalog_order():
    # With the catalog order, the sensitive projection's shards would always
    # occupy the quietest slots. Across seeds, the quietest slot must not be
    # monopolized by the most sensitive projection.
    shards = build_shards(profile_rows(), digital_projection_ids=["p_digital"], tier_rows=4, tier_cols=4)
    noise = [0.01, 0.02, 0.04, 0.08]
    available = [True] * 4
    quietest_occupants = set()
    for s in range(12):
        records = place_shards(
            shards, noise=noise, available=available, tiers_per_tile=2,
            policy="hardware_only", timestep=0, seed=s,
        )
        best = min(records, key=lambda r: (r.tile_noise_std, r.tile_id, r.tier_id))
        quietest_occupants.add(best.projection_id)
    assert quietest_occupants != {"p_sensitive"}


def test_fused_qkv_preserves_semantic_boundaries_and_matches_480_total():
    rows = []
    for block in range(12):
        rows.extend([
            {"projection_id": f"block_{block}/attn.c_attn", "out_features": 2304, "in_features": 768, "sensitivity_score_for_mapping": 1.0},
            {"projection_id": f"block_{block}/attn.c_proj", "out_features": 768, "in_features": 768, "sensitivity_score_for_mapping": 1.0},
            {"projection_id": f"block_{block}/mlp.c_fc", "out_features": 3072, "in_features": 768, "sensitivity_score_for_mapping": 1.0},
            {"projection_id": f"block_{block}/mlp.c_proj", "out_features": 768, "in_features": 3072, "sensitivity_score_for_mapping": 1.0},
        ])
    shards = build_shards(rows, digital_projection_ids=[], tier_rows=512, tier_cols=512)
    assert len(shards) == 480
    qkv = [s for s in shards if s.projection_id == "block_0/attn.c_attn"]
    assert len(qkv) == 12
    assert all(not (s.row_start < 768 < s.row_end) for s in qkv)
    assert all(not (s.row_start < 1536 < s.row_end) for s in qkv)


def test_negative_sensitivity_is_floored_for_mapping() -> None:
    rows = [
        {
            "projection_id": "block_0/attn.c_proj",
            "out_features": 4,
            "in_features": 4,
            "sensitivity_score_for_mapping": -0.25,
        }
    ]
    shards = build_shards(
        rows,
        digital_projection_ids=[],
        tier_rows=2,
        tier_cols=2,
        sensitivity_floor=0.0,
    )
    assert shards
    assert all(shard.sensitivity == 0.0 for shard in shards)
    assert all(shard.importance == 0.0 for shard in shards)
