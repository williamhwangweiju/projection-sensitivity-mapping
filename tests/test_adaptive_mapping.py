from src.mapping.adaptive import adaptive_step
from src.mapping.placement import place_shards
from src.mapping.sharding import build_shards


def test_adaptive_step_accepts_large_proxy_improvement():
    rows = [
        {"projection_id": "sensitive", "out_features": 2, "in_features": 2, "sensitivity_score_for_mapping": 10.0},
        {"projection_id": "robust", "out_features": 2, "in_features": 2, "sensitivity_score_for_mapping": 1.0},
    ]
    shards = build_shards(rows, digital_projection_ids=[], tier_rows=2, tier_cols=2)
    initial = place_shards(
        shards,
        noise=[0.01, 0.08],
        available=[True, True],
        tiers_per_tile=1,
        policy="static_sensitivity",
        timestep=0,
        seed=42,
    )
    # Reverse which tile is reliable. A remap should recover the sensitive shard.
    updated, diagnostics = adaptive_step(
        shards,
        initial,
        timestep=1,
        noise=[0.08, 0.01],
        available=[True, True],
        tiers_per_tile=1,
        seed=42,
        unavailable_noise_std=0.1,
        minimum_relative_improvement=0.0,
        migration_penalty_per_moved_weight_fraction=0.0,
        max_moved_weight_fraction=1.0,
        cooldown_satisfied=True,
    )
    assert diagnostics["accepted"] is True
    sensitive = next(row for row in updated if row.projection_id == "sensitive")
    assert sensitive.tile_id == 1


def test_adaptive_step_respects_cooldown():
    rows = [{"projection_id": "p", "out_features": 2, "in_features": 2, "sensitivity_score_for_mapping": 1.0}]
    shards = build_shards(rows, digital_projection_ids=[], tier_rows=2, tier_cols=2)
    initial = place_shards(shards, noise=[0.01], available=[True], tiers_per_tile=1, policy="static_sensitivity", timestep=0, seed=1)
    _, diagnostics = adaptive_step(
        shards,
        initial,
        timestep=1,
        noise=[0.02],
        available=[True],
        tiers_per_tile=1,
        seed=1,
        unavailable_noise_std=0.1,
        minimum_relative_improvement=0.0,
        migration_penalty_per_moved_weight_fraction=0.0,
        max_moved_weight_fraction=1.0,
        cooldown_satisfied=False,
    )
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "cooldown"
