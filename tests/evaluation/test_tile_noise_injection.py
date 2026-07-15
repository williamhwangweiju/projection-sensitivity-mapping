"""Tests for tile_noise_injection and noise_materialization.

Covers:
- Zero-sigma injection leaves weights unchanged.
- Non-zero sigma changes the expected weight slice.
- Weights are restored exactly after NoisedModelContext exits.
- Single-tile test: only slices on that tile change.
- Uniform-sigma policy-invariance test.
- Checksum verification.
- Determinism: same seeds → identical noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.noise_materialization import (
    build_sigma_map,
    compute_noise_reference_scale,
    generate_paired_noise_tensors,
)
from src.evaluation.placement_to_gpt2 import (
    GPT2_SMALL_HIDDEN,
    build_shard_assignments,
)
from src.evaluation.schemas import GPT2ShardAssignment
from src.evaluation.tile_noise_injection import (
    NoisedModelContext,
    apply_tile_noise,
    compute_weight_checksums,
    restore_projection_weights,
    save_projection_weights,
)


# ---------------------------------------------------------------------------
# Minimal model and assignment fixtures
# ---------------------------------------------------------------------------

class SimpleLinearBlock(nn.Module):
    """Tiny model with one Conv1D-style projection (as nn.Linear) for testing."""

    def __init__(self, in_f: int = 16, out_f: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _fake_assignment(
    *,
    shard_id: str = "test#r0c0",
    hf_module_path: str = "proj",
    tile_id: int = 0,
    tier_id: int = 0,
    canonical_row_start: int = 0,
    canonical_row_end: int = 16,
    canonical_col_start: int = 0,
    canonical_col_end: int = 16,
    weights_in_shard: int = 256,
    weights_in_projection: int = 256,
) -> GPT2ShardAssignment:
    return GPT2ShardAssignment(
        shard_id=shard_id,
        projection_id="test/proj",
        hf_module_path=hf_module_path,
        sim_module_path="layers.0.out_proj",
        sim_layer=0,
        qkv_component=None,
        canonical_row_start=canonical_row_start,
        canonical_row_end=canonical_row_end,
        canonical_col_start=canonical_col_start,
        canonical_col_end=canonical_col_end,
        tile_id=tile_id,
        tier_id=tier_id,
        policy="test",
        placement_seed=0,
        weights_in_shard=weights_in_shard,
        weights_in_projection=weights_in_projection,
        shard_weight=1.0,
        sensitivity_score=0.0,
    )


# ---------------------------------------------------------------------------
# Noise reference scale
# ---------------------------------------------------------------------------

class TestNoiseReferenceScale:
    def test_weight_std_method(self) -> None:
        W = torch.randn(32, 16) * 2.0
        r = compute_noise_reference_scale(W, method="weight_std")
        assert abs(r - float(W.std())) < 0.01

    def test_clip_threshold_method(self) -> None:
        W = torch.randn(32, 16)
        r_std = compute_noise_reference_scale(W, method="weight_std")
        r_clip = compute_noise_reference_scale(W, method="clip_threshold", clip_sigma=2.5)
        assert abs(r_clip - 2.5 * r_std) < 1e-6

    def test_floor_for_near_zero_weights(self) -> None:
        W = torch.zeros(8, 8)
        r = compute_noise_reference_scale(W, method="weight_std")
        assert r >= 1e-8


# ---------------------------------------------------------------------------
# Sigma map
# ---------------------------------------------------------------------------

class TestBuildSigmaMap:
    def test_zero_tile_noise_gives_zero_sigma(self) -> None:
        assignments = [_fake_assignment()]
        sigma_maps = build_sigma_map(
            assignments,
            tile_noise_at_timestep={0: 0.0},
            noise_reference_scales={"proj": 1.0},
        )
        assert "proj" in sigma_maps
        assert float(sigma_maps["proj"].abs().max()) == 0.0

    def test_nonzero_tile_noise_scales_correctly(self) -> None:
        assignments = [_fake_assignment(tile_id=5)]
        r_p = 2.0
        sigma_norm = 0.1
        sigma_maps = build_sigma_map(
            assignments,
            tile_noise_at_timestep={5: sigma_norm},
            noise_reference_scales={"proj": r_p},
        )
        expected = r_p * sigma_norm
        assert abs(float(sigma_maps["proj"][0, 0]) - expected) < 1e-6

    def test_partial_coverage_zeros_uncovered(self) -> None:
        # Shard covers only [0:8, 0:8] of a 16×16 map
        a = _fake_assignment(
            canonical_row_start=0, canonical_row_end=8,
            canonical_col_start=0, canonical_col_end=8,
        )
        sigma_maps = build_sigma_map(
            [a],
            tile_noise_at_timestep={0: 1.0},
            noise_reference_scales={"proj": 1.0},
        )
        cov = sigma_maps["proj"]
        assert float(cov[0, 0]) == 1.0
        assert float(cov[8, 8]) == 0.0 if cov.shape[0] > 8 else True


# ---------------------------------------------------------------------------
# Paired noise tensors
# ---------------------------------------------------------------------------

class TestGeneratePairedNoiseTensors:
    def test_deterministic(self) -> None:
        assignments = [_fake_assignment()]
        Z1 = generate_paired_noise_tensors(assignments, seed=42)
        Z2 = generate_paired_noise_tensors(assignments, seed=42)
        assert torch.allclose(Z1["proj"], Z2["proj"])

    def test_different_seeds(self) -> None:
        assignments = [_fake_assignment()]
        Z1 = generate_paired_noise_tensors(assignments, seed=42)
        Z2 = generate_paired_noise_tensors(assignments, seed=99)
        assert not torch.allclose(Z1["proj"], Z2["proj"])

    def test_shape_matches_shard_extent(self) -> None:
        a = _fake_assignment(
            canonical_row_start=0, canonical_row_end=20,
            canonical_col_start=0, canonical_col_end=10,
        )
        Z = generate_paired_noise_tensors([a], seed=0)
        assert Z["proj"].shape == (20, 10)


# ---------------------------------------------------------------------------
# Zero-noise injection leaves weights unchanged
# ---------------------------------------------------------------------------

class TestZeroNoiseInjection:
    def test_zero_sigma_does_not_change_weights(self) -> None:
        model = SimpleLinearBlock(in_f=16, out_f=32)
        W_before = model.proj.weight.data.clone()

        a = _fake_assignment(
            hf_module_path="proj",
            canonical_row_start=0, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )
        sigma_maps = {"proj": torch.zeros(32, 16)}
        Z_by_module = generate_paired_noise_tensors([a], seed=0)

        apply_tile_noise(model, assignments=[a], sigma_maps=sigma_maps, Z_by_module=Z_by_module)

        assert torch.allclose(model.proj.weight.data, W_before, atol=1e-7)


# ---------------------------------------------------------------------------
# Non-zero noise changes weights in expected slice
# ---------------------------------------------------------------------------

class TestNonZeroNoiseInjection:
    def test_noise_changes_only_covered_slice(self) -> None:
        model = SimpleLinearBlock(in_f=16, out_f=32)
        W_before = model.proj.weight.data.clone()

        # Cover only rows [0:8] of the 32×16 canonical weight
        a = _fake_assignment(
            hf_module_path="proj",
            canonical_row_start=0, canonical_row_end=8,
            canonical_col_start=0, canonical_col_end=16,
        )
        sigma_maps = {"proj": torch.zeros(32, 16)}
        sigma_maps["proj"][0:8, :] = 1.0  # large noise in covered slice

        # All-ones Z
        Z_by_module = {"proj": torch.ones(32, 16)}

        saved = save_projection_weights(model, ["proj"])
        apply_tile_noise(
            model, assignments=[a], sigma_maps=sigma_maps, Z_by_module=Z_by_module,
            saved_weights=saved,
        )

        W_after = model.proj.weight.data.clone()

        # Covered slice (canonical rows [0:8]) must differ from original
        # nn.Linear stores W as [out, in] = canonical directly
        W_canonical_before = W_before  # for nn.Linear: canonical = W directly
        W_canonical_after = W_after
        covered_diff = (W_canonical_after[:8, :] - W_canonical_before[:8, :]).abs()
        uncovered_diff = (W_canonical_after[8:, :] - W_canonical_before[8:, :]).abs()
        assert covered_diff.max() > 0.5
        assert uncovered_diff.max() < 1e-6


# ---------------------------------------------------------------------------
# Restoration test
# ---------------------------------------------------------------------------

class TestWeightRestoration:
    def test_context_manager_restores_weights(self) -> None:
        model = SimpleLinearBlock(in_f=16, out_f=32)
        W_before = model.proj.weight.data.clone()

        a = _fake_assignment(
            hf_module_path="proj",
            canonical_row_start=0, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )
        sigma_maps = {"proj": torch.ones(32, 16) * 2.0}
        Z_by_module = generate_paired_noise_tensors([a], seed=0)

        with NoisedModelContext(model, [a], sigma_maps, Z_by_module):
            # Inside: weights should be modified
            W_inside = model.proj.weight.data.clone()
            assert not torch.allclose(W_before, W_inside, atol=1e-5)

        # After exit: weights must be exactly restored
        W_after = model.proj.weight.data.clone()
        assert torch.allclose(W_before, W_after, atol=1e-7), \
            "Weights not exactly restored after NoisedModelContext exit."

    def test_restoration_on_exception(self) -> None:
        model = SimpleLinearBlock(in_f=16, out_f=32)
        W_before = model.proj.weight.data.clone()

        a = _fake_assignment(
            hf_module_path="proj",
            canonical_row_start=0, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )
        sigma_maps = {"proj": torch.ones(32, 16)}
        Z_by_module = generate_paired_noise_tensors([a], seed=0)

        try:
            with NoisedModelContext(model, [a], sigma_maps, Z_by_module):
                raise RuntimeError("Simulated error during inference")
        except RuntimeError:
            pass  # expected

        W_after = model.proj.weight.data.clone()
        assert torch.allclose(W_before, W_after, atol=1e-7)


# ---------------------------------------------------------------------------
# Single-tile test: only the targeted tile's shard changes
# ---------------------------------------------------------------------------

class TestSingleTileLocalization:
    def test_only_targeted_tile_shard_changes(self) -> None:
        model = SimpleLinearBlock(in_f=16, out_f=32)
        W_before = model.proj.weight.data.clone()

        # Two non-overlapping shards on different tiles
        a_hot = _fake_assignment(
            shard_id="test#r0c0",
            hf_module_path="proj",
            tile_id=0,
            canonical_row_start=0, canonical_row_end=16,
            canonical_col_start=0, canonical_col_end=16,
        )
        a_cold = _fake_assignment(
            shard_id="test#r1c0",
            hf_module_path="proj",
            tile_id=1,
            canonical_row_start=16, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )

        # Only tile 0 has noise
        sigma_maps = build_sigma_map(
            [a_hot, a_cold],
            tile_noise_at_timestep={0: 1.0, 1: 0.0},
            noise_reference_scales={"proj": 1.0},
        )
        Z_by_module = generate_paired_noise_tensors([a_hot, a_cold], seed=0)

        saved = save_projection_weights(model, ["proj"])
        apply_tile_noise(
            model, assignments=[a_hot, a_cold],
            sigma_maps=sigma_maps, Z_by_module=Z_by_module,
            saved_weights=saved,
        )

        W_after = model.proj.weight.data.clone()
        hot_diff = (W_after[:16, :] - W_before[:16, :]).abs()
        cold_diff = (W_after[16:, :] - W_before[16:, :]).abs()

        assert hot_diff.max() > 1e-5
        assert cold_diff.max() < 1e-6


# ---------------------------------------------------------------------------
# Uniform-noise policy-invariance test
# ---------------------------------------------------------------------------

class TestUniformNoisePolicyInvariance:
    def test_same_z_uniform_sigma_produces_same_noisy_weights(self) -> None:
        """When all tiles have the same sigma and Z is shared, all policies
        produce the same noisy weight tensor."""
        model = SimpleLinearBlock(in_f=16, out_f=32)

        # Policy A: tile 0 covers first half, tile 1 second half
        a1 = _fake_assignment(
            shard_id="pol_a#r0c0", hf_module_path="proj", tile_id=0,
            canonical_row_start=0, canonical_row_end=16,
            canonical_col_start=0, canonical_col_end=16,
        )
        a2 = _fake_assignment(
            shard_id="pol_a#r1c0", hf_module_path="proj", tile_id=1,
            canonical_row_start=16, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )

        # Policy B: tile 2 covers first half, tile 3 second half (different tiles, same uniform sigma)
        b1 = _fake_assignment(
            shard_id="pol_b#r0c0", hf_module_path="proj", tile_id=2,
            canonical_row_start=0, canonical_row_end=16,
            canonical_col_start=0, canonical_col_end=16,
        )
        b2 = _fake_assignment(
            shard_id="pol_b#r1c0", hf_module_path="proj", tile_id=3,
            canonical_row_start=16, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )

        uniform_sigma = 0.05
        tile_noise = {0: uniform_sigma, 1: uniform_sigma, 2: uniform_sigma, 3: uniform_sigma}

        sigma_a = build_sigma_map([a1, a2], tile_noise_at_timestep=tile_noise,
                                  noise_reference_scales={"proj": 1.0})
        sigma_b = build_sigma_map([b1, b2], tile_noise_at_timestep=tile_noise,
                                  noise_reference_scales={"proj": 1.0})

        # Both sigma maps should be uniform
        assert torch.allclose(sigma_a["proj"], sigma_b["proj"], atol=1e-7)

        # With the same Z, both policies produce the same noisy weights
        Z = generate_paired_noise_tensors([a1, a2], seed=42)

        saved = save_projection_weights(model, ["proj"])

        apply_tile_noise(model, assignments=[a1, a2], sigma_maps=sigma_a, Z_by_module=Z,
                         saved_weights=saved)
        W_policy_a = model.proj.weight.data.clone()
        restore_projection_weights(model, saved)

        apply_tile_noise(model, assignments=[b1, b2], sigma_maps=sigma_b, Z_by_module=Z,
                         saved_weights=saved)
        W_policy_b = model.proj.weight.data.clone()
        restore_projection_weights(model, saved)

        assert torch.allclose(W_policy_a, W_policy_b, atol=1e-7), \
            "Uniform-sigma policies must produce identical noisy weights with shared Z."


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

class TestWeightChecksums:
    def test_checksums_before_after_and_restored(self) -> None:
        model = SimpleLinearBlock(in_f=16, out_f=32)
        a = _fake_assignment(
            hf_module_path="proj",
            canonical_row_start=0, canonical_row_end=32,
            canonical_col_start=0, canonical_col_end=16,
        )
        sigma_maps = {"proj": torch.ones(32, 16) * 0.5}
        Z_by_module = generate_paired_noise_tensors([a], seed=0)

        saved_before = save_projection_weights(model, ["proj"])

        with NoisedModelContext(model, [a], sigma_maps, Z_by_module) as ctx:
            noisy_weights = ctx.noisy_weights

        saved_after = save_projection_weights(model, ["proj"])
        records = compute_weight_checksums(
            model, ["proj"],
            saved_before=saved_before,
            noisy_weights=noisy_weights,
            saved_after_restore=saved_after,
        )

        assert len(records) == 1
        r = records[0]
        assert r.weights_match_original, "Weight restoration checksum failed."
        assert r.checksum_before != r.checksum_after_noise, \
            "Noisy weights should have a different checksum than clean."
        assert r.checksum_before == r.checksum_restored, \
            "Restored weights checksum should match original."
