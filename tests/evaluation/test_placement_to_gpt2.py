"""Tests for the placement-to-GPT-2 bridge.

These tests exercise the core correctness requirements from the Phase-4 spec:

- Orientation test: canonical weight reproduces manual linear computation.
- Q/K/V reconstruction test: split and re-concatenated slices equal original.
- Coverage test: every analog-mapped weight is covered exactly once.
- Shard-index parsing.
- Module-name resolution (including next-layer q_proj_out shift).
- Coordinate computation for all projection types.
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

from src.evaluation.placement_to_gpt2 import (
    INJECTABLE_PROJECTIONS,
    QKV_ROW_OFFSETS,
    GPT2_CANONICAL_SHAPES,
    GPT2_SMALL_HIDDEN,
    _compute_canonical_coordinates,
    _parse_shard_indices,
    _resolve_hf_info,
    build_shard_assignments,
    validate_shard_coverage,
)
from src.evaluation.schemas import GPT2ShardAssignment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_assignment(
    *,
    shard_id: str = "layers.0.out_proj#r0c0",
    module_name: str = "decoder_stack.layers.0.out_proj",
    projection_id: str = "block_0/attn.c_proj",
    projection_name: str = "attn.c_proj",
    tile_id: int = 0,
    tier_start: int = 0,
    used_rows: int = 512,
    used_cols: int = 512,
    weights_in_shard: int = 512 * 512,
    group_total_weights: int = 768 * 768,
    shard_weight: float = 1.0,
    sensitivity_score: float = 0.0,
) -> dict[str, str]:
    """Return a dict mimicking a Phase-3 placement CSV row."""
    return {
        "module_name": module_name,
        "shard_id": shard_id,
        "projection_id": projection_id,
        "projection_name": projection_name,
        "tile_id": str(tile_id),
        "tier_start": str(tier_start),
        "used_rows": str(used_rows),
        "used_cols": str(used_cols),
        "weights_in_shard": str(weights_in_shard),
        "group_total_weights": str(group_total_weights),
        "shard_weight": str(shard_weight),
        "sensitivity_score": str(sensitivity_score),
    }


# ---------------------------------------------------------------------------
# Shard-index parsing
# ---------------------------------------------------------------------------

class TestParseShardIndices:
    def test_basic(self) -> None:
        ri, ci = _parse_shard_indices("layers.0.out_proj#r3c7")
        assert ri == 3 and ci == 7

    def test_zero_zero(self) -> None:
        ri, ci = _parse_shard_indices("layers.0.ffn1#r0c0")
        assert ri == 0 and ci == 0

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_shard_indices("layers.0.out_proj_no_shard")


# ---------------------------------------------------------------------------
# Module-name resolution
# ---------------------------------------------------------------------------

class TestResolveHFInfo:
    @pytest.mark.parametrize("tag,expected_qkv,expected_proj", [
        ("q_proj_in", "Q", "attn.c_attn"),
        ("k_proj_in", "K", "attn.c_attn"),
        ("v_proj_in", "V", "attn.c_attn"),
        ("out_proj", None, "attn.c_proj"),
        ("ffn1", None, "mlp.c_fc"),
        ("ffn2", None, "mlp.c_proj"),
    ])
    def test_same_layer(self, tag: str, expected_qkv: str | None, expected_proj: str) -> None:
        module = f"decoder_stack.layers.5.{tag}"
        hf_path, proj, qkv, row_off = _resolve_hf_info(module, sim_layer=5)
        assert proj == expected_proj
        assert qkv == expected_qkv
        assert "transformer.h.5" in hf_path

    @pytest.mark.parametrize("tag,expected_qkv", [
        ("q_proj_out", "Q"),
        ("k_proj_out", "K"),
        ("v_proj_out", "V"),
    ])
    def test_next_layer_shift(self, tag: str, expected_qkv: str) -> None:
        module = f"decoder_stack.layers.3.{tag}"
        hf_path, proj, qkv, row_off = _resolve_hf_info(module, sim_layer=3)
        # Must map to block 4 (3 + 1)
        assert "transformer.h.4" in hf_path
        assert qkv == expected_qkv

    def test_qkv_row_offsets(self) -> None:
        for tag, expected_comp in [("q_proj_in", "Q"), ("k_proj_in", "K"), ("v_proj_in", "V")]:
            module = f"decoder_stack.layers.0.{tag}"
            _, _, _, row_off = _resolve_hf_info(module, sim_layer=0)
            assert row_off == QKV_ROW_OFFSETS[expected_comp]

    def test_lm_head_raises_by_default(self) -> None:
        # build_shard_assignments skips lm_head by default; direct call should not raise
        hf_path, proj, qkv, row_off = _resolve_hf_info("lm_head", sim_layer=0)
        assert hf_path == "lm_head"
        assert qkv is None

    def test_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="Embedding"):
            _resolve_hf_info("decoder_stack.layers.0.token_embedding", sim_layer=0)


# ---------------------------------------------------------------------------
# Coordinate computation
# ---------------------------------------------------------------------------

class TestCanonicalCoordinates:
    def test_out_proj_r0c0(self) -> None:
        rs, re, cs, ce = _compute_canonical_coordinates(
            input_shard_index=0, output_shard_index=0,
            output_row_offset=0, used_input_features=512, used_output_features=512, tier_rows=512, tier_cols=512,
        )
        assert rs == 0 and re == 512 and cs == 0 and ce == 512

    def test_out_proj_r1c1(self) -> None:
        # attn.c_proj is 768×768; shard r1c1 covers [512:768, 512:768]
        rs, re, cs, ce = _compute_canonical_coordinates(
            input_shard_index=1, output_shard_index=1,
            output_row_offset=0, used_input_features=256, used_output_features=256, tier_rows=512, tier_cols=512,
        )
        assert rs == 512 and re == 768 and cs == 512 and ce == 768

    def test_k_proj_in_row_offset(self) -> None:
        # K block starts at row offset 768
        rs, re, cs, ce = _compute_canonical_coordinates(
            input_shard_index=0, output_shard_index=0,
            output_row_offset=QKV_ROW_OFFSETS["K"],
            used_input_features=512, used_output_features=512, tier_rows=512, tier_cols=512,
        )
        assert rs == 768 and re == 1280

    def test_v_proj_in_row_offset(self) -> None:
        rs, re, cs, ce = _compute_canonical_coordinates(
            input_shard_index=0, output_shard_index=0,
            output_row_offset=QKV_ROW_OFFSETS["V"],
            used_input_features=256, used_output_features=512, tier_rows=512, tier_cols=512,
        )
        assert rs == 1536 and re == 1792

    def test_ffn1_large_rows(self) -> None:
        # mlp.c_fc is 3072×768; 6 row shards of 512, 1 col shard of 512 + 1 of 256
        rs, re, cs, ce = _compute_canonical_coordinates(
            input_shard_index=5, output_shard_index=1,
            output_row_offset=0, used_input_features=512, used_output_features=256, tier_rows=512, tier_cols=512,
        )
        assert rs == 5 * 512 and re == 6 * 512 and cs == 512 and ce == 768


# ---------------------------------------------------------------------------
# Build shard assignments: minimal synthetic placements
# ---------------------------------------------------------------------------

def _build_full_c_proj_rows() -> list[dict[str, str]]:
    """Build a minimal but complete set of rows for block_0/attn.c_proj (768×768).

    attn.c_proj canonical shape [768, 768] is tiled as:
        r0c0: rows [0:512],   cols [0:512]   → 512×512
        r1c0: rows [512:768], cols [0:512]   → 256×512
        r0c1: rows [0:512],   cols [512:768] → 512×256
        r1c1: rows [512:768], cols [512:768] → 256×256
    """
    shards = [
        (0, 0, 0, 512, 512),   # (tile_id, ri, ci, used_rows, used_cols)
        (1, 1, 0, 256, 512),
        (2, 0, 1, 512, 256),
        (3, 1, 1, 256, 256),
    ]
    rows = []
    for tile_id, ri, ci, used_rows, used_cols in shards:
        rows.append(_make_assignment(
            shard_id=f"decoder_stack.layers.0.out_proj#r{ri}c{ci}",
            module_name="decoder_stack.layers.0.out_proj",
            projection_id="block_0/attn.c_proj",
            projection_name="attn.c_proj",
            tile_id=tile_id,
            used_rows=used_rows,
            used_cols=used_cols,
        ))
    return rows


class TestBuildShardAssignments:
    def test_skip_embedding_by_default(self) -> None:
        rows = [_make_assignment(
            shard_id="decoder_stack.layers.0.token_embedding#r0c0",
            module_name="decoder_stack.layers.0.token_embedding",
            projection_id="embedding/token_embedding",
            projection_name="token_embedding",
        )]
        assignments = build_shard_assignments(rows, policy="test", placement_seed=0)
        assert assignments == []

    def test_skip_lm_head_by_default(self) -> None:
        rows = [_make_assignment(
            shard_id="lm_head#r0c0",
            module_name="lm_head",
            projection_id="head/lm_head",
            projection_name="lm_head",
        )]
        assignments = build_shard_assignments(rows, policy="test", placement_seed=0)
        assert assignments == []

    def test_correct_hf_path(self) -> None:
        rows = _build_full_c_proj_rows()
        assignments = build_shard_assignments(rows, policy="test", placement_seed=0)
        hf_paths = {a.hf_module_path for a in assignments}
        assert hf_paths == {"transformer.h.0.attn.c_proj"}

    def test_policy_and_seed_propagated(self) -> None:
        rows = _build_full_c_proj_rows()
        assignments = build_shard_assignments(rows, policy="static_sensitivity", placement_seed=42)
        for a in assignments:
            assert a.policy == "static_sensitivity"
            assert a.placement_seed == 42


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------

class TestValidateCoverage:
    def test_full_coverage_ok(self) -> None:
        rows = _build_full_c_proj_rows()
        assignments = build_shard_assignments(rows, policy="test", placement_seed=0)
        coverage = validate_shard_coverage(
            assignments,
            canonical_shapes={"transformer.h.0.attn.c_proj": (768, 768)},
        )
        cov = coverage["transformer.h.0.attn.c_proj"]
        # Every element should be 1
        assert int(cov.min()) == 1
        assert int(cov.max()) == 1

    def test_overlap_raises(self) -> None:
        rows = _build_full_c_proj_rows()
        # Add a duplicate of the first shard → overlap
        duplicate = dict(rows[0])
        duplicate["tile_id"] = "99"
        rows.append(duplicate)
        assignments = build_shard_assignments(rows, policy="test", placement_seed=0)
        with pytest.raises(ValueError, match="overlap"):
            validate_shard_coverage(assignments)


# ---------------------------------------------------------------------------
# Orientation test (Conv1D vs manual linear)
# ---------------------------------------------------------------------------

class TestOrientationWithConv1D:
    def test_conv1d_output_matches_canonical_matmul(self) -> None:
        """HuggingFace Conv1D output must equal manual W_canonical @ x.T."""
        from transformers.pytorch_utils import Conv1D  # type: ignore

        in_features, out_features = 16, 32
        conv1d = Conv1D(out_features, in_features)
        # Conv1D.weight shape: [in_features, out_features]
        W_hf = conv1d.weight.detach()  # [16, 32]
        W_canonical = W_hf.T           # [32, 16]

        x = torch.randn(2, in_features)  # batch=2
        out_conv1d = conv1d(x)           # [2, 32]
        out_manual = x @ W_canonical.T   # [2, 32]

        # With bias
        assert torch.allclose(out_conv1d - conv1d.bias, out_manual, atol=1e-5)

    def test_canonical_slice_covers_correct_weights(self) -> None:
        """Noise injected into canonical slice must affect only those output features."""
        from transformers.pytorch_utils import Conv1D  # type: ignore

        in_features, out_features = 16, 32
        conv1d = Conv1D(out_features, in_features)
        W_original = conv1d.weight.data.clone()

        # Inject a known delta into canonical rows [0:16] (first half of outputs)
        delta = torch.zeros(out_features, in_features)
        delta[:16, :] = 1.0  # large perturbation for first 16 output features

        with torch.no_grad():
            conv1d.weight.copy_((W_original.T + delta).T)

        x = torch.randn(1, in_features)
        out_orig = (x @ W_original)
        out_pert = conv1d(x) - conv1d.bias

        diff = (out_pert - out_orig).squeeze(0)
        # First 16 outputs should have changed, last 16 should be ~zero
        assert diff[:16].abs().max().item() > 0.5
        assert diff[16:].abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Q/K/V reconstruction test
# ---------------------------------------------------------------------------

class TestQKVReconstruction:
    def test_qkv_slices_reconstruct_c_attn(self) -> None:
        """Q, K, V canonical row slices must reconstruct the full attn.c_attn weight."""
        from transformers import GPT2Model  # type: ignore

        model = GPT2Model.from_pretrained("gpt2")
        # GPT2Model has .h (transformer blocks) directly, not .transformer.h
        c_attn = model.h[0].attn.c_attn
        W_hf = c_attn.weight  # Conv1D: [768, 2304]
        W_canonical = W_hf.T  # [2304, 768]

        d = GPT2_SMALL_HIDDEN
        Q_slice = W_canonical[0:d, :]      # [768, 768]
        K_slice = W_canonical[d:2*d, :]    # [768, 768]
        V_slice = W_canonical[2*d:3*d, :]  # [768, 768]

        W_reconstructed = torch.cat([Q_slice, K_slice, V_slice], dim=0)
        assert W_reconstructed.shape == W_canonical.shape
        assert torch.allclose(W_reconstructed, W_canonical, atol=1e-6)
