"""Structural tests for the reviewer-requested hybrid baselines."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase4_quality.run_hybrid_baselines import (
    _completed_run,
    summarize_rows,
    summarize_rows_by_timestep,
)
from src.common.analog import ManualAnalogSettings
from src.common.projections import canonical_weight_bias, iter_gpt2_projections
from src.common.tabular import write_csv
from src.evaluation.aihwkit_gpt2 import (
    HybridAnalogModel,
    swap_in_clipped_digital_module,
)
from src.evaluation.hybrid_baselines import (
    build_hybrid_placement,
    parse_digital_weight_mode,
    parse_hybrid_designs,
    projection_compute_accounting,
)


def profile_rows():
    return [
        {
            "projection_id": "lm_head",
            "out_features": 8,
            "in_features": 4,
            "sensitivity_score_for_mapping": 0.9,
        },
        {
            "projection_id": "block_0/attn.c_proj",
            "out_features": 4,
            "in_features": 4,
            "sensitivity_score_for_mapping": 0.6,
        },
        {
            "projection_id": "block_0/mlp.c_fc",
            "out_features": 8,
            "in_features": 4,
            "sensitivity_score_for_mapping": 0.2,
        },
    ]


def config():
    return {
        "experiment": {"seed": 42, "placement_seed": 7},
        "hardware": {
            "num_tiles": 8,
            "tiers_per_tile": 2,
            "tier_shape": {"rows": 4, "cols": 4},
        },
        "phase3": {"mapping_timestep": 0, "sensitivity_floor": 0.0},
        "hybrid_baselines": {
            "mapping_policy": "hardware_only",
            "designs": [
                {"name": "hybrid_lm_head", "digital_projection_ids": ["lm_head"]},
                {
                    "name": "hybrid_top2",
                    "digital_projection_ids": ["lm_head", "block_0/attn.c_proj"],
                },
            ],
        },
    }


def trace():
    return SimpleNamespace(
        noise_std=np.asarray([[0.03, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08]]),
        available=np.ones((1, 8), dtype=bool),
    )


def analog_settings():
    return ManualAnalogSettings(
        clip_sigma=2.5,
        range_mode="peak_to_peak",
        reference_noise_std=0.023,
        tile_size=512,
        adc_dac_bits=8,
        output_bound=None,
        weight_scaling_omega=1.0,
        weight_scaling_columnwise=False,
    )


def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=40,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    model.eval()
    # Give the weights a clearly clippable tail.
    with torch.no_grad():
        model.transformer.wte.weight.mul_(1.0).add_(0.0)
        model.transformer.wte.weight[0, :] = 5.0
        model.transformer.h[0].attn.c_attn.weight[0, 0] = 3.0
    return model


def test_reviewer_designs_are_valid_and_nested():
    rows = profile_rows()
    designs = parse_hybrid_designs(config(), [row["projection_id"] for row in rows])
    assert [design.name for design in designs] == ["hybrid_lm_head", "hybrid_top2"]
    assert set(designs[0].digital_projection_ids) < set(designs[1].digital_projection_ids)
    assert all(design.mapping_policy is None for design in designs)
    assert designs[0].effective_policy("hardware_only") == "hardware_only"


def test_unknown_digital_projection_is_rejected():
    cfg = config()
    cfg["hybrid_baselines"]["designs"][0]["digital_projection_ids"] = ["missing"]
    with pytest.raises(ValueError, match="outside the Phase-1 universe"):
        parse_hybrid_designs(cfg, [row["projection_id"] for row in profile_rows()])


def test_per_design_mapping_policy_is_parsed_and_validated():
    cfg = config()
    cfg["hybrid_baselines"]["designs"].append(
        {
            "name": "hybrid_lm_head_sens",
            "digital_projection_ids": ["lm_head"],
            "mapping_policy": "static_sensitivity",
        }
    )
    designs = parse_hybrid_designs(cfg, [row["projection_id"] for row in profile_rows()])
    assert designs[-1].mapping_policy == "static_sensitivity"
    assert designs[-1].effective_policy("hardware_only") == "static_sensitivity"
    cfg["hybrid_baselines"]["designs"][-1]["mapping_policy"] = "best_effort"
    with pytest.raises(ValueError, match="mapping_policy must be one of"):
        parse_hybrid_designs(cfg, [row["projection_id"] for row in profile_rows()])


def test_digital_weight_mode_is_parsed_and_validated():
    cfg = config()
    assert parse_digital_weight_mode(cfg) == "unclipped"
    cfg["hybrid_baselines"]["digital_weight_mode"] = "Clipped"
    assert parse_digital_weight_mode(cfg) == "clipped"
    cfg["hybrid_baselines"]["digital_weight_mode"] = "quantized"
    with pytest.raises(ValueError, match="digital_weight_mode"):
        parse_digital_weight_mode(cfg)
    with pytest.raises(ValueError, match="digital_weight_mode"):
        HybridAnalogModel(
            object(),
            digital_projection_ids=[],
            settings=analog_settings(),
            include_lm_head_candidate=True,
            digital_weight_mode="quantized",
        )


def test_remaining_analog_set_is_remapped_without_digital_rows():
    rows = profile_rows()
    designs = parse_hybrid_designs(config(), [row["projection_id"] for row in rows])
    records = build_hybrid_placement(config(), rows, trace(), designs[0])
    assert {record.projection_id for record in records} == {
        "block_0/attn.c_proj",
        "block_0/mlp.c_fc",
    }
    assert len({(record.tile_id, record.tier_id) for record in records}) == len(records)
    assert all(record.policy == "hardware_only" for record in records)


def test_remainder_uses_the_design_policy_when_overridden():
    cfg = config()
    cfg["hybrid_baselines"]["designs"][0]["mapping_policy"] = "static_sensitivity"
    rows = profile_rows()
    designs = parse_hybrid_designs(cfg, [row["projection_id"] for row in rows])
    records = build_hybrid_placement(cfg, rows, trace(), designs[0])
    assert all(record.policy == "static_sensitivity" for record in records)
    # Sorted rule: the most important remaining shard sits on the quietest tile.
    quietest_tile = int(np.argmin(trace().noise_std[0]))
    top = max(records, key=lambda record: record.importance)
    assert top.tile_id == quietest_tile


def test_compute_accounting_is_exact():
    accounting = projection_compute_accounting(profile_rows(), ["lm_head"])
    assert accounting["total_projection_macs_per_token"] == 80
    assert accounting["digital_projection_macs_per_token"] == 32
    assert accounting["digital_projection_mac_fraction"] == pytest.approx(0.4)


def test_clipped_digital_swap_holds_clipped_weights_and_restores():
    model = tiny_gpt2()
    settings = analog_settings()
    handles = {
        handle.projection_id: handle
        for handle in iter_gpt2_projections(model, include_lm_head=True)
    }
    inputs = torch.randint(0, 40, (1, 12))
    with torch.no_grad():
        reference_logits = model(inputs).logits.clone()
    embedding_before = model.transformer.wte.weight.detach().clone()

    for projection_id in ("lm_head", "block_0/attn.c_attn"):
        handle = handles[projection_id]
        original_module = handle.module
        canonical, bias = canonical_weight_bias(original_module)
        threshold = 2.5 * canonical.std(unbiased=False)
        expected = canonical.clamp(min=-threshold, max=threshold)
        assert bool((canonical.abs() > threshold).any()), "test weights must clip"

        state = swap_in_clipped_digital_module(handle, settings, torch.device("cpu"))
        swapped = getattr(handle.parent, handle.attribute)
        assert swapped is state.digital_module
        assert isinstance(swapped, torch.nn.Linear)
        assert torch.allclose(swapped.weight.detach(), expected)
        if bias is None:
            assert swapped.bias is None
        else:
            assert torch.allclose(swapped.bias.detach(), bias)
        assert state.preprocessing["num_clipped"] > 0
        assert state.preprocessing["clip_threshold"] == pytest.approx(float(threshold))
        # Forward semantics: x @ clip(W)^T + b, independent of Conv1D/Linear layout.
        x = torch.randn(3, canonical.shape[1])
        with torch.no_grad():
            produced = swapped(x)
            wanted = x @ expected.T + (0.0 if bias is None else bias)
        assert torch.allclose(produced, wanted, atol=1e-6)

        # Restore the checkpoint module and confirm the model is unchanged.
        setattr(handle.parent, handle.attribute, original_module)

    with torch.no_grad():
        restored_logits = model(inputs).logits
    assert torch.equal(restored_logits, reference_logits)
    # The token embedding is never touched by a digital LM-head swap.
    assert torch.equal(model.transformer.wte.weight.detach(), embedding_before)


def test_clipped_lm_head_changes_logits_but_not_embedding():
    model = tiny_gpt2()
    handle = next(
        handle
        for handle in iter_gpt2_projections(model, include_lm_head=True)
        if handle.projection_id == "lm_head"
    )
    inputs = torch.randint(0, 40, (1, 12))
    with torch.no_grad():
        before = model(inputs).logits.clone()
    embedding_before = model.transformer.wte.weight.detach().clone()
    swap_in_clipped_digital_module(handle, analog_settings(), torch.device("cpu"))
    with torch.no_grad():
        after = model(inputs).logits
    assert not torch.allclose(before, after)
    assert torch.equal(model.transformer.wte.weight.detach(), embedding_before)
    assert model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr()


def test_summaries_report_both_paired_references():
    rows = [
        {
            "design": "hybrid_lm_head",
            "digital_projection_ids": "lm_head",
            "digital_weight_mode": "clipped",
            "policy": "hardware_only",
            "timestep": t,
            "realization": 0,
            "nll": 3.6 + 0.01 * t / 119,
            "nll_improvement_vs_all_analog": 0.05 + 0.01 * t / 119,
            "nll_improvement_vs_static_sensitivity": -0.02 + 0.03 * t / 119,
            "digital_projection_mac_fraction": 0.31,
            "analog_shards": 480,
        }
        for t in (0, 119)
    ]
    summary = summarize_rows(rows, seed=1)
    assert len(summary) == 1
    assert summary[0]["mean_nll_improvement_vs_all_analog"] == pytest.approx(0.055)
    assert summary[0]["mean_nll_improvement_vs_static_sensitivity"] == pytest.approx(-0.005)
    assert summary[0]["win_fraction_vs_all_analog"] == 1.0
    assert summary[0]["win_fraction_vs_static_sensitivity"] == 0.5
    by_timestep = summarize_rows_by_timestep(rows)
    assert [row["timestep"] for row in by_timestep] == [0, 119]
    assert by_timestep[1]["mean_nll_improvement_vs_static_sensitivity"] == pytest.approx(0.01)


def test_completed_run_reuses_only_exact_complete_artifacts(tmp_path: Path):
    signature = {"config_sha256": "same"}
    (tmp_path / "hybrid_baselines_run_signature.json").write_text(
        json.dumps(signature), encoding="utf-8"
    )
    (tmp_path / "hybrid_baselines_metadata.json").write_text("{}", encoding="utf-8")
    write_csv(
        tmp_path / "hybrid_baselines_by_condition.csv",
        [{"design": "hybrid_lm_head", "timestep": 0, "realization": 0}],
    )
    expected = {("hybrid_lm_head", 0, 0)}
    assert _completed_run(tmp_path, signature, expected) == (
        tmp_path / "hybrid_baselines_metadata.json"
    )

    with pytest.raises(ValueError, match="different run signature"):
        _completed_run(tmp_path, {"config_sha256": "changed"}, expected)
