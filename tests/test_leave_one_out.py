"""Dependency-light tests for the leave-one-out rank-agreement helper."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.profilers.leave_one_out import rank_agreement


def test_identical_rankings_are_perfect() -> None:
    x = [0.06, 0.03, 0.02, 0.01, 0.005, 0.001]
    out = rank_agreement(x, x)
    assert math.isclose(out["spearman_rho"], 1.0)
    assert math.isclose(out["kendall_tau"], 1.0)
    assert out["top5_overlap"] == 1.0


def test_reversed_rankings_are_anti_correlated() -> None:
    x = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    out = rank_agreement(x, x[::-1])
    assert math.isclose(out["spearman_rho"], -1.0)
    assert math.isclose(out["kendall_tau"], -1.0)


def test_scale_invariance_and_topk() -> None:
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=49)
    out = rank_agreement(x, 100.0 * x + 3.0)
    assert math.isclose(out["spearman_rho"], 1.0)
    assert out["top10_overlap"] == 1.0 and out["top20_overlap"] == 1.0
    assert out["n"] == 49.0


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        rank_agreement([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        rank_agreement([1.0, 2.0, 3.0], [1.0, 2.0])


# --------------------------------------------------------------------------
# Control-flow test of the profiler with a pure-torch stand-in for AIHWKit.
# The stub keeps every projection as an nn.Linear whose weight is overwritten
# in place, so the restore / un-restore bookkeeping, the paired noise seeding,
# and the summary statistics are exercised without the simulator.
# --------------------------------------------------------------------------

def _tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        vocab_size=97, n_positions=16, n_embd=16, n_layer=2, n_head=2, n_inner=32,
    )
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def test_profiler_control_flow_with_stubbed_analog(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from src.common.projections import canonical_weight_bias, iter_gpt2_projections
    from src.common.manual_weights import prepare_projection_weight
    import src.profilers.leave_one_out as loo

    class _StubState:
        def __init__(self, handle, module, clipped, bias, rng):
            self.handle = handle
            self.analog_module = module
            self.clipped_weight = clipped
            self.bias = bias
            self.preprocessing = {"programmed_range": float(rng)}

        @property
        def programmed_range(self):
            return float(self.preprocessing["programmed_range"])

    class _StubHybrid:
        """Replace every projection's module by a fresh nn.Linear holding clipped weights."""

        def __init__(self, model, *, digital_projection_ids, settings, include_lm_head_candidate,
                     phase1_projection_rows=None):
            self.model = model
            self.settings = settings
            self.include = include_lm_head_candidate
            self.states = {}
            self.original_modules = {}

        def convert(self):
            for handle in iter_gpt2_projections(self.model, include_lm_head=self.include):
                weight, bias = canonical_weight_bias(handle.module)
                prepared = prepare_projection_weight(weight, self.settings)
                lin = torch.nn.Linear(handle.in_features, handle.out_features, bias=bias is not None)
                with torch.no_grad():
                    lin.weight.copy_(prepared.clipped_weight)
                    if bias is not None:
                        lin.bias.copy_(bias)
                self.original_modules[handle.projection_id] = handle.module
                setattr(handle.parent, handle.attribute, lin)
                self.states[handle.projection_id] = _StubState(
                    handle, lin, prepared.clipped_weight.clone(), None if bias is None else bias.clone(),
                    prepared.preprocessing.to_dict()["programmed_range"])
            return self

        @property
        def analog_projection_ids(self):
            return tuple(self.states)

        def restore_nominal_weights(self):
            for s in self.states.values():
                with torch.no_grad():
                    s.analog_module.weight.copy_(s.clipped_weight)

        def assert_nominal_restored(self, atol=3e-6):
            for s in self.states.values():
                assert float((s.analog_module.weight - s.clipped_weight).abs().max()) <= atol

        def metadata(self):
            return {"analog_projection_count": len(self.states)}

    def _stub_set_weights(module, weight, bias, *, verify=True, verification_tolerance=3e-6):
        with torch.no_grad():
            module.weight.copy_(weight.to(module.weight.dtype))
            if bias is not None and module.bias is not None:
                module.bias.copy_(bias.to(module.bias.dtype))

    monkeypatch.setattr(loo, "HybridAnalogModel", _StubHybrid)
    monkeypatch.setattr(loo, "set_analog_weights_exact", _stub_set_weights)

    model = _tiny_gpt2()
    config = {
        "experiment": {"seed": 42},
        "model": {"device": "cpu"},
        "analog": {"clip_sigma": 2.5, "range_mode": "peak_to_peak", "reference_noise_std": 0.023,
                   "tile_size": 512, "adc_dac_bits": 8, "output_bound": None,
                   "weight_scaling_omega": 1.0, "weight_scaling_columnwise": False},
        "profiling": {"include_lm_head": True, "num_seeds": 2, "seed_stride": 1, "antithetic": True},
    }
    # two tiny batches
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 97, (2, 12), generator=g)
    batches = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}]
    phase1_rows = [
        {"projection_id": pid, "sensitivity_score_for_mapping": float(i + 1)}
        for i, pid in enumerate(h.projection_id for h in iter_gpt2_projections(model, include_lm_head=True))
    ]
    profiler = loo.LeaveOneOutProfiler(model, config, phase1_rows, restore_mode="nominal")
    assert profiler.hybrid.states, "stub conversion produced no projections"
    targets = list(profiler.hybrid.analog_projection_ids)[:3]
    result = profiler.run(batches, projection_ids=targets, log=False)

    assert result["profiled_projection_count"] == 3
    assert result["analog_projection_count"] == len(phase1_rows)
    assert all(r["projection_id"] in targets for r in result["projections"])
    for row in result["projections"]:
        assert len(row["realizations"]) == 2
        assert all(torch.isfinite(torch.tensor(x["delta_nll_loo"])) for x in row["realizations"])
        # NLL without p's noise must be the noisy NLL minus the LOO delta (bookkeeping consistency)
        for x in row["realizations"]:
            assert abs((x["nll_all_analog"] - x["nll_without_p"]) - x["delta_nll_loo"]) < 1e-9
    assert result["rank_agreement"]["n"] == 3.0
    # weights are restored to nominal after the run
    profiler.hybrid.assert_nominal_restored()

    # digital restore mode swaps the module back and forth without losing state
    profiler_d = loo.LeaveOneOutProfiler(_tiny_gpt2(), config, phase1_rows, restore_mode="digital")
    res_d = profiler_d.run(batches, projection_ids=targets[:1], log=False)
    assert res_d["profiled_projection_count"] == 1
    pid = targets[0]
    state = profiler_d.hybrid.states[pid]
    assert getattr(state.handle.parent, state.handle.attribute) is state.analog_module
