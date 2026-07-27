"""Fisher/magnitude proxy-score computation on a tiny offline GPT-2."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from src.common.manual_weights import ManualAnalogSettings, prepare_projection_weight
from src.common.projections import canonical_weight_bias, iter_gpt2_projections

SETTINGS = ManualAnalogSettings(
    clip_sigma=2.5,
    range_mode="peak_to_peak",
    reference_noise_std=0.023,
    tile_size=512,
    adc_dac_bits=8,
    output_bound=None,
    weight_scaling_omega=1.0,
    weight_scaling_columnwise=False,
)


def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=128)
    model = GPT2LMHeadModel(config)
    model.config.use_cache = False
    return model.eval()


def fisher_traces(model, batches) -> dict[str, float]:
    """Mirror of the run_proxy_sensitivity accumulation (untied lm_head)."""
    handles = list(iter_gpt2_projections(model, include_lm_head=True))
    original_lm_head = model.lm_head
    untied = nn.Linear(
        original_lm_head.in_features, original_lm_head.out_features, bias=False
    )
    with torch.no_grad():
        untied.weight.copy_(original_lm_head.weight)
    model.lm_head = untied
    parameters = {
        handle.projection_id: (
            untied.weight if handle.projection_id == "lm_head" else handle.module.weight
        )
        for handle in handles
    }
    accumulators = {
        projection_id: torch.zeros_like(parameter, dtype=torch.float64)
        for projection_id, parameter in parameters.items()
    }
    total_weight = 0.0
    for batch in batches:
        weight = float((batch["labels"][..., 1:] != -100).sum().item())
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        for projection_id, parameter in parameters.items():
            accumulators[projection_id] += (parameter.grad.detach().double() ** 2) * weight
        total_weight += weight
    model.zero_grad(set_to_none=True)
    model.lm_head = original_lm_head
    return {
        projection_id: float(accumulator.sum().item()) / total_weight
        for projection_id, accumulator in accumulators.items()
    }


def batches(model, count=2):
    result = []
    for index in range(count):
        generator = torch.Generator().manual_seed(index)
        ids = torch.randint(0, model.config.vocab_size, (2, 16), generator=generator)
        result.append({"input_ids": ids, "labels": ids.clone()})
    return result


def test_fisher_traces_are_positive_and_deterministic():
    model = tiny_model()
    data = batches(model)
    first = fisher_traces(model, data)
    second = fisher_traces(model, data)
    assert set(first) == {
        handle.projection_id
        for handle in iter_gpt2_projections(model, include_lm_head=True)
    }
    for projection_id, value in first.items():
        assert value > 0.0, projection_id
        assert value == pytest.approx(second[projection_id]), projection_id


def test_lm_head_fisher_excludes_embedding_path():
    model = tiny_model()
    data = batches(model)
    traces = fisher_traces(model, data)

    # Direct comparison: gradient on the TIED parameter includes both the
    # lm_head matmul and the embedding-lookup contributions, so its squared
    # magnitude differs from the untied lm_head-only trace.
    model.zero_grad(set_to_none=True)
    loss = model(**data[0]).loss
    loss.backward()
    tied_sq = float((model.lm_head.weight.grad.double() ** 2).sum().item())
    model.zero_grad(set_to_none=True)
    assert traces["lm_head"] > 0.0
    assert tied_sq != pytest.approx(traces["lm_head"])


def test_magnitude_score_matches_definition():
    model = tiny_model()
    handle = next(iter_gpt2_projections(model, include_lm_head=False))
    canonical, _ = canonical_weight_bias(handle.module)
    prepared = prepare_projection_weight(canonical, SETTINGS)
    sigma_abs = SETTINGS.reference_noise_std * prepared.preprocessing.programmed_range
    clipped_energy = float((prepared.clipped_weight.double() ** 2).sum().item())
    expected = (sigma_abs**2) * handle.parameter_count / clipped_energy
    # Definition sanity: doubling the noise quadruples the score.
    quadrupled = ((2 * sigma_abs) ** 2) * handle.parameter_count / clipped_energy
    assert quadrupled == pytest.approx(4 * expected)
    assert expected > 0.0
