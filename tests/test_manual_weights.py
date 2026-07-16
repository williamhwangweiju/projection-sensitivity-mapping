from __future__ import annotations

import torch

from src.common.manual_weights import (
    ManualAnalogSettings,
    materialize_manual_noise,
    prepare_projection_weight,
    projection_noise_seed,
)


def settings(range_mode: str = "peak_to_peak") -> ManualAnalogSettings:
    return ManualAnalogSettings(
        clip_sigma=2.5,
        range_mode=range_mode,
        reference_noise_std=0.023,
        tile_size=512,
        adc_dac_bits=8,
        output_bound=12.0,
        weight_scaling_omega=1.0,
        weight_scaling_columnwise=False,
    )


def test_manual_clipping_uses_population_std_and_peak_to_peak_range() -> None:
    weight = torch.tensor([[-10.0, -1.0, 0.0], [1.0, 2.0, 10.0]])
    prepared = prepare_projection_weight(weight, settings())
    expected_std = float(weight.std(unbiased=False).item())
    threshold = 2.5 * expected_std
    assert prepared.preprocessing.original_std == expected_std
    assert abs(prepared.preprocessing.clip_threshold - threshold) < 1e-6
    assert torch.all(prepared.clipped_weight <= threshold)
    assert torch.all(prepared.clipped_weight >= -threshold)
    expected_range = float(
        (prepared.clipped_weight.max() - prepared.clipped_weight.min()).item()
    )
    assert prepared.preprocessing.programmed_range == expected_range


def test_reference_noise_std_is_fraction_of_programmed_range() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((1024, 1024), generator=generator)
    prepared = prepare_projection_weight(weight, settings())
    z = torch.randn(weight.shape, generator=torch.Generator().manual_seed(8))
    _, noise = materialize_manual_noise(
        prepared.clipped_weight,
        0.023,
        prepared.preprocessing.programmed_range,
        z,
    )
    expected = 0.023 * prepared.preprocessing.programmed_range
    assert abs(float(noise.std(unbiased=False)) - expected) / expected < 0.01


def test_noise_is_not_reclipped() -> None:
    weight = torch.tensor([[-1.0, 0.0], [0.5, 1.0]])
    prepared = prepare_projection_weight(weight, settings())
    z = torch.full_like(weight, 100.0)
    noisy, _ = materialize_manual_noise(
        prepared.clipped_weight,
        0.023,
        prepared.preprocessing.programmed_range,
        z,
    )
    assert float(noisy.max()) > prepared.preprocessing.clip_threshold


def test_projection_seed_is_stable_and_projection_specific() -> None:
    assert projection_noise_seed(42, "block_0/attn.c_attn") == projection_noise_seed(
        42, "block_0/attn.c_attn"
    )
    assert projection_noise_seed(42, "block_0/attn.c_attn") != projection_noise_seed(
        42, "block_0/attn.c_proj"
    )
