#!/usr/bin/env python3
"""Fast AIHWKit API and deterministic-forward smoke test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped


def find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "src" / "common" / "analog.py").is_file():
            return candidate
    raise RuntimeError("Could not find repository root.")


ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.analog import (
    ManualAnalogSettings,
    get_analog_weights_exact,
    make_rpu_config,
    materialize_manual_noise,
    prepare_projection_weight,
    set_analog_weights_exact,
)
from src.common.config import load_yaml


def main(device_name: str) -> None:
    config = load_yaml(ROOT / "configs/full_pipeline/gpt2_3dcim.yaml")
    settings = ManualAnalogSettings.from_config(config)
    device = torch.device(device_name)
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn((24, 16), generator=generator)
    bias = torch.randn((24,), generator=generator)
    prepared = prepare_projection_weight(weight, settings)

    digital = nn.Linear(16, 24, bias=True)
    with torch.no_grad():
        digital.weight.copy_(prepared.clipped_weight)
        digital.bias.copy_(bias)
    analog = AnalogLinearMapped.from_digital(
        digital, rpu_config=make_rpu_config(settings)
    ).to(device)
    reference, _ = get_analog_weights_exact(analog)
    assert torch.allclose(reference, prepared.clipped_weight, atol=3e-6, rtol=0)

    z = torch.randn(weight.shape, generator=torch.Generator().manual_seed(456))
    noisy, noise = materialize_manual_noise(
        prepared.clipped_weight,
        settings.reference_noise_std,
        prepared.preprocessing.programmed_range,
        z,
    )
    set_analog_weights_exact(analog, noisy, bias, verify=True)
    expected_std = settings.reference_noise_std * prepared.preprocessing.programmed_range
    realized_std = float(noise.std(unbiased=False))
    assert abs(realized_std - expected_std) / expected_std < 0.12

    x = torch.randn((3, 16), generator=torch.Generator().manual_seed(789)).to(device)
    with torch.inference_mode():
        first = analog(x)
        second = analog(x)
    if not torch.equal(first, second):
        max_difference = float((first - second).abs().max().item())
        raise AssertionError(
            "Forward output is stochastic even though all internal noise is disabled: "
            f"max_difference={max_difference}"
        )

    set_analog_weights_exact(
        analog, prepared.clipped_weight, bias, verify=True
    )
    restored, _ = get_analog_weights_exact(analog)
    assert torch.allclose(restored, prepared.clipped_weight, atol=3e-6, rtol=0)
    print("AIHWKit shared-contract smoke test passed.")
    print(f"Device: {device}")
    print(f"Programmed range: {prepared.preprocessing.programmed_range:.8f}")
    print(f"Expected noise std: {expected_std:.8f}")
    print(f"Realized noise std: {realized_std:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    main(args.device)
