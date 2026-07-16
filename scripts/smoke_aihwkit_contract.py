#!/usr/bin/env python3
"""Small AIHWKit read/write/restore smoke test before expensive experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import ManualAnalogSettings, make_rpu_config, materialize_manual_noise, prepare_projection_weight
from src.evaluation.aihwkit_gpt2 import get_analog_weights_exact, set_analog_weights_exact


def main() -> None:
    try:
        from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
    except ImportError as exc:
        raise SystemExit("AIHWKit 1.1.0 is not installed.") from exc
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    settings = ManualAnalogSettings(
        clip_sigma=2.5,
        range_mode="peak_to_peak",
        reference_noise_std=0.023,
        tile_size=512,
        adc_dac_bits=8,
        output_bound=12.0,
        weight_scaling_omega=1.0,
        weight_scaling_columnwise=False,
    )
    digital = torch.nn.Linear(17, 13, bias=True).to(device)
    prepared = prepare_projection_weight(digital.weight.detach(), settings)
    with torch.no_grad():
        digital.weight.copy_(prepared.clipped_weight.to(device))
    analog = AnalogLinearMapped.from_digital(digital, rpu_config=make_rpu_config(settings)).to(device)
    analog.eval()
    set_analog_weights_exact(analog, prepared.clipped_weight, digital.bias.detach())
    clean, _ = get_analog_weights_exact(analog)
    clean_error = float((clean.cpu() - prepared.clipped_weight.cpu()).abs().max().item())
    if clean_error > 3e-6:
        raise RuntimeError(f"Clean exact-write error: {clean_error}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    z = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    noisy, noise = materialize_manual_noise(
        clean, 0.023, prepared.preprocessing.programmed_range, z
    )
    set_analog_weights_exact(analog, noisy, digital.bias.detach())
    noisy, _ = get_analog_weights_exact(analog)
    noise_error = float(((noisy.cpu() - clean.cpu()) - noise.cpu()).abs().max().item())
    if noise_error > 3e-6:
        raise RuntimeError(f"Noisy exact-write error: {noise_error}")
    set_analog_weights_exact(analog, clean, digital.bias.detach())
    restored, _ = get_analog_weights_exact(analog)
    restore_error = float((restored.cpu() - clean.cpu()).abs().max().item())
    if restore_error > 3e-6:
        raise RuntimeError(f"Restore error: {restore_error}")
    print("AIHWKit contract smoke test passed.")
    print(f"clean_error={clean_error:.3e} noise_error={noise_error:.3e} restore_error={restore_error:.3e}")


if __name__ == "__main__":
    main()
