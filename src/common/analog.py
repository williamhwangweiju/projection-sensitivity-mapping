"""AIHWKit configuration and exact logical-weight I/O shared by hybrid phases.

The AIHWKit imports are intentionally local so dependency-light structural tests
can run without installing the simulator. The implementation of exact mapped-
tile writes mirrors the validated current repository path and preserves clean
AIHWKit mapping scales when logical noisy weights are installed.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from src.common.manual_weights import (  # re-export
    ManualAnalogSettings,
    PreparedProjection,
    ProjectionPreprocessing,
    materialize_manual_noise,
    normalized_noise_to_absolute_sigma,
    prepare_projection_weight,
    projection_noise_seed,
    set_seed,
    tensor_checksum,
)


def make_rpu_config(settings: ManualAnalogSettings) -> Any:
    settings.validate()
    try:
        from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
        from aihwkit.simulator.configs import InferenceRPUConfig
        from aihwkit.simulator.configs.utils import MappingParameter, WeightClipParameter
        from aihwkit.simulator.parameters.enums import (
            BoundManagementType,
            NoiseManagementType,
            WeightClipType,
            WeightNoiseType,
        )
    except ImportError as exc:
        raise RuntimeError("AIHWKit 1.1.0 is required for analog execution.") from exc

    resolution = 1.0 / float(2**settings.adc_dac_bits - 2)
    rpu_config = InferenceRPUConfig()
    rpu_config.clip = WeightClipParameter(type=WeightClipType.NONE)
    rpu_config.mapping = MappingParameter(
        max_input_size=settings.tile_size,
        max_output_size=settings.tile_size,
        digital_bias=True,
        weight_scaling_omega=settings.weight_scaling_omega,
        weight_scaling_columnwise=settings.weight_scaling_columnwise,
    )
    rpu_config.forward.is_perfect = False
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = resolution
    rpu_config.forward.inp_sto_round = False
    rpu_config.forward.inp_asymmetry = 0.0
    rpu_config.forward.out_res = resolution
    rpu_config.forward.out_bound = settings.output_bound
    rpu_config.forward.out_sto_round = False
    rpu_config.forward.out_scale = 1.0
    rpu_config.forward.out_asymmetry = 0.0
    rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.inp_noise = 0.0
    rpu_config.forward.out_noise = 0.0
    rpu_config.forward.out_noise_std = 0.0
    rpu_config.forward.w_noise = 0.0
    rpu_config.forward.ir_drop = 0.0
    rpu_config.forward.out_nonlinearity = 0.0
    rpu_config.forward.out_nonlinearity_std = 0.0
    rpu_config.forward.slope_calibration = 0.0
    rpu_config.forward.v_offset_std = 0.0
    rpu_config.forward.r_series = 0.0
    rpu_config.forward.w_read_asymmetry_dtod = 0.0
    rpu_config.forward.w_noise_type = WeightNoiseType.NONE
    rpu_config.noise_model = PCMLikeNoiseModel(
        prog_noise_scale=0.0,
        read_noise_scale=0.0,
        drift_scale=0.0,
    )
    rpu_config.drift_compensation = None
    return rpu_config


def analog_configuration(settings: ManualAnalogSettings) -> dict[str, Any]:
    settings.validate()
    resolution = 1.0 / float(2**settings.adc_dac_bits - 2)
    return {
        "manual_preprocessing_version": 2,
        "clipping": "manual_projection_wide_symmetric_gaussian",
        "clip_sigma": settings.clip_sigma,
        "clip_order": "clip_once_before_noise",
        "reclip_after_noise": False,
        "programmed_range_mode": settings.range_mode,
        "noise_unit": "normalized_std_fraction_of_programmed_projection_range",
        "reference_noise_std": settings.reference_noise_std,
        "noise_distribution": "manual_additive_iid_gaussian_logical_weight_domain",
        "tile_size": settings.tile_size,
        "adc_dac_bits": settings.adc_dac_bits,
        "input_resolution": resolution,
        "output_resolution": resolution,
        "input_bound": 1.0,
        "output_bound": settings.output_bound,
        "input_stochastic_rounding": False,
        "output_stochastic_rounding": False,
        "bound_management": "ITERATIVE",
        "noise_management": "ABS_MAX",
        "forward_is_perfect": False,
        "weight_scaling_omega": settings.weight_scaling_omega,
        "weight_scaling_columnwise": settings.weight_scaling_columnwise,
        "internal_clipping": False,
        "internal_programming_noise": False,
        "internal_read_noise": False,
        "internal_drift": False,
        "internal_forward_weight_noise": False,
    }


def get_analog_weights_exact(module: Any) -> tuple[Tensor, Tensor | None]:
    weight, bias = module.get_weights(
        apply_weight_scaling=True,
        realistic=False,
    )
    weight = weight.detach().cpu().float().contiguous()
    bias = None if bias is None else bias.detach().cpu().float().contiguous()
    return weight, bias


def set_analog_weights_exact(
    module: Any,
    canonical_weight: Tensor,
    bias: Tensor | None,
    *,
    verification_tolerance: float = 3e-6,
    verify: bool = True,
) -> None:
    """Write logical weights while preserving clean AIHWKit mapping scales."""
    expected_shape = (module.out_features, module.in_features)
    if tuple(canonical_weight.shape) != expected_shape:
        raise ValueError(
            f"Weight shape {tuple(canonical_weight.shape)} does not match "
            f"analog layer shape {expected_shape}."
        )
    if verification_tolerance <= 0 or not math.isfinite(verification_tolerance):
        raise ValueError("verification_tolerance must be finite and positive.")
    if not bool(torch.isfinite(canonical_weight).all().item()):
        raise ValueError("canonical_weight contains non-finite values.")

    analog_module = getattr(module, "analog_module", None)
    required = ("in_sizes", "out_sizes", "array")
    if analog_module is None or any(not hasattr(analog_module, name) for name in required):
        raise TypeError(
            "Expected AnalogLinearMapped to expose TileModuleArray in_sizes, "
            "out_sizes, and array attributes."
        )

    in_start = 0
    tile_count = 0
    for in_size, in_tiles in zip(analog_module.in_sizes, analog_module.array):
        in_end = in_start + int(in_size)
        out_start = 0
        if len(in_tiles) != len(analog_module.out_sizes):
            raise RuntimeError("Mapped-tile structure does not match out_sizes.")
        for out_size, analog_tile in zip(analog_module.out_sizes, in_tiles):
            out_end = out_start + int(out_size)
            logical_slice = canonical_weight[
                out_start:out_end, in_start:in_end
            ].detach().to(
                device=torch.device(analog_tile.device),
                dtype=analog_tile.get_dtype(),
            ).contiguous()
            scales = analog_tile.get_scales()
            if scales is None:
                internal_slice = logical_slice
            else:
                scales = scales.detach().to(
                    device=logical_slice.device,
                    dtype=logical_slice.dtype,
                ).reshape(-1)
                if scales.numel() not in (1, int(out_size)):
                    raise ValueError("Unexpected mapped-tile scale shape.")
                if not bool(torch.isfinite(scales).all().item()) or bool(
                    (scales == 0).any().item()
                ):
                    raise ValueError("Analog mapping scales are invalid.")
                internal_slice = logical_slice / scales.view(-1, 1)
            analog_tile.set_weights(
                internal_slice,
                None,
                apply_weight_scaling=False,
                realistic=False,
            )
            tile_count += 1
            out_start = out_end
        if out_start != module.out_features:
            raise RuntimeError("Output partitions do not cover the full layer.")
        in_start = in_end
    if tile_count == 0 or in_start != module.in_features:
        raise RuntimeError("Input partitions do not cover the full layer.")

    digital_bias = getattr(analog_module, "bias", None)
    if bias is None:
        if digital_bias is not None:
            digital_bias.data.zero_()
    else:
        if digital_bias is None:
            raise ValueError("A bias was supplied, but the analog layer has no bias.")
        if tuple(bias.shape) != (module.out_features,):
            raise ValueError("Bias shape does not match analog layer output size.")
        digital_bias.data.copy_(
            bias.detach().to(device=digital_bias.device, dtype=digital_bias.dtype)
        )

    if verify:
        actual_weight, actual_bias = get_analog_weights_exact(module)
        expected_cpu = canonical_weight.detach().cpu().float().contiguous()
        max_error = float((actual_weight - expected_cpu).abs().max().item())
        if max_error > verification_tolerance:
            raise RuntimeError(
                "Logical weight write/read mismatch while preserving scales: "
                f"max_error={max_error:.8e}."
            )
        if bias is not None:
            if actual_bias is None:
                raise RuntimeError("AIHWKit did not retain the supplied digital bias.")
            bias_error = float(
                (actual_bias - bias.detach().cpu().float()).abs().max().item()
            )
            if bias_error > verification_tolerance:
                raise RuntimeError(f"AIHWKit bias mismatch: {bias_error:.8e}.")
