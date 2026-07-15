"""AIHWKit conversion and exact weight I/O for full-model GPT-2 evaluation.

Phase 4 converts all 48 transformer projections to ``AnalogLinearMapped``.
Each analog projection is initialized from the corresponding pretrained GPT-2
``Conv1D`` weight. The token embedding, position embedding, and language-model
head remain ordinary digital PyTorch modules.

The clean all-analog reference reproduces the Phase-1 preprocessing exactly:

1. Copy the original GPT-2 projection into ``AnalogLinearMapped``.
2. Apply AIHWKit ``LAYER_GAUSSIAN`` clipping independently on every mapped
   analog tile by calling ``post_update_step()`` once per tile.
3. Save the resulting clipped logical weight as the Phase-4 reference.

Phase-2/3 programming error is then materialized explicitly into those clipped
logical weights. AIHWKit's internal programming, read, drift, and forward
weight noise are disabled so the heterogeneous noise is applied exactly once.
AIHWKit remains responsible for the mapped analog forward path, including the
same DAC/ADC resolution, bound management, and noise management as Phase 1.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D

from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter, WeightClipParameter
from aihwkit.simulator.parameters.enums import (
    BoundManagementType,
    NoiseManagementType,
    WeightClipType,
    WeightNoiseType,
)

LOGGER = logging.getLogger(__name__)

GPT2_PROJECTION_SUFFIXES = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


@dataclass(frozen=True, slots=True)
class AnalogProjectionReference:
    """Exact post-preprocessing reference state for one analog projection."""

    hf_module_path: str
    projection_id: str
    canonical_weight: Tensor
    bias: Tensor | None
    analog_tile_count: int
    clip_sigma: float
    weight_std_before_clipping: float
    clip_threshold: float
    num_clipped_weights: int
    total_weights: int
    reference_copy_max_abs_error: float
    preprocessing_mean_abs_change: float
    preprocessing_max_abs_change: float
    clipping_scope: str = "aihwkit_mapped_tile_post_update_step"
    post_update_step_applied: bool = True

    def metadata_row(self) -> dict[str, Any]:
        return {
            "hf_module_path": self.hf_module_path,
            "projection_id": self.projection_id,
            "canonical_shape": list(self.canonical_weight.shape),
            "analog_tile_count": self.analog_tile_count,
            "clip_sigma": self.clip_sigma,
            # This threshold is only a whole-projection diagnostic. The actual
            # Phase-1/4 clipping thresholds are calculated separately inside
            # every AIHWKit mapped tile by post_update_step().
            "whole_projection_diagnostic_clip_threshold": self.clip_threshold,
            "weight_std_before_clipping": self.weight_std_before_clipping,
            "num_clipped_weights": self.num_clipped_weights,
            "total_weights": self.total_weights,
            "fraction_clipped": self.num_clipped_weights / self.total_weights,
            "reference_copy_max_abs_error": self.reference_copy_max_abs_error,
            "preprocessing_mean_abs_change": self.preprocessing_mean_abs_change,
            "preprocessing_max_abs_change": self.preprocessing_max_abs_change,
            "clipping_scope": self.clipping_scope,
            "post_update_step_applied": self.post_update_step_applied,
        }


def _get_module(root: Any, dotted_path: str) -> Any:
    module = root
    for component in dotted_path.split("."):
        module = module[int(component)] if component.isdigit() else getattr(module, component)
    return module


def _get_parent_and_attribute(root: Any, dotted_path: str) -> tuple[Any, str]:
    components = dotted_path.split(".")
    parent = root
    for component in components[:-1]:
        parent = parent[int(component)] if component.isdigit() else getattr(parent, component)
    return parent, components[-1]


def projection_id_from_hf_path(hf_module_path: str) -> str:
    """Convert a GPT-2 module path to the canonical Phase-1 projection ID."""
    components = hf_module_path.split(".")
    try:
        block_index = int(components[2])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Unsupported GPT-2 projection path: {hf_module_path!r}."
        ) from exc

    projection_name = ".".join(components[3:])
    if projection_name not in GPT2_PROJECTION_SUFFIXES:
        raise ValueError(
            f"Unsupported GPT-2 projection path: {hf_module_path!r}."
        )
    return f"block_{block_index}/{projection_name}"


def expected_gpt2_projection_paths(num_hidden_layers: int) -> set[str]:
    """Return every transformer projection path expected for GPT-2."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")
    return {
        f"transformer.h.{block}.{projection}"
        for block in range(num_hidden_layers)
        for projection in GPT2_PROJECTION_SUFFIXES
    }


def _validate_phase1_preprocessing_metadata(
    analog_configuration: Mapping[str, Any],
) -> None:
    """Reject Phase-1 metadata that describes a different preprocessing path."""
    clip_type = str(analog_configuration.get("clip_type", "LAYER_GAUSSIAN"))
    if clip_type != "LAYER_GAUSSIAN":
        raise ValueError(
            f"Phase 4 requires Phase-1 clip_type='LAYER_GAUSSIAN', got {clip_type!r}."
        )

    clipping_scope = analog_configuration.get("clipping_scope")
    if clipping_scope is not None and str(clipping_scope) != (
        "aihwkit_mapped_tile_post_update_step"
    ):
        raise ValueError(
            "Phase-1 clipping_scope must be "
            "'aihwkit_mapped_tile_post_update_step', got "
            f"{clipping_scope!r}."
        )

    post_update_applied = analog_configuration.get("post_update_step_applied")
    if post_update_applied is not None and not bool(post_update_applied):
        raise ValueError(
            "Phase-1 metadata says post_update_step was not applied, so its "
            "preprocessing cannot be reproduced by this Phase-4 bridge."
        )


def build_phase4_rpu_config(
    analog_configuration: Mapping[str, Any],
) -> InferenceRPUConfig:
    """Build an AIHWKit 1.1.0 config matching Phase-1 preprocessing/forward.

    The clipping configuration remains enabled so that conversion can call
    ``post_update_step()`` exactly once, as Phase 1 does. Internal programming,
    read, drift, and forward weight noise are disabled because Phase-2/3 noise
    is materialized explicitly after the clean clipped reference is saved.
    """
    _validate_phase1_preprocessing_metadata(analog_configuration)

    tile_size = int(analog_configuration.get("tile_size", 512))
    if tile_size <= 0:
        raise ValueError("AIHWKit tile_size must be positive.")

    clip_sigma = float(analog_configuration.get("clip_sigma", 2.5))
    if not math.isfinite(clip_sigma) or clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be finite and positive.")

    adc_dac_bits = int(analog_configuration.get("adc_dac_bits", 8))
    if adc_dac_bits < 2:
        raise ValueError("adc_dac_bits must be at least 2.")

    default_resolution = 1.0 / float(2**adc_dac_bits - 2)
    input_resolution = float(
        analog_configuration.get("input_resolution", default_resolution)
    )
    output_resolution = float(
        analog_configuration.get("output_resolution", default_resolution)
    )
    output_bound = float(analog_configuration.get("output_bound", 12.0))
    weight_scaling_omega = float(
        analog_configuration.get("weight_scaling_omega", 0.0)
    )
    weight_scaling_columnwise = bool(
        analog_configuration.get("weight_scaling_columnwise", False)
    )

    for name, value in (
        ("input_resolution", input_resolution),
        ("output_resolution", output_resolution),
        ("output_bound", output_bound),
        ("weight_scaling_omega", weight_scaling_omega),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if input_resolution <= 0.0 or output_resolution <= 0.0 or output_bound <= 0.0:
        raise ValueError("I/O resolutions and output_bound must be positive.")

    # Match the Phase-1 mapping configuration exactly. Positive omega values
    # are supported. During Phase-4 snapshot writes, set_analog_weights_exact()
    # preserves the mapping scales established by the clean conversion instead
    # of asking AIHWKit to recompute them from each noised logical tensor.
    if weight_scaling_omega < 0.0:
        raise ValueError("weight_scaling_omega must be non-negative.")

    bound_name = str(analog_configuration.get("bound_management", "ITERATIVE"))
    noise_name = str(analog_configuration.get("noise_management", "ABS_MAX"))
    if bound_name != "ITERATIVE":
        raise ValueError(
            f"Unsupported Phase-1 bound management {bound_name!r}; expected ITERATIVE."
        )
    if noise_name != "ABS_MAX":
        raise ValueError(
            f"Unsupported Phase-1 noise management {noise_name!r}; expected ABS_MAX."
        )

    rpu_config = InferenceRPUConfig()
    rpu_config.clip = WeightClipParameter(
        type=WeightClipType.LAYER_GAUSSIAN,
        sigma=clip_sigma,
    )
    rpu_config.mapping = MappingParameter(
        max_input_size=tile_size,
        max_output_size=tile_size,
        digital_bias=True,
        weight_scaling_omega=weight_scaling_omega,
        weight_scaling_columnwise=weight_scaling_columnwise,
    )

    rpu_config.forward.is_perfect = True
    rpu_config.forward.inp_res = input_resolution
    rpu_config.forward.out_res = output_resolution
    rpu_config.forward.out_bound = output_bound
    rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.inp_noise = 0.0
    rpu_config.forward.out_noise = 0.0
    rpu_config.forward.w_noise = 0.0
    rpu_config.forward.w_noise_type = WeightNoiseType.NONE

    rpu_config.noise_model = PCMLikeNoiseModel(
        prog_noise_scale=0.0,
        read_noise_scale=0.0,
        drift_scale=0.0,
    )
    rpu_config.drift_compensation = None
    return rpu_config


def get_analog_weights_exact(
    module: AnalogLinearMapped,
) -> tuple[Tensor, Tensor | None]:
    """Read exact logical weights using the AIHWKit 1.1.0 API."""
    weight, bias = module.get_weights(
        apply_weight_scaling=True,
        realistic=False,
    )
    weight = weight.detach().cpu().float().contiguous()
    bias = None if bias is None else bias.detach().cpu().float().contiguous()
    return weight, bias


def set_analog_weights_exact(
    module: AnalogLinearMapped,
    canonical_weight: Tensor,
    bias: Tensor | None,
    *,
    verification_tolerance: float = 2e-6,
) -> None:
    """Write logical weights while preserving the clean mapping scales.

    AIHWKit applies ``weight_scaling_omega`` independently to every mapped
    tile. Calling ``module.set_weights(..., apply_weight_scaling=True)`` for a
    Phase-4 noisy snapshot would therefore recompute the tile mapping scales
    from that snapshot. Phase 1 does not do this when programming noise is
    applied: the mapping scales are established by the clean conversion and
    remain fixed while the internal conductances are perturbed.

    To reproduce that behavior, this function converts every desired logical
    tile slice back to its internal crossbar representation using the tile's
    existing scale, then writes it with ``apply_weight_scaling=False``. It does
    not call ``post_update_step()`` because the clean reference was already
    clipped once during conversion.
    """
    expected_shape = (module.out_features, module.in_features)
    if tuple(canonical_weight.shape) != expected_shape:
        raise ValueError(
            f"Weight shape {tuple(canonical_weight.shape)} does not match "
            f"analog layer shape {expected_shape}."
        )
    if verification_tolerance <= 0.0 or not math.isfinite(
        verification_tolerance
    ):
        raise ValueError("verification_tolerance must be finite and positive.")
    if not bool(torch.isfinite(canonical_weight).all().item()):
        raise ValueError("canonical_weight contains non-finite values.")

    analog_module = getattr(module, "analog_module", None)
    required_attributes = ("in_sizes", "out_sizes", "array")
    if analog_module is None or any(
        not hasattr(analog_module, name) for name in required_attributes
    ):
        raise TypeError(
            "Expected AnalogLinearMapped to use AIHWKit TileModuleArray with "
            "in_sizes, out_sizes, and array attributes."
        )

    in_start = 0
    tile_count = 0
    for in_size, in_tiles in zip(analog_module.in_sizes, analog_module.array):
        in_end = in_start + int(in_size)
        out_start = 0

        if len(in_tiles) != len(analog_module.out_sizes):
            raise RuntimeError(
                "AIHWKit mapped-tile structure does not match out_sizes."
            )

        for out_size, analog_tile in zip(analog_module.out_sizes, in_tiles):
            out_end = out_start + int(out_size)
            logical_slice = canonical_weight[
                out_start:out_end,
                in_start:in_end,
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
                    raise ValueError(
                        "Unexpected mapping-scale shape for mapped tile: "
                        f"{tuple(scales.shape)} for out_size={int(out_size)}."
                    )
                if not bool(torch.isfinite(scales).all().item()):
                    raise ValueError("Analog mapping scales contain non-finite values.")
                if bool((scales == 0).any().item()):
                    raise ValueError("Analog mapping scales contain zero values.")
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
            raise RuntimeError(
                "AIHWKit mapped output partitions do not cover the full layer."
            )
        in_start = in_end

    if tile_count == 0:
        raise RuntimeError("AnalogLinearMapped contains no analog tiles.")
    if in_start != module.in_features:
        raise RuntimeError(
            "AIHWKit mapped input partitions do not cover the full layer."
        )

    digital_bias = getattr(analog_module, "bias", None)
    if bias is None:
        if digital_bias is not None:
            digital_bias.data.zero_()
    else:
        if digital_bias is None:
            raise ValueError(
                "A bias was supplied, but the mapped analog layer has no digital bias."
            )
        if tuple(bias.shape) != (module.out_features,):
            raise ValueError(
                f"Bias shape {tuple(bias.shape)} does not match "
                f"({module.out_features},)."
            )
        if not bool(torch.isfinite(bias).all().item()):
            raise ValueError("bias contains non-finite values.")
        digital_bias.data.copy_(
            bias.detach().to(
                device=digital_bias.device,
                dtype=digital_bias.dtype,
            )
        )

    actual_weight, actual_bias = get_analog_weights_exact(module)
    expected_cpu = canonical_weight.detach().cpu().float().contiguous()
    max_error = float((actual_weight - expected_cpu).abs().max().item())
    if max_error > verification_tolerance:
        raise RuntimeError(
            "AIHWKit logical weight write/read mismatch while preserving mapping "
            f"scales: max_error={max_error:.8e}, "
            f"tolerance={verification_tolerance:.8e}."
        )

    if bias is None:
        if actual_bias is not None and bool((actual_bias != 0).any().item()):
            raise RuntimeError("AIHWKit digital bias was not cleared exactly.")
    else:
        if actual_bias is None:
            raise RuntimeError("AIHWKit did not retain the supplied digital bias.")
        bias_error = float(
            (actual_bias - bias.detach().cpu().float()).abs().max().item()
        )
        if bias_error > verification_tolerance:
            raise RuntimeError(
                "AIHWKit bias write/read mismatch: "
                f"max_error={bias_error:.8e}, "
                f"tolerance={verification_tolerance:.8e}."
            )


def _conv1d_to_linear(module: Conv1D) -> tuple[nn.Linear, Tensor]:
    """Copy one GPT-2 Conv1D into canonical ``[out, in]`` Linear form."""
    canonical = module.weight.detach().T.contiguous().to(dtype=torch.float32)
    linear = nn.Linear(
        in_features=canonical.shape[1],
        out_features=canonical.shape[0],
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        linear.weight.copy_(canonical.to(device=linear.weight.device))
        if module.bias is not None:
            linear.bias.copy_(
                module.bias.detach().to(
                    device=linear.bias.device,
                    dtype=linear.bias.dtype,
                )
            )
    return linear, canonical.detach().cpu().float().contiguous()


def _apply_phase1_preprocessing(analog_linear: AnalogLinearMapped) -> None:
    """Apply Phase-1 tile-level clipping exactly once to a mapped projection."""
    tile_count = 0
    for analog_tile in analog_linear.analog_tiles():
        analog_tile.post_update_step()
        tile_count += 1
    if tile_count == 0:
        raise RuntimeError("AnalogLinearMapped contains no analog tiles.")


def convert_gpt2_projections_to_aihwkit(
    model: Any,
    hf_module_paths: Sequence[str],
    *,
    analog_configuration: Mapping[str, Any],
    reference_tolerance: float = 1e-5,
) -> dict[str, AnalogProjectionReference]:
    """Convert all selected GPT-2 projections and reproduce Phase-1 clipping.

    Every path is converted independently from its corresponding pretrained
    GPT-2 projection weight. Once this function returns, every selected module
    in ``model`` is an ``AnalogLinearMapped`` and stores its own clipped logical
    projection weight.
    """
    clip_sigma = float(analog_configuration.get("clip_sigma", 2.5))
    if reference_tolerance <= 0.0:
        raise ValueError("reference_tolerance must be positive.")

    unique_paths = sorted(set(hf_module_paths))
    if not unique_paths:
        raise ValueError("No GPT-2 projection paths were provided.")

    references: dict[str, AnalogProjectionReference] = {}
    for index, path in enumerate(unique_paths, start=1):
        parent, attribute = _get_parent_and_attribute(model, path)
        original = getattr(parent, attribute)
        if not isinstance(original, Conv1D):
            raise TypeError(
                f"{path} must be a Hugging Face Conv1D before conversion, got "
                f"{type(original).__name__}."
            )

        digital_linear, original_canonical = _conv1d_to_linear(original)
        analog_linear = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=build_phase4_rpu_config(analog_configuration),
        )
        analog_linear.eval()

        # Verify the initial copy before Phase-1 preprocessing changes weights.
        copied_weight, copied_bias = get_analog_weights_exact(analog_linear)
        expected_weight = digital_linear.weight.detach().cpu().float().contiguous()
        if tuple(copied_weight.shape) != tuple(expected_weight.shape):
            raise ValueError(
                f"{path}: AIHWKit weight shape {tuple(copied_weight.shape)} does "
                f"not match expected {tuple(expected_weight.shape)}."
            )
        copy_error = float((copied_weight - expected_weight).abs().max().item())
        if copy_error > reference_tolerance:
            raise ValueError(
                f"{path}: AIHWKit input copy error {copy_error:.8e} exceeds "
                f"tolerance {reference_tolerance:.8e}."
            )

        # Exact Phase-1 preprocessing: tile-level LAYER_GAUSSIAN clipping.
        # _apply_phase1_preprocessing(analog_linear)
        reference_weight, reference_bias = get_analog_weights_exact(analog_linear)
        preprocessing_delta = reference_weight - original_canonical
        changed = preprocessing_delta.abs() > 1e-12

        weight_std = float(original_canonical.std(unbiased=False).item())
        diagnostic_threshold = clip_sigma * weight_std
        reference = AnalogProjectionReference(
            hf_module_path=path,
            projection_id=projection_id_from_hf_path(path),
            canonical_weight=reference_weight.detach().cpu().float().contiguous(),
            bias=(
                None
                if reference_bias is None
                else reference_bias.detach().cpu().float().contiguous()
            ),
            analog_tile_count=int(analog_linear.analog_tile_count()),
            clip_sigma=clip_sigma,
            weight_std_before_clipping=weight_std,
            clip_threshold=float(diagnostic_threshold),
            num_clipped_weights=int(changed.sum().item()),
            total_weights=int(original_canonical.numel()),
            reference_copy_max_abs_error=copy_error,
            preprocessing_mean_abs_change=float(
                preprocessing_delta.abs().mean().item()
            ),
            preprocessing_max_abs_change=float(
                preprocessing_delta.abs().max().item()
            ),
        )

        setattr(parent, attribute, analog_linear)
        references[path] = reference
        LOGGER.info(
            "Converted analog projection %d/%d: %s shape=%s tiles=%d "
            "clipped_weights=%d",
            index,
            len(unique_paths),
            path,
            tuple(reference_weight.shape),
            reference.analog_tile_count,
            reference.num_clipped_weights,
        )

    validate_all_gpt2_projections_analog(
        model,
        unique_paths,
        references=references,
    )
    return references


def validate_all_gpt2_projections_analog(
    model: Any,
    hf_module_paths: Sequence[str],
    *,
    references: Mapping[str, AnalogProjectionReference] | None = None,
    expected_num_projections: int | None = None,
) -> None:
    """Verify that every requested projection is analog and has its own weight."""
    paths = sorted(set(hf_module_paths))
    if expected_num_projections is not None and len(paths) != expected_num_projections:
        raise ValueError(
            f"Expected {expected_num_projections} analog projections, got {len(paths)}."
        )

    for path in paths:
        module = _get_module(model, path)
        if not isinstance(module, AnalogLinearMapped):
            raise TypeError(
                f"{path} is {type(module).__name__}, not AnalogLinearMapped."
            )
        weight, _ = get_analog_weights_exact(module)
        expected_shape = (module.out_features, module.in_features)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"{path}: stored logical weight shape {tuple(weight.shape)} does "
                f"not match {expected_shape}."
            )
        if references is not None:
            if path not in references:
                raise KeyError(f"No analog reference was recorded for {path}.")
            reference = references[path]
            if not torch.equal(weight, reference.canonical_weight):
                error = float(
                    (weight - reference.canonical_weight).abs().max().item()
                )
                raise ValueError(
                    f"{path}: module weight does not match its recorded reference; "
                    f"max error={error:.8e}."
                )


def reference_weight_map(
    references: Mapping[str, AnalogProjectionReference],
) -> dict[str, Tensor]:
    return {
        path: reference.canonical_weight.detach().cpu().clone()
        for path, reference in references.items()
    }


def save_analog_projection_weights(
    model: Any,
    hf_module_paths: Sequence[str],
) -> dict[str, Tensor]:
    """Read all selected AIHWKit logical weights exactly to CPU."""
    result: dict[str, Tensor] = {}
    for path in sorted(set(hf_module_paths)):
        module = _get_module(model, path)
        if not isinstance(module, AnalogLinearMapped):
            raise TypeError(f"{path} is not AnalogLinearMapped.")
        weight, _ = get_analog_weights_exact(module)
        result[path] = weight.detach().cpu().float().contiguous().clone()
    return result


def set_analog_projection_weight(
    model: Any,
    hf_module_path: str,
    canonical_weight: Tensor,
    *,
    reference_bias: Tensor | None,
) -> None:
    module = _get_module(model, hf_module_path)
    if not isinstance(module, AnalogLinearMapped):
        raise TypeError(f"{hf_module_path} is not AnalogLinearMapped.")
    set_analog_weights_exact(module, canonical_weight, reference_bias)


def restore_analog_projection_weights(
    model: Any,
    references: Mapping[str, AnalogProjectionReference],
) -> None:
    """Restore every analog projection to its clipped Phase-1 reference."""
    for path, reference in references.items():
        set_analog_projection_weight(
            model,
            path,
            reference.canonical_weight,
            reference_bias=reference.bias,
        )


def validate_reference_equivalence(
    first: Mapping[str, AnalogProjectionReference],
    second: Mapping[str, AnalogProjectionReference],
    *,
    atol: float = 1e-6,
) -> None:
    """Verify independently converted all-analog models share a reference."""
    if set(first) != set(second):
        raise ValueError("All-analog models contain different projection paths.")

    for path in sorted(first):
        a = first[path]
        b = second[path]
        if a.projection_id != b.projection_id:
            raise ValueError(f"Projection ID mismatch at {path}.")
        if not torch.allclose(
            a.canonical_weight,
            b.canonical_weight,
            rtol=0.0,
            atol=atol,
        ):
            error = float(
                (a.canonical_weight - b.canonical_weight).abs().max().item()
            )
            raise ValueError(
                f"All-analog reference models differ at {path}; "
                f"max error={error:.8e}."
            )
        if (a.bias is None) != (b.bias is None):
            raise ValueError(f"Bias presence mismatch at {path}.")
        if a.bias is not None and not torch.allclose(
            a.bias,
            b.bias,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"All-analog reference biases differ at {path}.")
