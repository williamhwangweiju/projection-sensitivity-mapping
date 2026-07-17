"""AIHWKit-independent manual clipping and Gaussian noise pipeline.

This module is the single source of truth used by both Phase 1 and Phase 4.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ManualAnalogSettings:
    clip_sigma: float
    range_mode: str
    reference_noise_std: float
    tile_size: int
    adc_dac_bits: int
    output_bound: float | None
    weight_scaling_omega: float
    weight_scaling_columnwise: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ManualAnalogSettings":
        analog = config["analog"]
        return cls(
            clip_sigma=float(analog["clip_sigma"]),
            range_mode=str(analog.get("range_mode", "peak_to_peak")),
            reference_noise_std=float(analog["reference_noise_std"]),
            tile_size=int(analog["tile_size"]),
            adc_dac_bits=int(analog["adc_dac_bits"]),
            output_bound=(
                None
                if analog.get("output_bound") is None
                else float(analog["output_bound"])
            ),
            weight_scaling_omega=float(analog.get("weight_scaling_omega", 1.0)),
            weight_scaling_columnwise=bool(
                analog.get("weight_scaling_columnwise", False)
            ),
        )

    def validate(self) -> None:
        if not math.isfinite(self.clip_sigma) or self.clip_sigma <= 0:
            raise ValueError("clip_sigma must be finite and positive.")
        if self.range_mode not in {"peak_to_peak", "absmax"}:
            raise ValueError("range_mode must be peak_to_peak or absmax.")
        if not math.isfinite(self.reference_noise_std) or self.reference_noise_std < 0:
            raise ValueError("reference_noise_std must be finite and non-negative.")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if self.adc_dac_bits < 2:
            raise ValueError("adc_dac_bits must be at least two.")
        if self.output_bound is not None and (
            not math.isfinite(self.output_bound) or self.output_bound <= 0
        ):
            raise ValueError("output_bound must be null (AIHWKit default) or finite and positive.")
        if not math.isfinite(self.weight_scaling_omega) or self.weight_scaling_omega < 0:
            raise ValueError("weight_scaling_omega must be finite and non-negative.")


@dataclass(frozen=True)
class ProjectionPreprocessing:
    original_std: float
    clip_threshold: float
    programmed_range: float
    range_mode: str
    num_weights: int
    num_clipped: int
    fraction_clipped: float
    original_checksum: str
    clipped_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedProjection:
    clipped_weight: Tensor
    preprocessing: ProjectionPreprocessing


def tensor_checksum(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().float().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch using backend-compatible ranges."""

    raw_seed = int(seed)
    torch_seed = raw_seed % (2**63 - 1)
    numpy_seed = raw_seed % (2**32)

    random.seed(torch_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def projection_noise_seed(realization_seed: int, projection_id: str, realization_offset: int = 0) -> int:
    digest = hashlib.sha256(projection_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(realization_seed) + int(realization_offset) + offset) % (2**63 - 1))


def prepare_projection_weight(
    canonical_weight: Tensor,
    settings: ManualAnalogSettings,
) -> PreparedProjection:
    settings.validate()
    clean = canonical_weight.detach().cpu().float().contiguous()
    if clean.ndim != 2:
        raise ValueError("Projection weight must be canonical [out, in].")
    if not bool(torch.isfinite(clean).all().item()):
        raise ValueError("Projection weight contains non-finite values.")

    population_std = clean.std(unbiased=False)
    threshold = settings.clip_sigma * population_std
    if not bool(torch.isfinite(threshold).item()) or float(threshold) <= 0:
        raise ValueError("Projection has zero or invalid standard deviation.")
    clipped = clean.clamp(min=-threshold, max=threshold).contiguous()
    if settings.range_mode == "peak_to_peak":
        programmed_range = clipped.max() - clipped.min()
    else:
        programmed_range = clipped.abs().max()
    if not bool(torch.isfinite(programmed_range).item()) or float(programmed_range) <= 0:
        raise ValueError("Programmed range must be finite and positive.")

    changed = clipped.ne(clean)
    num_weights = int(clean.numel())
    num_clipped = int(changed.sum().item())
    metadata = ProjectionPreprocessing(
        original_std=float(population_std.item()),
        clip_threshold=float(threshold.item()),
        programmed_range=float(programmed_range.item()),
        range_mode=settings.range_mode,
        num_weights=num_weights,
        num_clipped=num_clipped,
        fraction_clipped=float(num_clipped / num_weights),
        original_checksum=tensor_checksum(clean),
        clipped_checksum=tensor_checksum(clipped),
    )
    return PreparedProjection(clipped_weight=clipped, preprocessing=metadata)


def normalized_noise_to_absolute_sigma(
    normalized_sigma: float | Tensor,
    programmed_range: float,
) -> float | Tensor:
    if not math.isfinite(programmed_range) or programmed_range <= 0:
        raise ValueError("programmed_range must be finite and positive.")
    if isinstance(normalized_sigma, Tensor):
        if not bool(torch.isfinite(normalized_sigma).all().item()):
            raise ValueError("normalized_sigma contains non-finite values.")
        if bool((normalized_sigma < 0).any().item()):
            raise ValueError("normalized_sigma cannot be negative.")
        return normalized_sigma * float(programmed_range)
    value = float(normalized_sigma)
    if not math.isfinite(value) or value < 0:
        raise ValueError("normalized_sigma must be finite and non-negative.")
    return value * float(programmed_range)


def materialize_manual_noise(
    clipped_weight: Tensor,
    normalized_sigma: float | Tensor,
    programmed_range: float,
    gaussian_field: Tensor,
) -> tuple[Tensor, Tensor]:
    reference = clipped_weight.detach().cpu().float().contiguous()
    z = gaussian_field.detach().cpu().float().contiguous()
    if z.shape != reference.shape:
        raise ValueError("Gaussian field and projection weight shapes differ.")
    absolute_sigma = normalized_noise_to_absolute_sigma(
        normalized_sigma, programmed_range
    )
    if isinstance(absolute_sigma, Tensor):
        sigma_map = absolute_sigma.detach().cpu().float().contiguous()
        if sigma_map.shape != reference.shape:
            raise ValueError("Per-weight sigma map has the wrong shape.")
    else:
        sigma_map = torch.full_like(reference, float(absolute_sigma))
    noise = sigma_map * z
    # Deliberately no post-noise clipping.
    return (reference + noise).contiguous(), noise.contiguous()
