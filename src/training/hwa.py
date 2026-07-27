"""Hardware-aware noise-injection training utilities.

Phase 0 fine-tunes the model under the same weight-noise model that Phases 1
and 4 deploy: symmetric clipping at ``clip_sigma`` population standard
deviations and additive i.i.d. Gaussian noise expressed as a fraction of the
projection's programmed range.

The scheme is perturb-forward / clean-update (Joshi et al., Nat. Commun.
2020): each wrapped projection computes its forward pass from a temporarily
clipped and noised copy of the live weight, so gradients are evaluated at the
perturbed point while optimizer updates apply to the clean parameters. The
parameters themselves are never mutated by the noise path.

This module is deliberately AIHWKit-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.common.manual_weights import projection_noise_seed
from src.common.projections import (
    ProjectionHandle,
    iter_gpt2_projections,
)

try:  # transformers is required by every runner but not by structural tests.
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None


@dataclass(frozen=True)
class HWANoiseSettings:
    """Noise-injection configuration shared by all wrapped projections."""

    clip_sigma: float
    range_mode: str
    noise_std_range: tuple[float, float]
    clip_in_forward: bool
    include_lm_head: bool
    exclude_projection_ids: frozenset[str]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HWANoiseSettings":
        analog = config["analog"]
        noise = config["hwa_training"]["noise"]
        low, high = (float(value) for value in noise["noise_std_range"])
        return cls(
            clip_sigma=float(analog["clip_sigma"]),
            range_mode=str(analog.get("range_mode", "peak_to_peak")),
            noise_std_range=(low, high),
            clip_in_forward=bool(noise.get("clip_in_forward", True)),
            include_lm_head=bool(noise.get("include_lm_head", True)),
            exclude_projection_ids=frozenset(
                str(value) for value in noise.get("exclude_projection_ids", [])
            ),
        )

    def validate(self) -> None:
        low, high = self.noise_std_range
        if not (0.0 <= low <= high):
            raise ValueError("noise_std_range must satisfy 0 <= low <= high.")
        if self.clip_sigma <= 0:
            raise ValueError("clip_sigma must be positive.")
        if self.range_mode not in {"peak_to_peak", "absmax"}:
            raise ValueError("range_mode must be peak_to_peak or absmax.")


def straight_through_clip(
    weight: Tensor, clip_sigma: float, range_mode: str
) -> tuple[Tensor, Tensor]:
    """Clip at ``clip_sigma`` population stds with straight-through gradients.

    Returns the clipped weight (whose backward pass treats the clip as the
    identity, so clipped coordinates keep receiving gradient) and the scalar
    programmed range of the clipped matrix. The threshold and range follow
    ``prepare_projection_weight`` exactly, but are recomputed per call because
    the weights move during training.
    """
    with torch.no_grad():
        detached = weight.detach()
        threshold = clip_sigma * detached.float().std(unbiased=False)
        clipped_detached = detached.clamp(-threshold, threshold)
        if range_mode == "peak_to_peak":
            programmed_range = clipped_detached.max() - clipped_detached.min()
        else:
            programmed_range = clipped_detached.abs().max()
    clipped = weight + (weight.clamp(-threshold, threshold) - weight).detach()
    return clipped, programmed_range


class NoisyProjection(nn.Module):
    """Forward-time clip + Gaussian weight noise around a wrapped projection.

    The wrapped module stays registered as a submodule, so parameter identity
    (and therefore optimizer state) is unaffected by wrapping. Noise is drawn
    from a per-projection deterministic generator; ``noise_enabled = False``
    bypasses the perturbation entirely and delegates to the wrapped module.
    """

    def __init__(
        self,
        wrapped: nn.Module,
        projection_id: str,
        settings: HWANoiseSettings,
        base_seed: int,
        parent: nn.Module,
        attribute: str,
    ) -> None:
        super().__init__()
        if Conv1D is not None and isinstance(wrapped, Conv1D):
            self._transpose_weight = True
        elif isinstance(wrapped, nn.Linear):
            self._transpose_weight = False
        else:
            raise TypeError(f"Unsupported projection module: {type(wrapped).__name__}")
        self.wrapped = wrapped
        self.projection_id = projection_id
        self.settings = settings
        self.base_seed = int(base_seed)
        self.noise_enabled = True
        self._generator: torch.Generator | None = None
        # Unwrap bookkeeping; excluded from the module tree on purpose.
        object.__setattr__(self, "unwrap_parent", parent)
        object.__setattr__(self, "unwrap_attribute", attribute)

    @property
    def weight(self) -> Tensor:
        """Clean underlying weight, for callers that address it directly.

        Transformers utilities (weight tying, output-embedding access) reach
        for ``.weight`` on projection modules; they must see the live clean
        Parameter, never a noised copy.
        """
        return self.wrapped.weight

    @property
    def bias(self) -> Tensor | None:
        return getattr(self.wrapped, "bias", None)

    def _generator_for(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(projection_noise_seed(self.base_seed, self.projection_id))
            self._generator = generator
        return self._generator

    def forward(self, x: Tensor) -> Tensor:
        if not self.noise_enabled:
            return self.wrapped(x)
        weight = self.wrapped.weight
        # Canonical [out, in] orientation; iid noise is orientation-invariant
        # but F.linear expects [out, in].
        canonical = weight.T if self._transpose_weight else weight
        if self.settings.clip_in_forward:
            effective, programmed_range = straight_through_clip(
                canonical, self.settings.clip_sigma, self.settings.range_mode
            )
        else:
            effective = canonical
            with torch.no_grad():
                detached = canonical.detach()
                if self.settings.range_mode == "peak_to_peak":
                    programmed_range = detached.max() - detached.min()
                else:
                    programmed_range = detached.abs().max()
        low, high = self.settings.noise_std_range
        if high > 0.0:
            generator = self._generator_for(weight.device)
            normalized_sigma = torch.empty(
                (), device=weight.device, dtype=torch.float32
            ).uniform_(low, high, generator=generator)
            field = torch.randn(
                effective.shape,
                device=weight.device,
                dtype=torch.float32,
                generator=generator,
            )
            effective = effective + (normalized_sigma * programmed_range) * field
        bias = getattr(self.wrapped, "bias", None)
        output = F.linear(x, effective.to(x.dtype), None)
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output

    def generator_state(self) -> Tensor | None:
        return None if self._generator is None else self._generator.get_state()

    def set_generator_state(self, state: Tensor, device: torch.device) -> None:
        generator = torch.Generator(device=device)
        generator.set_state(state)
        self._generator = generator


def candidate_handles(
    model: nn.Module, settings: HWANoiseSettings
) -> list[ProjectionHandle]:
    handles = list(
        iter_gpt2_projections(model, include_lm_head=settings.include_lm_head)
    )
    return [
        handle
        for handle in handles
        if handle.projection_id not in settings.exclude_projection_ids
    ]


def wrap_analog_candidates(
    model: nn.Module, settings: HWANoiseSettings, seed: int
) -> dict[str, NoisyProjection]:
    """Replace every analog-candidate projection with a NoisyProjection."""
    settings.validate()
    wrapped: dict[str, NoisyProjection] = {}
    for handle in candidate_handles(model, settings):
        wrapper = NoisyProjection(
            handle.module,
            handle.projection_id,
            settings,
            base_seed=seed,
            parent=handle.parent,
            attribute=handle.attribute,
        )
        setattr(handle.parent, handle.attribute, wrapper)
        wrapped[handle.projection_id] = wrapper
    if not wrapped:
        raise ValueError("No projection was wrapped for HWA training.")
    return wrapped


def unwrap_analog_candidates(
    model: nn.Module, wrapped: Mapping[str, NoisyProjection]
) -> None:
    """Restore every wrapped module; safe to call repeatedly."""
    for wrapper in wrapped.values():
        parent = wrapper.unwrap_parent
        attribute = wrapper.unwrap_attribute
        if getattr(parent, attribute) is wrapper:
            setattr(parent, attribute, wrapper.wrapped)


def set_noise_enabled(
    wrapped: Iterable[NoisyProjection] | Mapping[str, NoisyProjection],
    enabled: bool,
) -> None:
    values = wrapped.values() if isinstance(wrapped, Mapping) else wrapped
    for wrapper in values:
        wrapper.noise_enabled = bool(enabled)


def snapshot_generator_states(
    wrapped: Mapping[str, NoisyProjection],
) -> dict[str, Tensor | None]:
    """Capture every wrapper's noise-generator state.

    Use around evaluation passes so noisy evals do not advance the training
    noise streams (otherwise changing the eval cadence changes the training
    noise sequence).
    """
    return {
        projection_id: wrapper.generator_state()
        for projection_id, wrapper in wrapped.items()
    }


def restore_generator_states(
    wrapped: Mapping[str, NoisyProjection],
    states: Mapping[str, Tensor | None],
    device: torch.device,
) -> None:
    for projection_id, state in states.items():
        wrapper = wrapped.get(projection_id)
        if wrapper is None:
            continue
        if state is None:
            # The generator had not been created yet at snapshot time;
            # recreate it lazily from the deterministic per-projection seed.
            wrapper._generator = None
        else:
            wrapper.set_generator_state(state, device)
