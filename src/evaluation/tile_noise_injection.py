"""Apply and restore heterogeneous tile noise in all-analog GPT-2 models."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import torch

from .aihwkit_gpt2 import (
    AnalogProjectionReference,
    restore_analog_projection_weights,
    save_analog_projection_weights,
    set_analog_projection_weight,
)
from .placement_to_gpt2 import group_assignments_by_module
from .schemas import GPT2ShardAssignment, WeightChecksum


def _tensor_checksum(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    data = cpu.view(torch.uint8).numpy().tobytes()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}:".encode("utf-8")
    return hashlib.sha256(header + data).hexdigest()


def save_projection_weights(
    model: Any,
    hf_paths: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Read exact logical weights from AIHWKit mapped layers."""
    return save_analog_projection_weights(model, hf_paths)


def restore_projection_weights(
    model: Any,
    references: Mapping[str, AnalogProjectionReference],
) -> None:
    restore_analog_projection_weights(model, references)


def apply_tile_noise(
    model: Any,
    *,
    assignments: Sequence[GPT2ShardAssignment],
    sigma_maps: Mapping[str, torch.Tensor],
    Z_by_module: Mapping[str, torch.Tensor],
    references: Mapping[str, AnalogProjectionReference],
) -> dict[str, torch.Tensor]:
    """Apply one paired Gaussian realization to all analog projections."""
    by_module = group_assignments_by_module(assignments)
    noisy_weights: dict[str, torch.Tensor] = {}

    for path in by_module:
        if path not in references:
            raise KeyError(f"Missing all-analog reference for {path}.")
        if path not in sigma_maps:
            raise KeyError(f"Missing sigma map for {path}.")
        if path not in Z_by_module:
            raise KeyError(f"Missing paired Z tensor for {path}.")

        reference = references[path]
        clean = reference.canonical_weight.detach().cpu().float()
        sigma = sigma_maps[path].detach().cpu().float()
        noise_field = Z_by_module[path].detach().cpu().float()
        if tuple(clean.shape) != tuple(sigma.shape):
            raise ValueError(
                f"{path}: reference shape {tuple(clean.shape)} != sigma shape "
                f"{tuple(sigma.shape)}."
            )
        if tuple(clean.shape) != tuple(noise_field.shape):
            raise ValueError(
                f"{path}: reference shape {tuple(clean.shape)} != Z shape "
                f"{tuple(noise_field.shape)}."
            )
        if not bool(torch.isfinite(sigma).all().item()):
            raise ValueError(f"{path}: sigma map contains non-finite values.")
        if bool((sigma < 0).any().item()):
            raise ValueError(f"{path}: sigma map contains negative values.")
        if not bool(torch.isfinite(noise_field).all().item()):
            raise ValueError(f"{path}: paired noise tensor contains non-finite values.")

        noisy = (clean + sigma * noise_field).contiguous()
        set_analog_projection_weight(
            model,
            path,
            noisy,
            reference_bias=reference.bias,
        )
        noisy_weights[path] = noisy

    return noisy_weights


def compute_weight_checksums(
    model: Any,
    hf_paths: Sequence[str],
    *,
    saved_before: Mapping[str, torch.Tensor],
    noisy_weights: Mapping[str, torch.Tensor],
    saved_after_restore: Mapping[str, torch.Tensor] | None = None,
) -> list[WeightChecksum]:
    """Create checksum records and verify exact logical-weight restoration."""
    records: list[WeightChecksum] = []
    for path in sorted(set(hf_paths)):
        if path not in saved_before:
            raise KeyError(f"Missing pre-noise reference weight for {path}.")
        if path not in noisy_weights:
            raise KeyError(f"Missing noisy weight for {path}.")
        before = saved_before[path]
        noisy = noisy_weights[path]
        if saved_after_restore is None:
            restored = save_analog_projection_weights(model, [path])[path]
        else:
            if path not in saved_after_restore:
                raise KeyError(f"Missing restored weight for {path}.")
            restored = saved_after_restore[path]

        before_checksum = _tensor_checksum(before)
        noisy_checksum = _tensor_checksum(noisy)
        restored_checksum = _tensor_checksum(restored)
        records.append(
            WeightChecksum(
                hf_module_path=path,
                checksum_before=before_checksum,
                checksum_after_noise=noisy_checksum,
                checksum_restored=restored_checksum,
                weights_match_original=(before_checksum == restored_checksum),
            )
        )
    return records


class AnalogNoisedModelContext:
    """Inject paired tile noise and always restore the all-analog reference."""

    def __init__(
        self,
        model: Any,
        assignments: Sequence[GPT2ShardAssignment],
        sigma_maps: Mapping[str, torch.Tensor],
        Z_by_module: Mapping[str, torch.Tensor],
        references: Mapping[str, AnalogProjectionReference],
    ) -> None:
        self._model = model
        self._assignments = assignments
        self._sigma_maps = sigma_maps
        self._Z_by_module = Z_by_module
        self._references = references
        self._saved: dict[str, torch.Tensor] = {
            path: reference.canonical_weight.detach().cpu().clone()
            for path, reference in references.items()
        }
        self.noisy_weights: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "AnalogNoisedModelContext":
        try:
            self.noisy_weights = apply_tile_noise(
                self._model,
                assignments=self._assignments,
                sigma_maps=self._sigma_maps,
                Z_by_module=self._Z_by_module,
                references=self._references,
            )
        except Exception:
            restore_analog_projection_weights(self._model, self._references)
            raise
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        restore_analog_projection_weights(self._model, self._references)
        return False

    @property
    def saved_weights(self) -> dict[str, torch.Tensor]:
        return self._saved


# Backward-compatible import name used by the Phase-4 runner.
NoisedModelContext = AnalogNoisedModelContext
