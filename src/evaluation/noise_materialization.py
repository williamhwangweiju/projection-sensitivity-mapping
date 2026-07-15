"""Materialize Phase-2 tile noise as per-weight Phase-4 sigma maps.

For shard ``s`` of projection ``p`` at hardware snapshot ``t``::

    sigma_absolute[s, t] = calibration_scale[p] * tile_sigma[tile(s), t]

``tile_sigma`` is expressed in the same normalized units as AIHWKit's
``PCMLikeNoiseModel.prog_noise_scale``.  The calibration scale is measured in
Phase 1 from the effective logical weight error produced at the reference
programming-noise setting.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .placement_to_gpt2 import group_assignments_by_module
from .schemas import (
    GPT2ShardAssignment,
    NoiseInjectionRecord,
    ProjectionNoiseCalibration,
)


def load_phase1_noise_calibrations(
    path: str | Path,
) -> dict[str, ProjectionNoiseCalibration]:
    """Load per-projection empirical calibration from a Phase-1 JSON file."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    raw_records = payload.get("projection_noise_calibration")
    if raw_records is None:
        results = payload.get("results", {})
        if isinstance(results, Mapping):
            raw_records = results.get("projection_noise_calibration")
    if not isinstance(raw_records, list):
        raise ValueError(
            f"{resolved} must contain 'projection_noise_calibration' as a list."
        )

    calibrations: dict[str, ProjectionNoiseCalibration] = {}
    for index, row in enumerate(raw_records):
        if not isinstance(row, Mapping):
            raise TypeError(f"Calibration entry {index} must be a mapping.")
        calibration = ProjectionNoiseCalibration(
            projection_id=str(row["projection_id"]),
            hf_module_path=str(row["hf_module_path"]),
            reference_sigma_normalized=float(row["reference_sigma_normalized"]),
            measured_noise_std_absolute=float(row["measured_noise_std_absolute"]),
            measured_noise_rms_absolute=float(row["measured_noise_rms_absolute"]),
            noise_reference_scale=float(row["noise_reference_scale"]),
            calibration_source=str(
                row.get(
                    "calibration_source",
                    "phase1_aihwkit_empirical_programming_noise",
                )
            ),
            num_calibration_seeds=int(row.get("num_calibration_seeds", 1)),
        )
        if calibration.hf_module_path in calibrations:
            raise ValueError(
                f"Duplicate Phase-1 calibration for {calibration.hf_module_path}."
            )
        calibrations[calibration.hf_module_path] = calibration

    if not calibrations:
        raise ValueError("Phase-1 calibration list is empty.")
    return calibrations


def build_sigma_map(
    reference_weights: Mapping[str, torch.Tensor],
    assignments: Sequence[GPT2ShardAssignment],
    *,
    tile_noise_at_timestep: Mapping[int, float],
    calibrations: Mapping[str, ProjectionNoiseCalibration],
) -> dict[str, torch.Tensor]:
    """Build one absolute per-weight sigma tensor for each GPT-2 projection."""
    by_module = group_assignments_by_module(assignments)
    sigma_maps: dict[str, torch.Tensor] = {}

    for path, shards in by_module.items():
        if path not in reference_weights:
            raise KeyError(f"No all-analog reference weight for {path}.")
        if path not in calibrations:
            raise KeyError(f"No Phase-1 calibration for {path}.")
        calibration = calibrations[path]
        projection_ids = {shard.projection_id for shard in shards}
        if projection_ids != {calibration.projection_id}:
            raise ValueError(
                f"{path}: assignments use {sorted(projection_ids)}, but calibration "
                f"is for {calibration.projection_id!r}."
            )

        canonical_shape = tuple(int(value) for value in reference_weights[path].shape)
        sigma_map = torch.zeros(canonical_shape, dtype=torch.float32)
        filled = torch.zeros(canonical_shape, dtype=torch.bool)

        for shard in shards:
            if shard.tile_id not in tile_noise_at_timestep:
                raise KeyError(
                    f"No Phase-2 noise value for tile {shard.tile_id}, used by "
                    f"{shard.shard_id}."
                )
            normalized_sigma = float(tile_noise_at_timestep[shard.tile_id])
            if not math.isfinite(normalized_sigma) or normalized_sigma < 0.0:
                raise ValueError(
                    f"Invalid normalized sigma for tile {shard.tile_id}: "
                    f"{normalized_sigma}."
                )
            absolute_sigma = calibration.noise_reference_scale * normalized_sigma
            if not math.isfinite(absolute_sigma) or absolute_sigma < 0.0:
                raise ValueError(
                    f"Invalid absolute sigma for {shard.shard_id}: {absolute_sigma}."
                )

            rows = slice(shard.canonical_row_start, shard.canonical_row_end)
            cols = slice(shard.canonical_col_start, shard.canonical_col_end)
            if bool(filled[rows, cols].any().item()):
                raise ValueError(f"Overlapping sigma-map assignment at {shard.shard_id}.")
            sigma_map[rows, cols] = absolute_sigma
            filled[rows, cols] = True

        missing = int((~filled).sum().item())
        if missing:
            raise ValueError(f"{path}: sigma map leaves {missing} weights uncovered.")
        sigma_maps[path] = sigma_map

    return sigma_maps


def generate_paired_noise_tensors(
    reference_weights: Mapping[str, torch.Tensor],
    hf_paths: Sequence[str],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Generate deterministic canonical standard-normal tensors per projection."""
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(set(hf_paths)):
        if path not in reference_weights:
            raise KeyError(f"No all-analog reference weight for {path}.")
        shape = tuple(int(value) for value in reference_weights[path].shape)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_module_seed(seed, path))
        tensors[path] = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
    return tensors


def _module_seed(base_seed: int, module_path: str) -> int:
    digest = hashlib.sha256(module_path.encode("utf-8")).digest()[:8]
    offset = int.from_bytes(digest, "little", signed=False)
    return (int(base_seed) + offset) & 0x7FFF_FFFF_FFFF_FFFF


def build_noise_injection_records(
    assignments: Sequence[GPT2ShardAssignment],
    *,
    tile_noise_at_timestep: Mapping[int, float],
    tile_faulted_at_timestep: Mapping[int, bool],
    tile_available_at_timestep: Mapping[int, bool],
    calibrations: Mapping[str, ProjectionNoiseCalibration],
    timestep: int,
    noise_realization: int,
    noise_realization_seed: int,
    unavailable_action: str,
) -> list[NoiseInjectionRecord]:
    """Create one provenance record for every mapped shard."""
    records: list[NoiseInjectionRecord] = []
    for assignment in assignments:
        tile_id = assignment.tile_id
        path = assignment.hf_module_path
        for mapping_name, mapping in (
            ("noise", tile_noise_at_timestep),
            ("fault", tile_faulted_at_timestep),
            ("availability", tile_available_at_timestep),
        ):
            if tile_id not in mapping:
                raise KeyError(
                    f"No Phase-2 {mapping_name} value for tile {tile_id}, used by "
                    f"{assignment.shard_id}."
                )
        if path not in calibrations:
            raise KeyError(f"No Phase-1 calibration for {path}.")

        calibration = calibrations[path]
        if calibration.projection_id != assignment.projection_id:
            raise ValueError(
                f"{assignment.shard_id}: calibration is for "
                f"{calibration.projection_id!r}, assignment is "
                f"{assignment.projection_id!r}."
            )
        normalized_sigma = float(tile_noise_at_timestep[tile_id])
        if not math.isfinite(normalized_sigma) or normalized_sigma < 0.0:
            raise ValueError(
                f"Invalid normalized sigma for tile {tile_id}: {normalized_sigma}."
            )
        absolute_sigma = calibration.noise_reference_scale * normalized_sigma

        records.append(
            NoiseInjectionRecord(
                shard_id=assignment.shard_id,
                projection_id=assignment.projection_id,
                hf_module_path=path,
                tile_id=tile_id,
                tier_id=assignment.tier_id,
                policy=assignment.policy,
                timestep=int(timestep),
                noise_realization=int(noise_realization),
                noise_realization_seed=int(noise_realization_seed),
                tile_noise_std_normalized=normalized_sigma,
                noise_reference_scale=calibration.noise_reference_scale,
                tile_noise_std_absolute=absolute_sigma,
                noise_reference_source=calibration.calibration_source,
                is_faulted=bool(tile_faulted_at_timestep[tile_id]),
                is_available=bool(tile_available_at_timestep[tile_id]),
                unavailable_action=unavailable_action,
            )
        )
    return records
