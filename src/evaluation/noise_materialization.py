"""Build physical-placement-derived per-weight sigma maps for Phase 4."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.common.analog import materialize_manual_noise
from src.evaluation.aihwkit_gpt2 import AnalogProjectionReference


REQUIRED_PLACEMENT_COLUMNS = {
    "projection_id",
    "shard_id",
    "tile_id",
    "tier_id",
    "out_start",
    "out_end",
    "in_start",
    "in_end",
}


@dataclass(frozen=True)
class ProjectionNoiseMaterialization:
    noisy_weight: Tensor
    noise: Tensor
    normalized_sigma_map: Tensor
    absolute_sigma_map: Tensor
    assignment_summary: dict[str, Any]


def load_placement(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_PLACEMENT_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Placement {path} is missing columns: {missing}")
    if frame["shard_id"].duplicated().any():
        raise ValueError(f"Placement {path} contains duplicate shard IDs.")
    return frame


def build_normalized_sigma_map(
    reference: AnalogProjectionReference,
    placement: pd.DataFrame,
    tile_noise_std: np.ndarray,
    tile_available: np.ndarray,
    *,
    unavailable_action: str,
) -> tuple[Tensor, list[dict[str, Any]]]:
    rows = placement[placement["projection_id"] == reference.projection_id]
    if rows.empty:
        raise ValueError(f"Placement has no shards for {reference.projection_id}.")
    shape = tuple(reference.clipped_weight.shape)
    sigma_map = torch.full(shape, float("nan"), dtype=torch.float32)
    coverage = torch.zeros(shape, dtype=torch.int16)
    assignment_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        tile_id = int(row.tile_id)
        if tile_id < 0 or tile_id >= tile_noise_std.size:
            raise ValueError(f"Invalid tile_id {tile_id} in placement.")
        if not bool(tile_available[tile_id]):
            if unavailable_action == "error":
                raise RuntimeError(
                    f"{reference.projection_id} is mapped to unavailable tile {tile_id}."
                )
            raise ValueError(f"Unsupported unavailable_action: {unavailable_action!r}")
        out_start, out_end = int(row.out_start), int(row.out_end)
        in_start, in_end = int(row.in_start), int(row.in_end)
        if not (
            0 <= out_start < out_end <= shape[0]
            and 0 <= in_start < in_end <= shape[1]
        ):
            raise ValueError(
                f"Invalid shard bounds for {reference.projection_id}: "
                f"[{out_start}:{out_end}, {in_start}:{in_end}] vs {shape}."
            )
        normalized_sigma = float(tile_noise_std[tile_id])
        sigma_map[out_start:out_end, in_start:in_end] = normalized_sigma
        coverage[out_start:out_end, in_start:in_end] += 1
        assignment_rows.append(
            {
                "projection_id": reference.projection_id,
                "shard_id": str(row.shard_id),
                "tile_id": tile_id,
                "tier_id": int(row.tier_id),
                "out_start": out_start,
                "out_end": out_end,
                "in_start": in_start,
                "in_end": in_end,
                "normalized_noise_std": normalized_sigma,
                "programmed_range": float(
                    reference.preprocessing["programmed_range"]
                ),
                "absolute_noise_std": normalized_sigma
                * float(reference.preprocessing["programmed_range"]),
            }
        )
    if bool((coverage != 1).any().item()):
        zero = int((coverage == 0).sum().item())
        overlap = int((coverage > 1).sum().item())
        raise RuntimeError(
            f"{reference.projection_id}: placement coverage failure: "
            f"uncovered={zero}, overlapped={overlap}."
        )
    if not bool(torch.isfinite(sigma_map).all().item()):
        raise RuntimeError(f"{reference.projection_id}: sigma map is non-finite.")
    return sigma_map, assignment_rows


def materialize_projection_noise(
    reference: AnalogProjectionReference,
    normalized_sigma_map: Tensor,
    gaussian_field: Tensor,
) -> ProjectionNoiseMaterialization:
    noisy, noise = materialize_manual_noise(
        reference.clipped_weight,
        normalized_sigma_map,
        float(reference.preprocessing["programmed_range"]),
        gaussian_field,
    )
    absolute_sigma = normalized_sigma_map * float(
        reference.preprocessing["programmed_range"]
    )
    weighted_rms_normalized = float(
        torch.sqrt(torch.mean(normalized_sigma_map.square())).item()
    )
    summary = {
        "projection_id": reference.projection_id,
        "normalized_sigma_mean": float(normalized_sigma_map.mean().item()),
        "normalized_sigma_rms": weighted_rms_normalized,
        "normalized_sigma_min": float(normalized_sigma_map.min().item()),
        "normalized_sigma_max": float(normalized_sigma_map.max().item()),
        "absolute_sigma_mean": float(absolute_sigma.mean().item()),
        "absolute_sigma_rms": float(
            torch.sqrt(torch.mean(absolute_sigma.square())).item()
        ),
        "realized_noise_std": float(noise.std(unbiased=False).item()),
        "realized_noise_abs_mean": float(noise.abs().mean().item()),
        "realized_noise_abs_max": float(noise.abs().max().item()),
    }
    return ProjectionNoiseMaterialization(
        noisy_weight=noisy,
        noise=noise,
        normalized_sigma_map=normalized_sigma_map,
        absolute_sigma_map=absolute_sigma,
        assignment_summary=summary,
    )
