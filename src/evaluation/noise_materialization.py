"""Coordinate-preserving tile-noise materialization for hybrid GPT-2 models."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from src.common.analog import projection_noise_seed, set_analog_weights_exact
if TYPE_CHECKING:
    from src.evaluation.aihwkit_gpt2 import HybridAnalogModel


_INTEGER_FIELDS = {
    "timestep", "shard_index", "row_start", "row_end", "col_start", "col_end",
    "weight_count", "tile_id", "tier_id",
}
_FLOAT_FIELDS = {"shard_weight", "sensitivity", "importance", "tile_noise_std"}


def read_placement_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a Phase-3/5 placement CSV and restore numeric field types."""
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in _INTEGER_FIELDS:
            if field in item and item[field] != "":
                item[field] = int(item[field])
        for field in _FLOAT_FIELDS:
            if field in item and item[field] != "":
                item[field] = float(item[field])
        parsed.append(item)
    return parsed


def update_placement_noise(
    rows: Iterable[Mapping[str, Any]], tile_noise: Sequence[float], timestep: int
) -> list[dict[str, Any]]:
    """Keep physical assignments fixed while substituting the current tile state."""
    updated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        tile_id = int(row["tile_id"])
        row["timestep"] = int(timestep)
        row["tile_noise_std"] = float(tile_noise[tile_id])
        updated.append(row)
    return updated


def _paired_standard_normal(
    shape: torch.Size,
    *,
    base_seed: int,
    projection_id: str,
    realization: int,
) -> Tensor:
    """Return a policy-independent coordinate noise field for paired comparisons."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(projection_noise_seed(base_seed, projection_id, realization))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def apply_tile_noise(
    hybrid: "HybridAnalogModel",
    placement_rows: Iterable[Mapping[str, Any]],
    *,
    base_seed: int,
    realization: int,
    sign: float = 1.0,
) -> dict[str, float]:
    """Write one paired, tile-scaled logical-weight perturbation into the analog set.

    The standard-normal field is keyed only by projection and realization. Policies
    therefore see the same coordinate-level random field; only the tile-dependent
    scale assigned to each shard changes.
    """
    rows_by_projection: dict[str, list[Mapping[str, Any]]] = {}
    for row in placement_rows:
        rows_by_projection.setdefault(str(row["projection_id"]), []).append(row)

    expected = set(hybrid.analog_projection_ids)
    actual = set(rows_by_projection)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Placement/analog-set mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    normalized_noise_energy = 0.0
    total_weights = 0
    for projection_id, state in hybrid.states.items():
        module = state.analog_module
        nominal = state.clipped_weight
        standard = _paired_standard_normal(
            nominal.shape,
            base_seed=base_seed,
            projection_id=projection_id,
            realization=realization,
        ).to(
            device=nominal.device,
            dtype=nominal.dtype,
        )
        noisy = nominal.clone()
        covered = torch.zeros_like(nominal, dtype=torch.bool)
        for row in rows_by_projection[projection_id]:
            rs, re = int(row["row_start"]), int(row["row_end"])
            cs, ce = int(row["col_start"]), int(row["col_end"])
            if bool(covered[rs:re, cs:ce].any().item()):
                raise ValueError(f"Overlapping shard coordinates for {projection_id}.")
            normalized_std = float(row["tile_noise_std"])
            delta = (
                standard[rs:re, cs:ce]
                * normalized_std
                * float(state.programmed_range)
                * float(sign)
            )
            noisy[rs:re, cs:ce] += delta
            covered[rs:re, cs:ce] = True
            normalized_noise_energy += float(delta.square().sum().item())
            total_weights += int(delta.numel())
        if not bool(covered.all().item()):
            raise ValueError(f"Incomplete shard coverage for {projection_id}.")
        set_analog_weights_exact(
            module,
            noisy,
            state.bias,
            verify=False,
        )

    return {
        "injected_weight_count": float(total_weights),
        "injected_noise_rms": (
            (normalized_noise_energy / total_weights) ** 0.5 if total_weights else 0.0
        ),
    }
