"""Crossbar sharding and authoritative 3D-CIM mapping extraction.

IBM 3D-CIM linear weights use [input_features, output_features].  The mapping
row index therefore identifies an input-dimension fragment and the mapping
column index identifies an output-dimension fragment.  Phase 4 transposes
these coordinates into GPT-2 canonical [output_features, input_features].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .projection_catalog import (
    MappedModuleSpec,
    build_group_total_weights,
    iter_mappable_modules,
)


@dataclass(frozen=True, slots=True)
class ProjectionShard:
    """One crossbar-compatible shard from a logical projection group."""

    shard_id: str
    projection_id: str
    block_id: str
    projection_name: str
    execution_index: int

    # IBM mapping-grid indices: row=input dimension, col=output dimension.
    row_shard_index: int
    col_shard_index: int

    sim_input_start: int
    sim_input_end: int
    sim_output_start: int
    sim_output_end: int

    # Kept for compatibility with earlier Phase-3 consumers.  These are exact
    # simulator-coordinate ranges, not canonical GPT-2 coordinates.
    out_start: int
    out_end: int
    in_start: int
    in_end: int

    out_features: int
    in_features: int
    weights_in_shard: int
    weights_in_projection: int
    shard_weight: float
    sensitivity_score: float
    sensitivity_score_unit: str
    sensitivity_reference_noise_std: float


@dataclass(frozen=True, slots=True)
class MappedShardRecord:
    """One ProjectionShard plus authoritative 3D-CIM mapping metadata."""

    module_name: str
    shard: ProjectionShard
    tile_id: int
    tier_start: int
    tiers_used: int
    used_rows: int
    used_cols: int
    group_total_weights: int
    module_num_weights: int
    utilization: float


def extract_shards_from_3dcim_mapping(
    *,
    model: Any,
    mapped_specs: Sequence[MappedModuleSpec],
    tier_rows: int,
    tier_cols: int,
) -> list[MappedShardRecord]:
    """Extract exact shard extents from already-mapped 3D-CIM modules."""

    if tier_rows <= 0 or tier_cols <= 0:
        raise ValueError("tier_rows and tier_cols must be positive.")

    specs_by_name: Mapping[str, MappedModuleSpec] = {
        spec.module_name: spec for spec in mapped_specs
    }
    group_total_weights = build_group_total_weights(mapped_specs)

    records: list[MappedShardRecord] = []
    for module_name, module in iter_mappable_modules(
        model,
        include_embeddings=False,
        include_lm_head=False,
    ):
        if module_name not in specs_by_name:
            continue
        spec = specs_by_name[module_name]
        mapping = getattr(module, "mapping", None)
        if mapping is None:
            raise ValueError(f"Module {module_name} was not mapped.")

        group_total = group_total_weights[spec.group_key]
        if group_total <= 0:
            raise ValueError(f"Non-positive group total for {spec.group_key}.")

        for row_idx, row in enumerate(mapping):
            for col_idx, mapping_entry in enumerate(row):
                tile_idx, tier_idx, utilization, used_rows, used_cols = mapping_entry
                tile_id = int(tile_idx)
                tier_id = int(tier_idx)
                used_rows = int(used_rows)
                used_cols = int(used_cols)
                if not 1 <= used_rows <= tier_rows:
                    raise ValueError(
                        f"{module_name}#r{row_idx}c{col_idx}: invalid used_rows="
                        f"{used_rows} for tier_rows={tier_rows}."
                    )
                if not 1 <= used_cols <= tier_cols:
                    raise ValueError(
                        f"{module_name}#r{row_idx}c{col_idx}: invalid used_cols="
                        f"{used_cols} for tier_cols={tier_cols}."
                    )

                sim_input_start = row_idx * tier_rows
                sim_input_end = sim_input_start + used_rows
                sim_output_start = col_idx * tier_cols
                sim_output_end = sim_output_start + used_cols
                used_weights = used_rows * used_cols
                shard_id = f"{module_name}#r{row_idx}c{col_idx}"

                shard = ProjectionShard(
                    shard_id=shard_id,
                    projection_id=spec.projection_id,
                    block_id=spec.block_id,
                    projection_name=spec.projection_name,
                    execution_index=spec.execution_index,
                    row_shard_index=row_idx,
                    col_shard_index=col_idx,
                    sim_input_start=sim_input_start,
                    sim_input_end=sim_input_end,
                    sim_output_start=sim_output_start,
                    sim_output_end=sim_output_end,
                    out_start=sim_output_start,
                    out_end=sim_output_end,
                    in_start=sim_input_start,
                    in_end=sim_input_end,
                    out_features=used_cols,
                    in_features=used_rows,
                    weights_in_shard=used_weights,
                    weights_in_projection=group_total,
                    shard_weight=used_weights / group_total,
                    sensitivity_score=spec.sensitivity_score,
                    sensitivity_score_unit=spec.sensitivity_score_unit,
                    sensitivity_reference_noise_std=(
                        spec.sensitivity_reference_noise_std
                    ),
                )

                records.append(
                    MappedShardRecord(
                        module_name=module_name,
                        shard=shard,
                        tile_id=tile_id,
                        tier_start=tier_id,
                        tiers_used=1,
                        used_rows=used_rows,
                        used_cols=used_cols,
                        group_total_weights=group_total,
                        module_num_weights=spec.num_weights,
                        utilization=float(utilization),
                    )
                )

    if not records:
        raise ValueError("No transformer-projection shards were extracted.")
    return records


def mapped_shard_records_to_placement_rows(
    *,
    policy_name: str,
    records: Sequence[MappedShardRecord],
    trace_seed: int,
    placement_seed: int,
) -> list[dict[str, Any]]:
    """Convert mapped shards into the per-policy CSV consumed by Phase 4."""

    rows: list[dict[str, Any]] = []
    for record in records:
        shard = record.shard
        rows.append(
            {
                "policy": policy_name,
                "module_name": record.module_name,
                "shard_id": shard.shard_id,
                "projection_id": shard.projection_id,
                "block_id": shard.block_id,
                "projection_name": shard.projection_name,
                "execution_index": shard.execution_index,
                "row_shard_index": shard.row_shard_index,
                "col_shard_index": shard.col_shard_index,
                "sim_input_start": shard.sim_input_start,
                "sim_input_end": shard.sim_input_end,
                "sim_output_start": shard.sim_output_start,
                "sim_output_end": shard.sim_output_end,
                "tile_id": record.tile_id,
                "tier_start": record.tier_start,
                "tiers_used": record.tiers_used,
                "used_rows": record.used_rows,
                "used_cols": record.used_cols,
                "weights_in_shard": shard.weights_in_shard,
                "group_total_weights": record.group_total_weights,
                "module_num_weights": record.module_num_weights,
                "shard_weight": shard.shard_weight,
                "sensitivity_score": shard.sensitivity_score,
                "sensitivity_score_unit": shard.sensitivity_score_unit,
                "sensitivity_reference_noise_std": (
                    shard.sensitivity_reference_noise_std
                ),
                "shard_importance": (
                    shard.shard_weight * shard.sensitivity_score
                ),
                "utilization": record.utilization,
                "trace_seed": trace_seed,
                "placement_seed": placement_seed,
            }
        )
    return rows
