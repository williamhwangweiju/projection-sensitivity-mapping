"""Crossbar sharding and 3D-CIM mapping-extraction utilities."""

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
    """One crossbar-compatible shard from a parent projection."""

    shard_id: str
    projection_id: str
    block_id: str
    projection_name: str
    execution_index: int

    row_shard_index: int
    col_shard_index: int

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


@dataclass(frozen=True, slots=True)
class MappedShardRecord:
    """One ProjectionShard plus its authoritative 3D-CIM mapping metadata."""

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
) -> list[MappedShardRecord]:
    """Canonicalize the actual 3D-CIM module mappings without remapping them.

    The source simulator currently exposes fragment indices and used dimensions,
    but not exact source-tensor coordinate ranges. To preserve existing Phase 3
    behavior, out_start/out_end and in_start/in_end retain their prior fragment-
    identifier semantics. Phase 4 can later replace this single conversion point
    once exact tensor-coordinate reconstruction has been validated.
    """
    specs_by_name: Mapping[str, MappedModuleSpec] = {
        spec.module_name: spec
        for spec in mapped_specs
    }
    group_total_weights = build_group_total_weights(mapped_specs)

    records: list[MappedShardRecord] = []
    for module_name, module in iter_mappable_modules(model):
        spec = specs_by_name[module_name]
        mapping = getattr(module, "mapping", None)
        if mapping is None:
            raise ValueError(f"Module {module_name} was not mapped.")

        group_total = group_total_weights[spec.group_key]

        for row_idx, row in enumerate(mapping):
            for col_idx, mapping_entry in enumerate(row):
                tile_idx, tier_idx, utilization, used_rows, used_cols = mapping_entry
                tile_id = int(tile_idx)
                tier_id = int(tier_idx)
                used_rows = int(used_rows)
                used_cols = int(used_cols)
                used_weights = used_rows * used_cols

                shard_id = f"{module_name}#r{row_idx}c{col_idx}"
                shard_weight = used_weights / group_total if group_total > 0 else 0.0

                shard = ProjectionShard(
                    shard_id=shard_id,
                    projection_id=spec.projection_id,
                    block_id=spec.block_id,
                    projection_name=spec.projection_name,
                    execution_index=spec.execution_index,
                    row_shard_index=row_idx,
                    col_shard_index=col_idx,
                    out_start=row_idx,
                    out_end=row_idx + 1,
                    in_start=col_idx,
                    in_end=col_idx + 1,
                    out_features=used_rows,
                    in_features=used_cols,
                    weights_in_shard=used_weights,
                    weights_in_projection=group_total,
                    shard_weight=shard_weight,
                    sensitivity_score=spec.sensitivity_score,
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

    return records


def mapped_shard_records_to_placement_rows(
    *,
    policy_name: str,
    records: Sequence[MappedShardRecord],
) -> list[dict[str, Any]]:
    """Convert 3D-CIM mapped shards into the existing per-policy placement CSV."""
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
                "shard_importance": shard.shard_weight * shard.sensitivity_score,
                "utilization": record.utilization,
            }
        )
    return rows
