"""Coordinate-preserving projection sharding for one crossbar per physical tier."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ProjectionShard:
    shard_id: str
    projection_id: str
    shard_index: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    weight_count: int
    projection_weight_count: int
    shard_weight: float
    sensitivity: float
    importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def projection_row_regions(projection_id: str, out_features: int) -> list[tuple[int, int]]:
    """Return semantic row regions that may not be crossed by a physical shard.

    GPT-2 stores Q, K, and V in one fused ``attn.c_attn`` matrix. The IBM-style
    physical mapping treats them as three separate 768-output projections, so
    shard boundaries must preserve those semantic regions.
    """
    if projection_id.endswith("/attn.c_attn") and out_features % 3 == 0:
        width = out_features // 3
        return [(index * width, (index + 1) * width) for index in range(3)]
    return [(0, out_features)]


def count_projection_shards(
    projection_id: str, out_features: int, in_features: int, tier_rows: int, tier_cols: int
) -> int:
    import math
    col_shards = math.ceil(in_features / tier_cols)
    return sum(
        math.ceil((end - start) / tier_rows) * col_shards
        for start, end in projection_row_regions(projection_id, out_features)
    )


def build_shards(
    projection_rows: Iterable[Mapping[str, Any]],
    *,
    digital_projection_ids: Iterable[str],
    tier_rows: int,
    tier_cols: int,
    sensitivity_floor: float = 0.0,
) -> list[ProjectionShard]:
    digital = frozenset(digital_projection_ids)
    shards: list[ProjectionShard] = []
    for projection in projection_rows:
        projection_id = str(projection["projection_id"])
        if projection_id in digital:
            continue
        out_features = int(projection["out_features"])
        in_features = int(projection["in_features"])
        total = out_features * in_features
        sensitivity = max(
            float(projection["sensitivity_score_for_mapping"]),
            float(sensitivity_floor),
        )
        index = 0
        for region_start, region_end in projection_row_regions(projection_id, out_features):
            for row_start in range(region_start, region_end, tier_rows):
                row_end = min(row_start + tier_rows, region_end)
                for col_start in range(0, in_features, tier_cols):
                    col_end = min(col_start + tier_cols, in_features)
                    count = (row_end - row_start) * (col_end - col_start)
                    weight = count / total
                    shards.append(
                        ProjectionShard(
                            shard_id=f"{projection_id}#s{index:04d}",
                            projection_id=projection_id,
                            shard_index=index,
                            row_start=row_start,
                            row_end=row_end,
                            col_start=col_start,
                            col_end=col_end,
                            weight_count=count,
                            projection_weight_count=total,
                            shard_weight=weight,
                            sensitivity=sensitivity,
                            importance=sensitivity * weight,
                        )
                    )
                    index += 1
    validate_shards(shards)
    return shards


def validate_shards(shards: Iterable[ProjectionShard]) -> None:
    groups: dict[str, list[ProjectionShard]] = {}
    for shard in shards:
        groups.setdefault(shard.projection_id, []).append(shard)
    for projection_id, rows in groups.items():
        if abs(sum(row.shard_weight for row in rows) - 1.0) > 1e-9:
            raise ValueError(f"Shard weights do not sum to one for {projection_id}.")
        if sum(row.weight_count for row in rows) != rows[0].projection_weight_count:
            raise ValueError(f"Weight coverage mismatch for {projection_id}.")
