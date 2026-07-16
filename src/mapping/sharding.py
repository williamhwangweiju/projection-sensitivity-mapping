"""GPT-2 projection sharding onto 512x512 physical analog tiers."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


PROJECTION_ORDER = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


@dataclass(frozen=True)
class ProjectionSpec:
    projection_id: str
    block_index: int
    projection_name: str
    hf_module_path: str
    out_features: int
    in_features: int
    sensitivity: float


@dataclass(frozen=True)
class ProjectionShard:
    shard_id: str
    projection_id: str
    block_index: int
    projection_name: str
    hf_module_path: str
    logical_group: str
    out_start: int
    out_end: int
    in_start: int
    in_end: int
    weight_count: int
    projection_weight_count: int
    shard_weight: float
    sensitivity: float
    importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXPECTED_GPT2_SMALL_SHARDS_PER_BLOCK = 40
EXPECTED_GPT2_SMALL_TOTAL_SHARDS = 480


def projection_specs_from_phase1(
    phase1_payload: Mapping[str, Any],
    *,
    sensitivity_floor: float = 0.0,
) -> list[ProjectionSpec]:
    results = phase1_payload.get("results", phase1_payload)
    projections = results.get("projections")
    if not isinstance(projections, list):
        raise ValueError("Phase-1 results do not contain results.projections.")
    specs: list[ProjectionSpec] = []
    for record in projections:
        projection_id = str(record["projection_id"])
        block_index = int(
            record.get("block_index", str(record.get("block_label", "0")).split("_")[-1])
        )
        projection_name = str(
            record.get("projection_name", record.get("projection_label"))
        )
        shape = record.get("weight_shape_out_in")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"Missing canonical shape for {projection_id}.")
        sensitivity = max(float(record["sensitivity_score_for_mapping"]), sensitivity_floor)
        specs.append(
            ProjectionSpec(
                projection_id=projection_id,
                block_index=block_index,
                projection_name=projection_name,
                hf_module_path=str(record["hf_module_path"]),
                out_features=int(shape[0]),
                in_features=int(shape[1]),
                sensitivity=sensitivity,
            )
        )
    specs.sort(
        key=lambda spec: (
            spec.block_index,
            PROJECTION_ORDER.index(spec.projection_name),
        )
    )
    return specs


def _ranges(length: int, tile_size: int) -> list[tuple[int, int]]:
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive.")
    return [
        (start, min(start + tile_size, length))
        for start in range(0, length, tile_size)
    ]


def _group_ranges(spec: ProjectionSpec) -> list[tuple[str, int, int]]:
    """Return output-axis groups before physical 512x512 sharding.

    GPT-2's fused c_attn output is split into Q, K and V groups.  This prevents
    a physical shard from crossing semantic projection boundaries and reproduces
    the intended 12 c_attn tiers per block.
    """
    if spec.projection_name != "attn.c_attn":
        return [("full", 0, spec.out_features)]
    if spec.out_features % 3 != 0:
        raise ValueError("GPT-2 c_attn output dimension is not divisible by 3.")
    width = spec.out_features // 3
    return [
        ("q", 0, width),
        ("k", width, 2 * width),
        ("v", 2 * width, 3 * width),
    ]


def shard_projection(spec: ProjectionSpec, tile_size: int) -> list[ProjectionShard]:
    projection_weights = spec.out_features * spec.in_features
    shards: list[ProjectionShard] = []
    local_index = 0
    for group_name, group_start, group_end in _group_ranges(spec):
        group_length = group_end - group_start
        for local_out_start, local_out_end in _ranges(group_length, tile_size):
            out_start = group_start + local_out_start
            out_end = group_start + local_out_end
            for in_start, in_end in _ranges(spec.in_features, tile_size):
                count = (out_end - out_start) * (in_end - in_start)
                fraction = count / projection_weights
                importance = spec.sensitivity * fraction
                shards.append(
                    ProjectionShard(
                        shard_id=f"{spec.projection_id}/shard_{local_index:02d}",
                        projection_id=spec.projection_id,
                        block_index=spec.block_index,
                        projection_name=spec.projection_name,
                        hf_module_path=spec.hf_module_path,
                        logical_group=group_name,
                        out_start=out_start,
                        out_end=out_end,
                        in_start=in_start,
                        in_end=in_end,
                        weight_count=count,
                        projection_weight_count=projection_weights,
                        shard_weight=fraction,
                        sensitivity=spec.sensitivity,
                        importance=importance,
                    )
                )
                local_index += 1
    if sum(shard.weight_count for shard in shards) != projection_weights:
        raise RuntimeError(f"Shards do not exactly cover {spec.projection_id}.")
    return shards


def build_gpt2_shards(
    specs: Iterable[ProjectionSpec],
    tile_size: int,
    *,
    require_gpt2_small_contract: bool = True,
) -> list[ProjectionShard]:
    shards: list[ProjectionShard] = []
    specs_list = list(specs)
    for spec in specs_list:
        shards.extend(shard_projection(spec, tile_size))
    ids = [shard.shard_id for shard in shards]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate shard identifiers were generated.")
    if require_gpt2_small_contract:
        if len(specs_list) != 48:
            raise ValueError(f"Expected 48 GPT-2 projections, found {len(specs_list)}.")
        if len(shards) != EXPECTED_GPT2_SMALL_TOTAL_SHARDS:
            raise ValueError(
                f"Expected {EXPECTED_GPT2_SMALL_TOTAL_SHARDS} physical shards, "
                f"found {len(shards)}."
            )
        for block_index in range(12):
            count = sum(shard.block_index == block_index for shard in shards)
            if count != EXPECTED_GPT2_SMALL_SHARDS_PER_BLOCK:
                raise ValueError(
                    f"Block {block_index} has {count} shards; expected "
                    f"{EXPECTED_GPT2_SMALL_SHARDS_PER_BLOCK}."
                )
    return shards
