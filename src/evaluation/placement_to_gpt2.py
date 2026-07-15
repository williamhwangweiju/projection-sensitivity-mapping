"""Translate Phase-3 IBM 3D-CIM placements to GPT-2 weight slices.

IBM simulator fragments are indexed in ``[input_features, output_features]``
orientation. Phase 4 uses canonical ``[output_features, input_features]``
orientation, so simulator row fragments become canonical columns and simulator
column fragments become canonical rows.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import GPT2ShardAssignment


GPT2_SMALL_HIDDEN = 768
GPT2_SMALL_INNER = 3072
GPT2_SMALL_LAYERS = 12

INJECTABLE_PROJECTIONS = frozenset(
    {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}
)

_LAYER_RE = re.compile(r"layers\.(\d+)")
_SHARD_RE = re.compile(r"#r(\d+)c(\d+)$")


def gpt2_canonical_projection_shapes(
    *,
    hidden_size: int = GPT2_SMALL_HIDDEN,
    inner_size: int = GPT2_SMALL_INNER,
) -> dict[str, tuple[int, int]]:
    """Return canonical ``[out, in]`` shapes for one GPT-2 block."""
    if hidden_size <= 0 or inner_size <= 0:
        raise ValueError("hidden_size and inner_size must be positive.")
    return {
        "attn.c_attn": (3 * hidden_size, hidden_size),
        "attn.c_proj": (hidden_size, hidden_size),
        "mlp.c_fc": (inner_size, hidden_size),
        "mlp.c_proj": (hidden_size, inner_size),
    }


def expected_gpt2_module_shapes(
    *,
    num_hidden_layers: int = GPT2_SMALL_LAYERS,
    hidden_size: int = GPT2_SMALL_HIDDEN,
    inner_size: int = GPT2_SMALL_INNER,
) -> dict[str, tuple[int, int]]:
    """Return expected canonical shapes for all injectable GPT-2 modules."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")
    per_block = gpt2_canonical_projection_shapes(
        hidden_size=hidden_size,
        inner_size=inner_size,
    )
    return {
        f"transformer.h.{block}.{projection}": shape
        for block in range(num_hidden_layers)
        for projection, shape in per_block.items()
    }


def _parse_layer_index(module_name: str) -> int:
    match = _LAYER_RE.search(module_name)
    if match is None:
        raise ValueError(
            f"Cannot extract simulator layer index from {module_name!r}."
        )
    return int(match.group(1))


def _parse_shard_indices(shard_id: str) -> tuple[int, int]:
    """Return simulator ``(input_fragment, output_fragment)`` indices."""
    match = _SHARD_RE.search(shard_id)
    if match is None:
        raise ValueError(
            f"Cannot parse '#r<input>c<output>' from shard_id {shard_id!r}."
        )
    return int(match.group(1)), int(match.group(2))


def _resolve_hf_info(
    module_name: str,
    sim_layer: int,
    *,
    hidden_size: int,
    num_hidden_layers: int,
) -> tuple[str, str, str | None, int, int]:
    """Resolve a simulator module to GPT-2 path and Q/K/V row offset."""
    if "token_embedding" in module_name or "pos_embedding" in module_name:
        raise ValueError(f"Embedding is not injectable: {module_name!r}.")
    if module_name.endswith("lm_head"):
        raise ValueError("The LM head is intentionally kept digital in Phase 4.")

    if "q_proj_in" in module_name:
        if sim_layer != 0:
            raise ValueError(
                f"Unexpected q_proj_in outside simulator layer 0: {module_name!r}."
            )
        qkv, target_block, row_offset = "Q", 0, 0
    elif "k_proj_in" in module_name:
        if sim_layer != 0:
            raise ValueError(
                f"Unexpected k_proj_in outside simulator layer 0: {module_name!r}."
            )
        qkv, target_block, row_offset = "K", 0, hidden_size
    elif "v_proj_in" in module_name:
        if sim_layer != 0:
            raise ValueError(
                f"Unexpected v_proj_in outside simulator layer 0: {module_name!r}."
            )
        qkv, target_block, row_offset = "V", 0, 2 * hidden_size
    elif "q_proj_out" in module_name:
        qkv, target_block, row_offset = "Q", sim_layer + 1, 0
    elif "k_proj_out" in module_name:
        qkv, target_block, row_offset = "K", sim_layer + 1, hidden_size
    elif "v_proj_out" in module_name:
        qkv, target_block, row_offset = "V", sim_layer + 1, 2 * hidden_size
    elif "out_proj" in module_name:
        qkv, target_block, row_offset = None, sim_layer, 0
    elif "ffn1" in module_name:
        qkv, target_block, row_offset = None, sim_layer, 0
    elif "ffn2" in module_name:
        qkv, target_block, row_offset = None, sim_layer, 0
    else:
        raise ValueError(f"Unsupported simulator module: {module_name!r}.")

    if not 0 <= target_block < num_hidden_layers:
        raise ValueError(
            f"{module_name!r} resolves to GPT-2 block {target_block}, but the "
            f"model has blocks [0, {num_hidden_layers})."
        )

    if qkv is not None:
        projection = "attn.c_attn"
    elif "out_proj" in module_name:
        projection = "attn.c_proj"
    elif "ffn1" in module_name:
        projection = "mlp.c_fc"
    else:
        projection = "mlp.c_proj"

    hf_path = f"transformer.h.{target_block}.{projection}"
    return hf_path, projection, qkv, row_offset, target_block


def _compute_canonical_coordinates(
    *,
    input_shard_index: int,
    output_shard_index: int,
    output_row_offset: int,
    used_input_features: int,
    used_output_features: int,
    tier_rows: int,
    tier_cols: int,
) -> tuple[int, int, int, int]:
    """Convert IBM ``[input, output]`` coordinates to canonical ``[out, in]``."""
    canonical_row_start = output_row_offset + output_shard_index * tier_cols
    canonical_row_end = canonical_row_start + used_output_features
    canonical_col_start = input_shard_index * tier_rows
    canonical_col_end = canonical_col_start + used_input_features
    return (
        canonical_row_start,
        canonical_row_end,
        canonical_col_start,
        canonical_col_end,
    )


def load_placement_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load one Phase-3 placement CSV."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open(newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"Placement CSV is empty: {resolved}.")
    return rows


def _required_int(row: Mapping[str, Any], key: str, shard_id: str) -> int:
    if key not in row or str(row[key]).strip() == "":
        raise KeyError(f"{shard_id}: placement row is missing {key!r}.")
    return int(row[key])


def _required_float(row: Mapping[str, Any], key: str, shard_id: str) -> float:
    if key not in row or str(row[key]).strip() == "":
        raise KeyError(f"{shard_id}: placement row is missing {key!r}.")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{shard_id}: {key} must be finite, got {value}.")
    return value


def build_shard_assignments(
    placement_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    placement_seed: int,
    tier_rows: int = 512,
    tier_cols: int = 512,
    num_tiles: int = 72,
    tiers_per_tile: int = 8,
    num_hidden_layers: int = GPT2_SMALL_LAYERS,
    hidden_size: int = GPT2_SMALL_HIDDEN,
    inner_size: int = GPT2_SMALL_INNER,
    skip_embeddings: bool = True,
    skip_lm_head: bool = True,
) -> list[GPT2ShardAssignment]:
    """Convert Phase-3 rows into validated GPT-2 shard assignments."""
    if not policy:
        raise ValueError("policy cannot be empty.")
    if tier_rows <= 0 or tier_cols <= 0:
        raise ValueError("tier_rows and tier_cols must be positive.")
    if num_tiles <= 0 or tiers_per_tile <= 0:
        raise ValueError("num_tiles and tiers_per_tile must be positive.")

    expected_shapes = gpt2_canonical_projection_shapes(
        hidden_size=hidden_size,
        inner_size=inner_size,
    )
    assignments: list[GPT2ShardAssignment] = []

    for row in placement_rows:
        module_name = str(row.get("module_name", "")).strip()
        shard_id = str(row.get("shard_id", "")).strip()
        if not module_name or not shard_id:
            raise ValueError("Each placement row needs module_name and shard_id.")

        if skip_embeddings and (
            "token_embedding" in module_name or "pos_embedding" in module_name
        ):
            continue
        if skip_lm_head and module_name.endswith("lm_head"):
            continue

        csv_projection_name = str(row.get("projection_name", "")).strip()
        if csv_projection_name and csv_projection_name not in INJECTABLE_PROJECTIONS:
            continue

        sim_layer = _parse_layer_index(module_name)
        input_shard_idx, output_shard_idx = _parse_shard_indices(shard_id)
        hf_path, resolved_projection, qkv, row_offset, target_block = _resolve_hf_info(
            module_name,
            sim_layer,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )
        expected_projection_id = f"block_{target_block}/{resolved_projection}"

        if csv_projection_name and csv_projection_name != resolved_projection:
            raise ValueError(
                f"{shard_id}: Phase-3 projection_name={csv_projection_name!r}, "
                f"but Phase 4 resolves {resolved_projection!r}."
            )
        csv_projection_id = str(row.get("projection_id", "")).strip()
        if csv_projection_id and csv_projection_id != expected_projection_id:
            raise ValueError(
                f"{shard_id}: Phase-3 projection_id={csv_projection_id!r}, "
                f"expected {expected_projection_id!r}. Regenerate Phase-3 placements."
            )

        used_rows = _required_int(row, "used_rows", shard_id)
        used_cols = _required_int(row, "used_cols", shard_id)
        if not 1 <= used_rows <= tier_rows:
            raise ValueError(
                f"{shard_id}: used_rows={used_rows}, expected [1, {tier_rows}]."
            )
        if not 1 <= used_cols <= tier_cols:
            raise ValueError(
                f"{shard_id}: used_cols={used_cols}, expected [1, {tier_cols}]."
            )

        sim_input_start = input_shard_idx * tier_rows
        sim_input_end = sim_input_start + used_rows
        sim_output_start = output_shard_idx * tier_cols
        sim_output_end = sim_output_start + used_cols

        row_start, row_end, col_start, col_end = _compute_canonical_coordinates(
            input_shard_index=input_shard_idx,
            output_shard_index=output_shard_idx,
            output_row_offset=row_offset,
            used_input_features=used_rows,
            used_output_features=used_cols,
            tier_rows=tier_rows,
            tier_cols=tier_cols,
        )

        projection_shape = expected_shapes[resolved_projection]
        if row_end > projection_shape[0] or col_end > projection_shape[1]:
            raise ValueError(
                f"{shard_id}: canonical slice rows [{row_start}, {row_end}), "
                f"cols [{col_start}, {col_end}) exceeds {resolved_projection} "
                f"shape {projection_shape}."
            )

        tile_id = _required_int(row, "tile_id", shard_id)
        tier_key = "tier_start" if "tier_start" in row else "tier_id"
        tier_id = _required_int(row, tier_key, shard_id)
        if not 0 <= tile_id < num_tiles:
            raise ValueError(
                f"{shard_id}: tile_id={tile_id} outside [0, {num_tiles})."
            )
        if not 0 <= tier_id < tiers_per_tile:
            raise ValueError(
                f"{shard_id}: tier_id={tier_id} outside [0, {tiers_per_tile})."
            )

        weights_in_shard = int(row.get("weights_in_shard") or used_rows * used_cols)
        expected_shard_weights = used_rows * used_cols
        if weights_in_shard != expected_shard_weights:
            raise ValueError(
                f"{shard_id}: weights_in_shard={weights_in_shard}, but "
                f"used_rows*used_cols={expected_shard_weights}."
            )
        weights_in_projection = _required_int(row, "group_total_weights", shard_id)
        if weights_in_projection <= 0:
            raise ValueError(f"{shard_id}: group_total_weights must be positive.")

        shard_weight = weights_in_shard / weights_in_projection
        sensitivity_score = _required_float(row, "sensitivity_score", shard_id)

        assignments.append(
            GPT2ShardAssignment(
                shard_id=shard_id,
                projection_id=expected_projection_id,
                hf_module_path=hf_path,
                sim_module_path=module_name,
                sim_layer=sim_layer,
                target_gpt2_block=target_block,
                qkv_component=qkv,
                sim_input_start=sim_input_start,
                sim_input_end=sim_input_end,
                sim_output_start=sim_output_start,
                sim_output_end=sim_output_end,
                canonical_row_start=row_start,
                canonical_row_end=row_end,
                canonical_col_start=col_start,
                canonical_col_end=col_end,
                tile_id=tile_id,
                tier_id=tier_id,
                policy=policy,
                placement_seed=placement_seed,
                weights_in_shard=weights_in_shard,
                weights_in_projection=weights_in_projection,
                shard_weight=shard_weight,
                sensitivity_score=sensitivity_score,
            )
        )

    if not assignments:
        raise ValueError("No injectable GPT-2 assignments were created.")
    return assignments


def validate_shard_coverage(
    assignments: Sequence[GPT2ShardAssignment],
    *,
    canonical_shapes: Mapping[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Require every expected GPT-2 projection weight to be covered once."""
    expected_paths = set(canonical_shapes)
    actual_paths = {assignment.hf_module_path for assignment in assignments}
    missing_modules = expected_paths - actual_paths
    unexpected_modules = actual_paths - expected_paths
    if missing_modules or unexpected_modules:
        raise ValueError(
            f"Missing modules: {sorted(missing_modules)}; "
            f"unexpected modules: {sorted(unexpected_modules)}."
        )

    coverage_by_module = {
        path: np.zeros(shape, dtype=np.uint8)
        for path, shape in canonical_shapes.items()
    }
    for assignment in assignments:
        shape = canonical_shapes[assignment.hf_module_path]
        if not (
            0 <= assignment.canonical_row_start
            < assignment.canonical_row_end
            <= shape[0]
        ):
            raise ValueError(
                f"{assignment.shard_id}: invalid canonical row range for {shape}."
            )
        if not (
            0 <= assignment.canonical_col_start
            < assignment.canonical_col_end
            <= shape[1]
        ):
            raise ValueError(
                f"{assignment.shard_id}: invalid canonical column range for {shape}."
            )
        target = coverage_by_module[assignment.hf_module_path]
        target[
            assignment.canonical_row_start : assignment.canonical_row_end,
            assignment.canonical_col_start : assignment.canonical_col_end,
        ] += 1

    for path, coverage in coverage_by_module.items():
        missing = int((coverage == 0).sum())
        overlaps = int((coverage > 1).sum())
        if missing or overlaps:
            raise ValueError(
                f"{path}: missing_weights={missing}, overlapping_weights={overlaps}, "
                f"shape={coverage.shape}."
            )
    return coverage_by_module


def group_assignments_by_module(
    assignments: Sequence[GPT2ShardAssignment],
) -> dict[str, list[GPT2ShardAssignment]]:
    """Group assignments by Hugging Face module path in deterministic order."""
    result: dict[str, list[GPT2ShardAssignment]] = {}
    for assignment in assignments:
        result.setdefault(assignment.hf_module_path, []).append(assignment)
    for shards in result.values():
        shards.sort(
            key=lambda item: (
                item.canonical_row_start,
                item.canonical_col_start,
                item.shard_id,
            )
        )
    return dict(sorted(result.items()))
