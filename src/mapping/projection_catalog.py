"""Projection catalog and simulator-module metadata for Phase 3 mapping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


MAPPABLE_MODULE_TYPES = {"Linear", "Embedding"}

PROJECTION_EXECUTION_ORDER = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "lm_head",
)

EMBEDDING_EXECUTION_ORDER = (
    "token_embedding",
    "pos_embedding",
)

GPT2_SMALL_PROJECTION_SHAPES: dict[str, tuple[int, int]] = {
    "attn.c_attn": (2304, 768),
    "attn.c_proj": (768, 768),
    "mlp.c_fc": (3072, 768),
    "mlp.c_proj": (768, 3072),
    "lm_head": (50257, 768),
}


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Canonical logical projection entry used by lightweight mappers."""

    projection_id: str
    block_id: str
    projection_name: str
    execution_index: int

    out_features: int
    in_features: int
    num_weights: int

    sensitivity_score: float


@dataclass(frozen=True, slots=True)
class MappedModuleSpec:
    """Metadata for one 3D-CIM-mappable model module."""

    module_name: str
    block_id: str
    projection_name: str
    projection_id: str
    execution_index: int
    out_features: int
    in_features: int
    num_weights: int
    sensitivity_score: float

    @property
    def group_key(self) -> tuple[str, str]:
        """Return the Phase-1 projection group represented by this module."""
        return (self.block_id, self.projection_name)


def load_phase1_sensitivity_lookup(
    phase1_results_path: str | Path,
) -> dict[tuple[str, str], float]:
    """Load Phase 1 sensitivities keyed by canonical block/projection group."""
    path = Path(phase1_results_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    results = loaded.get("results", {})
    projections = results.get("projections", [])
    if not isinstance(projections, list):
        raise ValueError("Phase 1 results JSON must contain results.projections list.")

    lookup: dict[tuple[str, str], float] = {}
    for row in projections:
        if not isinstance(row, Mapping):
            continue

        block_id = str(row.get("block_id", "")).strip()
        projection_name = str(row.get("proj_name", "")).strip()
        if not block_id or not projection_name:
            continue

        if projection_name == "lm_head":
            block_id = "head"

        lookup[(block_id, projection_name)] = float(
            row.get("sensitivity_mean", 0.0)
        )

    if not lookup:
        raise ValueError("No Phase 1 sensitivity rows were loaded.")
    return lookup


def projection_group_id(block_id: str, projection_name: str) -> str:
    """Return the canonical logical projection identifier."""
    return f"{block_id}/{projection_name}"


def projection_execution_index(block_id: str, projection_name: str) -> int:
    """Return the simulator-module execution index used by baseline policies."""
    if block_id == "embedding":
        try:
            return -100 + EMBEDDING_EXECUTION_ORDER.index(projection_name)
        except ValueError:
            return -1

    if block_id == "head":
        try:
            projection_index = PROJECTION_EXECUTION_ORDER.index(projection_name)
        except ValueError:
            projection_index = 99
        return 10_000 * 100 + projection_index

    if not block_id.startswith("block_"):
        return 0

    block_index = int(block_id.removeprefix("block_"))
    try:
        projection_index = PROJECTION_EXECUTION_ORDER.index(projection_name)
    except ValueError:
        projection_index = 99
    return block_index * 100 + projection_index


def module_projection_metadata(
    module_name: str,
    sensitivity_lookup: Mapping[tuple[str, str], float],
    *,
    include_lm_head: bool,
) -> tuple[str, str, float]:
    """Map a 3D-CIM module name to its canonical Phase 1 projection group."""
    if "token_embedding" in module_name:
        return "embedding", "token_embedding", 0.0

    if "pos_embedding" in module_name:
        return "embedding", "pos_embedding", 0.0

    if module_name.endswith("lm_head"):
        sensitivity = (
            sensitivity_lookup.get(("head", "lm_head"), 0.0)
            if include_lm_head
            else 0.0
        )
        return "head", "lm_head", float(sensitivity)

    block_match = re.search(r"layers\.(\d+)", module_name)
    if block_match is None:
        raise ValueError(f"Cannot determine block index from module name: {module_name}")
    block_id = f"block_{int(block_match.group(1))}"

    if any(
        tag in module_name
        for tag in (
            "q_proj_in",
            "k_proj_in",
            "v_proj_in",
            "q_proj_out",
            "k_proj_out",
            "v_proj_out",
        )
    ):
        projection_name = "attn.c_attn"
    elif "out_proj" in module_name:
        projection_name = "attn.c_proj"
    elif "ffn1" in module_name:
        projection_name = "mlp.c_fc"
    elif "ffn2" in module_name:
        projection_name = "mlp.c_proj"
    else:
        raise ValueError(
            f"Unsupported module name for projection metadata: {module_name}"
        )

    sensitivity = sensitivity_lookup.get((block_id, projection_name), 0.0)
    return block_id, projection_name, float(sensitivity)


def iter_mappable_modules(model: Any) -> list[tuple[str, Any]]:
    """Return 3D-CIM Linear and Embedding modules in model traversal order."""
    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if type(module).__name__ in MAPPABLE_MODULE_TYPES:
            modules.append((name, module))
    return modules


def build_mapped_module_specs(
    modules: Sequence[tuple[str, Any]],
    *,
    sensitivity_lookup: Mapping[tuple[str, str], float],
    include_lm_head: bool,
) -> dict[str, MappedModuleSpec]:
    """Build canonical metadata for the actual 3D-CIM model modules."""
    specs: dict[str, MappedModuleSpec] = {}

    for module_name, module in modules:
        block_id, projection_name, sensitivity = module_projection_metadata(
            module_name,
            sensitivity_lookup,
            include_lm_head=include_lm_head,
        )
        out_features = int(module.weight.shape[1])
        in_features = int(module.weight.shape[0])
        projection_id = projection_group_id(block_id, projection_name)

        specs[module_name] = MappedModuleSpec(
            module_name=module_name,
            block_id=block_id,
            projection_name=projection_name,
            projection_id=projection_id,
            execution_index=projection_execution_index(block_id, projection_name),
            out_features=out_features,
            in_features=in_features,
            num_weights=out_features * in_features,
            sensitivity_score=sensitivity,
        )

    return specs


def build_group_total_weights(
    mapped_specs: Sequence[MappedModuleSpec],
) -> dict[tuple[str, str], int]:
    """Aggregate physical module weights by logical Phase 1 projection group."""
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for spec in mapped_specs:
        totals[spec.group_key] += spec.num_weights
    return dict(totals)


def order_modules_for_policy(
    modules: list[tuple[str, Any]],
    *,
    specs_by_name: Mapping[str, MappedModuleSpec],
    group_total_weights: Mapping[tuple[str, str], int],
    policy_name: str,
    seed: int,
) -> list[tuple[str, Any]]:
    """Order 3D-CIM modules exactly as required by a static baseline policy."""
    if policy_name == "random":
        rng = random.Random(seed)
        shuffled = list(modules)
        rng.shuffle(shuffled)
        return shuffled

    if policy_name == "static_sensitivity":
        decorated = []
        for index, (name, module) in enumerate(modules):
            spec = specs_by_name[name]
            group_total = group_total_weights.get(spec.group_key, spec.num_weights)
            module_fraction = spec.num_weights / group_total if group_total > 0 else 0.0
            module_importance = spec.sensitivity_score * module_fraction
            decorated.append(
                (-module_importance, spec.execution_index, index, name, module)
            )
        decorated.sort()
        return [(name, module) for _, _, _, name, module in decorated]

    return modules


def mapped_module_specs_to_rows(
    mapped_specs: Sequence[MappedModuleSpec],
) -> list[dict[str, Any]]:
    """Convert simulator-module specs into the existing Phase 3 catalog CSV rows."""
    group_total_weights = build_group_total_weights(mapped_specs)
    rows: list[dict[str, Any]] = []

    for spec in sorted(mapped_specs, key=lambda item: item.module_name):
        group_total = group_total_weights[spec.group_key]
        rows.append(
            {
                "module_name": spec.module_name,
                "projection_id": spec.projection_id,
                "block_id": spec.block_id,
                "projection_name": spec.projection_name,
                "execution_index": spec.execution_index,
                "out_features": spec.out_features,
                "in_features": spec.in_features,
                "num_weights": spec.num_weights,
                "group_total_weights": group_total,
                "module_weight_fraction": (
                    spec.num_weights / group_total
                    if group_total > 0
                    else 0.0
                ),
                "sensitivity_score": spec.sensitivity_score,
                "module_importance": (
                    spec.sensitivity_score * spec.num_weights / group_total
                    if group_total > 0
                    else 0.0
                ),
            }
        )

    return rows
