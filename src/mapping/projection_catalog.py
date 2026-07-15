"""Projection catalog and simulator-module metadata for Phase 3 mapping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


PROJECTION_EXECUTION_ORDER = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    projection_id: str
    block_id: str
    projection_name: str
    execution_index: int
    out_features: int
    in_features: int
    num_weights: int
    sensitivity_score: float
    sensitivity_score_unit: str
    sensitivity_reference_noise_std: float


@dataclass(frozen=True, slots=True)
class MappedModuleSpec:
    module_name: str
    block_id: str
    projection_name: str
    projection_id: str
    execution_index: int
    out_features: int
    in_features: int
    num_weights: int
    sensitivity_score: float
    sensitivity_score_unit: str
    sensitivity_reference_noise_std: float

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.block_id, self.projection_name)


def load_phase1_sensitivity_lookup(
    phase1_results_path: str | Path,
) -> dict[tuple[str, str], float]:
    """Load strict total-DeltaPPL sensitivity scores from Phase 1."""

    path = Path(phase1_results_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    results = loaded.get("results", {})
    projections = results.get("projections", []) if isinstance(results, Mapping) else []
    if not isinstance(projections, list):
        raise ValueError("Phase 1 JSON must contain results.projections as a list.")

    lookup: dict[tuple[str, str], float] = {}
    for index, row in enumerate(projections):
        if not isinstance(row, Mapping):
            raise TypeError(f"Phase-1 projection row {index} is not a mapping.")
        block_id = str(row.get("block_id", "")).strip()
        projection_name = str(row.get("proj_name", "")).strip()
        if projection_name == "lm_head":
            continue
        if not block_id or not projection_name:
            raise ValueError(f"Phase-1 projection row {index} lacks identifiers.")

        if "sensitivity_score_for_mapping" not in row:
            raise KeyError(
                f"{block_id}/{projection_name} lacks "
                "sensitivity_score_for_mapping. Rerun the updated Phase 1."
            )
        unit = str(row.get("sensitivity_score_unit", ""))
        if unit != "delta_ppl_total":
            raise ValueError(
                f"{block_id}/{projection_name} has unsupported sensitivity "
                f"unit {unit!r}; expected 'delta_ppl_total'."
            )
        score = float(row["sensitivity_score_for_mapping"])
        if not (-float("inf") < score < float("inf")):
            raise ValueError(f"Non-finite sensitivity for {block_id}/{projection_name}.")
        key = (block_id, projection_name)
        if key in lookup:
            raise ValueError(f"Duplicate Phase-1 sensitivity for {key}.")
        lookup[key] = score

    expected = {
        (f"block_{block}", projection)
        for block in range(12)
        for projection in PROJECTION_EXECUTION_ORDER
    }
    missing = expected - set(lookup)
    if missing:
        raise ValueError(f"Missing Phase-1 sensitivity rows: {sorted(missing)}")
    return lookup


def projection_group_id(block_id: str, projection_name: str) -> str:
    return f"{block_id}/{projection_name}"


def projection_execution_index(block_id: str, projection_name: str) -> int:
    block_index = int(block_id.removeprefix("block_"))
    return block_index * 100 + PROJECTION_EXECUTION_ORDER.index(projection_name)


def module_projection_metadata(
    module_name: str,
    sensitivity_lookup: Mapping[tuple[str, str], float],
    *,
    include_lm_head: bool = False,
    num_hidden_layers: int = 12,
) -> tuple[str, str, float]:
    """Map one IBM simulator module to the true GPT-2 projection group."""

    if "token_embedding" in module_name or "pos_embedding" in module_name:
        if include_lm_head:
            raise ValueError("Embeddings are digital and cannot be analog projections.")
        return "digital", "embedding", 0.0
    if module_name.endswith("lm_head"):
        if include_lm_head:
            raise ValueError(
                "This unified pipeline keeps lm_head digital. Set include_lm_head=false."
            )
        return "digital", "lm_head", 0.0

    match = re.search(r"layers\.(\d+)", module_name)
    if match is None:
        raise ValueError(f"Cannot determine simulator layer from {module_name!r}.")
    sim_layer = int(match.group(1))

    if any(tag in module_name for tag in ("q_proj_in", "k_proj_in", "v_proj_in")):
        if sim_layer != 0:
            raise ValueError(f"Unexpected *_proj_in outside layer 0: {module_name}")
        target_block, projection_name = 0, "attn.c_attn"
    elif any(tag in module_name for tag in ("q_proj_out", "k_proj_out", "v_proj_out")):
        target_block, projection_name = sim_layer + 1, "attn.c_attn"
    elif "out_proj" in module_name:
        target_block, projection_name = sim_layer, "attn.c_proj"
    elif "ffn1" in module_name:
        target_block, projection_name = sim_layer, "mlp.c_fc"
    elif "ffn2" in module_name:
        target_block, projection_name = sim_layer, "mlp.c_proj"
    else:
        raise ValueError(f"Unsupported simulator module: {module_name!r}")

    if not 0 <= target_block < num_hidden_layers:
        raise ValueError(
            f"{module_name!r} resolves to invalid GPT-2 block {target_block}."
        )
    block_id = f"block_{target_block}"
    key = (block_id, projection_name)
    if key not in sensitivity_lookup:
        raise KeyError(f"Missing Phase-1 sensitivity for {key}.")
    return block_id, projection_name, float(sensitivity_lookup[key])


def iter_mappable_modules(
    model: Any,
    *,
    include_embeddings: bool = False,
    include_lm_head: bool = False,
) -> list[tuple[str, Any]]:
    """Return analog transformer projections only by default."""

    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if type_name == "Embedding":
            if include_embeddings:
                modules.append((name, module))
            continue
        if type_name != "Linear":
            continue
        if name.endswith("lm_head") and not include_lm_head:
            continue
        if any(
            tag in name
            for tag in (
                "q_proj_in", "k_proj_in", "v_proj_in",
                "q_proj_out", "k_proj_out", "v_proj_out",
                "out_proj", "ffn1", "ffn2",
            )
        ):
            modules.append((name, module))
    return modules


def build_mapped_module_specs(
    modules: Sequence[tuple[str, Any]],
    *,
    sensitivity_lookup: Mapping[tuple[str, str], float],
    include_lm_head: bool,
    reference_noise_std: float,
    num_hidden_layers: int = 12,
) -> dict[str, MappedModuleSpec]:
    if reference_noise_std <= 0.0:
        raise ValueError("reference_noise_std must be positive.")

    specs: dict[str, MappedModuleSpec] = {}
    for module_name, module in modules:
        block_id, projection_name, sensitivity = module_projection_metadata(
            module_name,
            sensitivity_lookup,
            include_lm_head=include_lm_head,
            num_hidden_layers=num_hidden_layers,
        )
        if block_id == "digital":
            continue
        # IBM Linear weight orientation is [input_features, output_features].
        in_features = int(module.weight.shape[0])
        out_features = int(module.weight.shape[1])
        specs[module_name] = MappedModuleSpec(
            module_name=module_name,
            block_id=block_id,
            projection_name=projection_name,
            projection_id=projection_group_id(block_id, projection_name),
            execution_index=projection_execution_index(block_id, projection_name),
            out_features=out_features,
            in_features=in_features,
            num_weights=out_features * in_features,
            sensitivity_score=sensitivity,
            sensitivity_score_unit="delta_ppl_total",
            sensitivity_reference_noise_std=float(reference_noise_std),
        )
    if not specs:
        raise ValueError("No analog transformer projection specs were built.")
    return specs


def build_group_total_weights(
    mapped_specs: Sequence[MappedModuleSpec],
) -> dict[tuple[str, str], int]:
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
    if policy_name == "random":
        rng = random.Random(seed)
        result = list(modules)
        rng.shuffle(result)
        return result
    if policy_name == "static_sensitivity":
        return sorted(
            modules,
            key=lambda item: (
                -(
                    specs_by_name[item[0]].sensitivity_score
                    * specs_by_name[item[0]].num_weights
                    / group_total_weights[specs_by_name[item[0]].group_key]
                ),
                specs_by_name[item[0]].execution_index,
                item[0],
            ),
        )
    return list(modules)


def mapped_module_specs_to_rows(
    mapped_specs: Sequence[MappedModuleSpec],
) -> list[dict[str, Any]]:
    group_totals = build_group_total_weights(mapped_specs)
    rows: list[dict[str, Any]] = []
    for spec in sorted(mapped_specs, key=lambda item: item.module_name):
        group_total = group_totals[spec.group_key]
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
                "module_weight_fraction": spec.num_weights / group_total,
                "sensitivity_score": spec.sensitivity_score,
                "sensitivity_score_unit": spec.sensitivity_score_unit,
                "sensitivity_reference_noise_std": (
                    spec.sensitivity_reference_noise_std
                ),
                "module_importance": (
                    spec.sensitivity_score * spec.num_weights / group_total
                ),
            }
        )
    return rows
