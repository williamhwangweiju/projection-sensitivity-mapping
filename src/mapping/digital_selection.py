"""Budgeted projection-level digital protection selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.common.config import stable_id


@dataclass(frozen=True)
class DigitalCandidate:
    projection_id: str
    sensitivity: float
    parameter_count: int
    macs_per_token: int
    tied_to_embedding: bool = False

    def score(self, method: str) -> float:
        if method == "sensitivity_rank":
            return self.sensitivity
        if method == "sensitivity_per_parameter":
            return self.sensitivity / max(self.parameter_count, 1)
        if method == "sensitivity_per_mac":
            return self.sensitivity / max(self.macs_per_token, 1)
        raise ValueError(f"Unsupported digital selection method: {method}")


def candidates_from_profile(payload: Mapping[str, Any]) -> list[DigitalCandidate]:
    projections = payload.get("projections")
    if projections is None:
        projections = payload.get("results", {}).get("projections")
    if not isinstance(projections, list):
        raise ValueError("Phase 1 artifact does not contain a projections list.")
    result: list[DigitalCandidate] = []
    for row in projections:
        result.append(
            DigitalCandidate(
                projection_id=str(row["projection_id"]),
                sensitivity=float(row["sensitivity_score_for_mapping"]),
                parameter_count=int(row["parameter_count"]),
                macs_per_token=int(row["macs_per_token"]),
                tied_to_embedding=bool(row.get("tied_to_embedding", False)),
            )
        )
    return result


def select_by_count(
    candidates: Iterable[DigitalCandidate],
    *,
    method: str,
    count: int,
    forced: Iterable[str] = (),
) -> list[str]:
    candidate_list = list(candidates)
    known = {candidate.projection_id for candidate in candidate_list}
    forced_set = {str(value) for value in forced}
    unknown = forced_set - known
    if unknown:
        raise ValueError(f"Forced digital projections are absent from Phase 1: {sorted(unknown)}")
    ranked = sorted(candidate_list, key=lambda item: (-item.score(method), item.projection_id))
    selected = list(sorted(forced_set))
    for candidate in ranked:
        if len(selected) >= count:
            break
        if candidate.projection_id not in forced_set:
            selected.append(candidate.projection_id)
    return sorted(selected)


def select_by_fraction(
    candidates: Iterable[DigitalCandidate],
    *,
    method: str,
    fraction: float,
    cost_field: str,
    forced: Iterable[str] = (),
) -> list[str]:
    candidate_list = list(candidates)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Digital budget fractions must lie in [0, 1].")
    if cost_field not in {"parameter_count", "macs_per_token"}:
        raise ValueError(cost_field)
    total_cost = sum(getattr(candidate, cost_field) for candidate in candidate_list)
    target = fraction * total_cost
    forced_set = {str(value) for value in forced}
    selected: list[str] = []
    cost = 0
    by_id = {candidate.projection_id: candidate for candidate in candidate_list}
    for projection_id in sorted(forced_set):
        if projection_id not in by_id:
            raise ValueError(f"Unknown forced projection: {projection_id}")
        selected.append(projection_id)
        cost += getattr(by_id[projection_id], cost_field)
    ranked = sorted(candidate_list, key=lambda item: (-item.score(method), item.projection_id))
    for candidate in ranked:
        if candidate.projection_id in forced_set:
            continue
        if cost >= target:
            break
        selected.append(candidate.projection_id)
        cost += getattr(candidate, cost_field)
    return sorted(selected)


def operating_point_record(
    candidates: Iterable[DigitalCandidate],
    *,
    method: str,
    budget_type: str,
    budget_value: float,
    digital_projection_ids: Iterable[str],
) -> dict[str, Any]:
    candidate_list = list(candidates)
    selected = frozenset(digital_projection_ids)
    total_parameters = sum(item.parameter_count for item in candidate_list)
    total_macs = sum(item.macs_per_token for item in candidate_list)
    digital_parameters = sum(item.parameter_count for item in candidate_list if item.projection_id in selected)
    # GPT-2's LM head is tied to the digital token embedding. Report its logical
    # digital execution size, but do not count a second weight copy as incremental
    # digital storage when tied_to_embedding is true.
    total_incremental_storage = sum(
        item.parameter_count for item in candidate_list if not item.tied_to_embedding
    )
    digital_incremental_storage = sum(
        item.parameter_count
        for item in candidate_list
        if item.projection_id in selected and not item.tied_to_embedding
    )
    digital_macs = sum(item.macs_per_token for item in candidate_list if item.projection_id in selected)
    record = {
        "selection_method": method,
        "budget_type": budget_type,
        "budget_value": budget_value,
        "digital_projection_ids": sorted(selected),
        "analog_projection_ids": sorted(item.projection_id for item in candidate_list if item.projection_id not in selected),
        "digital_projection_count": len(selected),
        "analog_projection_count": len(candidate_list) - len(selected),
        "digital_parameter_count": digital_parameters,
        "total_parameter_count": total_parameters,
        "digital_parameter_fraction": digital_parameters / max(total_parameters, 1),
        "analog_parameter_fraction": 1.0 - digital_parameters / max(total_parameters, 1),
        "digital_incremental_storage_parameter_count": digital_incremental_storage,
        "total_incremental_storage_parameter_count": total_incremental_storage,
        "digital_incremental_storage_fraction": digital_incremental_storage / max(total_incremental_storage, 1),
        "digital_macs_per_token": digital_macs,
        "total_macs_per_token": total_macs,
        "digital_mac_fraction": digital_macs / max(total_macs, 1),
        "analog_mac_fraction": 1.0 - digital_macs / max(total_macs, 1),
    }
    record["digital_set_id"] = stable_id(
        "digital",
        {
            "method": method,
            "budget_type": budget_type,
            "budget_value": budget_value,
            "digital_projection_ids": record["digital_projection_ids"],
        },
    )
    return record
