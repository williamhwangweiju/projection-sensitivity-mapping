"""Pure helpers for deployment-time digital/analog hybrid baselines."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.mapping.placement import PlacementRecord, place_shards
from src.mapping.sharding import build_shards


_DESIGN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: Placement policies accepted for the analog remainder of a hybrid design.
REMAINDER_POLICIES = (
    "random",
    "sequential",
    "hardware_only",
    "static_sensitivity",
    "adversarial",
    "oracle_clairvoyant",
)

#: How projections retained in digital compute hold their weights.
#:
#: ``unclipped``: the original floating-point module of the checkpoint is kept
#: as-is (the configuration Sec. IV-A of the paper labels a diagnostic).
#: ``clipped``: the retained projection is a float32 module whose weights are
#: clipped at the same ``clip_sigma`` threshold the analog path uses, with no
#: quantization and no noise -- the function the HWA checkpoint was trained to
#: compute, i.e. the deployment-relevant digital comparator.
DIGITAL_WEIGHT_MODES = ("unclipped", "clipped")


@dataclass(frozen=True)
class HybridBaselineDesign:
    """A named set of projections retained in full-precision digital compute."""

    name: str
    digital_projection_ids: tuple[str, ...]
    #: Placement policy for the analog remainder; ``None`` means the section
    #: default (``hybrid_baselines.mapping_policy``).
    mapping_policy: str | None = None

    def effective_policy(self, default: str) -> str:
        return self.mapping_policy if self.mapping_policy is not None else str(default)


def parse_digital_weight_mode(config: Mapping[str, Any]) -> str:
    """Return the validated ``hybrid_baselines.digital_weight_mode``."""
    section = config.get("hybrid_baselines")
    if not isinstance(section, Mapping):
        raise ValueError("Config must define a hybrid_baselines mapping.")
    mode = str(section.get("digital_weight_mode", "unclipped")).strip().lower()
    if mode not in DIGITAL_WEIGHT_MODES:
        raise ValueError(
            f"hybrid_baselines.digital_weight_mode must be one of {DIGITAL_WEIGHT_MODES}; "
            f"got {mode!r}."
        )
    return mode


def parse_hybrid_designs(
    config: Mapping[str, Any],
    candidate_projection_ids: Iterable[str],
) -> list[HybridBaselineDesign]:
    """Parse and validate ``hybrid_baselines.designs`` from a pipeline config."""
    section = config.get("hybrid_baselines")
    if not isinstance(section, Mapping):
        raise ValueError("Config must define a hybrid_baselines mapping.")
    raw_designs = section.get("designs")
    if not isinstance(raw_designs, Sequence) or isinstance(raw_designs, (str, bytes)):
        raise ValueError("hybrid_baselines.designs must be a non-empty list.")
    default_policy = str(section.get("mapping_policy", "hardware_only"))
    if default_policy not in REMAINDER_POLICIES:
        raise ValueError(
            f"hybrid_baselines.mapping_policy must be one of {REMAINDER_POLICIES}; "
            f"got {default_policy!r}."
        )
    candidates = {str(value) for value in candidate_projection_ids}
    designs: list[HybridBaselineDesign] = []
    names: set[str] = set()
    for raw in raw_designs:
        if not isinstance(raw, Mapping):
            raise ValueError("Every hybrid baseline design must be a mapping.")
        name = str(raw.get("name", "")).strip()
        if not _DESIGN_NAME.fullmatch(name):
            raise ValueError(
                f"Invalid hybrid design name {name!r}; use letters, digits, '.', '_' or '-'."
            )
        if name in names:
            raise ValueError(f"Duplicate hybrid design name: {name}")
        raw_ids = raw.get("digital_projection_ids")
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise ValueError(f"{name}: digital_projection_ids must be a non-empty list.")
        digital_ids = tuple(dict.fromkeys(str(value) for value in raw_ids))
        if not digital_ids:
            raise ValueError(f"{name}: at least one digital projection is required.")
        unknown = sorted(set(digital_ids) - candidates)
        if unknown:
            raise ValueError(
                f"{name}: projections are outside the Phase-1 universe: {unknown}"
            )
        if len(digital_ids) == len(candidates):
            raise ValueError(f"{name}: at least one projection must remain analog.")
        raw_policy = raw.get("mapping_policy")
        policy: str | None = None
        if raw_policy is not None:
            policy = str(raw_policy).strip()
            if policy not in REMAINDER_POLICIES:
                raise ValueError(
                    f"{name}: mapping_policy must be one of {REMAINDER_POLICIES}; "
                    f"got {policy!r}."
                )
        designs.append(HybridBaselineDesign(name, digital_ids, policy))
        names.add(name)
    if not designs:
        raise ValueError("hybrid_baselines.designs must contain at least one design.")
    return designs


def projection_compute_accounting(
    projection_rows: Iterable[Mapping[str, Any]],
    digital_projection_ids: Iterable[str],
) -> dict[str, float | int]:
    """Return projection-weight/MAC fractions assigned to digital compute.

    For these dense projections, parameter count and per-token MAC count are
    both ``out_features * in_features``. The LM head is tied to the token
    embedding in GPT-2, so this is a compute split, not a claim that keeping it
    digital duplicates model storage.
    """
    digital = set(digital_projection_ids)
    total = 0
    digital_count = 0
    for row in projection_rows:
        count = int(row["out_features"]) * int(row["in_features"])
        total += count
        if str(row["projection_id"]) in digital:
            digital_count += count
    if total <= 0:
        raise ValueError("Phase-1 projection rows contain no projection weights.")
    return {
        "total_projection_macs_per_token": total,
        "digital_projection_macs_per_token": digital_count,
        "analog_projection_macs_per_token": total - digital_count,
        "digital_projection_mac_fraction": digital_count / total,
        "analog_projection_mac_fraction": (total - digital_count) / total,
    }


def build_hybrid_placement(
    config: Mapping[str, Any],
    projection_rows: Iterable[Mapping[str, Any]],
    trace: Any,
    design: HybridBaselineDesign,
) -> list[PlacementRecord]:
    """Remap the remaining analog shards for one hybrid design."""
    rows = list(projection_rows)
    section = config["hybrid_baselines"]
    policy = design.effective_policy(section.get("mapping_policy", "hardware_only"))
    mapping_timestep = int(section.get("mapping_timestep", config["phase3"].get("mapping_timestep", 0)))
    total_timesteps = int(trace.noise_std.shape[0])
    if not 0 <= mapping_timestep < total_timesteps:
        raise ValueError(
            f"Hybrid mapping timestep {mapping_timestep} is outside [0, {total_timesteps - 1}]."
        )
    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    tiers_per_tile = int(config["hardware"]["tiers_per_tile"])
    total_tiers = int(config["hardware"]["num_tiles"]) * tiers_per_tile
    sensitivity_floor = float(config.get("phase3", {}).get("sensitivity_floor", 0.0))
    shards = build_shards(
        rows,
        digital_projection_ids=design.digital_projection_ids,
        tier_rows=tier_rows,
        tier_cols=tier_cols,
        sensitivity_floor=sensitivity_floor,
    )
    if len(shards) > total_tiers:
        raise ValueError(
            f"{design.name}: remaining analog set needs {len(shards)} tiers but "
            f"the substrate provides {total_tiers}."
        )
    if not shards:
        raise ValueError(f"{design.name}: no analog shards remain.")
    if policy == "oracle_clairvoyant":
        noise = np.sqrt(np.square(trace.noise_std.astype(np.float64)).mean(axis=0))
        available = trace.available.all(axis=0)
    else:
        noise = trace.noise_std[mapping_timestep]
        available = trace.available[mapping_timestep]
    return place_shards(
        shards,
        noise=noise,
        available=available,
        tiers_per_tile=tiers_per_tile,
        policy=policy,
        timestep=mapping_timestep,
        seed=int(config["experiment"].get("placement_seed", config["experiment"]["seed"])),
    )


def placement_rows(records: Iterable[PlacementRecord]) -> list[dict[str, Any]]:
    """Return deterministic CSV-ready placement rows."""
    return [row.to_dict() for row in sorted(records, key=lambda value: value.shard_id)]


__all__ = [
    "DIGITAL_WEIGHT_MODES",
    "HybridBaselineDesign",
    "REMAINDER_POLICIES",
    "build_hybrid_placement",
    "parse_digital_weight_mode",
    "parse_hybrid_designs",
    "placement_rows",
    "projection_compute_accounting",
]
