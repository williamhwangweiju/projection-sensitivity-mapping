#!/usr/bin/env python3
"""Place shards by diagonal-Fisher proxy importance (the static_fisher policy)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.mapping.objective import placement_proxy
from src.mapping.placement import PlacementRecord, place_shards
from src.mapping.sharding import build_shards
from src.simulators.tile_fidelity import load_trace

POLICY = "static_fisher"
PROXY_FIELD = "fisher_score"


def proxy_profile_rows(
    projection_rows: Iterable[Mapping[str, Any]],
    proxy_rows: Iterable[Mapping[str, Any]],
    *,
    field: str = PROXY_FIELD,
) -> list[dict[str, Any]]:
    """Copy Phase-1 projection rows with the proxy score as the mapping score.

    Every analog projection must have a proxy score (the former
    ``sensitivity_overrides`` contract); the geometry fields are untouched.
    """
    scores = {str(row["projection_id"]): float(row[field]) for row in proxy_rows}
    rows: list[dict[str, Any]] = []
    for projection in projection_rows:
        projection_id = str(projection["projection_id"])
        if projection_id not in scores:
            raise ValueError(f"Proxy sidecar is missing analog projection {projection_id}.")
        row = dict(projection)
        row["sensitivity_score_for_mapping"] = scores[projection_id]
        rows.append(row)
    return rows


def place_static_fisher(
    projection_rows: Sequence[Mapping[str, Any]],
    proxy_rows: Sequence[Mapping[str, Any]],
    *,
    noise: Sequence[float],
    available: Sequence[bool],
    tiers_per_tile: int,
    tier_rows: int,
    tier_cols: int,
    sensitivity_floor: float = 0.0,
    timestep: int = 0,
    seed: int = 42,
) -> list[PlacementRecord]:
    """Fisher-importance shards placed by the sorted static rule, labeled ``static_fisher``."""
    shard_kwargs: dict[str, Any] = dict(
        digital_projection_ids=[],
        tier_rows=tier_rows,
        tier_cols=tier_cols,
        sensitivity_floor=sensitivity_floor,
    )
    measured = build_shards(projection_rows, **shard_kwargs)
    fisher = build_shards(proxy_profile_rows(projection_rows, proxy_rows), **shard_kwargs)
    if [shard.shard_id for shard in measured] != [shard.shard_id for shard in fisher]:
        raise AssertionError("Fisher shards must share the measured shard geometry.")
    records = place_shards(
        fisher,
        noise=noise,
        available=available,
        tiers_per_tile=tiers_per_tile,
        # Same sorted rule as the proposed policy; only the importance input differs.
        policy="static_sensitivity",
        timestep=timestep,
        seed=seed,
    )
    return [replace(record, policy=POLICY) for record in records]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    config_path: Path,
    phase1_path: Path,
    trace_path: Path,
    proxy_path: Path,
    phase3_manifest_path: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    proxy = load_json(proxy_path)
    trace = load_trace(str(trace_path))
    cfg = config["phase3"]
    mapping_timestep = int(cfg.get("mapping_timestep", 0))
    root = resolve_path(output_root if output_root is not None else cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)

    records = place_static_fisher(
        profile["projections"],
        proxy["projections"],
        noise=trace.noise_std[mapping_timestep],
        available=trace.available[mapping_timestep],
        tiers_per_tile=int(config["hardware"]["tiers_per_tile"]),
        tier_rows=int(config["hardware"]["tier_shape"]["rows"]),
        tier_cols=int(config["hardware"]["tier_shape"]["cols"]),
        sensitivity_floor=float(cfg.get("sensitivity_floor", 0.0)),
        timestep=mapping_timestep,
        seed=int(config["experiment"].get("placement_seed", 42)),
    )
    rows = [record.to_dict() for record in sorted(records, key=lambda row: row.shard_id)]
    placement_path = root / f"placement_{POLICY}.csv"
    write_rows(placement_path, rows)
    entry = {
        "policy": POLICY,
        "placement_path": str(placement_path),
        "analog_shards": len(rows),
        # The policy's own objective, on Fisher importance. Phase 4 recomputes
        # its proxy_variance diagnostic on the measured channel for every policy.
        "proxy_variance": placement_proxy(records, variance=True),
        "proxy_noise": placement_proxy(records, variance=False),
    }

    base: dict[str, Any] = {}
    placements: list[dict[str, Any]] = []
    if phase3_manifest_path is not None:
        base = load_json(phase3_manifest_path)
        placements = [dict(item) for item in base["placements"] if str(item["policy"]) != POLICY]
    placements.append(entry)

    total_tiers = int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    manifest_path = root / "phase3_manifest_fisher.json"
    save_json(
        manifest_path,
        {
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": file_sha256(config_path),
            "phase1_path": str(phase1_path),
            "trace_path": str(trace_path),
            "proxy_path": str(proxy_path),
            "proxy_field": PROXY_FIELD,
            "base_phase3_manifest": None if phase3_manifest_path is None else str(phase3_manifest_path),
            "analog_projection_count": len(profile["projections"]),
            "required_tiers": int(base.get("required_tiers", len(rows))),
            "available_tiers": int(base.get("available_tiers", total_tiers)),
            "placements": placements,
        },
    )
    write_rows(root / "phase3_summary_fisher.csv", placements)
    print(f"static_fisher placement saved to: {placement_path}")
    print(f"Phase 3 manifest with static_fisher: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place shards by diagonal-Fisher importance (static_fisher).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True, help="Measured Phase-1 profile (geometry and measured scores).")
    parser.add_argument("--trace", type=Path, required=True, help="Phase-2 trace.npz.")
    parser.add_argument("--proxy", type=Path, required=True, help="Proxy sidecar from run_proxy_sensitivity.py.")
    parser.add_argument("--phase3-manifest", type=Path, help="Existing Phase-3 manifest whose placements are carried over.")
    parser.add_argument("--output-root", type=Path, help="Defaults to phase3.output_root from the config.")
    args = parser.parse_args()
    main(args.config, args.phase1, args.trace, args.proxy, args.phase3_manifest, args.output_root)
