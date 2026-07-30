#!/usr/bin/env python3
"""Create static all-analog placements for every configured policy."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.mapping.objective import placement_proxy
from src.mapping.placement import place_shards
from src.mapping.sharding import build_shards, count_projection_shards
from src.simulators.tile_fidelity import load_trace

# Policies whose shard importance comes from the proxy-sensitivity sidecar
# rather than the measured Phase-1 score. Shard geometry (and therefore shard
# IDs) is identical across importance channels.
PROXY_IMPORTANCE_POLICIES = {"static_fisher": "fisher_score"}

# Policies that place against the full trace trajectory (RMS tile noise and
# all-time availability) instead of the mapping-timestep snapshot. Phase 4
# re-reads tile noise from the trace at evaluation time, so only the physical
# assignment carries over.
CLAIRVOYANT_POLICIES = {"oracle_clairvoyant"}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    config_path: Path,
    phase1_path: Path,
    trace_path: Path,
    proxy_path: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    trace = load_trace(str(trace_path))
    cfg = config["phase3"]
    mapping_timestep = int(cfg.get("mapping_timestep", 0))
    policies = [str(value) for value in cfg["policies"]]
    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    total_tiers = int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    sensitivity_floor = float(cfg.get("sensitivity_floor", 0.0))

    # The deployment is all-analog: every profiled projection is a candidate
    # and none is protected digitally. Capacity must hold up front.
    required_tiers = sum(
        count_projection_shards(
            str(row["projection_id"]),
            int(row["out_features"]),
            int(row["in_features"]),
            tier_rows,
            tier_cols,
        )
        for row in profile["projections"]
    )
    if required_tiers > total_tiers:
        raise ValueError(
            f"All-analog deployment needs {required_tiers} tiers but the "
            f"substrate provides {total_tiers}."
        )

    proxy_policies = [policy for policy in policies if policy in PROXY_IMPORTANCE_POLICIES]
    proxy_scores: dict[str, dict[str, float]] = {}
    if proxy_policies:
        if proxy_path is None:
            raise ValueError(
                f"Policies {proxy_policies} require a proxy sensitivity artifact "
                "(run_proxy_sensitivity.py); pass --proxy."
            )
        proxy_rows = load_json(proxy_path)["projections"]
        for policy in proxy_policies:
            field = PROXY_IMPORTANCE_POLICIES[policy]
            proxy_scores[policy] = {
                str(row["projection_id"]): float(row[field]) for row in proxy_rows
            }

    output_root = resolve_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    shards = build_shards(
        profile["projections"],
        digital_projection_ids=[],
        tier_rows=tier_rows,
        tier_cols=tier_cols,
        sensitivity_floor=sensitivity_floor,
    )
    proxy_shards = {
        policy: build_shards(
            profile["projections"],
            digital_projection_ids=[],
            tier_rows=tier_rows,
            tier_cols=tier_cols,
            sensitivity_floor=sensitivity_floor,
            sensitivity_overrides=proxy_scores[policy],
        )
        for policy in proxy_policies
    }

    snapshot_noise = trace.noise_std[mapping_timestep]
    snapshot_available = trace.available[mapping_timestep]
    trajectory_noise = np.sqrt(np.square(trace.noise_std.astype(np.float64)).mean(axis=0))
    trajectory_available = trace.available.all(axis=0)

    manifest: list[dict] = []
    for policy in policies:
        clairvoyant = policy in CLAIRVOYANT_POLICIES
        records = place_shards(
            proxy_shards.get(policy, shards),
            noise=trajectory_noise if clairvoyant else snapshot_noise,
            available=trajectory_available if clairvoyant else snapshot_available,
            tiers_per_tile=int(config["hardware"]["tiers_per_tile"]),
            policy=policy,
            timestep=mapping_timestep,
            seed=int(config["experiment"].get("placement_seed", 42)),
        )
        rows = [record.to_dict() for record in sorted(records, key=lambda row: row.shard_id)]
        path = output_root / f"placement_{policy}.csv"
        write_rows(path, rows)
        manifest.append({
            "policy": policy,
            "placement_path": str(path),
            "analog_shards": len(rows),
            "proxy_variance": placement_proxy(records, variance=True),
            "proxy_noise": placement_proxy(records, variance=False),
        })

    manifest_path = output_root / "phase3_manifest.json"
    save_json(
        manifest_path,
        {
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "phase1_path": str(phase1_path),
            "trace_path": str(trace_path),
            "proxy_path": None if proxy_path is None else str(proxy_path),
            "analog_projection_count": len(profile["projections"]),
            "required_tiers": required_tiers,
            "available_tiers": total_tiers,
            "placements": manifest,
        },
    )
    write_rows(output_root / "phase3_summary.csv", manifest)
    print(f"Phase 3 manifest saved to: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--proxy",
        type=Path,
        help="Proxy sensitivity sidecar; required when policies include static_fisher.",
    )
    args = parser.parse_args()
    main(args.config, args.phase1, args.trace, args.proxy)
