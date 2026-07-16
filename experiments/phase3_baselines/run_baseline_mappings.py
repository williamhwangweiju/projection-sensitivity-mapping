#!/usr/bin/env python3
"""Create static placements for every digital-protection operating point."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.mapping.objective import placement_proxy
from src.mapping.placement import place_shards
from src.mapping.sharding import build_shards
from src.simulators.tile_fidelity import load_trace


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(config_path: Path, phase1_path: Path, operating_points_path: Path, trace_path: Path) -> Path:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    operating_points = load_json(operating_points_path)["operating_points"]
    trace = load_trace(str(trace_path))
    cfg = config["phase3"]
    mapping_timestep = int(cfg.get("mapping_timestep", 0))
    policies = [str(value) for value in cfg["policies"]]
    output_root = resolve_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    skipped: list[dict] = []
    for point in operating_points:
        digital_set_id = point["digital_set_id"]
        if not bool(point.get("capacity_feasible", True)):
            skipped.append({
                "digital_set_id": digital_set_id,
                "reason": "analog_shards_exceed_available_tiers",
                "analog_shard_count": point.get("analog_shard_count"),
                "available_physical_tiers": point.get("available_physical_tiers"),
            })
            continue
        shards = build_shards(
            profile["projections"],
            digital_projection_ids=point["digital_projection_ids"],
            tier_rows=int(config["hardware"]["tier_shape"]["rows"]),
            tier_cols=int(config["hardware"]["tier_shape"]["cols"]),
            sensitivity_floor=float(config["phase3"].get("sensitivity_floor", 0.0)),
        )
        point_dir = output_root / digital_set_id
        point_dir.mkdir(parents=True, exist_ok=True)
        save_json(point_dir / "digital_operating_point.json", point)
        for policy in policies:
            records = place_shards(
                shards,
                noise=trace.noise_std[mapping_timestep],
                available=trace.available[mapping_timestep],
                tiers_per_tile=int(config["hardware"]["tiers_per_tile"]),
                policy=policy,
                timestep=mapping_timestep,
                seed=int(config["experiment"].get("placement_seed", 42)),
            )
            rows = [record.to_dict() for record in sorted(records, key=lambda row: row.shard_id)]
            path = point_dir / f"placement_{policy}.csv"
            write_rows(path, rows)
            manifest.append({
                "digital_set_id": digital_set_id,
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
            "operating_points_path": str(operating_points_path),
            "trace_path": str(trace_path),
            "placements": manifest,
            "skipped_operating_points": skipped,
        },
    )
    write_rows(output_root / "phase3_summary.csv", manifest)
    print(f"Phase 3 manifest saved to: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.phase1, args.operating_points, args.trace)
