#!/usr/bin/env python3
"""Run migration-aware adaptive analog placement for each digital operating point."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.tabular import write_csv
from src.evaluation.noise_materialization import read_placement_csv
from src.mapping.adaptive import adaptive_step
from src.mapping.placement import PlacementRecord
from src.mapping.sharding import build_shards
from src.simulators.tile_fidelity import load_trace


def records_from_rows(rows: list[dict[str, Any]]) -> list[PlacementRecord]:
    fields = PlacementRecord.__dataclass_fields__
    return [PlacementRecord(**{field: row[field] for field in fields}) for row in rows]


def selected_points(points: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    budget_types = set(str(value) for value in cfg.get("evaluate_budget_types", []))
    methods = set(str(value) for value in cfg.get("evaluate_selection_methods", []))
    result = [
        point for point in points
        if bool(point.get("capacity_feasible", True))
        and (not budget_types or str(point["budget_type"]) in budget_types)
        and (not methods or str(point["selection_method"]) in methods)
    ]
    maximum = cfg.get("max_operating_points")
    if maximum is not None:
        result = result[: int(maximum)]
    if not result:
        raise ValueError("No digital operating points matched Phase-5 filters.")
    return result


def main(
    config_path: Path,
    phase1_path: Path,
    operating_points_path: Path,
    trace_path: Path,
    phase3_manifest_path: Path,
) -> Path:
    config = load_yaml(config_path)
    cfg = config["phase5"]
    profile = load_json(phase1_path)
    points = selected_points(
        load_json(operating_points_path)["operating_points"], cfg
    )
    phase3 = load_json(phase3_manifest_path)["placements"]
    initial_paths = {
        row["digital_set_id"]: Path(row["placement_path"])
        for row in phase3
        if row["policy"] == "static_sensitivity"
    }
    points = [point for point in points if str(point["digital_set_id"]) in initial_paths]
    if not points:
        raise ValueError("All selected Phase-5 operating points were capacity-infeasible or lacked static placements.")
    trace = load_trace(str(trace_path))
    output_root = resolve_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    unavailable_noise_std = float(
        cfg.get("unavailable_noise_std", config["phase2"]["fidelity_model"]["max_noise_std"])
    )
    check_interval = max(int(cfg.get("check_interval", 1)), 1)
    cooldown = max(int(cfg.get("cooldown_timesteps", 0)), 0)
    for point in points:
        digital_set_id = str(point["digital_set_id"])
        if digital_set_id not in initial_paths:
            raise FileNotFoundError(
                f"No static-sensitivity initialization for {digital_set_id}."
            )
        shards = build_shards(
            profile["projections"],
            digital_projection_ids=point["digital_projection_ids"],
            tier_rows=int(config["hardware"]["tier_shape"]["rows"]),
            tier_cols=int(config["hardware"]["tier_shape"]["cols"]),
            sensitivity_floor=float(config["phase3"].get("sensitivity_floor", 0.0)),
        )
        current = records_from_rows(read_placement_csv(initial_paths[digital_set_id]))
        last_remap = -10**9
        all_placements: list[dict[str, Any]] = []
        point_events: list[dict[str, Any]] = []
        for timestep in range(trace.noise_std.shape[0]):
            should_check = timestep > 0 and timestep % check_interval == 0
            if should_check:
                current, diagnostics = adaptive_step(
                    shards,
                    current,
                    timestep=timestep,
                    noise=trace.noise_std[timestep],
                    available=trace.available[timestep],
                    tiers_per_tile=int(config["hardware"]["tiers_per_tile"]),
                    seed=int(config["experiment"].get("placement_seed", 42)),
                    unavailable_noise_std=unavailable_noise_std,
                    minimum_relative_improvement=float(cfg["minimum_relative_proxy_improvement"]),
                    migration_penalty_per_moved_weight_fraction=float(cfg["migration_penalty_per_moved_weight_fraction"]),
                    max_moved_weight_fraction=float(cfg["max_moved_weight_fraction"]),
                    cooldown_satisfied=(timestep - last_remap >= cooldown),
                )
                if bool(diagnostics["accepted"]):
                    last_remap = timestep
            else:
                from src.mapping.adaptive import refresh_placement_state
                current = refresh_placement_state(
                    current,
                    timestep=timestep,
                    noise=trace.noise_std[timestep],
                    available=trace.available[timestep],
                    unavailable_noise_std=unavailable_noise_std,
                )
                diagnostics = {
                    "accepted": False,
                    "reason": "not_scheduled",
                    "current_proxy_variance": None,
                    "candidate_proxy_variance": None,
                    "absolute_proxy_improvement": None,
                    "relative_proxy_improvement": None,
                    "moved_shards": 0.0,
                    "moved_weights": 0.0,
                    "moved_bytes_fp32": 0.0,
                    "moved_weight_fraction": 0.0,
                    "migration_penalty": 0.0,
                    "penalized_gain": None,
                }
            all_placements.extend(
                {
                    "digital_set_id": digital_set_id,
                    **record.to_dict(),
                }
                for record in current
            )
            event = {
                "digital_set_id": digital_set_id,
                "selection_method": point["selection_method"],
                "digital_parameter_fraction": point["digital_parameter_fraction"],
                "digital_mac_fraction": point["digital_mac_fraction"],
                "timestep": timestep,
                **diagnostics,
            }
            point_events.append(event)
            event_rows.append(event)
        point_dir = output_root / digital_set_id
        placement_path = write_csv(point_dir / "adaptive_placements.csv", all_placements)
        events_path = write_csv(point_dir / "adaptive_events.csv", point_events)
        manifest.append({
            "digital_set_id": digital_set_id,
            "adaptive_placement_path": str(placement_path),
            "adaptive_events_path": str(events_path),
            "remap_events": sum(int(bool(row["accepted"])) for row in point_events),
            "total_moved_bytes_fp32": sum(float(row["moved_bytes_fp32"] or 0.0) for row in point_events if bool(row["accepted"])),
        })
    write_csv(output_root / "adaptive_event_summary.csv", event_rows)
    manifest_path = output_root / "phase5_manifest.json"
    save_json(manifest_path, {
        "repository_commit": git_commit(REPO_ROOT),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "phase1_path": str(phase1_path),
        "operating_points_path": str(operating_points_path),
        "trace_path": str(trace_path),
        "phase3_manifest_path": str(phase3_manifest_path),
        "adaptive_runs": manifest,
        "policy": "adaptive_sensitivity",
        "sensitivity_mode": "fixed_offline_phase1",
    })
    print(f"Phase 5 adaptive mapping complete: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.phase1, args.operating_points, args.trace, args.phase3_manifest)
