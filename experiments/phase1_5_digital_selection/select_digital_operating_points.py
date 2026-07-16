#!/usr/bin/env python3
"""Generate budgeted digital-protection operating points from Phase 1."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_json, load_yaml, resolve_path, save_json
from src.mapping.digital_selection import (
    candidates_from_profile,
    operating_point_record,
    select_by_count,
    select_by_fraction,
)
from src.mapping.sharding import count_projection_shards


def main(config_path: Path, phase1_path: Path) -> Path:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    candidates = candidates_from_profile(profile)
    cfg = config["digital_selection"]
    forced = [str(value) for value in cfg.get("forced_digital", [])]
    methods = [str(value) for value in cfg["methods"]]
    records: list[dict[str, Any]] = []
    explicit_sets = cfg.get("explicit_sets", {})
    for name, projection_ids in explicit_sets.items():
        record = operating_point_record(
            candidates,
            method=f"explicit:{name}",
            budget_type="explicit",
            budget_value=float(len(projection_ids)),
            digital_projection_ids=projection_ids,
        )
        records.append(record)
    for method in methods:
        for count in cfg["budgets"].get("projection_counts", []):
            selected = select_by_count(candidates, method=method, count=int(count), forced=forced)
            records.append(operating_point_record(candidates, method=method, budget_type="projection_count", budget_value=float(count), digital_projection_ids=selected))
        for fraction in cfg["budgets"].get("parameter_fractions", []):
            selected = select_by_fraction(candidates, method=method, fraction=float(fraction), cost_field="parameter_count", forced=forced)
            records.append(operating_point_record(candidates, method=method, budget_type="parameter_fraction", budget_value=float(fraction), digital_projection_ids=selected))
        for fraction in cfg["budgets"].get("mac_fractions", []):
            selected = select_by_fraction(candidates, method=method, fraction=float(fraction), cost_field="macs_per_token", forced=forced)
            records.append(operating_point_record(candidates, method=method, budget_type="mac_fraction", budget_value=float(fraction), digital_projection_ids=selected))

    # Preserve explicitly named operating points when another selector yields the same set.
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        unique.setdefault(tuple(record["digital_projection_ids"]), record)
    records = list(unique.values())
    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    total_slots = int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    profile_by_id = {str(row["projection_id"]): row for row in profile["projections"]}
    for record in records:
        analog_shards = 0
        for projection_id in record["analog_projection_ids"]:
            row = profile_by_id[projection_id]
            analog_shards += count_projection_shards(
                projection_id,
                int(row["out_features"]),
                int(row["in_features"]),
                tier_rows,
                tier_cols,
            )
        record["analog_shard_count"] = analog_shards
        record["available_physical_tiers"] = total_slots
        record["capacity_feasible"] = analog_shards <= total_slots
    records = sorted(unique.values(), key=lambda r: (r["digital_mac_fraction"], r["digital_parameter_fraction"], r["digital_set_id"]))
    output_root = resolve_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "digital_operating_points.json"
    save_json(
        json_path,
        {
            "phase1_path": str(phase1_path),
            "profile_mapping_unit": profile["mapping_sensitivity_unit"],
            "operating_points": records,
        },
    )
    csv_path = output_root / "digital_operating_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "digital_set_id", "selection_method", "budget_type", "budget_value",
            "digital_projection_count", "digital_parameter_fraction",
            "digital_incremental_storage_fraction", "digital_mac_fraction",
            "analog_shard_count", "available_physical_tiers", "capacity_feasible",
            "digital_projection_ids",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record[key] for key in fields if key != "digital_projection_ids"}, "digital_projection_ids": ";".join(record["digital_projection_ids"])})
    print(f"Digital operating points saved to: {json_path}")
    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.phase1)
