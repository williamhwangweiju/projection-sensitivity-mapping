#!/usr/bin/env python3
"""Run Phase 3: assign all 480 GPT-2 shards to physical 3D-CIM tiers.

The default workflow does not import IBM ``threedsim`` because performance
simulation is disabled.  The 72x8 physical slot model remains explicit and the
placement CSVs are authoritative for Phase 4.  This makes the pipeline directly
runnable in Colab without an unnecessary simulator submodule.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "src" / "mapping" / "sharding.py").is_file():
            return candidate
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_yaml, resolve_path
from src.mapping.objective import evaluate_placement
from src.mapping.placement import assign_policy, build_slots
from src.mapping.sharding import build_gpt2_shards, projection_specs_from_phase1
from src.simulators.tile_fidelity import load_fidelity_trace


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(
    config_path: Path,
    phase1_results: Path,
    phase2_trace: Path,
    phase2_metadata: Path | None,
    seed: int,
    output_dir: Path | None,
    overwrite: bool,
) -> Path:
    config = load_yaml(config_path)
    phase3 = config["phase3"]
    if bool(phase3.get("run_performance_simulation", False)):
        raise NotImplementedError(
            "This corrected Colab overlay handles physical placement and quality. "
            "Set run_performance_simulation=false; run IBM 3D-SiM separately for "
            "latency/energy after exporting the placement CSVs."
        )
    if output_dir is None:
        output_dir = (
            resolve_path(REPO_ROOT, phase3["output_root"])
            / str(phase3["name"])
            / f"seed_{seed}"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1 = _read_json(phase1_results)
    trace = load_fidelity_trace(phase2_trace)
    noise_std = trace["noise_std"].astype(np.float64)
    available = trace["available"].astype(bool)
    faulted = trace["faulted"].astype(bool)

    mapping_timestep = int(phase3.get("mapping_timestep", 0))
    if not 0 <= mapping_timestep < noise_std.shape[0]:
        raise ValueError("phase3.mapping_timestep is outside the Phase-2 trace.")

    sensitivity_floor = float(phase3.get("sensitivity_floor", 0.0))
    specs = projection_specs_from_phase1(
        phase1, sensitivity_floor=sensitivity_floor
    )
    shards = build_gpt2_shards(
        specs,
        tile_size=int(phase3["tier_rows"]),
        require_gpt2_small_contract=True,
    )
    slots = build_slots(
        noise_std[mapping_timestep],
        available[mapping_timestep],
        int(phase3["tiers_per_tile"]),
    )

    policies = ["random", "sequential", "hardware_only", "static_sensitivity"]
    filenames = phase3["placement_filenames"]
    assignments_by_policy = {}
    mapping_rows = []
    for policy in policies:
        policy_seed = int(seed) if policy == "random" else int(
            config["experiment"].get("placement_seed", seed)
        )
        assignments = assign_policy(
            policy,
            shards,
            slots,
            seed=policy_seed,
            mapping_timestep=mapping_timestep,
        )
        assignments_by_policy[policy] = assignments
        frame = pd.DataFrame([assignment.to_dict() for assignment in assignments])
        frame.sort_values(
            ["block_index", "projection_id", "out_start", "in_start"],
            inplace=True,
        )
        frame.to_csv(output_dir / str(filenames[policy]), index=False)

        metrics = evaluate_placement(
            assignments,
            noise_std[mapping_timestep],
            reference_noise_std=float(phase3["reference_noise_std"]),
            faulted=faulted[mapping_timestep],
            available=available[mapping_timestep],
        )
        mapping_rows.append(
            {"policy": policy, "timestep": mapping_timestep, **metrics}
        )

    mapping_frame = pd.DataFrame(mapping_rows)
    mapping_frame.to_csv(output_dir / "mapping_timestep_summary.csv", index=False)

    proxy_rows = []
    for timestep in range(noise_std.shape[0]):
        for policy, assignments in assignments_by_policy.items():
            proxy_rows.append(
                {
                    "timestep": timestep,
                    "policy": policy,
                    **evaluate_placement(
                        assignments,
                        noise_std[timestep],
                        reference_noise_std=float(phase3["reference_noise_std"]),
                        faulted=faulted[timestep],
                        available=available[timestep],
                    ),
                }
            )
    proxy_frame = pd.DataFrame(proxy_rows)
    proxy_frame.to_csv(output_dir / "placement_proxy_over_time.csv", index=False)

    at_mapping = mapping_frame.set_index("policy")[
        "sensitivity_weighted_variance_proxy"
    ]
    static_value = float(at_mapping["static_sensitivity"])
    tolerance = max(1e-12, abs(static_value) * 1e-10)
    for baseline in ("random", "sequential", "hardware_only"):
        if static_value > float(at_mapping[baseline]) + tolerance:
            raise AssertionError(
                "Shard-level static_sensitivity placement failed its construction "
                f"contract: {static_value} > {baseline}={at_mapping[baseline]}"
            )

    metadata = {
        "seed": int(seed),
        "mapping_timestep": mapping_timestep,
        "num_projections": len(specs),
        "num_shards": len(shards),
        "num_tiles": int(phase3["num_tiles"]),
        "tiers_per_tile": int(phase3["tiers_per_tile"]),
        "occupied_tiers": len(shards),
        "total_tiers": int(phase3["num_tiles"]) * int(phase3["tiers_per_tile"]),
        "mapping_method": {
            "random": "canonical shards paired with randomly permuted physical slots",
            "sequential": "canonical shards paired with canonical physical slots",
            "hardware_only": "canonical shards paired with ascending tile variance",
            "static_sensitivity": (
                "descending shard sensitivity*weight_fraction paired with ascending "
                "tile variance; rearrangement-optimal for the reported proxy"
            ),
        },
        "sensitivity_field": str(phase3["sensitivity_field"]),
        "sensitivity_unit": str(phase3["sensitivity_unit"]),
        "noise_unit": "normalized_std_fraction_of_programmed_projection_range",
        "phase1_results": str(phase1_results.resolve()),
        "phase2_trace": str(phase2_trace.resolve()),
        "phase2_metadata": None if phase2_metadata is None else str(phase2_metadata.resolve()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("Phase 3 placement completed.")
    print(f"Projections: {len(specs)} | Shards: {len(shards)}")
    print(mapping_frame.to_string(index=False))
    print(f"Results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "full_pipeline" / "gpt2_3dcim.yaml",
    )
    parser.add_argument("--phase1-results", type=Path, required=True)
    parser.add_argument("--phase2-trace", type=Path, required=True)
    parser.add_argument("--phase2-metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        args.config,
        args.phase1_results,
        args.phase2_trace,
        args.phase2_metadata,
        args.seed,
        args.output_dir,
        args.overwrite,
    )
