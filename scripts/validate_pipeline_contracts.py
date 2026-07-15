#!/usr/bin/env python3
"""Validate Phase 1-3 artifacts before Phase 4 quality evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.simulators.tile_fidelity import TileFidelityTrace

POLICIES = ("random", "sequential", "hardware_only", "static_sensitivity")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON mapping.")
    return dict(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1-results", type=Path, required=True)
    parser.add_argument("--phase2-trace", type=Path, required=True)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text()) or {}
    phase1 = load_json(args.phase1_results)
    results = phase1.get("results", {})
    projections = results.get("projections", []) if isinstance(results, Mapping) else []
    if not isinstance(projections, list) or len(projections) != 48:
        raise ValueError("Phase 1 must contain exactly 48 projection sensitivity rows.")
    for row in projections:
        if row.get("sensitivity_score_unit") != "delta_ppl_total":
            raise ValueError("Phase-1 mapping sensitivity must use delta_ppl_total.")
        if "sensitivity_score_for_mapping" not in row:
            raise ValueError("A Phase-1 row lacks sensitivity_score_for_mapping.")

    calibrations = phase1.get("projection_noise_calibration")
    if not isinstance(calibrations, list) or len(calibrations) != 48:
        raise ValueError(
            "Phase 1 must contain exactly 48 projection_noise_calibration records."
        )
    paths = {str(row["hf_module_path"]) for row in calibrations}
    if len(paths) != 48:
        raise ValueError("Phase-1 calibration module paths are not unique.")
    profiling_cfg = config.get("profiling", {})
    fidelity_cfg = config.get("fidelity_model", {})
    phase3_cfg = config.get("phase3", {})

    expected_reference_sigma = float(
        profiling_cfg.get("programming_noise_scale", 0.023)
    )
    phase2_reference_sigma = float(
        fidelity_cfg.get("reference_noise_std", expected_reference_sigma)
    )
    phase3_reference_sigma = float(
        phase3_cfg.get("reference_noise_std", expected_reference_sigma)
    )

    reference_sigmas = []
    for row in calibrations:
        reference_sigma = float(row["reference_sigma_normalized"])
        reference_sigmas.append(reference_sigma)
        if not math.isclose(
            reference_sigma,
            expected_reference_sigma,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            projection_id = row.get("projection_id", "<unknown>")
            raise ValueError(
                "Phase-1 calibration reference sigma mismatch for "
                f"{projection_id}: found {reference_sigma!r}, expected "
                f"{expected_reference_sigma!r} from "
                "profiling.programming_noise_scale."
            )
        if float(row["noise_reference_scale"]) <= 0.0:
            raise ValueError("Every Phase-1 noise_reference_scale must be positive.")

    phase4_aihwkit = config.get("phase4", {}).get("aihwkit", {})
    if phase4_aihwkit.get("forward_backend") != "aihwkit":
        raise ValueError("Phase 4 must use the AIHWKit forward backend.")
    if not bool(phase4_aihwkit.get("all_transformer_projections_analog", False)):
        raise ValueError("Phase 4 must make all transformer projections analog.")
    if bool(phase4_aihwkit.get("internal_programming_noise", False)):
        raise ValueError("Phase 4 internal programming noise must remain disabled.")

    trace = TileFidelityTrace.load_npz(args.phase2_trace)
    hardware = config["hardware"]
    num_tiles = int(hardware["num_tiles"])
    tiers_per_tile = int(hardware["tiers_per_tile"])
    if trace.noise_std.shape[1] != num_tiles:
        raise ValueError("Phase-2 tile count does not match unified config.")
    if str(trace.metadata.get("noise_unit", "")) != (
        "pcmlike_prog_noise_scale_equivalent"
    ):
        raise ValueError("Phase-2 trace has the wrong noise unit.")
    if list(map(int, trace.tile_ids)) != list(range(num_tiles)):
        raise ValueError("Phase-2 tile IDs must be contiguous 0..num_tiles-1.")

    expected_projection_ids = {
        f"block_{block}/{projection}"
        for block in range(12)
        for projection in (
            "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"
        )
    }
    logical_signatures: set[tuple[tuple[str, str], ...]] = set()
    for policy in POLICIES:
        path = args.phase3_dir / f"placement_{policy}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 480:
            raise ValueError(f"{path} must contain 480 tier rows, got {len(rows)}.")
        projection_ids = {row["projection_id"] for row in rows}
        if projection_ids != expected_projection_ids:
            missing = expected_projection_ids - projection_ids
            extra = projection_ids - expected_projection_ids
            raise ValueError(f"{path}: missing={sorted(missing)}, extra={sorted(extra)}")
        occupied: set[tuple[int, int]] = set()
        signature = []
        for row in rows:
            tile_id = int(row["tile_id"])
            tier_id = int(row["tier_start"])
            if not 0 <= tile_id < num_tiles:
                raise ValueError(f"{path}: invalid tile_id {tile_id}.")
            if not 0 <= tier_id < tiers_per_tile:
                raise ValueError(f"{path}: invalid tier_id {tier_id}.")
            slot = (tile_id, tier_id)
            if slot in occupied:
                raise ValueError(f"{path}: duplicate physical slot {slot}.")
            occupied.add(slot)
            if row.get("sensitivity_score_unit") != "delta_ppl_total":
                raise ValueError(f"{path}: wrong sensitivity score unit.")
            signature.append((row["shard_id"], row["projection_id"]))
        logical_signatures.add(tuple(sorted(signature)))

    if len(logical_signatures) != 1:
        raise ValueError("Phase-3 policies do not contain the same logical shards.")

    total_capacity = num_tiles * tiers_per_tile
    print("Pipeline contracts validated")
    print("  Phase-1 sensitivities: 48 total-DeltaPPL rows")
    print("  Phase-1 calibrations: 48")
    print(f"  Reference programming-noise scale: {expected_reference_sigma:.12g}")
    print(f"  Phase-2 trace: {trace.num_timesteps} x {trace.num_tiles}")
    print(f"  Phase-3 projection tiers: 480 / {total_capacity}")
    print(f"  Nominal utilization: {480 / total_capacity:.4%}")


if __name__ == "__main__":
    main()