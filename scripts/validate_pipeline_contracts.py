#!/usr/bin/env python3
"""Validate cross-phase units, projection metadata and physical placements."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def fail(message: str) -> None:
    raise AssertionError(message)


def main(
    config_path: Path,
    phase1_results: Path,
    phase2_trace: Path,
    phase3_dir: Path,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase1 = json.loads(phase1_results.read_text(encoding="utf-8"))
    with np.load(phase2_trace, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}

    analog_reference = float(config["analog"]["reference_noise_std"])
    phase2_reference = float(config["fidelity_model"]["reference_noise_std"])
    phase3_reference = float(config["phase3"]["reference_noise_std"])
    if not (
        math.isclose(analog_reference, phase2_reference, abs_tol=1e-12)
        and math.isclose(analog_reference, phase3_reference, abs_tol=1e-12)
    ):
        fail("Reference noise standard deviations differ across phases.")

    analog_metadata = phase1["metadata"]["analog_configuration"]
    expected = {
        "clipping": "manual_projection_wide_symmetric_gaussian",
        "clip_order": "clip_once_before_noise",
        "reclip_after_noise": False,
        "noise_unit": "normalized_std_fraction_of_programmed_projection_range",
        "forward_is_perfect": False,
        "internal_clipping": False,
        "internal_programming_noise": False,
        "internal_read_noise": False,
        "internal_drift": False,
    }
    for key, value in expected.items():
        if analog_metadata.get(key) != value:
            fail(f"Phase-1 analog contract mismatch for {key}: {analog_metadata.get(key)!r}")
    if float(analog_metadata["clip_sigma"]) != float(config["analog"]["clip_sigma"]):
        fail("Phase-1 clip_sigma does not match current config.")
    if str(analog_metadata["programmed_range_mode"]) != str(
        config["analog"]["range_mode"]
    ):
        fail("Phase-1 range mode does not match current config.")

    projections = phase1["results"]["projections"]
    if len(projections) != 48:
        fail(f"Expected 48 Phase-1 projections, found {len(projections)}.")
    projection_shapes = {}
    for record in projections:
        preprocessing = record.get("preprocessing", {})
        for field in (
            "original_checksum",
            "clipped_checksum",
            "clip_threshold",
            "programmed_range",
        ):
            if field not in preprocessing:
                fail(f"{record['projection_id']} is missing preprocessing.{field}.")
        if float(preprocessing["programmed_range"]) <= 0:
            fail(f"{record['projection_id']} has invalid programmed range.")
        projection_shapes[str(record["projection_id"])] = tuple(
            int(value) for value in record["weight_shape_out_in"]
        )

    if trace["noise_std"].ndim != 2:
        fail("Phase-2 noise_std is not [timesteps, tiles].")
    if trace["noise_std"].shape[1] != int(config["hardware"]["num_tiles"]):
        fail("Phase-2 tile count does not match hardware.num_tiles.")

    filenames = config["phase3"]["placement_filenames"]
    policies = ["random", "sequential", "hardware_only", "static_sensitivity"]
    for policy in policies:
        path = phase3_dir / str(filenames[policy])
        if not path.is_file():
            fail(f"Missing placement: {path}")
        frame = pd.read_csv(path)
        if len(frame) != 480:
            fail(f"{policy} placement has {len(frame)} rows; expected 480.")
        if frame["shard_id"].nunique() != 480:
            fail(f"{policy} placement does not contain 480 unique shards.")
        if frame[["tile_id", "tier_id"]].duplicated().any():
            fail(f"{policy} placement reuses a physical tier.")
        for projection_id, group in frame.groupby("projection_id"):
            if projection_id not in projection_shapes:
                fail(f"Unknown projection in placement: {projection_id}")
            shape = projection_shapes[projection_id]
            coverage = np.zeros(shape, dtype=np.int8)
            for row in group.itertuples(index=False):
                coverage[
                    int(row.out_start) : int(row.out_end),
                    int(row.in_start) : int(row.in_end),
                ] += 1
            if not np.all(coverage == 1):
                fail(f"{policy}/{projection_id} does not cover every weight once.")

    mapping_summary = pd.read_csv(phase3_dir / "mapping_timestep_summary.csv")
    values = mapping_summary.set_index("policy")[
        "sensitivity_weighted_variance_proxy"
    ]
    static = float(values["static_sensitivity"])
    for baseline in ("random", "sequential", "hardware_only"):
        if static > float(values[baseline]) + max(1e-12, abs(static) * 1e-10):
            fail(f"Static sensitivity proxy exceeds {baseline} at mapping time.")

    print("Pipeline contracts validated successfully.")
    print("  48 projections")
    print("  480 unique physical shards per policy")
    print("  identical manual clipping/noise units across Phases 1-4")
    print("  static sensitivity placement is proxy-optimal at mapping timestep")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1-results", type=Path, required=True)
    parser.add_argument("--phase2-trace", type=Path, required=True)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.phase1_results, args.phase2_trace, args.phase3_dir)
