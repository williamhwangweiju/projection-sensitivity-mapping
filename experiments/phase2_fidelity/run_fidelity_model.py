#!/usr/bin/env python3
"""Generate a Phase 2 tile-fidelity trace in the shared normalized noise unit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_yaml, resolve_path, save_json
from src.simulators.hardware import HardwareConfig
from src.simulators.tile_fidelity import TileFidelityModel, save_trace


def main(config_path: Path, seed: int | None = None) -> Path:
    config = load_yaml(config_path)
    hardware = HardwareConfig.from_config(config)
    hardware.validate()
    seed = int(config["experiment"]["seed"] if seed is None else seed)
    trace = TileFidelityModel(hardware, config["phase2"], seed).generate()
    phase = config["phase2"]
    output_dir = resolve_path(phase["output_root"]) / str(phase["name"]) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.npz"
    save_trace(str(trace_path), trace)
    save_json(
        output_dir / "metadata.json",
        {
            "seed": seed,
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "noise_unit": "logical_weight_std/programmed_projection_range",
            "num_tiles": hardware.num_tiles,
            "tiers_per_tile": hardware.tiers_per_tile,
            "num_timesteps": int(trace.noise_std.shape[0]),
            "mean_noise_initial": float(trace.noise_std[0].mean()),
            "mean_noise_final": float(trace.noise_std[-1].mean()),
            "final_faulted_tiles": int(trace.faulted[-1].sum()),
        },
    )
    with (output_dir / "timestep_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestep", "mean_noise_std", "mean_fidelity", "faulted_tiles", "available_tiles"])
        writer.writeheader()
        for t in range(trace.noise_std.shape[0]):
            writer.writerow({
                "timestep": t,
                "mean_noise_std": float(trace.noise_std[t].mean()),
                "mean_fidelity": float(trace.fidelity_score[t].mean()),
                "faulted_tiles": int(trace.faulted[t].sum()),
                "available_tiles": int(trace.available[t].sum()),
            })
    print(f"Phase 2 trace saved to: {trace_path}")
    return trace_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    main(args.config, args.seed)
