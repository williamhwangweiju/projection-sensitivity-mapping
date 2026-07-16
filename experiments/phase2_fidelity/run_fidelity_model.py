#!/usr/bin/env python3
"""Run Phase 2: generate a time-varying normalized tile-noise trace."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "src" / "simulators" / "tile_fidelity.py").is_file():
            return candidate
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_yaml, resolve_path
from src.simulators.tile_fidelity import generate_fidelity_trace


def main(config_path: Path, seed: int, output_dir: Path | None, overwrite: bool) -> Path:
    config = load_yaml(config_path)
    if output_dir is None:
        phase2 = config["phase2"]
        output_dir = (
            resolve_path(REPO_ROOT, phase2["output_root"])
            / str(phase2["name"])
            / f"seed_{seed}"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)

    print(f"Loading configuration from {config_path}")
    print(f"Generating trace with seed {seed}")
    trace = generate_fidelity_trace(config, seed)
    trace.save(output_dir)
    metadata = trace.metadata
    print("Phase 2 fidelity simulation completed.")
    print(
        f"Tiles: {metadata['num_tiles']} | Timesteps: {metadata['num_timesteps']}"
    )
    print(
        f"Mean noise: {metadata['initial_mean_noise_std']:.6f} -> "
        f"{metadata['final_mean_noise_std']:.6f} "
        f"({100.0 * metadata['mean_noise_increase_fraction']:.3f}%)"
    )
    print(
        f"Mean fidelity score: {metadata['initial_mean_fidelity_score']:.6f} -> "
        f"{metadata['final_mean_fidelity_score']:.6f}"
    )
    print(
        "Initial/final tile-rank correlation: "
        f"{metadata['initial_final_tile_rank_spearman']:.6f}"
    )
    print(
        f"Scheduled faults: {metadata['num_scheduled_faults']} | "
        f"Final faulted tiles: {metadata['num_final_faulted_tiles']}"
    )
    print(f"Final available tiles: {metadata['num_final_available_tiles']}")
    print(f"Results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "full_pipeline" / "gpt2_3dcim.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config, args.seed, args.output_dir, args.overwrite)
