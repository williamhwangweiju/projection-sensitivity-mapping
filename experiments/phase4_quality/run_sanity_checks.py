#!/usr/bin/env python3
"""Zero-noise and uniform-noise invariance checks for the all-analog hybrid."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import ManualAnalogSettings
from src.common.config import load_json, load_yaml, save_json
from src.common.dataset import build_causal_lm_batches
from src.common.model_loading import load_model_and_tokenizer
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel
from src.evaluation.hybrid_quality import evaluate_noisy_placement, evaluate_nominal_hybrid
from src.evaluation.noise_materialization import read_placement_csv


def with_uniform_noise(rows: list[dict[str, Any]], value: float) -> list[dict[str, Any]]:
    return [{**row, "tile_noise_std": float(value)} for row in rows]


def main(
    config_path: Path,
    phase1_path: Path,
    phase3_manifest_path: Path,
    output_path: Path,
) -> Path:
    config = load_yaml(config_path)
    phase1_profile = load_json(phase1_path)
    manifest = load_json(phase3_manifest_path)["placements"]
    paths = {str(row["policy"]): Path(row["placement_path"]) for row in manifest}
    policies = [str(value) for value in config["phase4"]["policies"]]
    missing = set(policies) - set(paths)
    if missing:
        raise FileNotFoundError(f"Missing placement policies: {sorted(missing)}")

    device = torch.device(str(config["model"]["device"]))
    model, tokenizer, _ = load_model_and_tokenizer(config, device=device)
    smoke_config = deepcopy(config)
    smoke_config["dataset"] = deepcopy(config.get("evaluation_dataset", config["dataset"]))
    # Keep sanity checks cheap even if the paper config uses the full corpus.
    smoke_config["dataset"]["max_tokens"] = min(
        int(smoke_config["dataset"].get("max_tokens") or 4096), 4096
    )
    batches, _ = build_causal_lm_batches(smoke_config, tokenizer)

    hybrid = HybridAnalogModel(
        model,
        digital_projection_ids=[],
        settings=ManualAnalogSettings.from_config(config),
        include_lm_head_candidate=bool(config["profiling"].get("include_lm_head", False)),
        phase1_projection_rows=phase1_profile["projections"],
    ).convert()
    try:
        nominal_nll, nominal_ppl, _ = evaluate_nominal_hybrid(hybrid, batches, device)
        zero_results: dict[str, float] = {}
        uniform_results: dict[str, float] = {}
        for policy in policies:
            rows = read_placement_csv(paths[policy])
            zero = evaluate_noisy_placement(
                hybrid,
                batches,
                device,
                with_uniform_noise(rows, 0.0),
                base_seed=int(config["experiment"]["seed"]),
                realization=0,
            )
            uniform = evaluate_noisy_placement(
                hybrid,
                batches,
                device,
                with_uniform_noise(rows, float(config["analog"]["reference_noise_std"])),
                base_seed=int(config["experiment"]["seed"]),
                realization=0,
            )
            zero_results[policy] = float(zero["nll"])
            uniform_results[policy] = float(uniform["nll"])
        zero_tolerance = float(config["phase4"].get("sanity_zero_nll_tolerance", 1e-6))
        uniform_tolerance = float(config["phase4"].get("sanity_uniform_nll_tolerance", 1e-6))
        zero_max_error = max(abs(value - nominal_nll) for value in zero_results.values())
        uniform_spread = max(uniform_results.values()) - min(uniform_results.values())
        if zero_max_error > zero_tolerance:
            raise RuntimeError(
                f"Zero-noise check failed: max NLL error {zero_max_error:.3e} > {zero_tolerance:.3e}"
            )
        if uniform_spread > uniform_tolerance:
            raise RuntimeError(
                f"Uniform-noise invariance failed: NLL spread {uniform_spread:.3e} > {uniform_tolerance:.3e}"
            )
        payload = {
            "nominal_nll": nominal_nll,
            "nominal_ppl": nominal_ppl,
            "zero_noise_nll_by_policy": zero_results,
            "zero_noise_max_error": zero_max_error,
            "uniform_noise_nll_by_policy": uniform_results,
            "uniform_noise_spread": uniform_spread,
            "passed": True,
        }
        save_json(output_path, payload)
        print(f"Hybrid quality sanity checks passed: {output_path}")
        return output_path
    finally:
        hybrid.restore_digital_modules()
        hybrid = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/results/phase4_hybrid_quality/sanity_checks.json")
    args = parser.parse_args()
    main(args.config, args.phase1, args.phase3_manifest, args.output)
