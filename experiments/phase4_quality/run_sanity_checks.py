#!/usr/bin/env python3
"""End-to-end zero-noise and uniform-noise invariance checks for one digital set."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import ManualAnalogSettings
from src.common.config import load_json, load_yaml, save_json
from src.common.dataset import build_causal_lm_batches
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel
from src.evaluation.hybrid_quality import evaluate_noisy_placement, evaluate_nominal_hybrid
from src.evaluation.noise_materialization import read_placement_csv


def with_uniform_noise(rows: list[dict[str, Any]], value: float) -> list[dict[str, Any]]:
    return [{**row, "tile_noise_std": float(value)} for row in rows]


def main(
    config_path: Path,
    phase1_path: Path,
    operating_points_path: Path,
    phase3_manifest_path: Path,
    digital_set_id: str,
    output_path: Path,
) -> Path:
    config = load_yaml(config_path)
    phase1_profile = load_json(phase1_path)
    points = {
        point["digital_set_id"]: point
        for point in load_json(operating_points_path)["operating_points"]
    }
    if digital_set_id not in points:
        raise KeyError(f"Unknown digital_set_id: {digital_set_id}")
    point = points[digital_set_id]
    manifest = load_json(phase3_manifest_path)["placements"]
    paths = {
        row["policy"]: Path(row["placement_path"])
        for row in manifest
        if row["digital_set_id"] == digital_set_id
    }
    policies = [str(value) for value in config["phase4"]["policies"]]
    missing = set(policies) - set(paths)
    if missing:
        raise FileNotFoundError(f"Missing placement policies: {sorted(missing)}")

    model_name = str(config["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    device = torch.device(str(config["model"]["device"]))
    model.to(device).eval()
    smoke_config = deepcopy(config)
    smoke_config["dataset"] = deepcopy(config.get("evaluation_dataset", config["dataset"]))
    # Keep sanity checks cheap even if the paper config uses the full corpus.
    smoke_config["dataset"]["max_tokens"] = min(
        int(smoke_config["dataset"].get("max_tokens") or 4096), 4096
    )
    batches, _ = build_causal_lm_batches(smoke_config, tokenizer)

    hybrid = HybridAnalogModel(
        model,
        digital_projection_ids=point["digital_projection_ids"],
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
            "digital_set_id": digital_set_id,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    parser.add_argument("--digital-set-id", required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/results/phase4_hybrid_quality/sanity_checks.json")
    args = parser.parse_args()
    main(args.config, args.phase1, args.operating_points, args.phase3_manifest, args.digital_set_id, args.output)
