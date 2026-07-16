#!/usr/bin/env python3
"""Optional measured greedy digital promotion using nominal hybrid NLL gain/cost."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
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
from src.evaluation.hybrid_quality import evaluate_nominal_hybrid
from src.mapping.digital_selection import candidates_from_profile, operating_point_record
from src.mapping.sharding import count_projection_shards


def evaluate_set(
    model: Any,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    config: dict[str, Any],
    digital: set[str],
    phase1_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    hybrid = HybridAnalogModel(
        model,
        digital_projection_ids=digital,
        settings=ManualAnalogSettings.from_config(config),
        include_lm_head_candidate=bool(config["profiling"].get("include_lm_head", False)),
        phase1_projection_rows=phase1_rows,
    ).convert()
    try:
        nll, ppl, _ = evaluate_nominal_hybrid(hybrid, batches, device)
        return nll, ppl
    finally:
        hybrid.restore_digital_modules()


def add_capacity(record: dict[str, Any], profile: dict[str, Any], config: dict[str, Any]) -> None:
    by_id = {str(row["projection_id"]): row for row in profile["projections"]}
    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    count = sum(
        count_projection_shards(
            projection_id,
            int(by_id[projection_id]["out_features"]),
            int(by_id[projection_id]["in_features"]),
            tier_rows,
            tier_cols,
        )
        for projection_id in record["analog_projection_ids"]
    )
    slots = int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    record["analog_shard_count"] = count
    record["available_physical_tiers"] = slots
    record["capacity_feasible"] = count <= slots


def main(config_path: Path, phase1_path: Path, output_path: Path, append_to: Path | None) -> Path:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    cfg = config["digital_selection"].get("greedy_marginal", {})
    forced = set(str(value) for value in cfg.get("forced_digital", config["digital_selection"].get("forced_digital", [])))
    cost_field = str(cfg.get("cost_field", "macs_per_token"))
    if cost_field not in {"macs_per_token", "parameter_count"}:
        raise ValueError("greedy_marginal.cost_field must be macs_per_token or parameter_count")
    max_promotions = int(cfg.get("max_promotions", 4))
    pool_size = int(cfg.get("candidate_pool_size", 12))
    candidates = candidates_from_profile(profile)
    by_id = {candidate.projection_id: candidate for candidate in candidates}
    unknown = forced - set(by_id)
    if unknown:
        raise ValueError(f"Unknown forced digital projections: {sorted(unknown)}")
    pool = sorted(candidates, key=lambda item: (-item.sensitivity, item.projection_id))[:pool_size]

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
    calibration = deepcopy(config)
    batches, dataset_metadata = build_causal_lm_batches(calibration, tokenizer)

    selected = set(forced)
    current_nll, current_ppl = evaluate_set(model, batches, device, config, selected, profile["projections"])
    records: list[dict[str, Any]] = []
    base = operating_point_record(
        candidates,
        method=f"greedy_marginal_per_{cost_field}",
        budget_type="greedy_step",
        budget_value=0.0,
        digital_projection_ids=selected,
    )
    base.update({"measured_nominal_nll": current_nll, "measured_nominal_ppl": current_ppl, "marginal_nll_gain": 0.0, "promoted_projection_id": None})
    add_capacity(base, profile, config)
    records.append(base)

    for step in range(1, max_promotions + 1):
        trials: list[tuple[float, float, str, float, float]] = []
        for candidate in pool:
            if candidate.projection_id in selected:
                continue
            trial_set = selected | {candidate.projection_id}
            nll, ppl = evaluate_set(model, batches, device, config, trial_set, profile["projections"])
            gain = current_nll - nll
            cost = float(getattr(candidate, cost_field))
            utility = gain / max(cost, 1.0)
            trials.append((utility, gain, candidate.projection_id, nll, ppl))
        if not trials:
            break
        utility, gain, projection_id, next_nll, next_ppl = max(trials, key=lambda item: (item[0], item[1], item[2]))
        selected.add(projection_id)
        current_nll, current_ppl = next_nll, next_ppl
        record = operating_point_record(
            candidates,
            method=f"greedy_marginal_per_{cost_field}",
            budget_type="greedy_step",
            budget_value=float(step),
            digital_projection_ids=selected,
        )
        record.update({
            "measured_nominal_nll": current_nll,
            "measured_nominal_ppl": current_ppl,
            "marginal_nll_gain": gain,
            "marginal_gain_per_cost": utility,
            "promoted_projection_id": projection_id,
        })
        add_capacity(record, profile, config)
        records.append(record)
        print(f"step={step} promoted={projection_id} gain={gain:.8f} NLL={current_nll:.8f}")

    payload: dict[str, Any] = {
        "phase1_path": str(phase1_path),
        "dataset": dataset_metadata,
        "selection_method": f"greedy_marginal_per_{cost_field}",
        "operating_points": records,
    }
    if append_to is not None and append_to.is_file():
        existing = load_json(append_to)
        combined = list(existing["operating_points"])
        seen = {tuple(row["digital_projection_ids"]) for row in combined}
        combined.extend(row for row in records if tuple(row["digital_projection_ids"]) not in seen)
        payload = {**existing, "operating_points": combined, "greedy_marginal_source": str(output_path)}
        save_json(append_to, payload)
        print(f"Appended measured greedy points to: {append_to}")
    save_json(output_path, {
        "phase1_path": str(phase1_path),
        "dataset": dataset_metadata,
        "selection_method": f"greedy_marginal_per_{cost_field}",
        "operating_points": records,
    })
    print(f"Measured greedy selection saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/results/phase1_5_digital_selection/greedy_marginal_points.json")
    parser.add_argument("--append-to", type=Path)
    args = parser.parse_args()
    main(args.config, args.phase1, args.output, args.append_to)
