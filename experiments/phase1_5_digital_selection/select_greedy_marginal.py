#!/usr/bin/env python3
"""Automatically select digital GPT-2 projections using measured marginal NLL gain.

The search starts with no hard-coded digital projections unless the configuration
explicitly supplies a constraint. At every step it temporarily promotes each
remaining candidate to digital, measures the nominal hybrid NLL on calibration
data, and chooses the projection with the largest measured quality gain per unit
of digital cost. The produced sequence is a nested set of operating points that
can be evaluated on held-out data in Phases 4 and 5.

This is a greedy search rather than an exhaustive search: GPT-2 Small has 49
projection candidates when the tied LM head is included, so testing all 2^49
partitions is intractable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_json, load_yaml, save_json
from src.mapping.digital_selection import candidates_from_profile, operating_point_record
from src.mapping.sharding import count_projection_shards


def build_search_hybrid(
    model: Any,
    config: dict[str, Any],
    forced: set[str],
    phase1_rows: list[dict[str, Any]],
) -> Any:
    """Convert the full analog candidate set exactly once.

    Only the permanently-forced digital projections are excluded from
    conversion. Every greedy trial is then realized by temporarily swapping
    the candidate projections' original digital modules into the forward
    graph (O(1) module swaps per trial) instead of rebuilding and
    re-verifying the entire hybrid conversion for every candidate set.
    """
    from src.common.analog import ManualAnalogSettings
    from src.evaluation.aihwkit_gpt2 import HybridAnalogModel

    return HybridAnalogModel(
        model,
        digital_projection_ids=forced,
        settings=ManualAnalogSettings.from_config(config),
        include_lm_head_candidate=bool(config["profiling"].get("include_lm_head", False)),
        phase1_projection_rows=phase1_rows,
    ).convert()


def add_capacity(record: dict[str, Any], profile: dict[str, Any], config: dict[str, Any]) -> None:
    """Attach physical-tier feasibility to an operating point."""
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


def _candidate_pool(candidates: list[Any], pool_size: Any) -> list[Any]:
    """Return every candidate by default; optional truncation is a speed ablation."""
    ranked = sorted(candidates, key=lambda item: (-item.sensitivity, item.projection_id))
    if pool_size is None:
        return ranked
    value = int(pool_size)
    return ranked if value <= 0 else ranked[:value]




def _write_operating_points_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "digital_set_id",
        "selection_method",
        "budget_type",
        "budget_value",
        "promoted_projection_id",
        "digital_projection_count",
        "digital_parameter_fraction",
        "digital_incremental_storage_fraction",
        "digital_mac_fraction",
        "measured_nominal_nll",
        "measured_nominal_ppl",
        "delta_nll_nominal_vs_digital",
        "marginal_nll_gain",
        "marginal_gain_per_cost",
        "analog_shard_count",
        "available_physical_tiers",
        "capacity_feasible",
        "digital_projection_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: (
                        ";".join(record.get(field, []))
                        if field == "digital_projection_ids"
                        else record.get(field)
                    )
                    for field in fields
                }
            )


def _append_records(
    path: Path,
    records: list[dict[str, Any]],
    source: Path,
    recommendation: dict[str, Any],
) -> None:
    existing = load_json(path)
    combined = list(existing.get("operating_points", []))
    seen = {tuple(row["digital_projection_ids"]) for row in combined}
    for row in records:
        key = tuple(row["digital_projection_ids"])
        if key not in seen:
            combined.append(row)
            seen.add(key)
    combined.sort(
        key=lambda row: (
            float(row.get("digital_mac_fraction", 0.0)),
            float(row.get("digital_parameter_fraction", 0.0)),
            str(row["digital_set_id"]),
        )
    )
    existing["operating_points"] = combined
    existing["measured_greedy_source"] = str(source)
    existing["recommended_digital_set_id"] = recommendation["digital_set_id"]
    existing["recommended_digital_projection_ids"] = recommendation[
        "digital_projection_ids"
    ]
    existing["recommendation_reason"] = recommendation["recommendation_reason"]
    save_json(path, existing)
    _write_operating_points_csv(path.with_suffix(".csv"), combined)


def main(config_path: Path, phase1_path: Path, output_path: Path, append_to: Path | None) -> Path:
    # Heavy ML dependencies are imported lazily so --help and structural tests
    # remain usable in dependency-light environments.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.common.dataset import build_causal_lm_batches
    from src.common.metrics import evaluate_nll_ppl

    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    selection_cfg = config["digital_selection"]
    cfg = selection_cfg.get("greedy_marginal", {})

    # Empty by default: lm_head and every transformer projection are candidates.
    forced = set(str(value) for value in cfg.get("forced_digital", selection_cfg.get("forced_digital", [])))
    cost_field = str(cfg.get("cost_field", "macs_per_token"))
    if cost_field not in {"macs_per_token", "parameter_count"}:
        raise ValueError("greedy_marginal.cost_field must be macs_per_token or parameter_count")
    objective = str(cfg.get("objective", "gain_per_cost"))
    if objective not in {"gain_per_cost", "nll_gain"}:
        raise ValueError("greedy_marginal.objective must be gain_per_cost or nll_gain")

    max_promotions = int(cfg.get("max_promotions", 8))
    target_delta = cfg.get("target_delta_nll_vs_digital")
    target_delta = None if target_delta is None else float(target_delta)
    minimum_gain = float(cfg.get("minimum_marginal_nll_gain", 0.0))
    max_mac_fraction = float(cfg.get("max_digital_mac_fraction", 1.0))
    max_parameter_fraction = float(cfg.get("max_digital_parameter_fraction", 1.0))

    candidates = candidates_from_profile(profile)
    by_id = {candidate.projection_id: candidate for candidate in candidates}
    unknown = forced - set(by_id)
    if unknown:
        raise ValueError(f"Unknown forced digital projections: {sorted(unknown)}")
    pool = _candidate_pool(candidates, cfg.get("candidate_pool_size"))

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
    if "digital_selection_dataset" in config:
        calibration["dataset"] = deepcopy(config["digital_selection_dataset"])
    batches, dataset_metadata = build_causal_lm_batches(calibration, tokenizer)
    digital_nll, digital_ppl, predicted_tokens = evaluate_nll_ppl(model, batches, device)

    hybrid = build_search_hybrid(model, config, forced, profile["projections"])
    measured_cache: dict[frozenset[str], tuple[float, float]] = {}

    def measure_digital_set(digital: set[str]) -> tuple[float, float]:
        """Nominal hybrid NLL/PPL with the given projections executed digitally.

        A trial differs from the converted hybrid only in which modules sit in
        the forward graph, so each measurement costs one evaluation pass plus
        O(|digital|) module swaps. Nominal evaluation is deterministic for a
        fixed digital set, so results are memoized.
        """
        key = frozenset(digital)
        if key in measured_cache:
            return measured_cache[key]
        with hybrid.temporarily_digital(digital):
            nll, ppl, _ = evaluate_nll_ppl(model, batches, device)
        measured_cache[key] = (nll, ppl)
        return nll, ppl

    try:
        selected = set(forced)
        current_nll, current_ppl = measure_digital_set(selected)
        records: list[dict[str, Any]] = []

        def make_record(
            *,
            step: int,
            promoted_projection_id: str | None,
            gain: float,
            utility: float,
            nll: float,
            ppl: float,
            evaluated_candidates: int,
        ) -> dict[str, Any]:
            record = operating_point_record(
                candidates,
                method=f"greedy_measured_{objective}_per_{cost_field}",
                budget_type="greedy_step",
                budget_value=float(step),
                digital_projection_ids=selected,
            )
            record.update(
                {
                    "measured_nominal_nll": nll,
                    "measured_nominal_ppl": ppl,
                    "digital_reference_nll": digital_nll,
                    "digital_reference_ppl": digital_ppl,
                    "delta_nll_nominal_vs_digital": nll - digital_nll,
                    "marginal_nll_gain": gain,
                    "marginal_gain_per_cost": utility,
                    "promoted_projection_id": promoted_projection_id,
                    "evaluated_candidate_promotions": evaluated_candidates,
                    "selection_objective": objective,
                    "selection_cost_field": cost_field,
                }
            )
            add_capacity(record, profile, config)
            return record

        initial = make_record(
            step=0,
            promoted_projection_id=None,
            gain=0.0,
            utility=0.0,
            nll=current_nll,
            ppl=current_ppl,
            evaluated_candidates=0,
        )
        records.append(initial)
        print(
            "step=0 digital=[] "
            f"NLL={current_nll:.8f} delta_vs_digital={current_nll - digital_nll:.8f} "
            f"capacity_feasible={initial['capacity_feasible']}"
        )

        for step in range(1, max_promotions + 1):
            current_record = records[-1]
            target_met = (
                target_delta is not None
                and current_record["capacity_feasible"]
                and float(current_record["delta_nll_nominal_vs_digital"]) <= target_delta
            )
            if target_met:
                print(
                    f"Stopping: target delta NLL <= {target_delta:.8f} reached at step {step - 1}."
                )
                break

            trials: list[dict[str, Any]] = []
            for candidate in pool:
                if candidate.projection_id in selected:
                    continue
                trial_set = selected | {candidate.projection_id}
                cost_record = operating_point_record(
                    candidates,
                    method="trial",
                    budget_type="trial",
                    budget_value=float(step),
                    digital_projection_ids=trial_set,
                )
                if float(cost_record["digital_mac_fraction"]) > max_mac_fraction:
                    continue
                if float(cost_record["digital_parameter_fraction"]) > max_parameter_fraction:
                    continue

                nll, ppl = measure_digital_set(trial_set)
                gain = current_nll - nll
                cost = float(getattr(candidate, cost_field))
                utility = gain if objective == "nll_gain" else gain / max(cost, 1.0)
                trials.append(
                    {
                        "projection_id": candidate.projection_id,
                        "nll": nll,
                        "ppl": ppl,
                        "gain": gain,
                        "utility": utility,
                        "cost": cost,
                    }
                )

            if not trials:
                print("Stopping: no remaining candidate satisfies the configured digital budget.")
                break

            best = max(
                trials,
                key=lambda item: (
                    float(item["utility"]),
                    float(item["gain"]),
                    -float(item["cost"]),
                    str(item["projection_id"]),
                ),
            )
            if float(best["gain"]) <= minimum_gain:
                print(
                    "Stopping: best measured promotion does not exceed "
                    f"minimum_marginal_nll_gain={minimum_gain:.8f}."
                )
                break

            selected.add(str(best["projection_id"]))
            current_nll = float(best["nll"])
            current_ppl = float(best["ppl"])
            record = make_record(
                step=step,
                promoted_projection_id=str(best["projection_id"]),
                gain=float(best["gain"]),
                utility=float(best["utility"]),
                nll=current_nll,
                ppl=current_ppl,
                evaluated_candidates=len(trials),
            )
            records.append(record)
            print(
                f"step={step} promoted={best['projection_id']} "
                f"gain={best['gain']:.8f} NLL={current_nll:.8f} "
                f"delta_vs_digital={current_nll - digital_nll:.8f} "
                f"digital_mac_fraction={record['digital_mac_fraction']:.6f} "
                f"capacity_feasible={record['capacity_feasible']}"
            )

    finally:
        hybrid.restore_digital_modules()

    feasible_records = [row for row in records if bool(row["capacity_feasible"])]
    target_records = (
        []
        if target_delta is None
        else [
            row
            for row in feasible_records
            if float(row["delta_nll_nominal_vs_digital"]) <= target_delta
        ]
    )
    if target_records:
        recommendation = min(
            target_records,
            key=lambda row: (
                float(row["digital_mac_fraction"]),
                float(row["digital_parameter_fraction"]),
            ),
        )
        recommendation_reason = "minimum digital cost meeting calibration delta-NLL target"
    elif feasible_records:
        recommendation = min(
            feasible_records,
            key=lambda row: (
                float(row["delta_nll_nominal_vs_digital"]),
                float(row["digital_mac_fraction"]),
            ),
        )
        recommendation_reason = "lowest measured calibration delta NLL among feasible searched points"
    else:
        recommendation = records[-1]
        recommendation_reason = "no searched point was capacity feasible"
    recommendation = {
        **recommendation,
        "recommendation_reason": recommendation_reason,
    }

    payload: dict[str, Any] = {
        "phase1_path": str(phase1_path),
        "dataset": dataset_metadata,
        "predicted_tokens": predicted_tokens,
        "digital_reference_nll": digital_nll,
        "digital_reference_ppl": digital_ppl,
        "selection_method": f"greedy_measured_{objective}_per_{cost_field}",
        "forced_digital_projection_ids": sorted(forced),
        "candidate_projection_ids": sorted(by_id),
        "candidate_pool_projection_ids": [candidate.projection_id for candidate in pool],
        "operating_points": records,
        "recommended_operating_point": recommendation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    _write_operating_points_csv(output_path.with_suffix(".csv"), records)
    if append_to is not None and append_to.is_file():
        _append_records(append_to, records, output_path, recommendation)
        print(f"Appended measured greedy points to: {append_to}")
    print(f"Measured greedy selection saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/results/phase1_5_digital_selection/greedy_marginal_points.json",
    )
    parser.add_argument("--append-to", type=Path)
    args = parser.parse_args()
    main(args.config, args.phase1, args.output, args.append_to)
