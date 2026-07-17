#!/usr/bin/env python3
"""Evaluate hybrid digital/analog GPT-2 quality for static placement policies."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import ManualAnalogSettings, analog_configuration, set_seed
from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.dataset import build_causal_lm_batches
from src.common.metrics import evaluate_nll_ppl, summarize
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel
from src.evaluation.hybrid_quality import (
    evaluate_noisy_placement,
    evaluate_nominal_hybrid,
    write_csv,
)
from src.evaluation.noise_materialization import read_placement_csv, update_placement_noise
from src.mapping.objective import placement_proxy
from src.mapping.placement import PlacementRecord
from src.simulators.tile_fidelity import load_trace


def representative_timesteps(trace: Any, requested: Iterable[int] | None) -> list[int]:
    total = int(trace.noise_std.shape[0])
    if requested:
        result = sorted({max(0, min(int(value), total - 1)) for value in requested})
        if result:
            return result
    values = {0, total // 2, total - 1}
    onsets = sorted(int(value) for value in trace.fault_onset if int(value) >= 0)
    if onsets:
        first = onsets[0]
        values.add(max(0, first - 1))
        values.add(min(total - 1, first))
    return sorted(values)


def select_points(points: list[dict[str, Any]], cfg: dict[str, Any], requested_ids: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    budget_types = set(str(value) for value in cfg.get("evaluate_budget_types", []))
    methods = set(str(value) for value in cfg.get("evaluate_selection_methods", []))
    for point in points:
        if not bool(point.get("capacity_feasible", True)):
            continue
        if requested_ids and point["digital_set_id"] not in requested_ids:
            continue
        if budget_types and str(point["budget_type"]) not in budget_types:
            continue
        method = str(point["selection_method"])
        if methods and method not in methods:
            continue
        selected.append(point)
    # Deterministic budget order: fewest digital projections first, so the
    # subset below always spans the frontier from cheapest to final point.
    selected.sort(
        key=lambda point: (
            int(point["digital_projection_count"]),
            float(point["digital_mac_fraction"]),
            float(point["digital_parameter_fraction"]),
            str(point["digital_set_id"]),
        )
    )
    maximum = cfg.get("max_operating_points")
    if maximum is not None and 0 < int(maximum) < len(selected):
        indices = np.linspace(0, len(selected) - 1, int(maximum))
        unique_indices = sorted({int(round(value)) for value in indices})
        selected = [selected[index] for index in unique_indices]
    if not selected:
        raise ValueError("No digital operating points matched the Phase-4 filters.")
    return selected


def _records_for_proxy(rows: list[dict[str, Any]]) -> list[PlacementRecord]:
    fields = PlacementRecord.__dataclass_fields__
    return [PlacementRecord(**{key: row[key] for key in fields}) for row in rows]


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 4000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["digital_set_id"]), str(row["policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for (digital_set_id, policy), group in sorted(groups.items()):
        delta_total = [float(row["delta_nll_total"]) for row in group]
        delta_tile = [float(row["delta_nll_tile"]) for row in group]
        total_summary = summarize(delta_total)
        tile_summary = summarize(delta_tile)
        lo, hi = _bootstrap_mean_ci(delta_tile, seed)
        first = group[0]
        output.append({
            "digital_set_id": digital_set_id,
            "selection_method": first["selection_method"],
            "digital_projection_count": first["digital_projection_count"],
            "digital_parameter_fraction": first["digital_parameter_fraction"],
            "digital_incremental_storage_fraction": first.get("digital_incremental_storage_fraction", first["digital_parameter_fraction"]),
            "digital_mac_fraction": first["digital_mac_fraction"],
            "policy": policy,
            "evaluations": len(group),
            "mean_delta_nll_total": total_summary["mean"],
            "std_delta_nll_total": total_summary["std"],
            "mean_delta_nll_tile": tile_summary["mean"],
            "std_delta_nll_tile": tile_summary["std"],
            "bootstrap_ci95_delta_nll_tile_low": lo,
            "bootstrap_ci95_delta_nll_tile_high": hi,
            "mean_ppl": float(np.mean([float(row["ppl_from_mean_nll"]) for row in group])),
            "mean_proxy_variance": float(np.mean([float(row["proxy_variance"]) for row in group])),
        })
    return output


def paired_differences(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["digital_set_id"]), int(row["timestep"]), int(row["realization"]))
        keyed[key][str(row["policy"])] = row
    comparisons = (
        ("hardware_only", "static_sensitivity"),
        ("sequential", "static_sensitivity"),
        ("random", "static_sensitivity"),
    )
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    wins: dict[tuple[str, str, str], int] = defaultdict(int)
    for (digital_set_id, _, _), policy_rows in keyed.items():
        for baseline, method in comparisons:
            if baseline not in policy_rows or method not in policy_rows:
                continue
            difference = float(policy_rows[baseline]["delta_nll_tile"]) - float(policy_rows[method]["delta_nll_tile"])
            key = (digital_set_id, baseline, method)
            grouped[key].append(difference)
            wins[key] += int(difference > 0)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        digital_set_id, baseline, method = key
        lo, hi = _bootstrap_mean_ci(values, seed)
        output.append({
            "digital_set_id": digital_set_id,
            "baseline_policy": baseline,
            "method_policy": method,
            "paired_samples": len(values),
            "mean_nll_improvement": float(np.mean(values)),
            "median_nll_improvement": float(np.median(values)),
            "std_nll_improvement": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "win_fraction": wins[key] / len(values),
        })
    return output


def main(
    config_path: Path,
    phase1_path: Path,
    operating_points_path: Path,
    trace_path: Path,
    phase3_manifest_path: Path,
    requested_ids: set[str] | None = None,
) -> Path:
    config = load_yaml(config_path)
    cfg = config["phase4"]
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    profile = load_json(phase1_path)
    operating_points = load_json(operating_points_path)["operating_points"]
    points = select_points(operating_points, cfg, requested_ids or set())
    manifest_payload = load_json(phase3_manifest_path)
    manifest = manifest_payload["placements"]
    available_point_ids = {str(row["digital_set_id"]) for row in manifest}
    points = [point for point in points if str(point["digital_set_id"]) in available_point_ids]
    if not points:
        raise ValueError("All selected Phase-4 operating points were capacity-infeasible or lacked placements.")

    # Every selected operating point is evaluated. Within each point the four
    # policies compare placements for one identical analog projection set, and
    # phase4.max_operating_points controls how many points span the
    # quality-versus-budget frontier (evenly spaced, endpoints included).
    print(f"Phase 4 evaluating {len(points)} digital operating point(s):")
    for point in points:
        print(
            f"  {point['digital_set_id']} | "
            f"digital_projections={point['digital_projection_count']} | "
            f"digital_mac_fraction={point['digital_mac_fraction']:.6f}"
        )

    points_by_id = {point["digital_set_id"]: point for point in points}
    placement_paths = {
        (row["digital_set_id"], row["policy"]): Path(row["placement_path"])
        for row in manifest
        if row["digital_set_id"] in points_by_id
    }
    # Placements are static within Phase 4. Parse each CSV once rather than on
    # every timestep/realization, which materially reduces overhead in long runs.
    placement_cache = {
        key: read_placement_csv(path) for key, path in placement_paths.items()
    }
    trace = load_trace(str(trace_path))
    timesteps = representative_timesteps(trace, cfg.get("timesteps"))
    policies = [str(value) for value in cfg["policies"]]
    realizations = int(cfg["num_realizations"])
    antithetic = bool(cfg.get("antithetic", False))
    unavailable_noise_std = float(cfg.get("unavailable_noise_std", config["phase2"]["fidelity_model"]["max_noise_std"]))

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

    evaluation_config = deepcopy(config)
    evaluation_config["dataset"] = deepcopy(config.get("evaluation_dataset", config["dataset"]))
    batches, dataset_metadata = build_causal_lm_batches(evaluation_config, tokenizer)
    digital_nll, digital_ppl, token_count = evaluate_nll_ppl(model, batches, device)

    settings = ManualAnalogSettings.from_config(config)
    settings.validate()
    all_rows: list[dict[str, Any]] = []
    nominal_rows: list[dict[str, Any]] = []
    for point in points:
        digital_set_id = str(point["digital_set_id"])
        hybrid = HybridAnalogModel(
            model,
            digital_projection_ids=point["digital_projection_ids"],
            settings=settings,
            include_lm_head_candidate=bool(config["profiling"].get("include_lm_head", False)),
            phase1_projection_rows=profile["projections"],
        ).convert()
        try:
            nominal_nll, nominal_ppl, _ = evaluate_nominal_hybrid(hybrid, batches, device)
            nominal_rows.append({
                "digital_set_id": digital_set_id,
                "selection_method": point["selection_method"],
                "digital_projection_ids": ";".join(point["digital_projection_ids"]),
                "digital_projection_count": point["digital_projection_count"],
                "digital_parameter_fraction": point["digital_parameter_fraction"],
                "digital_incremental_storage_fraction": point.get("digital_incremental_storage_fraction", point["digital_parameter_fraction"]),
                "digital_mac_fraction": point["digital_mac_fraction"],
                "digital_nll": digital_nll,
                "digital_ppl": digital_ppl,
                "nominal_hybrid_nll": nominal_nll,
                "nominal_hybrid_ppl": nominal_ppl,
                "delta_nll_nominal_vs_digital": nominal_nll - digital_nll,
                "delta_ppl_nominal_vs_digital": nominal_ppl - digital_ppl,
                "analog_projection_count": len(hybrid.analog_projection_ids),
            })
            for timestep in timesteps:
                current_noise = np.asarray(trace.noise_std[timestep], dtype=np.float64).copy()
                unavailable = ~np.asarray(trace.available[timestep], dtype=bool)
                current_noise[unavailable] = unavailable_noise_std
                for realization in range(realizations):
                    for policy in policies:
                        static_rows = placement_cache.get((digital_set_id, policy))
                        if static_rows is None:
                            raise FileNotFoundError(
                                f"Missing Phase-3 placement for {digital_set_id}/{policy}."
                            )
                        current_rows = update_placement_noise(static_rows, current_noise, timestep)
                        result = evaluate_noisy_placement(
                            hybrid,
                            batches,
                            device,
                            current_rows,
                            base_seed=seed,
                            realization=realization,
                            antithetic=antithetic,
                        )
                        unavailable_shards = sum(
                            int(not bool(trace.available[timestep, int(row["tile_id"])]))
                            for row in current_rows
                        )
                        faulted_shards = sum(
                            int(bool(trace.faulted[timestep, int(row["tile_id"])]))
                            for row in current_rows
                        )
                        proxy = placement_proxy(_records_for_proxy(current_rows), variance=True)
                        row = {
                            "digital_set_id": digital_set_id,
                            "selection_method": point["selection_method"],
                            "budget_type": point["budget_type"],
                            "budget_value": point["budget_value"],
                            "digital_projection_count": point["digital_projection_count"],
                            "digital_parameter_fraction": point["digital_parameter_fraction"],
                            "digital_incremental_storage_fraction": point.get("digital_incremental_storage_fraction", point["digital_parameter_fraction"]),
                            "digital_mac_fraction": point["digital_mac_fraction"],
                            "policy": policy,
                            "timestep": timestep,
                            "realization": realization,
                            "nll": result["nll"],
                            "ppl_from_mean_nll": result["ppl_from_mean_nll"],
                            "ppl_mean": result["ppl_mean"],
                            "digital_nll": digital_nll,
                            "digital_ppl": digital_ppl,
                            "nominal_hybrid_nll": nominal_nll,
                            "nominal_hybrid_ppl": nominal_ppl,
                            "delta_nll_total": result["nll"] - digital_nll,
                            "delta_ppl_total": result["ppl_from_mean_nll"] - digital_ppl,
                            "delta_nll_tile": result["nll"] - nominal_nll,
                            "delta_ppl_tile": result["ppl_from_mean_nll"] - nominal_ppl,
                            "proxy_variance": proxy,
                            "injected_noise_rms": result["injected_noise_rms"],
                            "faulted_shards": faulted_shards,
                            "unavailable_shards": unavailable_shards,
                            "predicted_tokens": int(result["predicted_tokens"]),
                        }
                        all_rows.append(row)
                        print(
                            f"digital={digital_set_id} t={timestep} real={realization} "
                            f"policy={policy} NLL={row['nll']:.6f} "
                            f"DeltaNLL(total)={row['delta_nll_total']:.6f} "
                            f"DeltaNLL(tile)={row['delta_nll_tile']:.6f}"
                        )
        finally:
            hybrid.restore_digital_modules()
            hybrid = None
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            output_root = resolve_path(cfg["output_root"])
            output_root.mkdir(parents=True, exist_ok=True)

            write_csv(
                output_root / "hybrid_quality_by_policy.partial.csv",
                all_rows,
            )

    output_root = resolve_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = write_csv(output_root / "hybrid_quality_by_policy.csv", all_rows)
    # The partial file is a crash checkpoint written after each operating
    # point; once the final artifact exists it is redundant and would only
    # confuse later analysis, so remove it.
    (output_root / "hybrid_quality_by_policy.partial.csv").unlink(missing_ok=True)
    nominal_path = write_csv(output_root / "nominal_hybrid_frontier.csv", nominal_rows)
    summaries = summarize_rows(all_rows, seed)
    summary_path = write_csv(output_root / "hybrid_quality_summary.csv", summaries)
    paired = paired_differences(all_rows, seed)
    paired_path = write_csv(output_root / "paired_policy_summary.csv", paired) if paired else None
    metadata_path = output_root / "phase4_metadata.json"
    save_json(metadata_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_commit(REPO_ROOT),
        "experiment_seed": seed,
        "placement_seed": int(config["experiment"].get("placement_seed", seed)),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "phase1_path": str(phase1_path),
        "operating_points_path": str(operating_points_path),
        "trace_path": str(trace_path),
        "phase3_manifest_path": str(phase3_manifest_path),
        "digital_reference": {"nll": digital_nll, "ppl": digital_ppl, "predicted_tokens": token_count},
        "dataset": dataset_metadata,
        "analog_configuration": analog_configuration(settings),
        "evaluated_digital_set_ids": [point["digital_set_id"] for point in points],
        "timesteps": timesteps,
        "realizations": realizations,
        "antithetic": antithetic,
        "artifacts": {
            "quality": str(quality_path),
            "nominal_frontier": str(nominal_path),
            "summary": str(summary_path),
            "paired_summary": None if paired_path is None else str(paired_path),
        },
    })
    print(f"Phase 4 complete: {metadata_path}")
    return metadata_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    parser.add_argument("--digital-set-id", action="append", default=[])
    args = parser.parse_args()
    main(
        args.config,
        args.phase1,
        args.operating_points,
        args.trace,
        args.phase3_manifest,
        set(args.digital_set_id),
    )
