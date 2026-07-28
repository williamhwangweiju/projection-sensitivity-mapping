#!/usr/bin/env python3
"""Evaluate all-analog GPT-2 quality under every static placement policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import ManualAnalogSettings, analog_configuration, set_seed
from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.dataset import build_causal_lm_batches
from src.common.metrics import evaluate_nll_ppl, summarize
from src.common.model_loading import load_model_and_tokenizer
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


def _records_for_proxy(
    rows: list[dict[str, Any]],
    measured_sensitivity: dict[str, float],
    sensitivity_floor: float,
) -> list[PlacementRecord]:
    """Re-weight placement rows with measured Phase-1 sensitivity.

    static_fisher placement CSVs carry Fisher importance in their
    sensitivity/importance columns; re-weighting every policy with the
    measured score keeps the proxy_variance column comparable across
    policies.
    """
    fields = PlacementRecord.__dataclass_fields__
    records = []
    for row in rows:
        values = {key: row[key] for key in fields}
        sensitivity = max(
            float(measured_sensitivity[str(row["projection_id"])]),
            float(sensitivity_floor),
        )
        values["sensitivity"] = sensitivity
        values["importance"] = sensitivity * float(row["shard_weight"])
        records.append(PlacementRecord(**values))
    return records


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


def summarize_rows(
    rows: list[dict[str, Any]], seed: int, digital_nll: float
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    for policy, group in sorted(groups.items()):
        delta_total = [float(row["delta_nll_total"]) for row in group]
        delta_tile = [float(row["delta_nll_tile"]) for row in group]
        total_summary = summarize(delta_total)
        tile_summary = summarize(delta_tile)
        lo, hi = _bootstrap_mean_ci(delta_tile, seed)
        degraded_nll = digital_nll + total_summary["mean"]
        output.append({
            "policy": policy,
            "evaluations": len(group),
            "mean_delta_nll_total": total_summary["mean"],
            "std_delta_nll_total": total_summary["std"],
            "mean_delta_nll_tile": tile_summary["mean"],
            "std_delta_nll_tile": tile_summary["std"],
            "bootstrap_ci95_delta_nll_tile_low": lo,
            "bootstrap_ci95_delta_nll_tile_high": hi,
            "mean_degraded_nll": degraded_nll,
            "mean_degraded_ppl_from_nll": math.exp(degraded_nll),
            "mean_proxy_variance": float(np.mean([float(row["proxy_variance"]) for row in group])),
        })
    return output


def paired_differences(
    rows: list[dict[str, Any]],
    seed: int,
    method_policies: Iterable[str] = ("static_sensitivity", "static_fisher"),
    baseline_policies: Iterable[str] = ("hardware_only", "sequential", "random"),
) -> list[dict[str, Any]]:
    """Paired within-run comparisons over shared (timestep, realization) noise.

    The bootstrap intervals resample correlated samples from one hardware
    trace and are descriptive; cross-trace inference lives in
    scripts/aggregate_multiseed.py.
    """
    keyed: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        keyed[(int(row["timestep"]), int(row["realization"]))][str(row["policy"])] = row
    comparisons = tuple(
        (baseline, method)
        for method in method_policies
        for baseline in baseline_policies
        if baseline != method
    )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    wins: dict[tuple[str, str], int] = defaultdict(int)
    for policy_rows in keyed.values():
        for baseline, method in comparisons:
            if baseline not in policy_rows or method not in policy_rows:
                continue
            difference = float(policy_rows[baseline]["delta_nll_tile"]) - float(policy_rows[method]["delta_nll_tile"])
            grouped[(baseline, method)].append(difference)
            wins[(baseline, method)] += int(difference > 0)
    output: list[dict[str, Any]] = []
    for (baseline, method), values in sorted(grouped.items()):
        lo, hi = _bootstrap_mean_ci(values, seed)
        output.append({
            "baseline_policy": baseline,
            "method_policy": method,
            "paired_samples": len(values),
            "mean_nll_improvement": float(np.mean(values)),
            "median_nll_improvement": float(np.median(values)),
            "std_nll_improvement": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "win_fraction": wins[(baseline, method)] / len(values),
        })
    return output


def main(
    config_path: Path,
    phase1_path: Path,
    trace_path: Path,
    phase3_manifest_path: Path,
) -> Path:
    config = load_yaml(config_path)
    cfg = config["phase4"]
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    profile = load_json(phase1_path)
    manifest_payload = load_json(phase3_manifest_path)
    placement_paths = {
        str(row["policy"]): Path(row["placement_path"])
        for row in manifest_payload["placements"]
    }

    trace = load_trace(str(trace_path))
    timesteps = representative_timesteps(trace, cfg.get("timesteps"))
    policies = [str(value) for value in cfg["policies"]]
    missing = sorted(set(policies) - set(placement_paths))
    if missing:
        raise FileNotFoundError(f"Missing Phase-3 placements for policies: {missing}")
    # Placements are static within Phase 4; parse each CSV once.
    placement_cache = {policy: read_placement_csv(placement_paths[policy]) for policy in policies}
    realizations = int(cfg["num_realizations"])
    antithetic = bool(cfg.get("antithetic", False))
    unavailable_noise_std = float(cfg.get("unavailable_noise_std", config["phase2"]["fidelity_model"]["max_noise_std"]))

    device = torch.device(str(config["model"]["device"]))
    model, tokenizer, model_source = load_model_and_tokenizer(config, device=device)

    evaluation_config = deepcopy(config)
    evaluation_config["dataset"] = deepcopy(config.get("evaluation_dataset", config["dataset"]))
    batches, dataset_metadata = build_causal_lm_batches(evaluation_config, tokenizer)
    digital_nll, digital_ppl, token_count = evaluate_nll_ppl(model, batches, device)
    print(f"Digital reference: NLL={digital_nll:.6f} PPL={digital_ppl:.4f}")

    # Release every torch-cached block from the reference evaluation before
    # AIHWKit allocates its tiles: RPUCuda uses raw cudaMalloc/cuBLAS outside
    # torch's caching allocator, and reserved-but-unused torch memory has
    # caused CUBLAS initialization failures at the first tile conversion.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    settings = ManualAnalogSettings.from_config(config)
    settings.validate()
    measured_sensitivity = {
        str(row["projection_id"]): float(row["sensitivity_score_for_mapping"])
        for row in profile["projections"]
    }
    proxy_floor = float(config.get("phase3", {}).get("sensitivity_floor", 0.0))
    output_root = resolve_path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    # Resume from the crash-checkpoint CSV: completed (policy, timestep,
    # realization) evaluations are loaded and skipped. Noise fields are keyed
    # by seed, projection, and realization, so skipping is order-independent.
    partial_path = output_root / "hybrid_quality_by_policy.partial.csv"
    completed: set[tuple[str, int, int]] = set()
    if partial_path.is_file():
        with partial_path.open("r", newline="", encoding="utf-8") as stream:
            loaded = list(csv.DictReader(stream))
        if loaded:
            all_rows.extend(loaded)
            completed = {
                (str(row["policy"]), int(row["timestep"]), int(row["realization"]))
                for row in loaded
            }
            print(
                f"Resuming Phase 4: {len(loaded)} completed evaluation(s) "
                f"loaded from {partial_path.name}."
            )

    # The deployment is all-analog: every profiled projection converts.
    hybrid = HybridAnalogModel(
        model,
        digital_projection_ids=[],
        settings=settings,
        include_lm_head_candidate=bool(config["profiling"].get("include_lm_head", False)),
        phase1_projection_rows=profile["projections"],
    ).convert()
    try:
        nominal_nll, nominal_ppl, _ = evaluate_nominal_hybrid(hybrid, batches, device)
        nominal_record = {
            "analog_projection_count": len(hybrid.analog_projection_ids),
            "digital_nll": digital_nll,
            "digital_ppl": digital_ppl,
            "nominal_hybrid_nll": nominal_nll,
            "nominal_hybrid_ppl": nominal_ppl,
            "delta_nll_nominal_vs_digital": nominal_nll - digital_nll,
            "delta_ppl_nominal_vs_digital": nominal_ppl - digital_ppl,
        }
        print(
            f"Nominal all-analog hybrid: NLL={nominal_nll:.6f} PPL={nominal_ppl:.4f} "
            f"(delta vs digital {nominal_nll - digital_nll:+.6f})"
        )
        for timestep in timesteps:
            current_noise = np.asarray(trace.noise_std[timestep], dtype=np.float64).copy()
            unavailable = ~np.asarray(trace.available[timestep], dtype=bool)
            current_noise[unavailable] = unavailable_noise_std
            for realization in range(realizations):
                for policy in policies:
                    if (policy, timestep, realization) in completed:
                        continue
                    current_rows = update_placement_noise(
                        placement_cache[policy], current_noise, timestep
                    )
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
                    proxy = placement_proxy(
                        _records_for_proxy(current_rows, measured_sensitivity, proxy_floor),
                        variance=True,
                    )
                    row = {
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
                        f"t={timestep} real={realization} policy={policy} "
                        f"NLL={row['nll']:.6f} "
                        f"DeltaNLL(total)={row['delta_nll_total']:.6f} "
                        f"DeltaNLL(tile)={row['delta_nll_tile']:.6f}"
                    )
            # Checkpoint after every completed timestep block so a preempted
            # session loses at most one timestep of work.
            if all_rows:
                write_csv(partial_path, all_rows)
    finally:
        hybrid.restore_digital_modules()
        hybrid = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if all_rows:
            write_csv(partial_path, all_rows)

    quality_path = write_csv(output_root / "hybrid_quality_by_policy.csv", all_rows)
    partial_path.unlink(missing_ok=True)
    nominal_path = save_json(output_root / "nominal_reference.json", nominal_record)
    summaries = summarize_rows(all_rows, seed, float(digital_nll))
    summary_path = write_csv(output_root / "hybrid_quality_summary.csv", summaries)
    method_policies = [
        str(value)
        for value in cfg.get("method_policies", ["static_sensitivity", "static_fisher"])
        if str(value) in set(policies)
    ]
    baseline_policies = [
        str(value)
        for value in cfg.get("baseline_policies", ["hardware_only", "sequential", "random"])
        if str(value) in set(policies)
    ]
    paired = paired_differences(all_rows, seed, method_policies, baseline_policies)
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
        "trace_path": str(trace_path),
        "phase3_manifest_path": str(phase3_manifest_path),
        "digital_reference": {
            "nll": digital_nll,
            "ppl": digital_ppl,
            "predicted_tokens": token_count,
        },
        "nominal_reference": nominal_record,
        "model_source": model_source,
        "dataset": dataset_metadata,
        "analog_configuration": analog_configuration(settings),
        "timesteps": timesteps,
        "realizations": realizations,
        "antithetic": antithetic,
        "artifacts": {
            "quality": str(quality_path),
            "nominal_reference": str(nominal_path),
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
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase3-manifest", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.phase1, args.trace, args.phase3_manifest)
