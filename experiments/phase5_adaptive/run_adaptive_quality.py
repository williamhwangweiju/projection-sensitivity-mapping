#!/usr/bin/env python3
"""Measure GPT-2 quality under the migration-aware adaptive placement policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any
import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import ManualAnalogSettings, set_seed
from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.dataset import build_causal_lm_batches
from src.common.metrics import evaluate_nll_ppl
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel
from src.evaluation.hybrid_quality import evaluate_noisy_placement, evaluate_nominal_hybrid, write_csv
from src.evaluation.noise_materialization import read_placement_csv
from src.mapping.objective import placement_proxy
from src.mapping.placement import PlacementRecord
from src.simulators.tile_fidelity import load_trace
from experiments.phase4_quality.run_hybrid_quality import representative_timesteps, summarize_rows


def _records_for_proxy(rows: list[dict[str, Any]]) -> list[PlacementRecord]:
    fields = PlacementRecord.__dataclass_fields__
    return [PlacementRecord(**{field: row[field] for field in fields}) for row in rows]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(
    config_path: Path,
    operating_points_path: Path,
    trace_path: Path,
    phase5_manifest_path: Path,
    phase4_quality_csv: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    cfg = config["phase5"]
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    points = {
        point["digital_set_id"]: point
        for point in load_json(operating_points_path)["operating_points"]
    }
    phase5_payload = load_json(phase5_manifest_path)
    adaptive_runs = phase5_payload["adaptive_runs"]
    phase1_profile = load_json(phase5_payload["phase1_path"])
    trace = load_trace(str(trace_path))
    timesteps = representative_timesteps(trace, cfg.get("quality_timesteps"))
    realizations = int(cfg.get("quality_num_realizations", config["phase4"]["num_realizations"]))
    antithetic = bool(cfg.get("quality_antithetic", config["phase4"].get("antithetic", False)))

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
    digital_nll, digital_ppl, predicted_tokens = evaluate_nll_ppl(model, batches, device)
    settings = ManualAnalogSettings.from_config(config)

    rows: list[dict[str, Any]] = []
    for run in adaptive_runs:
        digital_set_id = str(run["digital_set_id"])
        point = points[digital_set_id]
        all_placements = read_placement_csv(run["adaptive_placement_path"])
        by_timestep: dict[int, list[dict[str, Any]]] = {}
        for row in all_placements:
            by_timestep.setdefault(int(row["timestep"]), []).append(row)
        hybrid = HybridAnalogModel(
            model,
            digital_projection_ids=point["digital_projection_ids"],
            settings=settings,
            include_lm_head_candidate=bool(config["profiling"].get("include_lm_head", False)),
            phase1_projection_rows=phase1_profile["projections"],
        ).convert()
        try:
            nominal_nll, nominal_ppl, _ = evaluate_nominal_hybrid(hybrid, batches, device)
            for timestep in timesteps:
                if timestep not in by_timestep:
                    raise ValueError(f"Adaptive placement has no timestep {timestep}.")
                placement_rows = by_timestep[timestep]
                proxy = placement_proxy(_records_for_proxy(placement_rows), variance=True)
                for realization in range(realizations):
                    result = evaluate_noisy_placement(
                        hybrid,
                        batches,
                        device,
                        placement_rows,
                        base_seed=seed,
                        realization=realization,
                        antithetic=antithetic,
                    )
                    row = {
                        "digital_set_id": digital_set_id,
                        "selection_method": point["selection_method"],
                        "budget_type": point["budget_type"],
                        "budget_value": point["budget_value"],
                        "digital_projection_count": point["digital_projection_count"],
                        "digital_parameter_fraction": point["digital_parameter_fraction"],
                        "digital_incremental_storage_fraction": point.get(
                            "digital_incremental_storage_fraction",
                            point["digital_parameter_fraction"],
                        ),
                        "digital_mac_fraction": point["digital_mac_fraction"],
                        "policy": "adaptive_sensitivity",
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
                        "predicted_tokens": int(result["predicted_tokens"]),
                    }
                    rows.append(row)
                    print(
                        f"adaptive digital={digital_set_id} t={timestep} real={realization} "
                        f"NLL={row['nll']:.6f} DeltaNLL(total)={row['delta_nll_total']:.6f}"
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
    quality_path = write_csv(output_root / "adaptive_quality.csv", rows)
    summary_path = write_csv(output_root / "adaptive_quality_summary.csv", summarize_rows(rows, seed))
    comparison_path: Path | None = None
    if phase4_quality_csv is not None and phase4_quality_csv.is_file():
        static_rows = _read_csv(phase4_quality_csv)
        static_index = {
            (row["digital_set_id"], int(row["timestep"]), int(row["realization"]), row["policy"]): row
            for row in static_rows
        }
        comparisons: list[dict[str, Any]] = []
        for row in rows:
            for baseline in ("static_sensitivity", "hardware_only"):
                key = (
                    row["digital_set_id"], int(row["timestep"]), int(row["realization"]), baseline
                )
                if key not in static_index:
                    continue
                baseline_row = static_index[key]
                comparisons.append({
                    "digital_set_id": row["digital_set_id"],
                    "timestep": row["timestep"],
                    "realization": row["realization"],
                    "baseline_policy": baseline,
                    "adaptive_minus_baseline_nll": float(row["nll"]) - float(baseline_row["nll"]),
                    "baseline_minus_adaptive_tile_delta_nll": float(baseline_row["delta_nll_tile"]) - float(row["delta_nll_tile"]),
                })
        if comparisons:
            comparison_path = write_csv(output_root / "adaptive_vs_static_paired.csv", comparisons)
    metadata_path = output_root / "phase5_quality_metadata.json"
    save_json(metadata_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_commit(REPO_ROOT),
        "config_sha256": file_sha256(config_path),
        "experiment_seed": seed,
        "placement_seed": int(config["experiment"].get("placement_seed", seed)),
        "config_path": str(config_path),
        "operating_points_path": str(operating_points_path),
        "trace_path": str(trace_path),
        "phase5_manifest_path": str(phase5_manifest_path),
        "phase4_quality_csv": None if phase4_quality_csv is None else str(phase4_quality_csv),
        "digital_reference": {"nll": digital_nll, "ppl": digital_ppl, "predicted_tokens": predicted_tokens},
        "dataset": dataset_metadata,
        "timesteps": timesteps,
        "realizations": realizations,
        "artifacts": {
            "quality": str(quality_path),
            "summary": str(summary_path),
            "comparison": None if comparison_path is None else str(comparison_path),
        },
    })
    print(f"Phase 5 adaptive quality complete: {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase5-manifest", type=Path, required=True)
    parser.add_argument("--phase4-quality-csv", type=Path)
    args = parser.parse_args()
    main(
        args.config,
        args.operating_points,
        args.trace,
        args.phase5_manifest,
        args.phase4_quality_csv,
    )
