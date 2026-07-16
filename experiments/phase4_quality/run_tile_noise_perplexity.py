#!/usr/bin/env python3
"""Run Phase 4: all-projection tile-specific quality evaluation."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "src" / "evaluation" / "noise_materialization.py").is_file():
            return candidate
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.analog import (
    ManualAnalogSettings,
    analog_configuration,
    projection_noise_seed,
    set_analog_weights_exact,
    tensor_checksum,
)
from src.common.config import load_yaml, resolve_path
from src.common.dataset import build_lm_batches
from src.common.metrics import evaluate_nll_ppl, summarize
from src.evaluation.aihwkit_gpt2 import (
    convert_all_transformer_projections,
    reference_checksums,
    restore_all_references,
    restore_original_digital_modules,
)
from src.evaluation.noise_materialization import (
    build_normalized_sigma_map,
    load_placement,
    materialize_projection_noise,
)
from src.simulators.tile_fidelity import load_fidelity_trace


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_timesteps(
    requested: Any,
    faulted: np.ndarray,
) -> list[int]:
    num_timesteps = faulted.shape[0]
    if requested is not None:
        values = sorted({int(value) for value in requested})
        if any(value < 0 or value >= num_timesteps for value in values):
            raise ValueError("An evaluation timestep is outside the Phase-2 trace.")
        return values
    first_fault = None
    indices = np.flatnonzero(faulted.any(axis=1))
    if indices.size:
        first_fault = int(indices[0])
    values = {0, num_timesteps // 2, num_timesteps - 1}
    if first_fault is not None:
        values.add(first_fault)
        values.add(max(0, first_fault - 1))
    return sorted(values)


def _aggregate_quality(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        "nll",
        "ppl",
        "delta_nll_total",
        "delta_ppl_total",
        "delta_nll_tile_noise",
        "delta_ppl_tile_noise",
    ]
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_tuple))
        row["n"] = int(len(group))
        for metric in metric_columns:
            stats = summarize([float(value) for value in group[metric]])
            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_outputs(
    quality: pd.DataFrame,
    pairs: Iterable[Iterable[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    difference_rows = []
    for timestep in sorted(quality["timestep"].unique()):
        subset = quality[quality["timestep"] == timestep]
        for pair in pairs:
            policy_a, policy_b = [str(value) for value in pair]
            a = subset[subset["policy"] == policy_a].set_index("realization")
            b = subset[subset["policy"] == policy_b].set_index("realization")
            common = sorted(set(a.index) & set(b.index))
            for realization in common:
                difference_rows.append(
                    {
                        "timestep": int(timestep),
                        "realization": int(realization),
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "difference_delta_nll": float(
                            a.loc[realization, "delta_nll_tile_noise"]
                            - b.loc[realization, "delta_nll_tile_noise"]
                        ),
                        "difference_delta_ppl": float(
                            a.loc[realization, "delta_ppl_tile_noise"]
                            - b.loc[realization, "delta_ppl_tile_noise"]
                        ),
                    }
                )
    differences = pd.DataFrame(difference_rows)
    summary_rows = []
    if not differences.empty:
        for key, group in differences.groupby(
            ["timestep", "policy_a", "policy_b"], sort=True
        ):
            timestep, policy_a, policy_b = key
            nll = summarize(group["difference_delta_nll"].astype(float).tolist())
            ppl = summarize(group["difference_delta_ppl"].astype(float).tolist())
            summary_rows.append(
                {
                    "timestep": int(timestep),
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "n_pairs": int(len(group)),
                    **{f"difference_delta_nll_{name}": value for name, value in nll.items()},
                    **{f"difference_delta_ppl_{name}": value for name, value in ppl.items()},
                    "fraction_policy_b_better_nll": float(
                        (group["difference_delta_nll"] > 0).mean()
                    ),
                    "fraction_policy_b_better_ppl": float(
                        (group["difference_delta_ppl"] > 0).mean()
                    ),
                }
            )
    return differences, pd.DataFrame(summary_rows)


def main(
    config_path: Path,
    phase1_results: Path,
    phase2_trace: Path,
    phase2_metadata: Path | None,
    phase3_dir: Path,
    output_dir: Path | None,
    seed: int,
    overwrite: bool,
) -> Path:
    config = load_yaml(config_path)
    evaluation = config["evaluation"]
    if bool(evaluation.get("compute_kl", False)) or bool(
        evaluation.get("compute_agreement", False)
    ):
        raise NotImplementedError(
            "The corrected reference path currently supports NLL/PPL only. Set "
            "compute_kl=false and compute_agreement=false."
        )
    phase4 = config["phase4"]
    if output_dir is None:
        output_dir = (
            resolve_path(REPO_ROOT, phase4["output_root"])
            / str(phase4["name"])
            / f"seed_{seed}"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1 = _read_json(phase1_results)
    trace = load_fidelity_trace(phase2_trace)
    noise_std = trace["noise_std"].astype(np.float64)
    faulted = trace["faulted"].astype(bool)
    available = trace["available"].astype(bool)

    settings = ManualAnalogSettings.from_config(config)
    trace_reference = float(config["fidelity_model"]["reference_noise_std"])
    if not math.isclose(
        settings.reference_noise_std, trace_reference, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "analog.reference_noise_std and fidelity_model.reference_noise_std "
            "must match exactly."
        )

    model_name = str(config["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    device = torch.device(str(config["model"]["device"]))
    model.to(device)
    batches, dataset_metadata = build_lm_batches(config, tokenizer)

    digital_nll, digital_ppl, token_count = evaluate_nll_ppl(model, batches, device)
    if bool(evaluation.get("strict_phase1_compatibility", True)):
        phase1_baseline = phase1.get("baseline", {})
        phase1_nll = float(phase1_baseline.get("digital_nll", digital_nll))
        tolerance = float(evaluation.get("digital_nll_compatibility_tolerance", 1e-4))
        if abs(phase1_nll - digital_nll) > tolerance:
            raise ValueError(
                "Phase-1 and Phase-4 digital NLL values differ beyond tolerance: "
                f"{phase1_nll} vs {digital_nll}. Use the same model, dataset and "
                "preprocessing configuration."
            )
        current_analog = analog_configuration(settings)
        phase1_analog = phase1.get("metadata", {}).get("analog_configuration", {})
        compatibility_fields = (
            "clip_sigma",
            "programmed_range_mode",
            "reference_noise_std",
            "tile_size",
            "adc_dac_bits",
            "input_resolution",
            "output_resolution",
            "output_bound",
            "forward_is_perfect",
            "weight_scaling_omega",
            "weight_scaling_columnwise",
            "internal_clipping",
            "internal_programming_noise",
            "internal_read_noise",
            "internal_drift",
        )
        for field in compatibility_fields:
            if phase1_analog.get(field) != current_analog.get(field):
                raise ValueError(
                    f"Phase-1/Phase-4 analog configuration mismatch for {field}: "
                    f"{phase1_analog.get(field)!r} vs {current_analog.get(field)!r}."
                )
    references = convert_all_transformer_projections(model, config, phase1)
    try:
        analog_reference_nll, analog_reference_ppl, reference_tokens = evaluate_nll_ppl(
            model, batches, device
        )
        if reference_tokens != token_count:
            raise RuntimeError("Digital and analog reference token counts differ.")

        filenames = config["phase3"]["placement_filenames"]
        policies = [str(value) for value in evaluation["policies"]]
        placements = {
            policy: load_placement(phase3_dir / str(filenames[policy]))
            for policy in policies
        }
        timesteps = _selected_timesteps(evaluation.get("timesteps"), faulted)
        num_realizations = int(config["noise"]["num_realizations"])
        realization_stride = int(config["noise"].get("realization_seed_stride", 1))
        unavailable_action = str(evaluation.get("unavailable_action", "error"))

        quality_rows: list[dict[str, Any]] = []
        assignment_rows: list[dict[str, Any]] = []
        injection_rows: list[dict[str, Any]] = []
        checksum_rows: list[dict[str, Any]] = []

        for realization in range(num_realizations):
            realization_seed = int(seed) + realization * realization_stride
            for timestep in timesteps:
                for policy in policies:
                    try:
                        for projection_id, reference in references.items():
                            normalized_map, rows = build_normalized_sigma_map(
                                reference,
                                placements[policy],
                                noise_std[timestep],
                                available[timestep],
                                unavailable_action=unavailable_action,
                            )
                            projection_seed = projection_noise_seed(
                                realization_seed, projection_id
                            )
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(projection_seed)
                            z = torch.randn(
                                reference.clipped_weight.shape,
                                generator=generator,
                                dtype=torch.float32,
                            )
                            materialized = materialize_projection_noise(
                                reference, normalized_map, z
                            )
                            set_analog_weights_exact(
                                reference.analog_module,
                                materialized.noisy_weight,
                                reference.bias,
                                verify=False,
                            )
                            for row in rows:
                                assignment_rows.append(
                                    {
                                        "timestep": timestep,
                                        "realization": realization,
                                        "policy": policy,
                                        **row,
                                    }
                                )
                            injection_rows.append(
                                {
                                    "timestep": timestep,
                                    "realization": realization,
                                    "realization_seed": realization_seed,
                                    "projection_noise_seed": projection_seed,
                                    "policy": policy,
                                    **materialized.assignment_summary,
                                }
                            )
                            checksum_rows.append(
                                {
                                    "timestep": timestep,
                                    "realization": realization,
                                    "policy": policy,
                                    "projection_id": projection_id,
                                    "reference_checksum": reference.preprocessing[
                                        "clipped_checksum"
                                    ],
                                    "noisy_checksum": tensor_checksum(
                                        materialized.noisy_weight
                                    ),
                                }
                            )

                        nll, ppl, evaluated_tokens = evaluate_nll_ppl(
                            model, batches, device
                        )
                        if evaluated_tokens != token_count:
                            raise RuntimeError("Token count changed during Phase 4.")
                        row = {
                            "timestep": timestep,
                            "realization": realization,
                            "realization_seed": realization_seed,
                            "policy": policy,
                            "nll": nll,
                            "ppl": ppl,
                            "digital_nll": digital_nll,
                            "digital_ppl": digital_ppl,
                            "analog_reference_nll": analog_reference_nll,
                            "analog_reference_ppl": analog_reference_ppl,
                            "delta_nll_total": nll - digital_nll,
                            "delta_ppl_total": ppl - digital_ppl,
                            "delta_nll_tile_noise": nll - analog_reference_nll,
                            "delta_ppl_tile_noise": ppl - analog_reference_ppl,
                            "num_faulted_tiles": int(faulted[timestep].sum()),
                            "mean_tile_noise_std": float(noise_std[timestep].mean()),
                        }
                        quality_rows.append(row)
                        print(
                            f"t={timestep} real={realization} policy={policy} "
                            f"NLL={nll:.6f} PPL={ppl:.4f} "
                            f"DeltaNLL(tile)={row['delta_nll_tile_noise']:.8f} "
                            f"DeltaPPL(tile)={row['delta_ppl_tile_noise']:.6f}",
                            flush=True,
                        )
                    finally:
                        restore_all_references(references)

        quality = pd.DataFrame(quality_rows)
        quality.to_csv(output_dir / "quality_results.csv", index=False)
        _aggregate_quality(quality, ["policy"]).to_csv(
            output_dir / "quality_by_policy.csv", index=False
        )
        _aggregate_quality(quality, ["timestep", "policy"]).to_csv(
            output_dir / "quality_by_timestep.csv", index=False
        )
        differences, paired_summary = _paired_outputs(
            quality, evaluation["policy_pairs"]
        )
        differences.to_csv(output_dir / "paired_policy_differences.csv", index=False)
        paired_summary.to_csv(output_dir / "paired_policy_summary.csv", index=False)
        pd.DataFrame(assignment_rows).to_csv(
            output_dir / "projection_noise_assignments.csv", index=False
        )
        pd.DataFrame(injection_rows).to_csv(
            output_dir / "tile_noise_injection_records.csv", index=False
        )
        pd.DataFrame(checksum_rows).to_csv(
            output_dir / "weight_checksums.csv", index=False
        )

        reference_payload = {
            "digital": {
                "nll": digital_nll,
                "ppl": digital_ppl,
                "token_count": token_count,
            },
            "all_analog_clipped_noise_free": {
                "nll": analog_reference_nll,
                "ppl": analog_reference_ppl,
                "delta_nll_vs_digital": analog_reference_nll - digital_nll,
                "delta_ppl_vs_digital": analog_reference_ppl - digital_ppl,
            },
            "analog_configuration": analog_configuration(settings),
            "projections": [reference.metadata() for reference in references.values()],
            "readback_checksums": reference_checksums(references),
        }
        (output_dir / "reference_analog_conversion.json").write_text(
            json.dumps(reference_payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "timesteps": timesteps,
            "num_realizations": num_realizations,
            "policies": policies,
            "noise_unit": "normalized_std_fraction_of_programmed_projection_range",
            "quality_comparison_reference": "all_analog_clipped_noise_free",
            "phase1_results": str(phase1_results.resolve()),
            "phase2_trace": str(phase2_trace.resolve()),
            "phase2_metadata": None
            if phase2_metadata is None
            else str(phase2_metadata.resolve()),
            "phase3_dir": str(phase3_dir.resolve()),
            "dataset": dataset_metadata,
            "analog_configuration": analog_configuration(settings),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
    finally:
        restore_original_digital_modules(references)

    print("Phase 4 quality evaluation completed.")
    print(
        f"Digital NLL/PPL {digital_nll:.8f} / {digital_ppl:.6f} | "
        f"all-analog clipped reference {analog_reference_nll:.8f} / "
        f"{analog_reference_ppl:.6f}"
    )
    print(f"Results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "full_pipeline" / "gpt2_3dcim.yaml",
    )
    parser.add_argument("--phase1-results", type=Path, required=True)
    parser.add_argument("--phase2-trace", type=Path, required=True)
    parser.add_argument("--phase2-metadata", type=Path, default=None)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        args.config,
        args.phase1_results,
        args.phase2_trace,
        args.phase2_metadata,
        args.phase3_dir,
        args.output_dir,
        args.seed,
        args.overwrite,
    )
