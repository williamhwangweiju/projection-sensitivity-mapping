#!/usr/bin/env python3
"""Evaluate deployment-time hybrids that retain top projections digitally.

The model checkpoint, evaluation corpus, hardware traces, analog conversion,
and noise fields match Phase 4. For each configured design, the named digital
projections are kept in digital compute and every other Phase-1 projection is
remapped from scratch to the hardware substrate with the design's remainder
policy (``mapping_policy``, default ``hybrid_baselines.mapping_policy``).

``hybrid_baselines.digital_weight_mode`` selects how the retained digital
projections hold their weights:

* ``unclipped`` -- the checkpoint's original floating-point modules (the
  configuration the paper labels an off-distribution diagnostic);
* ``clipped`` -- float32 modules holding the 2.5-sigma-clipped weights the HWA
  checkpoint was trained through, no quantization, no noise (the
  deployment-relevant digital comparator).

Every condition is paired with the archived all-analog Phase-4 row of the same
remainder policy (``--all-analog-quality``) and, when present, with the
all-analog ``static_sensitivity`` row (the proposed policy). Two checkpoint-level
references are measured once per run: every projection digital and unclipped
(the 42.91 diagnostic of the paper) and, in ``clipped`` mode, every projection
digital and clipped.

Resumable: each completed condition is appended to a partial CSV, the
checkpoint references and per-design nominal passes are cached in a partial
JSON, and rerunning with the same configuration skips everything already done.
"""
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
from typing import Any, Iterable, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase4_quality.run_hybrid_quality import representative_timesteps
from src.common.analog import ManualAnalogSettings, analog_configuration, set_seed
from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.dataset import build_causal_lm_batches
from src.common.metrics import evaluate_nll_ppl
from src.common.model_loading import load_model_and_tokenizer
from src.common.tabular import write_csv
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel
from src.evaluation.hybrid_baselines import (
    HybridBaselineDesign,
    build_hybrid_placement,
    parse_digital_weight_mode,
    parse_hybrid_designs,
    placement_rows,
    projection_compute_accounting,
)
from src.evaluation.hybrid_quality import evaluate_noisy_placement, evaluate_nominal_hybrid
from src.evaluation.noise_materialization import update_placement_noise
from src.mapping.objective import placement_proxy
from src.simulators.tile_fidelity import load_trace

PROPOSED_POLICY = "static_sensitivity"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _reference_by_policy_condition(
    path: Path | None,
) -> dict[tuple[str, int, int], dict[str, str]]:
    """Index the archived all-analog Phase-4 rows by (policy, timestep, realization)."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"All-analog reference CSV does not exist: {path}")
    result: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in _read_csv(path):
        key = (str(row["policy"]), int(row["timestep"]), int(row["realization"]))
        if key in result:
            raise ValueError(f"Duplicate all-analog reference condition: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"No rows in all-analog reference CSV: {path}")
    return result


def _bootstrap_mean_ci(
    values: Iterable[float], seed: int, samples: int = 4000
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _floats(rows: Iterable[Mapping[str, Any]], column: str) -> list[float]:
    return [
        float(row[column])
        for row in rows
        if str(row.get(column, "")) != ""
    ]


def summarize_rows(rows: list[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Summarize absolute quality and paired improvements by hybrid design."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["design"])].append(row)
    output: list[dict[str, Any]] = []
    for design, group in sorted(groups.items()):
        nll = np.asarray([float(row["nll"]) for row in group], dtype=np.float64)
        first = group[0]
        record: dict[str, Any] = {
            "design": design,
            "digital_projection_ids": first["digital_projection_ids"],
            "digital_weight_mode": first.get("digital_weight_mode", "unclipped"),
            "mapping_policy": first["policy"],
            "evaluations": len(group),
            "mean_nll": float(nll.mean()),
            "std_nll": float(nll.std(ddof=1)) if nll.size > 1 else 0.0,
            "ppl_from_mean_nll": math.exp(float(nll.mean())),
            "digital_projection_mac_fraction": float(
                first["digital_projection_mac_fraction"]
            ),
            "analog_shards": int(first["analog_shards"]),
        }
        for label, column in (
            ("vs_all_analog", "nll_improvement_vs_all_analog"),
            ("vs_static_sensitivity", "nll_improvement_vs_static_sensitivity"),
        ):
            improvements = _floats(group, column)
            ci_low, ci_high = _bootstrap_mean_ci(improvements, seed)
            record[f"mean_nll_improvement_{label}"] = (
                float(np.mean(improvements)) if improvements else ""
            )
            record[f"bootstrap_ci95_improvement_{label}_low"] = ci_low if improvements else ""
            record[f"bootstrap_ci95_improvement_{label}_high"] = ci_high if improvements else ""
            record[f"win_fraction_{label}"] = (
                sum(value > 0 for value in improvements) / len(improvements)
                if improvements
                else ""
            )
        output.append(record)
    return output


def summarize_rows_by_timestep(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-(design, timestep) means of the paired improvements."""
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["design"]), int(row["timestep"]))].append(row)
    output: list[dict[str, Any]] = []
    for (design, timestep), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "design": design,
            "timestep": timestep,
            "evaluations": len(group),
            "mean_nll": float(np.mean([float(row["nll"]) for row in group])),
        }
        for label, column in (
            ("vs_all_analog", "nll_improvement_vs_all_analog"),
            ("vs_static_sensitivity", "nll_improvement_vs_static_sensitivity"),
        ):
            improvements = _floats(group, column)
            record[f"mean_nll_improvement_{label}"] = (
                float(np.mean(improvements)) if improvements else ""
            )
            record[f"win_fraction_{label}"] = (
                sum(value > 0 for value in improvements) / len(improvements)
                if improvements
                else ""
            )
        output.append(record)
    return output


def _run_signature(
    config_path: Path,
    phase1_path: Path,
    trace_path: Path,
    all_analog_quality_path: Path | None,
    designs: list[HybridBaselineDesign],
    default_policy: str,
    digital_weight_mode: str,
    timesteps: list[int],
    realizations: int,
) -> dict[str, Any]:
    return {
        "implementation_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(config_path),
        "phase1_sha256": file_sha256(phase1_path),
        "trace_sha256": file_sha256(trace_path),
        "all_analog_quality_sha256": (
            None
            if all_analog_quality_path is None
            else file_sha256(all_analog_quality_path)
        ),
        "digital_weight_mode": digital_weight_mode,
        "designs": [
            {
                "name": design.name,
                "digital_projection_ids": list(design.digital_projection_ids),
                "mapping_policy": design.effective_policy(default_policy),
            }
            for design in designs
        ],
        "timesteps": timesteps,
        "realizations": realizations,
    }


def _completed_run(
    output_root: Path,
    signature: Mapping[str, Any],
    expected: set[tuple[str, int, int]],
) -> Path | None:
    """Return prior metadata when an identical finalized run is complete."""
    final_signature_path = output_root / "hybrid_baselines_run_signature.json"
    if not final_signature_path.is_file():
        return None
    stored = load_json(final_signature_path)
    if stored != signature:
        raise ValueError(
            "Existing finalized hybrid baselines have a different run signature. "
            f"Use a new output directory or archive {output_root}."
        )
    quality_path = output_root / "hybrid_baselines_by_condition.csv"
    metadata_path = output_root / "hybrid_baselines_metadata.json"
    if not quality_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            "Final hybrid-baseline signature exists but finalized artifacts are missing: "
            f"{output_root}"
        )
    rows = _read_csv(quality_path)
    observed = {
        (str(row["design"]), int(row["timestep"]), int(row["realization"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            "Final hybrid-baseline artifact does not contain the expected conditions: "
            f"expected={len(expected)}, observed={len(rows)}"
        )
    print(f"Reusing completed hybrid baselines: {metadata_path}")
    return metadata_path


class _ReferenceCache:
    """Checkpoint-level references and per-design nominal passes, cached on disk
    so that a restarted run does not repeat them."""

    def __init__(self, path: Path, signature: Mapping[str, Any]) -> None:
        self.path = path
        self.signature = dict(signature)
        self.payload: dict[str, Any] = {"signature": self.signature, "references": {}, "nominal": {}}
        if path.is_file():
            stored = load_json(path)
            if stored.get("signature") == self.signature:
                self.payload = stored
                self.payload.setdefault("references", {})
                self.payload.setdefault("nominal", {})
            else:
                print(f"Ignoring reference cache with a different signature: {path}")

    def get_reference(self, name: str) -> dict[str, Any] | None:
        value = self.payload["references"].get(name)
        return dict(value) if isinstance(value, dict) else None

    def set_reference(self, name: str, value: Mapping[str, Any]) -> None:
        self.payload["references"][name] = dict(value)
        save_json(self.path, self.payload)

    def get_nominal(self, design: str) -> dict[str, Any] | None:
        value = self.payload["nominal"].get(design)
        return dict(value) if isinstance(value, dict) else None

    def set_nominal(self, design: str, value: Mapping[str, Any]) -> None:
        self.payload["nominal"][design] = dict(value)
        save_json(self.path, self.payload)


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main(
    config_path: Path,
    phase1_path: Path,
    trace_path: Path,
    *,
    output_dir: Path | None = None,
    all_analog_quality_path: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    section = config.get("hybrid_baselines")
    if not isinstance(section, Mapping):
        raise ValueError("Config must define hybrid_baselines.")
    profile = load_json(phase1_path)
    projection_rows_source = list(profile["projections"])
    candidate_ids = [str(row["projection_id"]) for row in projection_rows_source]
    designs = parse_hybrid_designs(config, candidate_ids)
    digital_weight_mode = parse_digital_weight_mode(config)
    default_policy = str(section.get("mapping_policy", "hardware_only"))
    if digital_weight_mode != "clipped":
        print(
            "Warning: digital_weight_mode is 'unclipped' (the checkpoint's original "
            "floating-point modules); the deployment-relevant comparator is 'clipped'."
        )

    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    trace = load_trace(str(trace_path))
    timesteps = representative_timesteps(
        trace, section.get("timesteps", config.get("phase4", {}).get("timesteps"))
    )
    realizations = int(
        section.get("num_realizations", config.get("phase4", {}).get("num_realizations", 1))
    )
    if realizations <= 0:
        raise ValueError("hybrid_baselines.num_realizations must be positive.")
    antithetic = bool(
        section.get("antithetic", config.get("phase4", {}).get("antithetic", False))
    )
    unavailable_noise_std = float(
        section.get(
            "unavailable_noise_std",
            config.get("phase4", {}).get(
                "unavailable_noise_std",
                config["phase2"]["fidelity_model"]["max_noise_std"],
            ),
        )
    )
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else resolve_path(section["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)

    signature = _run_signature(
        config_path,
        phase1_path,
        trace_path,
        all_analog_quality_path,
        designs,
        default_policy,
        digital_weight_mode,
        timesteps,
        realizations,
    )
    expected_conditions = {
        (design.name, timestep, realization)
        for design in designs
        for timestep in timesteps
        for realization in range(realizations)
    }
    completed_metadata = _completed_run(output_root, signature, expected_conditions)
    if completed_metadata is not None:
        return completed_metadata
    signature_path = output_root / "hybrid_baselines.partial_signature.json"
    partial_path = output_root / "hybrid_baselines.partial.csv"
    if partial_path.is_file():
        stored = load_json(signature_path) if signature_path.is_file() else None
        if stored != signature:
            raise ValueError(
                "Existing hybrid baseline partial results have a different run signature. "
                f"Move or remove {partial_path} before changing the run configuration."
            )
    else:
        save_json(signature_path, signature)
    cache = _ReferenceCache(output_root / "hybrid_baselines_references.partial.json", signature)

    all_rows: list[dict[str, Any]] = _read_csv(partial_path) if partial_path.is_file() else []
    completed = {
        (str(row["design"]), int(row["timestep"]), int(row["realization"]))
        for row in all_rows
    }
    if completed:
        print(f"Resuming hybrid baselines: {len(completed)} conditions already complete.")

    reference = _reference_by_policy_condition(all_analog_quality_path)
    requested_conditions = {
        (timestep, realization)
        for timestep in timesteps
        for realization in range(realizations)
    }
    if reference:
        for design in designs:
            policy = design.effective_policy(default_policy)
            missing = sorted(
                condition
                for condition in requested_conditions
                if (policy, *condition) not in reference
            )
            if missing:
                raise ValueError(
                    f"All-analog reference lacks {policy!r} rows for the paired "
                    f"conditions {missing} needed by design {design.name!r}."
                )
    proposed_available = bool(reference) and all(
        (PROPOSED_POLICY, *condition) in reference for condition in requested_conditions
    )
    if reference and not proposed_available:
        print(
            f"Note: all-analog reference has no complete {PROPOSED_POLICY!r} rows; "
            "the vs-proposed columns will be empty."
        )

    device = torch.device(str(config["model"]["device"]))
    model, tokenizer, model_source = load_model_and_tokenizer(config, device=device)
    evaluation_config = deepcopy(config)
    evaluation_config["dataset"] = deepcopy(
        config.get("evaluation_dataset", config["dataset"])
    )
    batches, dataset_metadata = build_causal_lm_batches(evaluation_config, tokenizer)
    available_eval_batches = len(batches)
    max_eval_batches = section.get("max_eval_batches")
    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)
        if max_eval_batches <= 0:
            raise ValueError("hybrid_baselines.max_eval_batches must be positive when set.")
        batches = batches[:max_eval_batches]
    dataset_metadata = dict(dataset_metadata) | {
        "windows_available": available_eval_batches,
        "max_eval_batches": max_eval_batches,
        "windows_used": len(batches),
    }

    settings = ManualAnalogSettings.from_config(config)
    settings.validate()
    include_lm_head_candidate = bool(config["profiling"].get("include_lm_head", False))

    # Reference 1: the HWA checkpoint with every projection digital and
    # unclipped (the paper's 42.91 "diagnostic").
    checkpoint_digital = cache.get_reference("checkpoint_digital")
    if checkpoint_digital is None:
        nll_value, ppl_value, token_count = evaluate_nll_ppl(model, batches, device)
        checkpoint_digital = {
            "nll": float(nll_value),
            "ppl": float(ppl_value),
            "predicted_tokens": int(token_count),
            "digital_weight_mode": "unclipped",
        }
        cache.set_reference("checkpoint_digital", checkpoint_digital)
    print(
        "HWA checkpoint, every projection digital (unclipped): "
        f"NLL={checkpoint_digital['nll']:.6f} PPL={checkpoint_digital['ppl']:.4f}"
    )
    _cleanup_cuda()

    # Reference 2 (clipped mode): every projection digital and clipped -- the
    # float32 deployment of the function the checkpoint was trained through.
    checkpoint_clipped_digital: dict[str, Any] | None = None
    if digital_weight_mode == "clipped":
        checkpoint_clipped_digital = cache.get_reference("checkpoint_clipped_digital")
        if checkpoint_clipped_digital is None:
            all_digital = HybridAnalogModel(
                model,
                digital_projection_ids=candidate_ids,
                settings=settings,
                include_lm_head_candidate=include_lm_head_candidate,
                phase1_projection_rows=projection_rows_source,
                digital_weight_mode="clipped",
            ).convert()
            try:
                nll_value, ppl_value, token_count = evaluate_nll_ppl(model, batches, device)
            finally:
                all_digital.restore_digital_modules()
                all_digital = None
                _cleanup_cuda()
            checkpoint_clipped_digital = {
                "nll": float(nll_value),
                "ppl": float(ppl_value),
                "predicted_tokens": int(token_count),
                "digital_weight_mode": "clipped",
            }
            cache.set_reference("checkpoint_clipped_digital", checkpoint_clipped_digital)
        print(
            "HWA checkpoint, every projection digital (clipped): "
            f"NLL={checkpoint_clipped_digital['nll']:.6f} "
            f"PPL={checkpoint_clipped_digital['ppl']:.4f}"
        )

    nominal_records: list[dict[str, Any]] = []
    design_metadata: list[dict[str, Any]] = []

    for design in designs:
        policy = design.effective_policy(default_policy)
        design_conditions = {
            (design.name, timestep, realization)
            for timestep in timesteps
            for realization in range(realizations)
        }
        cached_nominal = cache.get_nominal(design.name)
        if design_conditions <= completed and cached_nominal is not None:
            # Every condition of this design is already on disk; no GPU work.
            nominal_records.append(cached_nominal["record"])
            design_metadata.append(cached_nominal["metadata"])
            print(f"{design.name}: all {len(design_conditions)} conditions already complete.")
            continue

        records = build_hybrid_placement(config, projection_rows_source, trace, design)
        static_rows = placement_rows(records)
        placement_path = write_csv(
            output_root / f"placement_{design.name}_{policy}.csv", static_rows
        )
        accounting = projection_compute_accounting(
            projection_rows_source, design.digital_projection_ids
        )
        hybrid = HybridAnalogModel(
            model,
            digital_projection_ids=design.digital_projection_ids,
            settings=settings,
            include_lm_head_candidate=include_lm_head_candidate,
            phase1_projection_rows=projection_rows_source,
            digital_weight_mode=digital_weight_mode,
        ).convert()
        try:
            analog_ids = set(hybrid.analog_projection_ids)
            placement_ids = {str(row["projection_id"]) for row in static_rows}
            if analog_ids != placement_ids:
                raise ValueError(
                    f"{design.name}: placement/model analog projection mismatch: "
                    f"missing={sorted(analog_ids - placement_ids)}, "
                    f"extra={sorted(placement_ids - analog_ids)}"
                )
            if digital_weight_mode == "clipped" and set(
                hybrid.clipped_digital_projection_ids
            ) != set(design.digital_projection_ids):
                raise RuntimeError(
                    f"{design.name}: clipped-digital swap incomplete: "
                    f"{sorted(hybrid.clipped_digital_projection_ids)}"
                )
            if cached_nominal is None:
                nominal_nll, nominal_ppl, _ = evaluate_nominal_hybrid(hybrid, batches, device)
                nominal = {
                    "design": design.name,
                    "digital_projection_ids": ";".join(design.digital_projection_ids),
                    "digital_weight_mode": digital_weight_mode,
                    "mapping_policy": policy,
                    "analog_projection_count": len(hybrid.analog_projection_ids),
                    "analog_shards": len(static_rows),
                    **accounting,
                    "nominal_nll": float(nominal_nll),
                    "nominal_ppl": float(nominal_ppl),
                    "checkpoint_digital_nll": checkpoint_digital["nll"],
                    "checkpoint_digital_ppl": checkpoint_digital["ppl"],
                    "checkpoint_clipped_digital_nll": (
                        "" if checkpoint_clipped_digital is None else checkpoint_clipped_digital["nll"]
                    ),
                    "checkpoint_clipped_digital_ppl": (
                        "" if checkpoint_clipped_digital is None else checkpoint_clipped_digital["ppl"]
                    ),
                }
                metadata_entry = {
                    **hybrid.metadata(),
                    "name": design.name,
                    "mapping_policy": policy,
                    "placement_path": str(placement_path),
                    "analog_shards": len(static_rows),
                    **accounting,
                }
                cached_nominal = {"record": nominal, "metadata": metadata_entry}
                cache.set_nominal(design.name, cached_nominal)
            nominal = cached_nominal["record"]
            nominal_nll = float(nominal["nominal_nll"])
            nominal_records.append(nominal)
            design_metadata.append(cached_nominal["metadata"])
            print(
                f"{design.name}: digital={list(design.digital_projection_ids)} "
                f"({digital_weight_mode}), remainder={policy}, "
                f"nominal NLL={nominal_nll:.6f} PPL={float(nominal['nominal_ppl']):.4f}, "
                f"digital MAC fraction={float(accounting['digital_projection_mac_fraction']):.1%}"
            )
            for timestep in timesteps:
                current_noise = np.asarray(trace.noise_std[timestep], dtype=np.float64).copy()
                current_noise[~np.asarray(trace.available[timestep], dtype=bool)] = (
                    unavailable_noise_std
                )
                for realization in range(realizations):
                    key = (design.name, timestep, realization)
                    if key in completed:
                        continue
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
                    baseline = reference.get((policy, timestep, realization))
                    baseline_nll = "" if baseline is None else float(baseline["nll"])
                    baseline_ppl = (
                        "" if baseline is None else float(baseline["ppl_from_mean_nll"])
                    )
                    proposed = (
                        reference.get((PROPOSED_POLICY, timestep, realization))
                        if proposed_available
                        else None
                    )
                    proposed_nll = "" if proposed is None else float(proposed["nll"])
                    row = {
                        "design": design.name,
                        "digital_projection_ids": ";".join(design.digital_projection_ids),
                        "digital_weight_mode": digital_weight_mode,
                        "policy": policy,
                        "timestep": timestep,
                        "realization": realization,
                        "nll": result["nll"],
                        "ppl_from_mean_nll": result["ppl_from_mean_nll"],
                        "ppl_mean": result["ppl_mean"],
                        "nominal_nll": nominal_nll,
                        "nominal_ppl": float(nominal["nominal_ppl"]),
                        "delta_nll_tile": result["nll"] - nominal_nll,
                        "all_analog_policy": policy,
                        "all_analog_nll": baseline_nll,
                        "all_analog_ppl_from_mean_nll": baseline_ppl,
                        "nll_improvement_vs_all_analog": (
                            "" if baseline is None else float(baseline_nll) - result["nll"]
                        ),
                        "ppl_reduction_vs_all_analog": (
                            ""
                            if baseline is None
                            else float(baseline_ppl) - result["ppl_from_mean_nll"]
                        ),
                        "static_sensitivity_nll": proposed_nll,
                        "nll_improvement_vs_static_sensitivity": (
                            "" if proposed is None else float(proposed_nll) - result["nll"]
                        ),
                        "checkpoint_digital_nll": checkpoint_digital["nll"],
                        "checkpoint_digital_ppl": checkpoint_digital["ppl"],
                        "checkpoint_clipped_digital_nll": (
                            ""
                            if checkpoint_clipped_digital is None
                            else checkpoint_clipped_digital["nll"]
                        ),
                        "digital_projection_mac_fraction": accounting[
                            "digital_projection_mac_fraction"
                        ],
                        "analog_projection_count": len(hybrid.analog_projection_ids),
                        "analog_shards": len(static_rows),
                        "proxy_variance": placement_proxy(records, variance=True),
                        "injected_noise_rms": result["injected_noise_rms"],
                        "faulted_shards": sum(
                            int(bool(trace.faulted[timestep, int(item["tile_id"])]))
                            for item in current_rows
                        ),
                        "unavailable_shards": sum(
                            int(not bool(trace.available[timestep, int(item["tile_id"])]))
                            for item in current_rows
                        ),
                        "predicted_tokens": int(result["predicted_tokens"]),
                    }
                    all_rows.append(row)
                    completed.add(key)
                    write_csv(partial_path, all_rows)
                    print(
                        f"design={design.name} t={timestep} real={realization} "
                        f"NLL={row['nll']:.6f} improvement_vs_all_analog="
                        f"{row['nll_improvement_vs_all_analog']} "
                        f"vs_static_sensitivity={row['nll_improvement_vs_static_sensitivity']}"
                    )
        finally:
            hybrid.restore_digital_modules()
            hybrid = None
            _cleanup_cuda()

    quality_path = write_csv(output_root / "hybrid_baselines_by_condition.csv", all_rows)
    summary_path = write_csv(
        output_root / "hybrid_baselines_summary.csv", summarize_rows(all_rows, seed)
    )
    by_timestep_path = write_csv(
        output_root / "hybrid_baselines_summary_by_timestep.csv",
        summarize_rows_by_timestep(all_rows),
    )
    nominal_path = write_csv(output_root / "hybrid_baselines_nominal.csv", nominal_records)
    partial_path.unlink(missing_ok=True)
    signature_path.unlink(missing_ok=True)
    cache.path.unlink(missing_ok=True)
    metadata_path = output_root / "hybrid_baselines_metadata.json"
    digital_semantics = {
        "unclipped": (
            "Named projections retain the original floating-point modules of the "
            "shared HWA checkpoint (unclipped); remaining projections use the "
            "unchanged Phase-4 analog conversion and are remapped from scratch."
        ),
        "clipped": (
            "Named projections are float32 modules holding the weights clipped at "
            "clip_sigma population std, exactly as the analog conversion path "
            "(no quantization, no noise); remaining projections use the unchanged "
            "Phase-4 analog conversion and are remapped from scratch."
        ),
    }[digital_weight_mode]
    save_json(
        metadata_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": git_commit(REPO_ROOT),
            "experiment_type": "post_training_deployment_hybrid",
            "digital_weight_mode": digital_weight_mode,
            "digital_semantics": digital_semantics,
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "phase1_path": str(phase1_path.resolve()),
            "phase1_sha256": file_sha256(phase1_path),
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": file_sha256(trace_path),
            "all_analog_quality_path": (
                None if all_analog_quality_path is None else str(all_analog_quality_path.resolve())
            ),
            "model_source": model_source,
            "dataset": dataset_metadata,
            "checkpoint_digital_reference": checkpoint_digital,
            "checkpoint_clipped_digital_reference": checkpoint_clipped_digital,
            "analog_configuration": analog_configuration(settings),
            "default_mapping_policy": default_policy,
            "timesteps": timesteps,
            "realizations": realizations,
            "antithetic": antithetic,
            "designs": design_metadata,
            "artifacts": {
                "quality": str(quality_path),
                "summary": str(summary_path),
                "summary_by_timestep": str(by_timestep_path),
                "nominal": str(nominal_path),
            },
        },
    )
    save_json(output_root / "hybrid_baselines_run_signature.json", signature)
    print(f"Hybrid baselines complete: {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml",
    )
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--all-analog-quality",
        type=Path,
        default=None,
        help="Optional paired Phase-4 hybrid_quality_by_policy.csv reference.",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.phase1,
        args.trace,
        output_dir=args.output_dir,
        all_analog_quality_path=args.all_analog_quality,
    )
