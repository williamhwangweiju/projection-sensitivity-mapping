#!/usr/bin/env python3
"""Run full-model AIHWKit Phase-4 tile-level GPT-2 validation.

The runner uses a frozen Phase-2 hardware snapshot for each complete dataset
pass. Static Phase-3 placements remain fixed across timesteps. A paired
standard-normal field is reused across policies and timesteps for each noise
realization.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import datasets
import torch
import transformers
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Initialize AIHWKit using the same working import path as Phase 1.
import src.profilers.aihwkit_profiler as _aihwkit_bootstrap  # noqa: F401,E402

# Import NumPy only after AIHWKit has initialized.
import numpy as np  # noqa: E402

from src.evaluation.aihwkit_gpt2 import (  # noqa: E402
    convert_gpt2_projections_to_aihwkit,
    reference_weight_map,
    validate_reference_equivalence,
)
from src.evaluation.noise_materialization import (  # noqa: E402
    build_noise_injection_records,
    build_sigma_map,
    generate_paired_noise_tensors,
    load_phase1_noise_calibrations,
)
from src.evaluation.perplexity_evaluator import (  # noqa: E402
    compute_nll,
    evaluate_quality_pair,
)
from src.evaluation.placement_to_gpt2 import (  # noqa: E402
    build_shard_assignments,
    expected_gpt2_module_shapes,
    load_placement_csv,
    validate_shard_coverage,
)
from src.evaluation.schemas import (  # noqa: E402
    GPT2ShardAssignment,
    PairedPolicyDifference,
    QualityMetrics,
)
from src.evaluation.tile_noise_injection import (  # noqa: E402
    NoisedModelContext,
    compute_weight_checksums,
    save_projection_weights,
)
from src.simulators.tile_fidelity import TileFidelityTrace  # noqa: E402


LOGGER = logging.getLogger("phase4_quality")
DEFAULT_OUTPUT_ROOT = Path("data/results/phase4_quality")
DEFAULT_PLACEMENT_FILENAMES = {
    "random": "placement_random.csv",
    "sequential": "placement_sequential.csv",
    "hardware_only": "placement_hardware_only.csv",
    "static_sensitivity": "placement_static_sensitivity.csv",
}


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 all-analog AIHWKit GPT-2 quality evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--phase1-results", type=Path, default=None)
    parser.add_argument("--phase2-trace", type=Path, default=None)
    parser.add_argument("--phase2-metadata", type=Path, default=None)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-realizations", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timesteps", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(arguments)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration YAML must contain a mapping.")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def resolve_latest_phase1_results(root: Path = REPOSITORY_ROOT) -> Path:
    candidates = sorted(
        (root / "data/results/phase1_sensitivity").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No Phase-1 result JSON was found.")
    return candidates[-1]


def resolve_latest_phase2_trace(root: Path = REPOSITORY_ROOT) -> Path:
    candidates = sorted(
        (root / "data/results/phase2_fidelity/fidelity_traces").glob(
            "*/seed_*/trace.npz"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No Phase-2 trace.npz was found.")
    return candidates[-1]


def resolve_latest_phase3_dir(root: Path = REPOSITORY_ROOT) -> Path:
    candidates = sorted(
        (root / "data/results/phase3_baselines").glob("*/seed_*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No Phase-3 placement directory was found.")
    return candidates[-1]


def resolve_output_directory(
    config: Mapping[str, Any],
    *,
    cli_output_dir: Path | None,
    seed: int,
) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir.expanduser().resolve()
    phase4 = config.get("phase4", {}) or {}
    if not isinstance(phase4, Mapping):
        raise ValueError("phase4 must be a mapping.")
    root = Path(phase4.get("output_root", DEFAULT_OUTPUT_ROOT)).expanduser()
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    name = str(phase4.get("name", "phase4_quality"))
    return root.resolve() / name / f"seed_{seed}"


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_max_tokens(value: Any) -> int | None:
    return None if value is None else int(value)


def make_window(
    token_ids: Sequence[int],
    start: int,
    sequence_length: int,
    previous_end: int,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Create one Phase-1-compatible fixed-length evaluation window."""
    end = min(start + sequence_length, len(token_ids))
    tokens = list(token_ids[start:end])
    target_length = min(end - previous_end, len(tokens))
    padding = sequence_length - len(tokens)
    input_ids = tokens + [pad_token_id] * padding
    attention_mask = [1] * len(tokens) + [0] * padding
    labels = list(input_ids)
    for index in range(len(tokens) - target_length):
        labels[index] = -100
    for index in range(len(tokens), sequence_length):
        labels[index] = -100
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    valid_targets = int((batch["labels"][1:] != -100).sum().item())
    return batch, end, valid_targets


def build_batches(
    dataset_cfg: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Build exactly the same fixed-window protocol used by Phase 1."""
    from datasets import load_dataset

    dataset_name = str(dataset_cfg.get("name", "Salesforce/wikitext"))
    dataset_config = str(dataset_cfg.get("config", "wikitext-103-raw-v1"))
    dataset_split = str(dataset_cfg.get("split", "test"))
    sequence_length = int(dataset_cfg.get("sequence_length", 1024))
    stride = int(dataset_cfg.get("stride", sequence_length))
    batch_size = int(dataset_cfg.get("batch_size", 1))
    max_tokens = parse_max_tokens(dataset_cfg.get("max_tokens"))
    separator = str(dataset_cfg.get("document_separator", "\n\n"))
    drop_incomplete = bool(dataset_cfg.get("drop_incomplete_final_sequence", True))

    if sequence_length < 2 or stride <= 0 or batch_size <= 0:
        raise ValueError("Invalid sequence_length, stride, or batch_size.")

    raw_dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    token_ids: list[int] = []
    for sample in raw_dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text + separator, add_special_tokens=False))
        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break

    windows: list[dict[str, torch.Tensor]] = []
    predicted_tokens = 0
    previous_end = 0
    start = 0
    while start < len(token_ids):
        remaining = len(token_ids) - start
        if remaining < sequence_length and drop_incomplete:
            break
        if remaining < 2:
            break
        window, end, valid_targets = make_window(
            token_ids,
            start,
            sequence_length,
            previous_end,
            int(tokenizer.pad_token_id),
        )
        windows.append(window)
        predicted_tokens += valid_targets
        previous_end = end
        if end >= len(token_ids):
            break
        start += stride

    if not windows:
        raise ValueError("Dataset preprocessing produced no evaluation windows.")
    batches = [
        {
            key: torch.stack(
                [window[key] for window in windows[index : index + batch_size]]
            )
            for key in ("input_ids", "attention_mask", "labels")
        }
        for index in range(0, len(windows), batch_size)
    ]
    metadata = {
        "name": dataset_name,
        "config": dataset_config,
        "split": dataset_split,
        "sequence_length": sequence_length,
        "stride": stride,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "document_separator": separator,
        "drop_incomplete_final_sequence": drop_incomplete,
        "collected_tokens": len(token_ids),
        "num_windows": len(windows),
        "num_batches": len(batches),
        "predicted_tokens_per_pass": predicted_tokens,
    }
    return batches, metadata


def _extract_seed(payload: Mapping[str, Any], fallback: int) -> int:
    for key in ("trace_seed", "placement_seed", "seed", "experiment_seed"):
        value = payload.get(key)
        if isinstance(value, (int, np.integer)):
            return int(value)
    for value in payload.values():
        if isinstance(value, Mapping):
            found = _extract_seed(value, fallback=-1)
            if found >= 0:
                return found
    return fallback


def _trace_arrays(
    trace: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    noise = np.asarray(trace.noise_std, dtype=np.float64)
    if noise.ndim != 2:
        raise ValueError(f"trace.noise_std must be 2D, got shape {noise.shape}.")
    faulted_raw = getattr(trace, "faulted", None)
    faulted = (
        np.asarray(faulted_raw, dtype=bool)
        if faulted_raw is not None
        else np.zeros_like(noise, dtype=bool)
    )
    available_raw = getattr(trace, "available", None)
    if available_raw is None and hasattr(trace, "unavailable"):
        available_raw = ~np.asarray(trace.unavailable, dtype=bool)
    available = (
        np.asarray(available_raw, dtype=bool)
        if available_raw is not None
        else np.ones_like(noise, dtype=bool)
    )
    if faulted.shape != noise.shape or available.shape != noise.shape:
        raise ValueError(
            "Trace noise, faulted, and available arrays must have identical shapes."
        )
    raw_ids = getattr(trace, "tile_ids", None)
    tile_ids = (
        [int(value) for value in raw_ids]
        if raw_ids is not None
        else list(range(noise.shape[1]))
    )
    if len(tile_ids) != noise.shape[1] or len(set(tile_ids)) != len(tile_ids):
        raise ValueError("Trace tile_ids are invalid.")
    return noise, faulted, available, tile_ids


def select_representative_timesteps(
    noise: np.ndarray,
    faulted: np.ndarray,
    *,
    override: Sequence[int] | None,
) -> list[int]:
    n_timesteps = int(noise.shape[0])
    if override is not None:
        selected = sorted(set(int(value) for value in override))
        invalid = [value for value in selected if not 0 <= value < n_timesteps]
        if invalid:
            raise ValueError(f"Timesteps out of range [0, {n_timesteps}): {invalid}.")
        return selected

    selected = {0, n_timesteps // 2, n_timesteps - 1}
    any_fault = faulted.any(axis=1)
    fault_indices = np.flatnonzero(any_fault)
    if fault_indices.size:
        first_fault = int(fault_indices[0])
        selected.add(max(0, first_fault - 1))
        selected.add(first_fault)
    return sorted(selected)


def _validate_dataset_compatibility(
    phase1_payload: Mapping[str, Any],
    current_metadata: Mapping[str, Any],
    *,
    strict: bool,
) -> None:
    phase1_dataset = (
        phase1_payload.get("metadata", {}).get("dataset", {})
        if isinstance(phase1_payload.get("metadata", {}), Mapping)
        else {}
    )
    keys = (
        "name",
        "config",
        "split",
        "sequence_length",
        "stride",
        "batch_size",
        "max_tokens",
        "document_separator",
        "drop_incomplete_final_sequence",
    )
    mismatches = {
        key: (phase1_dataset.get(key), current_metadata.get(key))
        for key in keys
        if phase1_dataset.get(key) != current_metadata.get(key)
    }
    if mismatches and strict:
        raise ValueError(f"Phase-1/Phase-4 dataset mismatch: {mismatches}.")
    if mismatches:
        LOGGER.warning("Phase-1/Phase-4 dataset mismatch for debug run: %s", mismatches)


def _assignment_signature(
    assignments: Sequence[GPT2ShardAssignment],
) -> set[tuple[Any, ...]]:
    return {
        (
            assignment.shard_id,
            assignment.projection_id,
            assignment.hf_module_path,
            assignment.canonical_row_start,
            assignment.canonical_row_end,
            assignment.canonical_col_start,
            assignment.canonical_col_end,
        )
        for assignment in assignments
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s", path)
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, indent=2, sort_keys=True, allow_nan=False)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence.")
    mean = float(array.mean())
    if array.size > 1:
        std = float(array.std(ddof=1))
        half_width = 1.96 * std / math.sqrt(array.size)
    else:
        std = 0.0
        half_width = 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_quality(records: Sequence[QualityMetrics]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[QualityMetrics]] = {}
    for record in records:
        buckets.setdefault((record.timestep, record.policy), []).append(record)
    rows: list[dict[str, Any]] = []
    for (timestep, policy), group in sorted(buckets.items()):
        nll = _summarize([record.nll for record in group])
        ppl = _summarize([record.ppl for record in group])
        delta = _summarize([record.delta_nll for record in group])
        delta_ppl = _summarize([record.delta_ppl for record in group])
        kl = _summarize([record.kl_divergence for record in group])
        agreement = _summarize([record.next_token_agreement for record in group])
        rows.append(
            {
                "timestep": timestep,
                "policy": policy,
                "n_realizations": len(group),
                "nll_mean": nll["mean"],
                "nll_std": nll["std"],
                "nll_ci95_low": nll["ci95_low"],
                "nll_ci95_high": nll["ci95_high"],
                "ppl_mean": ppl["mean"],
                "ppl_std": ppl["std"],
                "ppl_ci95_low": ppl["ci95_low"],
                "ppl_ci95_high": ppl["ci95_high"],
                "delta_nll_mean": delta["mean"],
                "delta_nll_std": delta["std"],
                "delta_nll_ci95_low": delta["ci95_low"],
                "delta_nll_ci95_high": delta["ci95_high"],
                "delta_ppl_mean": delta_ppl["mean"],
                "delta_ppl_std": delta_ppl["std"],
                "delta_ppl_ci95_low": delta_ppl["ci95_low"],
                "delta_ppl_ci95_high": delta_ppl["ci95_high"],
                "kl_mean": kl["mean"],
                "kl_std": kl["std"],
                "agreement_mean": agreement["mean"],
                "agreement_std": agreement["std"],
                "num_faulted_shards": group[0].num_faulted_shards,
                "num_unavailable_shards": group[0].num_unavailable_shards,
                "mean_tile_noise_normalized": group[0].mean_tile_noise_normalized,
            }
        )
    return rows


def compute_paired_differences(
    records: Sequence[QualityMetrics],
    policy_pairs: Sequence[tuple[str, str]],
) -> list[PairedPolicyDifference]:
    indexed: dict[tuple[int, int, int, int], dict[str, QualityMetrics]] = {}
    for record in records:
        key = (
            record.trace_seed,
            record.placement_seed,
            record.timestep,
            record.noise_realization_seed,
        )
        indexed.setdefault(key, {})[record.policy] = record

    differences: list[PairedPolicyDifference] = []
    for (_, _, timestep, _), by_policy in sorted(indexed.items()):
        for policy_a, policy_b in policy_pairs:
            if policy_a not in by_policy or policy_b not in by_policy:
                continue
            a = by_policy[policy_a]
            b = by_policy[policy_b]
            differences.append(
                PairedPolicyDifference(
                    timestep=timestep,
                    noise_realization=a.noise_realization,
                    noise_realization_seed=a.noise_realization_seed,
                    trace_seed=a.trace_seed,
                    placement_seed=a.placement_seed,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    delta_nll_a=a.delta_nll,
                    delta_nll_b=b.delta_nll,
                    delta_ppl_a=a.delta_ppl,
                    delta_ppl_b=b.delta_ppl,
                    difference=a.delta_nll - b.delta_nll,
                    difference_delta_ppl=a.delta_ppl - b.delta_ppl,
                )
            )
    return differences


def aggregate_paired_differences(
    differences: Sequence[PairedPolicyDifference],
) -> list[dict[str, Any]]:
    buckets: dict[
        tuple[int, str, str], list[PairedPolicyDifference]
    ] = {}
    for record in differences:
        buckets.setdefault(
            (record.timestep, record.policy_a, record.policy_b), []
        ).append(record)
    rows: list[dict[str, Any]] = []
    for (timestep, policy_a, policy_b), group in sorted(buckets.items()):
        nll_values = [record.difference for record in group]
        ppl_values = [record.difference_delta_ppl for record in group]
        nll_summary = _summarize(nll_values)
        ppl_summary = _summarize(ppl_values)
        rows.append(
            {
                "timestep": timestep,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "n_pairs": len(group),
                "difference_delta_nll_mean": nll_summary["mean"],
                "difference_delta_nll_std": nll_summary["std"],
                "difference_delta_nll_ci95_low": nll_summary["ci95_low"],
                "difference_delta_nll_ci95_high": nll_summary["ci95_high"],
                "difference_delta_ppl_mean": ppl_summary["mean"],
                "difference_delta_ppl_std": ppl_summary["std"],
                "difference_delta_ppl_ci95_low": ppl_summary["ci95_low"],
                "difference_delta_ppl_ci95_high": ppl_summary["ci95_high"],
                "fraction_policy_b_better_nll": float(
                    np.mean(np.asarray(nll_values, dtype=np.float64) > 0.0)
                ),
                "fraction_policy_b_better_ppl": float(
                    np.mean(np.asarray(ppl_values, dtype=np.float64) > 0.0)
                ),
            }
        )
    return rows


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_arguments(arguments)
    configure_logging(args.log_level)
    config = load_yaml(args.config) if args.config is not None else {}

    experiment_cfg = config.get("experiment", {}) or {}
    model_cfg = config.get("model", {}) or {}
    dataset_cfg = dict(config.get("dataset", {}) or {})
    phase1_cfg = config.get("phase1", {}) or {}
    phase2_cfg = config.get("phase2", {}) or {}
    phase3_cfg = config.get("phase3", {}) or {}
    phase4_cfg = config.get("phase4", {}) or {}
    noise_cfg = config.get("noise", {}) or {}
    evaluation_cfg = config.get("evaluation", {}) or {}
    output_cfg = config.get("output", {}) or {}

    seed = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    device = torch.device(args.device or model_cfg.get("device", "cpu"))
    num_realizations = int(
        args.num_realizations
        if args.num_realizations is not None
        else noise_cfg.get("num_realizations", 10)
    )
    seed_stride = int(noise_cfg.get("realization_seed_stride", 1))
    if num_realizations <= 0 or seed_stride <= 0:
        raise ValueError("num_realizations and seed_stride must be positive.")
    if args.max_tokens is not None:
        dataset_cfg["max_tokens"] = args.max_tokens

    phase1_path = (
        args.phase1_results
        or (Path(phase1_cfg["results_path"]) if phase1_cfg.get("results_path") else None)
        or resolve_latest_phase1_results()
    ).expanduser().resolve()
    phase2_trace_path = (
        args.phase2_trace
        or (Path(phase2_cfg["trace_path"]) if phase2_cfg.get("trace_path") else None)
        or resolve_latest_phase2_trace()
    ).expanduser().resolve()
    phase2_metadata_path = (
        args.phase2_metadata
        or (
            Path(phase2_cfg["metadata_path"])
            if phase2_cfg.get("metadata_path")
            else phase2_trace_path.parent / "metadata.json"
        )
    ).expanduser().resolve()
    phase3_dir = (
        args.phase3_dir
        or (Path(phase3_cfg["results_dir"]) if phase3_cfg.get("results_dir") else None)
        or resolve_latest_phase3_dir()
    ).expanduser().resolve()
    output_dir = resolve_output_directory(config, cli_output_dir=args.output_dir, seed=seed)
    prepare_output_directory(output_dir, overwrite=args.overwrite)

    phase1_payload = load_json(phase1_path)
    phase2_metadata = load_json(phase2_metadata_path) if phase2_metadata_path.exists() else {}
    phase3_metadata_path = phase3_dir / "metadata.json"
    phase3_metadata = load_json(phase3_metadata_path) if phase3_metadata_path.exists() else {}

    trace = TileFidelityTrace.load_npz(phase2_trace_path)
    trace_noise, trace_faulted, trace_available, tile_ids = _trace_arrays(trace)
    trace_noise_unit = str(trace.metadata.get("noise_unit", ""))
    if trace_noise_unit != "pcmlike_prog_noise_scale_equivalent":
        raise ValueError(
            "Phase-2 trace noise_unit must be "
            "'pcmlike_prog_noise_scale_equivalent', got "
            f"{trace_noise_unit!r}."
        )

    num_tiles = int(phase3_cfg.get("num_tiles", 72))
    tiers_per_tile = int(phase3_cfg.get("tiers_per_tile", 8))
    tier_rows = int(phase3_cfg.get("tier_rows", phase3_cfg.get("tile_size", 512)))
    tier_cols = int(phase3_cfg.get("tier_cols", phase3_cfg.get("tile_size", 512)))
    if trace_noise.shape[1] != num_tiles:
        raise ValueError(
            f"Phase-2 trace has {trace_noise.shape[1]} tiles, Phase 3/4 config "
            f"expects {num_tiles}."
        )

    selected_timesteps = select_representative_timesteps(
        trace_noise,
        trace_faulted,
        override=args.timesteps or evaluation_cfg.get("timesteps"),
    )
    trace_seed = _extract_seed(phase2_metadata, fallback=int(phase2_cfg.get("trace_seed", seed)))
    placement_seed = _extract_seed(
        phase3_metadata,
        fallback=int(experiment_cfg.get("placement_seed", seed)),
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(model_cfg.get("name", "gpt2"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    reference_model = AutoModelForCausalLM.from_pretrained(model_name).float().to(device)
    noisy_model = AutoModelForCausalLM.from_pretrained(model_name).float().to(device)
    for model in (reference_model, noisy_model):
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.eval()

    batches, dataset_metadata = build_batches(dataset_cfg, tokenizer)
    strict_compatibility = bool(
        evaluation_cfg.get("strict_phase1_compatibility", args.max_tokens is None)
    )
    _validate_dataset_compatibility(
        phase1_payload,
        dataset_metadata,
        strict=strict_compatibility,
    )

    digital_nll, digital_ppl = compute_nll(reference_model, batches, device)
    phase1_baseline = phase1_payload.get("baseline", {})
    phase1_clean_ppl = phase1_baseline.get("clean_ppl") if isinstance(phase1_baseline, Mapping) else None
    if phase1_clean_ppl is not None and not math.isclose(
        digital_ppl,
        float(phase1_clean_ppl),
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        message = (
            f"Phase-4 digital PPL {digital_ppl:.8f} does not match Phase-1 "
            f"PPL {float(phase1_clean_ppl):.8f}."
        )
        if strict_compatibility:
            raise ValueError(message)
        LOGGER.warning(message)

    n_layers = int(reference_model.config.n_layer)
    hidden_size = int(reference_model.config.n_embd)
    inner_size = int(reference_model.config.n_inner or 4 * hidden_size)
    if n_layers != 12 or hidden_size != 768 or inner_size != 3072:
        raise ValueError(
            "This Phase-4 bridge currently targets GPT-2 small: "
            f"received layers={n_layers}, hidden={hidden_size}, inner={inner_size}."
        )

    policies = list(
        evaluation_cfg.get("policies", list(DEFAULT_PLACEMENT_FILENAMES))
    )
    placement_filenames = dict(DEFAULT_PLACEMENT_FILENAMES)
    placement_filenames.update(phase3_cfg.get("placement_filenames", {}) or {})
    canonical_shapes = expected_gpt2_module_shapes(
        num_hidden_layers=n_layers,
        hidden_size=hidden_size,
        inner_size=inner_size,
    )
    assignments_by_policy: dict[str, list[GPT2ShardAssignment]] = {}
    baseline_signature: set[tuple[Any, ...]] | None = None
    for policy in policies:
        if policy not in placement_filenames:
            raise ValueError(f"No placement filename configured for policy {policy!r}.")
        rows = load_placement_csv(phase3_dir / placement_filenames[policy])
        assignments = build_shard_assignments(
            rows,
            policy=policy,
            placement_seed=placement_seed,
            tier_rows=tier_rows,
            tier_cols=tier_cols,
            num_tiles=num_tiles,
            tiers_per_tile=tiers_per_tile,
            num_hidden_layers=n_layers,
            hidden_size=hidden_size,
            inner_size=inner_size,
        )
        validate_shard_coverage(assignments, canonical_shapes=canonical_shapes)
        signature = _assignment_signature(assignments)
        if baseline_signature is None:
            baseline_signature = signature
        elif signature != baseline_signature:
            raise ValueError(
                f"Policy {policy!r} maps a different logical shard set. Policies "
                "may differ only in physical tile/tier assignment."
            )
        assignments_by_policy[policy] = assignments

    first_assignments = next(iter(assignments_by_policy.values()))
    hf_paths = sorted({assignment.hf_module_path for assignment in first_assignments})
    calibrations = load_phase1_noise_calibrations(phase1_path)
    missing_calibrations = set(hf_paths) - set(calibrations)
    if missing_calibrations:
        raise ValueError(f"Missing Phase-1 calibrations: {sorted(missing_calibrations)}.")

    analog_configuration = (
        phase1_payload.get("metadata", {}).get("analog_configuration", {})
        if isinstance(phase1_payload.get("metadata", {}), Mapping)
        else {}
    )
    phase1_tile_size = int(analog_configuration.get("tile_size", tier_rows))
    clip_sigma = float(analog_configuration.get("clip_sigma", 2.5))
    if phase1_tile_size != tier_rows or phase1_tile_size != tier_cols:
        raise ValueError(
            "Phase-1 AIHWKit tile size and Phase-3/4 tier shape differ: "
            f"Phase 1={phase1_tile_size}, Phase 3/4=({tier_rows}, {tier_cols})."
        )
    phase4_runtime_cfg = phase4_cfg.get("aihwkit", {}) or {}
    if not isinstance(phase4_runtime_cfg, Mapping):
        raise ValueError("phase4.aihwkit must be a mapping.")
    if str(phase4_runtime_cfg.get("forward_backend", "aihwkit")) != "aihwkit":
        raise ValueError("This runner requires phase4.aihwkit.forward_backend=aihwkit.")
    if bool(phase4_runtime_cfg.get("internal_programming_noise", False)):
        raise ValueError(
            "Phase 4 must disable AIHWKit internal programming noise because "
            "Phase-2/3 noise is materialized explicitly into the weights."
        )

    reference_analog = convert_gpt2_projections_to_aihwkit(
        reference_model,
        hf_paths,
        analog_configuration=analog_configuration,
    )
    noisy_analog = convert_gpt2_projections_to_aihwkit(
        noisy_model,
        hf_paths,
        analog_configuration=analog_configuration,
    )
    validate_reference_equivalence(reference_analog, noisy_analog)
    reference_weights = reference_weight_map(reference_analog)
    reference_conversion_metadata = {
        path: reference.metadata_row()
        for path, reference in reference_analog.items()
    }

    reference_nll, reference_ppl = compute_nll(reference_model, batches, device)
    noisy_reference_nll, noisy_reference_ppl = compute_nll(
        noisy_model, batches, device
    )
    reference_tolerance = float(
        phase4_runtime_cfg.get("reference_nll_tolerance", 1e-6)
    )
    if not math.isclose(
        noisy_reference_nll,
        reference_nll,
        rel_tol=0.0,
        abs_tol=reference_tolerance,
    ):
        raise RuntimeError(
            "The independently converted all-analog reference models do not "
            "produce the same NLL: "
            f"{reference_nll:.10f} vs {noisy_reference_nll:.10f}."
        )

    # With uniform tile noise, all policies must produce identical per-weight
    # sigma maps because policies differ only in physical placement.
    uniform_tile_noise = {tile_id: 0.023 for tile_id in tile_ids}
    uniform_sigma_maps: dict[str, dict[str, torch.Tensor]] = {}
    for policy, assignments in assignments_by_policy.items():
        uniform_sigma_maps[policy] = build_sigma_map(
            reference_weights,
            assignments,
            tile_noise_at_timestep=uniform_tile_noise,
            calibrations=calibrations,
        )
    first_policy = policies[0]
    for policy in policies[1:]:
        for path in hf_paths:
            if not torch.equal(
                uniform_sigma_maps[first_policy][path],
                uniform_sigma_maps[policy][path],
            ):
                raise RuntimeError(
                    "Uniform-noise policy invariance failed for "
                    f"{first_policy} vs {policy} at {path}."
                )

    LOGGER.info(
        "Digital NLL/PPL %.8f / %.6f | all-analog AIHWKit reference %.8f / %.6f",
        digital_nll,
        digital_ppl,
        reference_nll,
        reference_ppl,
    )

    save_assignments = bool(output_cfg.get("save_assignment_csv", True))
    if save_assignments:
        write_csv(
            [
                assignment.to_row()
                for assignments in assignments_by_policy.values()
                for assignment in assignments
            ],
            output_dir / "projection_noise_assignments.csv",
        )

    compute_kl = bool(evaluation_cfg.get("compute_kl", True))
    compute_agreement = bool(evaluation_cfg.get("compute_agreement", True))
    unavailable_action = str(evaluation_cfg.get("unavailable_action", "error"))
    if unavailable_action not in {"error", "skip"}:
        raise ValueError("unavailable_action must be 'error' or 'skip'.")

    quality_records: list[QualityMetrics] = []
    injection_rows: list[dict[str, Any]] = []
    checksum_rows: list[dict[str, Any]] = []

    tile_index = {tile_id: index for index, tile_id in enumerate(tile_ids)}
    for realization in range(num_realizations):
        realization_seed = seed + realization * seed_stride
        paired_noise = generate_paired_noise_tensors(
            reference_weights,
            hf_paths,
            seed=realization_seed,
        )

        for timestep in selected_timesteps:
            tile_noise = {
                tile_id: float(trace_noise[timestep, index])
                for tile_id, index in tile_index.items()
            }
            tile_faulted = {
                tile_id: bool(trace_faulted[timestep, index])
                for tile_id, index in tile_index.items()
            }
            tile_available = {
                tile_id: bool(trace_available[timestep, index])
                for tile_id, index in tile_index.items()
            }

            for policy, assignments in assignments_by_policy.items():
                unavailable = [
                    assignment
                    for assignment in assignments
                    if not tile_available[assignment.tile_id]
                ]
                if unavailable:
                    message = (
                        f"Policy {policy} at timestep {timestep} uses "
                        f"{len(unavailable)} unavailable shards."
                    )
                    if unavailable_action == "error":
                        raise RuntimeError(message)
                    LOGGER.warning("Skipping: %s", message)
                    continue

                sigma_maps = build_sigma_map(
                    reference_weights,
                    assignments,
                    tile_noise_at_timestep=tile_noise,
                    calibrations=calibrations,
                )
                total_weights = sum(item.weights_in_shard for item in assignments)
                mean_noise = sum(
                    item.weights_in_shard * tile_noise[item.tile_id]
                    for item in assignments
                ) / total_weights
                n_faulted = sum(
                    1 for item in assignments if tile_faulted[item.tile_id]
                )
                n_unavailable = sum(
                    1 for item in assignments if not tile_available[item.tile_id]
                )

                with NoisedModelContext(
                    noisy_model,
                    assignments,
                    sigma_maps,
                    paired_noise,
                    noisy_analog,
                ) as context:
                    if compute_kl or compute_agreement:
                        metrics = evaluate_quality_pair(
                            reference_model,
                            noisy_model,
                            batches,
                            device,
                            compute_kl=compute_kl,
                            compute_agreement=compute_agreement,
                        )
                    else:
                        noisy_nll, noisy_ppl = compute_nll(
                            noisy_model, batches, device
                        )
                        metrics = {
                            "reference_nll": reference_nll,
                            "reference_ppl": reference_ppl,
                            "nll": noisy_nll,
                            "ppl": noisy_ppl,
                            "delta_nll": noisy_nll - reference_nll,
                            "delta_ppl": noisy_ppl - reference_ppl,
                            "kl_divergence": 0.0,
                            "next_token_agreement": 0.0,
                        }
                    noisy_snapshot = dict(context.noisy_weights)
                    saved_before = dict(context.saved_weights)

                if not math.isclose(
                    metrics["reference_nll"],
                    reference_nll,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        "Streaming reference NLL changed during evaluation: "
                        f"{metrics['reference_nll']} vs {reference_nll}."
                    )

                if bool(output_cfg.get("save_checksums", True)):
                    saved_after = save_projection_weights(noisy_model, hf_paths)
                    checksum_records = compute_weight_checksums(
                        noisy_model,
                        hf_paths,
                        saved_before=saved_before,
                        noisy_weights=noisy_snapshot,
                        saved_after_restore=saved_after,
                    )
                    for checksum in checksum_records:
                        if not checksum.weights_match_original:
                            raise RuntimeError(
                                f"Weight restoration failed for {checksum.hf_module_path}."
                            )
                        row = checksum.to_row()
                        row.update(
                            {
                                "policy": policy,
                                "timestep": timestep,
                                "noise_realization": realization,
                                "noise_realization_seed": realization_seed,
                            }
                        )
                        checksum_rows.append(row)

                quality_records.append(
                    QualityMetrics(
                        policy=policy,
                        timestep=timestep,
                        noise_realization=realization,
                        noise_realization_seed=realization_seed,
                        trace_seed=trace_seed,
                        placement_seed=placement_seed,
                        digital_nll=digital_nll,
                        digital_ppl=digital_ppl,
                        reference_nll=reference_nll,
                        reference_ppl=reference_ppl,
                        nll=metrics["nll"],
                        ppl=metrics["ppl"],
                        delta_nll=metrics["delta_nll"],
                        delta_ppl=metrics["delta_ppl"],
                        kl_divergence=metrics["kl_divergence"],
                        next_token_agreement=metrics["next_token_agreement"],
                        num_faulted_shards=n_faulted,
                        num_unavailable_shards=n_unavailable,
                        mean_tile_noise_normalized=float(mean_noise),
                        hardware_state_mode="operational_degradation",
                        unavailable_action=unavailable_action,
                    )
                )

                if bool(output_cfg.get("save_injection_records", True)):
                    injection_rows.extend(
                        record.to_row()
                        for record in build_noise_injection_records(
                            assignments,
                            tile_noise_at_timestep=tile_noise,
                            tile_faulted_at_timestep=tile_faulted,
                            tile_available_at_timestep=tile_available,
                            calibrations=calibrations,
                            timestep=timestep,
                            noise_realization=realization,
                            noise_realization_seed=realization_seed,
                            unavailable_action=unavailable_action,
                        )
                    )

                latest = quality_records[-1]
                LOGGER.info(
                    "t=%d real=%d policy=%s NLL=%.6f PPL=%.4f DeltaNLL=%.6f "
                    "DeltaPPL=%.4f KL=%.6f agree=%.4f",
                    timestep,
                    realization,
                    policy,
                    latest.nll,
                    latest.ppl,
                    latest.delta_nll,
                    latest.delta_ppl,
                    latest.kl_divergence,
                    latest.next_token_agreement,
                )

    if not quality_records:
        raise RuntimeError("No Phase-4 quality records were produced.")

    policy_pairs = [
        tuple(pair)
        for pair in evaluation_cfg.get(
            "policy_pairs",
            [
                ["hardware_only", "static_sensitivity"],
                ["random", "static_sensitivity"],
                ["sequential", "static_sensitivity"],
            ],
        )
    ]
    paired = compute_paired_differences(quality_records, policy_pairs)

    write_csv([record.to_row() for record in quality_records], output_dir / "quality_by_policy.csv")
    write_csv(aggregate_quality(quality_records), output_dir / "quality_by_timestep.csv")
    write_csv([record.to_row() for record in paired], output_dir / "paired_policy_differences.csv")
    write_csv(aggregate_paired_differences(paired), output_dir / "paired_policy_summary.csv")
    if injection_rows:
        write_csv(injection_rows, output_dir / "tile_noise_injection_records.csv")
    if checksum_rows:
        write_csv(checksum_rows, output_dir / "weight_checksums.csv")
    write_json(reference_conversion_metadata, output_dir / "reference_analog_conversion.json")

    effective_config = dict(config)
    effective_config["resolved"] = {
        "phase1_results": phase1_path,
        "phase2_trace": phase2_trace_path,
        "phase2_metadata": phase2_metadata_path,
        "phase3_directory": phase3_dir,
        "output_directory": output_dir,
        "noise_base_seed": seed,
        "trace_seed": trace_seed,
        "placement_seed": placement_seed,
        "num_realizations": num_realizations,
        "selected_timesteps": selected_timesteps,
        "device": str(device),
        "num_tiles": num_tiles,
        "tiers_per_tile": tiers_per_tile,
        "tier_shape": [tier_rows, tier_cols],
        "clip_sigma": clip_sigma,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_json_ready(effective_config), stream, sort_keys=False)

    import aihwkit

    metadata = {
        "phase": "phase4_quality",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "aihwkit": getattr(aihwkit, "__version__", None),
        },
        "model": model_name,
        "dataset": dataset_metadata,
        "digital_baseline": {"nll": digital_nll, "ppl": digital_ppl},
        "phase4_reference": {
            "mode": "all_transformer_projections_aihwkit_analog_without_tile_noise",
            "clip_sigma": clip_sigma,
            "internal_programming_noise": False,
            "internal_read_noise": False,
            "internal_drift": False,
            "nll": reference_nll,
            "ppl": reference_ppl,
        },
        "hardware": {
            "num_tiles": num_tiles,
            "tiers_per_tile": tiers_per_tile,
            "tier_rows": tier_rows,
            "tier_cols": tier_cols,
        },
        "evaluation": {
            "policies": policies,
            "timesteps": selected_timesteps,
            "num_realizations": num_realizations,
            "paired_noise_across_policies": True,
            "paired_noise_across_timesteps": True,
            "unavailable_action": unavailable_action,
            "forward_backend": "aihwkit",
            "all_transformer_projections_analog": True,
            "embeddings_analog": False,
            "lm_head_analog": False,
            "freeze_noise_during_dataset_pass": True,
            "reference_model_equivalence_passed": True,
            "uniform_noise_policy_invariance_passed": True,
            "reference_nll_tolerance": reference_tolerance,
        },
    }
    write_json(metadata, output_dir / "metadata.json")
    LOGGER.info("Phase 4 complete: %s", output_dir)


if __name__ == "__main__":
    main()
