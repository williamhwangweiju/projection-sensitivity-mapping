#!/usr/bin/env python3
"""Run Phase 3 static baselines using the IBM 3D-CIM simulator."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from threedsim.accelerator import Accelerator, AcceleratorConfig  # noqa: E402
from threedsim.inference import fast_trace_decoder, schedule_execution  # noqa: E402
from threedsim.mapping import MapStrategy, Mapper, Strategy  # noqa: E402
from threedsim.models import DecoderOnlyTransformer  # noqa: E402
from threedsim.modules import TransformerDecoderLayer  # noqa: E402
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable, make_use_linear  # noqa: E402

from src.mapping.objective import build_policy_summary, evaluate_placement_over_trace  # noqa: E402
from src.mapping.placement import (  # noqa: E402
    build_placement_from_mapped_shards,
    mapped_shards_with_placement_to_rows,
)
from src.mapping.projection_catalog import (  # noqa: E402
    MappedModuleSpec,
    build_group_total_weights,
    build_mapped_module_specs,
    iter_mappable_modules,
    load_phase1_sensitivity_lookup,
    mapped_module_specs_to_rows,
    order_modules_for_policy,
)
from src.mapping.sharding import (  # noqa: E402
    extract_shards_from_3dcim_mapping,
    mapped_shard_records_to_placement_rows,
)
from src.simulators.tile_fidelity import TileFidelityTrace  # noqa: E402


LOGGER = logging.getLogger("phase3_baselines")

DEFAULT_OUTPUT_ROOT = Path("data/results/phase3_baselines")
CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.json"
PROJECTION_CATALOG_FILENAME = "projection_catalog.csv"
PROJECTION_SHARDS_FILENAME = "projection_shards.csv"
TIMESTEP_METRICS_FILENAME = "timestep_metrics.csv"
POLICY_SUMMARY_FILENAME = "policy_summary.csv"
PLACEMENT_FILENAMES = {
    "random": "placement_random.csv",
    "sequential": "placement_sequential.csv",
    "hardware_only": "placement_hardware_only.csv",
    "static_sensitivity": "placement_static_sensitivity.csv",
}



def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: IBM 3D-CIM static mapping baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--phase1-results", type=Path, default=None)
    parser.add_argument("--phase2-trace", type=Path, default=None)
    parser.add_argument("--phase2-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-lm-head", action="store_true")
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_optional_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("Phase 3 config file must contain a YAML mapping.")
    return dict(loaded)


def resolve_latest_phase1_results() -> Path:
    candidates = sorted(
        (REPOSITORY_ROOT / "data/results/phase1_sensitivity").glob("*.json"),
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No Phase 1 results found in data/results/phase1_sensitivity."
        )
    return candidates[-1]


def resolve_latest_phase2_trace() -> Path:
    candidates = sorted(
        (REPOSITORY_ROOT / "data/results/phase2_fidelity/fidelity_traces").glob(
            "*/seed_*/trace.npz"
        ),
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No Phase 2 traces found in data/results/phase2_fidelity/fidelity_traces."
        )
    return candidates[-1]


def resolve_output_directory(
    *,
    cli_output_dir: Path | None,
    config: Mapping[str, Any],
    experiment_name: str,
    seed: int,
) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir.expanduser().resolve()

    phase3_cfg = config.get("phase3", {}) or {}
    if not isinstance(phase3_cfg, Mapping):
        raise ValueError("phase3 must be a mapping.")
    raw_output_root = phase3_cfg.get(
        "output_root",
        DEFAULT_OUTPUT_ROOT,
    )
    output_root = Path(raw_output_root).expanduser()
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    return output_root.resolve() / experiment_name / f"seed_{seed}"

def prepare_output_directory(output_directory: Path, *, overwrite: bool) -> None:
    if output_directory.exists():
        if not output_directory.is_dir():
            raise ValueError(
                f"Output path exists but is not a directory: {output_directory}"
            )
        if any(output_directory.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "Output directory already exists and is not empty: "
                    f"{output_directory}. Use --overwrite."
                )
            shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV output: {output_path}")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_yaml(data: Mapping[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            _json_ready(data),
            file,
            sort_keys=False,
            default_flow_style=False,
        )


def save_json(data: Mapping[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(data), file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value



def get_hardware_mapping(
    trace: TileFidelityTrace,
    phase2_metadata: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if "hardware" in trace.metadata and isinstance(trace.metadata["hardware"], Mapping):
        return dict(trace.metadata["hardware"])

    if phase2_metadata is not None:
        model_metadata = phase2_metadata.get("model_metadata", {})
        if isinstance(model_metadata, Mapping):
            hardware = model_metadata.get("hardware")
            if isinstance(hardware, Mapping):
                return dict(hardware)

    hardware_cfg = config.get("hardware")
    if isinstance(hardware_cfg, Mapping):
        return dict(hardware_cfg)

    raise ValueError("Unable to resolve hardware mapping from trace metadata or config.")


def resolve_reference_noise_std(
    *,
    cli_override: float | None,
    config: Mapping[str, Any],
    trace: TileFidelityTrace,
) -> float:
    if cli_override is not None:
        if cli_override <= 0.0:
            raise ValueError("reference_noise_std override must be positive.")
        return float(cli_override)

    phase3_cfg = config.get("phase3", {})
    if isinstance(phase3_cfg, Mapping):
        raw_value = phase3_cfg.get("reference_noise_std")
        if raw_value is not None:
            value = float(raw_value)
            if value <= 0.0:
                raise ValueError("phase3.reference_noise_std must be positive.")
            return value

    fidelity_metadata = trace.metadata.get("fidelity_model")
    if isinstance(fidelity_metadata, Mapping):
        value = fidelity_metadata.get("reference_noise_std")
        if value is not None:
            value = float(value)
            if value <= 0.0:
                raise ValueError("trace metadata reference_noise_std must be positive.")
            return value

    raise ValueError(
        "Unable to resolve reference_noise_std from CLI/config/trace metadata."
    )


def build_effective_config(
    *,
    original_config: Mapping[str, Any],
    phase1_results_path: Path,
    phase2_trace_path: Path,
    phase2_metadata_path: Path | None,
    output_directory: Path,
    experiment_name: str,
    seed: int,
    include_lm_head: bool,
    reference_noise_std: float,
) -> dict[str, Any]:
    effective_config = copy.deepcopy(dict(original_config))

    experiment_cfg = effective_config.get("experiment")
    if experiment_cfg is None:
        experiment_cfg = {}
        effective_config["experiment"] = experiment_cfg
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = dict(experiment_cfg)
        effective_config["experiment"] = experiment_cfg
    experiment_cfg["name"] = experiment_name
    experiment_cfg["seed"] = seed
    experiment_cfg["resolved_output_directory"] = str(output_directory)

    phase1_cfg = effective_config.get("phase1")
    if phase1_cfg is None:
        phase1_cfg = {}
        effective_config["phase1"] = phase1_cfg
    if not isinstance(phase1_cfg, dict):
        phase1_cfg = dict(phase1_cfg)
        effective_config["phase1"] = phase1_cfg
    phase1_cfg["results_path"] = str(phase1_results_path)

    phase2_cfg = effective_config.get("phase2")
    if phase2_cfg is None:
        phase2_cfg = {}
        effective_config["phase2"] = phase2_cfg
    if not isinstance(phase2_cfg, dict):
        phase2_cfg = dict(phase2_cfg)
        effective_config["phase2"] = phase2_cfg
    phase2_cfg["trace_path"] = str(phase2_trace_path)
    if phase2_metadata_path is not None:
        phase2_cfg["metadata_path"] = str(phase2_metadata_path)

    phase3_cfg = effective_config.get("phase3")
    if phase3_cfg is None:
        phase3_cfg = {}
        effective_config["phase3"] = phase3_cfg
    if not isinstance(phase3_cfg, dict):
        phase3_cfg = dict(phase3_cfg)
        effective_config["phase3"] = phase3_cfg
    phase3_cfg["include_lm_head"] = include_lm_head
    phase3_cfg["reference_noise_std"] = reference_noise_std

    return effective_config


def make_model(
    *,
    vocab_size: int = 50257,
    d_model: int = 768,
    num_layers: int = 12,
    nhead: int = 12,
    dim_feedforward: int = 3072,
    max_seq_length: int = 12,
) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        TransformerDecoderLayer,
        num_layers=num_layers,
        decoder_layer_kwargs={
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
        },
        embedding_layer_kwargs={
            "vocab_size": vocab_size,
            "embedding_dim": d_model,
            "max_seq_length": max_seq_length,
        },
        device="meta",
    )


def build_tile_priority(
    trace: TileFidelityTrace,
    *,
    policy_name: str,
    seed: int,
) -> list[int]:
    tile_ids = [
        int(tile_id)
        for tile_id in trace.tile_ids.tolist()
        if bool(trace.available[0, int(tile_id)])
    ]

    if not tile_ids:
        raise ValueError("No tiles are available at timestep 0.")

    if policy_name == "random":
        rng = random.Random(seed)
        rng.shuffle(tile_ids)
        return tile_ids

    if policy_name in {"hardware_only", "static_sensitivity"}:
        noise = trace.noise_std[0]
        return sorted(tile_ids, key=lambda tile_id: (float(noise[tile_id]), tile_id))

    return tile_ids


def map_model_for_policy(
    *,
    model: Any,
    accelerator: Accelerator,
    policy_name: str,
    sensitivity_lookup: Mapping[tuple[str, str], float],
    tile_priority: list[int],
    seed: int,
    reference_noise_std: float,
    num_hidden_layers: int,
) -> list[MappedModuleSpec]:
    """Map only the analog transformer projections.

    Token/position embeddings and lm_head remain digital and are excluded from
    both accelerator capacity and Phase-4 weight injection.
    """
    mapper = Mapper(
        accelerator=accelerator,
        model=model,
        map_strategy=MapStrategy(
            strategy=Strategy.GREEDY_IN_ORDER,
            split_ffn=True,
            stack_embedding=False,
        ),
    )

    modules = iter_mappable_modules(
        model,
        include_embeddings=False,
        include_lm_head=False,
    )
    specs_by_name = build_mapped_module_specs(
        modules,
        sensitivity_lookup=sensitivity_lookup,
        include_lm_head=False,
        reference_noise_std=reference_noise_std,
        num_hidden_layers=num_hidden_layers,
    )
    group_total_weights = build_group_total_weights(list(specs_by_name.values()))
    ordered_modules = order_modules_for_policy(
        modules,
        specs_by_name=specs_by_name,
        group_total_weights=group_total_weights,
        policy_name=policy_name,
        seed=seed,
    )

    mapped_specs: list[MappedModuleSpec] = []
    for module_name, module in ordered_modules:
        mapping, _ = mapper.shape_to_mapping(
            inp_shape=tuple(module.weight.shape),
            utilization=1.0,
            tile_indices=tile_priority,
        )
        module.set_mapping(mapping)
        mapped_specs.append(specs_by_name[module_name])

    return mapped_specs

def build_effective_model(
    *,
    accelerator: Accelerator,
    num_layers: int,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    vocab_size: int,
    target_len: int,
) -> DecoderOnlyTransformer:
    model = DecoderOnlyTransformer(
        TransformerDecoderLayer,
        num_layers=num_layers,
        decoder_layer_kwargs={
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
        },
        embedding_layer_kwargs={
            "vocab_size": vocab_size,
            "embedding_dim": d_model,
            "max_seq_length": target_len,
        },
        device="meta",
    )
    assign_acc(model, accelerator)
    return model


def run_policy(
    *,
    policy_name: str,
    trace: TileFidelityTrace,
    sensitivity_lookup: Mapping[tuple[str, str], float],
    accelerator_cfg: AcceleratorConfig,
    seed: int,
    trace_seed: int,
    model_cfg: Mapping[str, int],
    inference_cfg: Mapping[str, int],
    reference_noise_std: float,
    run_performance_simulation: bool,
) -> dict[str, Any]:
    accelerator = Accelerator(accelerator_cfg, device="meta")
    model = build_effective_model(
        accelerator=accelerator,
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        dim_feedforward=model_cfg["dim_feedforward"],
        vocab_size=model_cfg["vocab_size"],
        target_len=inference_cfg["target_len"],
    )

    tile_priority = build_tile_priority(trace, policy_name=policy_name, seed=seed)
    available_tiles_at_mapping = set(tile_priority)
    mapped_specs = map_model_for_policy(
        model=model,
        accelerator=accelerator,
        policy_name=policy_name,
        sensitivity_lookup=sensitivity_lookup,
        tile_priority=tile_priority,
        seed=seed,
        reference_noise_std=reference_noise_std,
        num_hidden_layers=model_cfg["num_layers"],
    )

    mapped_shard_records = extract_shards_from_3dcim_mapping(
        model=model,
        mapped_specs=mapped_specs,
        tier_rows=int(accelerator_cfg.tier_shape[0]),
        tier_cols=int(accelerator_cfg.tier_shape[1]),
    )
    expected_tiers = int(model_cfg["num_layers"]) * 40
    if len(mapped_shard_records) != expected_tiers:
        raise ValueError(
            f"Expected {expected_tiers} GPT-2 projection tiers, extracted "
            f"{len(mapped_shard_records)}. Check Q/K/V and FFN sharding."
        )
    total_capacity = int(accelerator_cfg.tiles) * int(accelerator_cfg.tiers)
    if len(mapped_shard_records) > total_capacity:
        raise ValueError(
            f"Projection demand {len(mapped_shard_records)} exceeds capacity "
            f"{total_capacity}."
        )

    shards = [record.shard for record in mapped_shard_records]
    placement = build_placement_from_mapped_shards(
        policy_name=policy_name,
        records=mapped_shard_records,
        tiers_per_tile=accelerator_cfg.tiers,
        num_tiles=accelerator_cfg.tiles,
        available_tiles=available_tiles_at_mapping,
    )
    placement_rows = mapped_shard_records_to_placement_rows(
        policy_name=policy_name,
        records=mapped_shard_records,
        trace_seed=trace_seed,
        placement_seed=seed,
    )

    quality_rows = evaluate_placement_over_trace(
        placement=placement,
        shards=shards,
        trace=trace,
        reference_noise_std=float(reference_noise_std),
    )
    summary = build_policy_summary(
        timestep_rows=quality_rows,
        placements={policy_name: placement},
    )[0]

    simulator = {
        "execution_time_ns": None,
        "energy_nj": None,
        "peak_memory_bytes": None,
        "flops": None,
        "energy_breakdown": None,
        "latency_breakdown": None,
    }
    if run_performance_simulation:
        # Performance simulation is optional because this unified quality
        # workflow deliberately keeps embeddings and lm_head digital.  Enable
        # only after the local 3D-SiM fork models those digital operations.
        fill_name_fields(model)
        make_traceable(model, is_traceable=True)
        make_use_linear(model, use_linear=True)
        fast_traced = fast_trace_decoder(
            model,
            start_len=inference_cfg["start_len"],
            target_len=inference_cfg["target_len"],
            bsz=inference_cfg["batch_size"],
        )
        (
            execution_time_ns,
            scratchpad_memory,
            peak_memory_bytes,
            energy_nj,
            flops,
            energy_breakdown,
            latency_breakdown,
        ) = schedule_execution(
            fast_traced.graph,
            accelerator=model.accelerator,
            copy_and_cleanup_graph=False,
            communication=True,
        )
        simulator = {
            "execution_time_ns": int(execution_time_ns),
            "energy_nj": float(energy_nj),
            "peak_memory_bytes": float(peak_memory_bytes),
            "flops": float(flops),
            "scratchpad_memory_trace": _json_ready(scratchpad_memory),
            "energy_breakdown": _json_ready(energy_breakdown),
            "latency_breakdown": _json_ready(latency_breakdown),
        }

    summary.update(
        {
            "simulator_execution_time_ns": simulator["execution_time_ns"],
            "simulator_energy_nj": simulator["energy_nj"],
            "simulator_peak_memory_bytes": simulator["peak_memory_bytes"],
            "simulator_flops": simulator["flops"],
            "lm_head_quality_included": False,
            "embeddings_analog_mapped": False,
            "available_tiles_at_mapping": len(available_tiles_at_mapping),
            "required_projection_tiers": len(mapped_shard_records),
            "total_tier_capacity": total_capacity,
        }
    )

    return {
        "projection_catalog_rows": mapped_module_specs_to_rows(mapped_specs),
        "placement_rows": placement_rows,
        "mapped_shard_records": mapped_shard_records,
        "placement": placement,
        "quality_rows": quality_rows,
        "policy_summary": summary,
        "simulator": simulator,
    }

def run_experiment(args: argparse.Namespace) -> Path:
    config = load_optional_yaml(args.config)

    phase1_cfg = config.get("phase1", {})
    phase2_cfg = config.get("phase2", {})
    if phase1_cfg is None:
        phase1_cfg = {}
    if phase2_cfg is None:
        phase2_cfg = {}
    if not isinstance(phase1_cfg, Mapping) or not isinstance(phase2_cfg, Mapping):
        raise ValueError("phase1 and phase2 sections must be mappings.")

    phase1_results_path = (
        args.phase1_results
        or (Path(phase1_cfg["results_path"]) if phase1_cfg.get("results_path") else None)
        or resolve_latest_phase1_results()
    )
    phase2_trace_path = (
        args.phase2_trace
        or (Path(phase2_cfg["trace_path"]) if phase2_cfg.get("trace_path") else None)
        or resolve_latest_phase2_trace()
    )
    phase2_metadata_path = (
        args.phase2_metadata
        or (Path(phase2_cfg["metadata_path"]) if phase2_cfg.get("metadata_path") else None)
    )
    if phase2_metadata_path is None:
        candidate = phase2_trace_path.with_name("metadata.json")
        phase2_metadata_path = candidate if candidate.exists() else None

    phase2_metadata = None
    if phase2_metadata_path is not None:
        with phase2_metadata_path.expanduser().resolve().open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if not isinstance(loaded, Mapping):
            raise ValueError("Phase 2 metadata JSON must contain a mapping.")
        phase2_metadata = dict(loaded)

    experiment_cfg = config.get("experiment", {}) or {}
    phase3_cfg = config.get("phase3", {}) or {}
    if not isinstance(experiment_cfg, Mapping) or not isinstance(phase3_cfg, Mapping):
        raise ValueError("experiment and phase3 sections must be mappings.")
    seed = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    experiment_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(phase3_cfg.get("name", "phase3_baselines")).strip(),
    ).strip("._-")
    if not experiment_name:
        raise ValueError("experiment.name does not contain usable path characters.")

    output_directory = resolve_output_directory(
        cli_output_dir=args.output_dir,
        config=config,
        experiment_name=experiment_name,
        seed=seed,
    )
    prepare_output_directory(output_directory, overwrite=args.overwrite)

    trace = TileFidelityTrace.load_npz(phase2_trace_path)
    noise_unit = str(trace.metadata.get("noise_unit", ""))
    if noise_unit != "pcmlike_prog_noise_scale_equivalent":
        raise ValueError(
            "Phase-2 trace noise_unit must be "
            "'pcmlike_prog_noise_scale_equivalent', got "
            f"{noise_unit!r}."
        )
    sensitivity_lookup = load_phase1_sensitivity_lookup(phase1_results_path)
    hardware_mapping = get_hardware_mapping(trace, phase2_metadata, config)
    reference_noise_std = resolve_reference_noise_std(
        cli_override=None,
        config=config,
        trace=trace,
    )
    include_lm_head = bool(args.include_lm_head or phase3_cfg.get("include_lm_head", False))
    if include_lm_head:
        raise ValueError("Unified pipeline requires include_lm_head=false.")

    raw_tier_shape = hardware_mapping.get(
        "tier_shape",
        {"rows": 512, "cols": 512},
    )

    if isinstance(raw_tier_shape, Mapping):
        tier_rows = int(raw_tier_shape.get("rows", 512))
        tier_cols = int(raw_tier_shape.get("cols", 512))
    else:
        if len(raw_tier_shape) != 2:
            raise ValueError(
                "hardware.tier_shape must contain exactly two values: "
                "[rows, cols]."
            )
        tier_rows = int(raw_tier_shape[0])
        tier_cols = int(raw_tier_shape[1])

    accelerator_cfg = AcceleratorConfig(
        tiles=int(
            hardware_mapping.get(
                "num_tiles",
                hardware_mapping.get(
                    "tiles",
                    trace.noise_std.shape[1],
                ),
            )
        ),
        tiers=int(
            hardware_mapping.get(
                "tiers_per_tile",
                hardware_mapping.get("tiers", 8),
            )
        ),
        tier_shape=(tier_rows, tier_cols),
        kv_caching=bool(phase3_cfg.get("kv_caching", False)),
    )

    model_cfg = {
        "num_layers": int(phase3_cfg.get("num_layers", 12)),
        "d_model": int(phase3_cfg.get("d_model", 768)),
        "nhead": int(phase3_cfg.get("nhead", 12)),
        "dim_feedforward": int(phase3_cfg.get("dim_feedforward", 3072)),
        "vocab_size": int(phase3_cfg.get("vocab_size", 50257)),
    }
    inference_cfg = {
        "batch_size": int(phase3_cfg.get("batch_size", 1)),
        "start_len": int(phase3_cfg.get("start_len", 1)),
        "target_len": int(phase3_cfg.get("target_len", 12)),
    }

    policies = ["random", "sequential", "hardware_only", "static_sensitivity"]
    all_quality_rows: list[dict[str, Any]] = []
    policy_summary_rows: list[dict[str, Any]] = []
    all_shard_rows: list[dict[str, Any]] = []
    projection_catalog_rows: list[dict[str, Any]] | None = None

    for policy_name in policies:
        LOGGER.info("Running policy: %s", policy_name)
        policy_result = run_policy(
            policy_name=policy_name,
            trace=trace,
            sensitivity_lookup=sensitivity_lookup,
            accelerator_cfg=accelerator_cfg,
            seed=seed,
            model_cfg=model_cfg,
            inference_cfg=inference_cfg,
            trace_seed=int(trace.metadata.get("seed", seed)),
            reference_noise_std=reference_noise_std,
            run_performance_simulation=bool(
                phase3_cfg.get("run_performance_simulation", False)
            ),
        )

        if projection_catalog_rows is None:
            projection_catalog_rows = policy_result["projection_catalog_rows"]

        write_csv(
            policy_result["placement_rows"],
            output_directory / PLACEMENT_FILENAMES[policy_name],
        )

        placement = policy_result["placement"]
        all_shard_rows.extend(
            mapped_shards_with_placement_to_rows(
                policy_name=policy_name,
                records=policy_result["mapped_shard_records"],
                placement=placement,
            )
        )

        all_quality_rows.extend(policy_result["quality_rows"])
        policy_summary_rows.append(policy_result["policy_summary"])

    metadata = {
        "phase": "phase3_baselines",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "seed": seed,
        "inputs": {
            "phase1_results": str(phase1_results_path.expanduser().resolve()),
            "phase2_trace": str(phase2_trace_path.expanduser().resolve()),
            "phase2_metadata": (
                None
                if phase2_metadata_path is None
                else str(phase2_metadata_path.expanduser().resolve())
            ),
        },
        "summary": {
            "num_policies": len(policies),
            "num_timesteps": trace.num_timesteps,
            "num_tiles": accelerator_cfg.tiles,
            "tiers_per_tile": accelerator_cfg.tiers,
            "initial_available_tiles": int(np.sum(trace.available[0])),
            "reference_noise_std": reference_noise_std,
            "include_lm_head_in_quality_objective": False,
            "embeddings_and_lm_head": "digital",
            "sensitivity_score_unit": "delta_ppl_total",
            "proxy_noise_scaling": "variance_ratio_squared",
            "proxy_name": "sensitivity_weighted_variance_proxy",
        },
        "artifacts": {
            "projection_catalog": PROJECTION_CATALOG_FILENAME,
            "projection_shards": PROJECTION_SHARDS_FILENAME,
            "timestep_metrics": TIMESTEP_METRICS_FILENAME,
            "policy_summary": POLICY_SUMMARY_FILENAME,
            **PLACEMENT_FILENAMES,
        },
        "hardware": {
            "tiles": accelerator_cfg.tiles,
            "tiers": accelerator_cfg.tiers,
            "tier_shape": list(accelerator_cfg.tier_shape),
            "kv_caching": accelerator_cfg.kv_caching,
            "dram_bandwidth": accelerator_cfg.dram_bandwidth,
            "dram_active_power": accelerator_cfg.dram_active_power,
            "dram_inactive_power": accelerator_cfg.dram_inactive_power,
        },
    }
    save_json(metadata, output_directory / METADATA_FILENAME)

    if projection_catalog_rows is None:
        raise RuntimeError("No projection catalog rows were generated.")

    write_csv(projection_catalog_rows, output_directory / PROJECTION_CATALOG_FILENAME)
    write_csv(all_shard_rows, output_directory / PROJECTION_SHARDS_FILENAME)
    write_csv(all_quality_rows, output_directory / TIMESTEP_METRICS_FILENAME)
    write_csv(policy_summary_rows, output_directory / POLICY_SUMMARY_FILENAME)

    effective_config = build_effective_config(
        original_config=config,
        phase1_results_path=phase1_results_path.expanduser().resolve(),
        phase2_trace_path=phase2_trace_path.expanduser().resolve(),
        phase2_metadata_path=(
            None
            if phase2_metadata_path is None
            else phase2_metadata_path.expanduser().resolve()
        ),
        output_directory=output_directory,
        experiment_name=experiment_name,
        seed=seed,
        include_lm_head=include_lm_head,
        reference_noise_std=reference_noise_std,
    )
    save_yaml(effective_config, output_directory / CONFIG_FILENAME)

    LOGGER.info("Phase 3 completed.")
    LOGGER.info("Results saved to: %s", output_directory)
    return output_directory


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_arguments(arguments)
    configure_logging(args.log_level)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
