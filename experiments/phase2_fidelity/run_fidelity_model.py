#!/usr/bin/env python3
"""Run the Phase 2 heterogeneous tile-fidelity simulation.

This experiment:

1. Loads a Phase 2 YAML configuration.
2. Builds the hardware and tile-fidelity models.
3. Generates a deterministic time-varying fidelity trace.
4. Saves the trace as a compressed NPZ file.
5. Saves effective configuration and experiment metadata.
6. Produces per-tile and per-timestep CSV summaries.

Run from the repository root:

    python experiments/phase2_fidelity/run_fidelity_model.py \
        --config configs/phase2_fidelity/mixed.yaml

Override the configured seed:

    python experiments/phase2_fidelity/run_fidelity_model.py \
        --config configs/phase2_fidelity/mixed.yaml \
        --seed 7

By default, outputs are saved under:

    data/fidelity_traces/<experiment_name>/seed_<seed>/
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


# Allow this script to be executed directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from src.simulators.hardware import (  # noqa: E402
    HardwareConfigurationError,
)
from src.simulators.tile_fidelity import (  # noqa: E402
    FIDELITY_CLASS_TO_CODE,
    FIDELITY_CODE_TO_CLASS,
    FidelityConfigurationError,
    FidelitySimulationError,
    TileFidelityModel,
    TileFidelityTrace,
)


LOGGER = logging.getLogger("phase2_fidelity")

DEFAULT_OUTPUT_ROOT = Path("data/results/phase2_fidelity/fidelity_traces")
TRACE_FILENAME = "trace.npz"
CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.json"
TILE_SUMMARY_FILENAME = "tile_summary.csv"
TIMESTEP_SUMMARY_FILENAME = "timestep_summary.csv"


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a heterogeneous, time-varying hardware "
            "tile-fidelity trace."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Phase 2 YAML configuration.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the seed defined under experiment.seed in the "
            "configuration."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact output directory. When omitted, the runner uses "
            "<output_root>/<experiment_name>/seed_<seed>."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )

    return parser.parse_args(arguments)


def configure_logging(level: str) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
    )


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the top-level YAML configuration."""

    resolved_path = config_path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {resolved_path}"
        )

    with resolved_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if loaded is None:
        raise ValueError(
            f"Configuration file is empty: {resolved_path}"
        )

    if not isinstance(loaded, Mapping):
        raise ValueError(
            "The top-level YAML configuration must be a mapping."
        )

    return dict(loaded)


def get_experiment_name(
    config: Mapping[str, Any],
    config_path: Path,
) -> str:
    """Resolve a filesystem-safe experiment name."""

    experiment_config = config.get("experiment", {})

    if experiment_config is None:
        experiment_config = {}

    if not isinstance(experiment_config, Mapping):
        raise ValueError("The experiment section must be a mapping.")

    raw_name = experiment_config.get(
        "name",
        config_path.stem,
    )

    if not isinstance(raw_name, str):
        raise ValueError("experiment.name must be a string.")

    normalized_name = raw_name.strip()

    if not normalized_name:
        raise ValueError("experiment.name cannot be empty.")

    filesystem_safe_name = re.sub(
        pattern=r"[^A-Za-z0-9._-]+",
        repl="_",
        string=normalized_name,
    ).strip("._-")

    if not filesystem_safe_name:
        raise ValueError(
            "experiment.name does not contain any usable characters."
        )

    return filesystem_safe_name


def resolve_output_directory(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    experiment_name: str,
    seed: int,
    cli_output_dir: Path | None,
) -> Path:
    """Determine the directory for all experiment artifacts."""

    if cli_output_dir is not None:
        return cli_output_dir.expanduser().resolve()

    experiment_config = config.get("experiment", {})

    if experiment_config is None:
        experiment_config = {}

    if not isinstance(experiment_config, Mapping):
        raise ValueError("The experiment section must be a mapping.")

    raw_output_root = experiment_config.get(
        "output_root",
        DEFAULT_OUTPUT_ROOT,
    )

    if not isinstance(raw_output_root, (str, Path)):
        raise ValueError(
            "experiment.output_root must be a path-like value."
        )

    output_root = Path(raw_output_root).expanduser()

    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root

    return (
        output_root.resolve()
        / experiment_name
        / f"seed_{seed}"
    )


def prepare_output_directory(
    output_directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Create an empty output directory."""

    if output_directory.exists():
        if not output_directory.is_dir():
            raise ValueError(
                "Output path exists but is not a directory: "
                f"{output_directory}"
            )

        contains_files = any(output_directory.iterdir())

        if contains_files and not overwrite:
            raise FileExistsError(
                "Output directory already exists and is not empty: "
                f"{output_directory}\n"
                "Use --overwrite to replace it."
            )

        if contains_files and overwrite:
            LOGGER.warning(
                "Removing existing output directory: %s",
                output_directory,
            )
            shutil.rmtree(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)


def build_effective_config(
    original_config: Mapping[str, Any],
    *,
    seed: int,
    experiment_name: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Create the exact configuration associated with this run."""

    effective_config = copy.deepcopy(dict(original_config))

    experiment_config = effective_config.get("experiment")

    if experiment_config is None:
        experiment_config = {}
        effective_config["experiment"] = experiment_config

    if not isinstance(experiment_config, dict):
        experiment_config = dict(experiment_config)
        effective_config["experiment"] = experiment_config

    experiment_config["name"] = experiment_name
    experiment_config["seed"] = seed
    experiment_config["resolved_output_directory"] = str(
        output_directory
    )

    return effective_config


def save_yaml(
    data: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Save a mapping as readable YAML."""

    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(
            _json_ready(data),
            output_file,
            sort_keys=False,
            default_flow_style=False,
        )


def save_json(
    data: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Save a mapping as formatted JSON."""

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            _json_ready(data),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")


def write_tile_summary(
    trace: TileFidelityTrace,
    output_path: Path,
) -> None:
    """Write one summary row for every tile."""

    fieldnames = [
        "tile_id",
        "thermal_zone",
        "initial_fidelity_class",
        "base_noise_std",
        "drift_rate",
        "fault_onset_timestep",
        "fault_noise_increase_fraction",
        "initial_noise_std",
        "final_noise_std",
        "mean_noise_std",
        "max_noise_std",
        "initial_fidelity_score",
        "final_fidelity_score",
        "final_dynamic_fidelity_class",
        "final_available",
        "final_faulted",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for tile_index, tile_id_value in enumerate(
            trace.tile_ids
        ):
            initial_class_code = int(
                trace.initial_fidelity_class[tile_index]
            )
            final_class_code = int(
                trace.dynamic_fidelity_class[
                    -1,
                    tile_index,
                ]
            )

            fault_onset = int(
                trace.fault_onset_timestep[tile_index]
            )

            writer.writerow(
                {
                    "tile_id": int(tile_id_value),
                    "thermal_zone": int(
                        trace.thermal_zone[tile_index]
                    ),
                    "initial_fidelity_class": (
                        FIDELITY_CODE_TO_CLASS[
                            initial_class_code
                        ].value
                    ),
                    "base_noise_std": float(
                        trace.base_noise_std[tile_index]
                    ),
                    "drift_rate": float(
                        trace.drift_rate[tile_index]
                    ),
                    "fault_onset_timestep": (
                        fault_onset
                        if fault_onset >= 0
                        else ""
                    ),
                    "fault_noise_increase_fraction": float(
                        trace.fault_noise_increase_fraction[
                            tile_index
                        ]
                    ),
                    "initial_noise_std": float(
                        trace.noise_std[0, tile_index]
                    ),
                    "final_noise_std": float(
                        trace.noise_std[-1, tile_index]
                    ),
                    "mean_noise_std": float(
                        np.mean(
                            trace.noise_std[:, tile_index]
                        )
                    ),
                    "max_noise_std": float(
                        np.max(
                            trace.noise_std[:, tile_index]
                        )
                    ),
                    "initial_fidelity_score": float(
                        trace.fidelity_score[
                            0,
                            tile_index,
                        ]
                    ),
                    "final_fidelity_score": float(
                        trace.fidelity_score[
                            -1,
                            tile_index,
                        ]
                    ),
                    "final_dynamic_fidelity_class": (
                        FIDELITY_CODE_TO_CLASS[
                            final_class_code
                        ].value
                    ),
                    "final_available": bool(
                        trace.available[-1, tile_index]
                    ),
                    "final_faulted": bool(
                        trace.faulted[-1, tile_index]
                    ),
                }
            )


def write_timestep_summary(
    trace: TileFidelityTrace,
    output_path: Path,
) -> None:
    """Write aggregate tile statistics for every timestep."""

    high_code = FIDELITY_CLASS_TO_CODE[
        FIDELITY_CODE_TO_CLASS[0]
    ]
    medium_code = FIDELITY_CLASS_TO_CODE[
        FIDELITY_CODE_TO_CLASS[1]
    ]
    low_code = FIDELITY_CLASS_TO_CODE[
        FIDELITY_CODE_TO_CLASS[2]
    ]

    fieldnames = [
        "timestep",
        "mean_noise_std",
        "std_noise_std",
        "min_noise_std",
        "max_noise_std",
        "mean_fidelity_score",
        "min_fidelity_score",
        "max_fidelity_score",
        "available_tiles",
        "faulted_tiles",
        "high_fidelity_tiles",
        "medium_fidelity_tiles",
        "low_fidelity_tiles",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for timestep_index, timestep_value in enumerate(
            trace.timesteps
        ):
            timestep_noise = trace.noise_std[
                timestep_index
            ]
            timestep_fidelity = trace.fidelity_score[
                timestep_index
            ]
            timestep_classes = (
                trace.dynamic_fidelity_class[
                    timestep_index
                ]
            )

            writer.writerow(
                {
                    "timestep": int(timestep_value),
                    "mean_noise_std": float(
                        np.mean(timestep_noise)
                    ),
                    "std_noise_std": float(
                        np.std(timestep_noise)
                    ),
                    "min_noise_std": float(
                        np.min(timestep_noise)
                    ),
                    "max_noise_std": float(
                        np.max(timestep_noise)
                    ),
                    "mean_fidelity_score": float(
                        np.mean(timestep_fidelity)
                    ),
                    "min_fidelity_score": float(
                        np.min(timestep_fidelity)
                    ),
                    "max_fidelity_score": float(
                        np.max(timestep_fidelity)
                    ),
                    "available_tiles": int(
                        np.count_nonzero(
                            trace.available[timestep_index]
                        )
                    ),
                    "faulted_tiles": int(
                        np.count_nonzero(
                            trace.faulted[timestep_index]
                        )
                    ),
                    "high_fidelity_tiles": int(
                        np.count_nonzero(
                            timestep_classes == high_code
                        )
                    ),
                    "medium_fidelity_tiles": int(
                        np.count_nonzero(
                            timestep_classes == medium_code
                        )
                    ),
                    "low_fidelity_tiles": int(
                        np.count_nonzero(
                            timestep_classes == low_code
                        )
                    ),
                }
            )


def build_run_summary(
    trace: TileFidelityTrace,
) -> dict[str, Any]:
    """Compute concise aggregate metrics for the run."""

    initial_noise = trace.noise_std[0]
    final_noise = trace.noise_std[-1]

    initial_mean_noise = float(np.mean(initial_noise))
    final_mean_noise = float(np.mean(final_noise))

    if initial_mean_noise > 0.0:
        mean_noise_change_percent = (
            (final_mean_noise - initial_mean_noise)
            / initial_mean_noise
            * 100.0
        )
    else:
        mean_noise_change_percent = None

    rank_correlation = spearman_rank_correlation(
        initial_noise,
        final_noise,
    )

    selected_fault_tiles = np.flatnonzero(
        trace.fault_onset_timestep >= 0
    )

    return {
        "num_tiles": trace.num_tiles,
        "num_timesteps": trace.num_timesteps,
        "initial_mean_noise_std": initial_mean_noise,
        "final_mean_noise_std": final_mean_noise,
        "mean_noise_change_percent": (
            float(mean_noise_change_percent)
            if mean_noise_change_percent is not None
            else None
        ),
        "initial_min_noise_std": float(
            np.min(initial_noise)
        ),
        "initial_max_noise_std": float(
            np.max(initial_noise)
        ),
        "final_min_noise_std": float(
            np.min(final_noise)
        ),
        "final_max_noise_std": float(
            np.max(final_noise)
        ),
        "initial_mean_fidelity_score": float(
            np.mean(trace.fidelity_score[0])
        ),
        "final_mean_fidelity_score": float(
            np.mean(trace.fidelity_score[-1])
        ),
        "initial_to_final_rank_correlation": (
            rank_correlation
        ),
        "scheduled_fault_count": int(
            selected_fault_tiles.size
        ),
        "selected_fault_tile_ids": (
            selected_fault_tiles.astype(int).tolist()
        ),
        "final_faulted_tile_count": int(
            np.count_nonzero(trace.faulted[-1])
        ),
        "final_available_tile_count": int(
            np.count_nonzero(trace.available[-1])
        ),
        "final_unavailable_tile_count": int(
            trace.num_tiles
            - np.count_nonzero(trace.available[-1])
        ),
    }


def spearman_rank_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    """Compute Spearman rank correlation without SciPy."""

    first_array = np.asarray(
        first,
        dtype=np.float64,
    )
    second_array = np.asarray(
        second,
        dtype=np.float64,
    )

    if first_array.shape != second_array.shape:
        raise ValueError(
            "Rank-correlation inputs must have matching shapes."
        )

    if first_array.ndim != 1:
        raise ValueError(
            "Rank-correlation inputs must be one-dimensional."
        )

    if first_array.size < 2:
        return None

    first_ranks = rankdata(first_array)
    second_ranks = rankdata(second_array)

    first_std = float(np.std(first_ranks))
    second_std = float(np.std(second_ranks))

    if first_std == 0.0 or second_std == 0.0:
        if np.allclose(first_array, second_array):
            return 1.0

        return None

    correlation_matrix = np.corrcoef(
        first_ranks,
        second_ranks,
    )

    correlation = float(correlation_matrix[0, 1])

    if not np.isfinite(correlation):
        return None

    return correlation


def rankdata(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to values, including ties."""

    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.empty(
        values.size,
        dtype=np.float64,
    )

    start_index = 0

    while start_index < values.size:
        end_index = start_index + 1

        while (
            end_index < values.size
            and sorted_values[end_index]
            == sorted_values[start_index]
        ):
            end_index += 1

        average_rank = (
            start_index + end_index - 1
        ) / 2.0

        ranks[
            order[start_index:end_index]
        ] = average_rank

        start_index = end_index

    return ranks


def build_metadata(
    *,
    config_path: Path,
    output_directory: Path,
    experiment_name: str,
    seed: int,
    trace: TileFidelityTrace,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build metadata describing the generated artifacts."""

    return {
        "phase": "phase2_fidelity",
        "experiment_name": experiment_name,
        "seed": seed,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "config_path": str(config_path.resolve()),
        "output_directory": str(output_directory),
        "artifacts": {
            "trace": TRACE_FILENAME,
            "effective_config": CONFIG_FILENAME,
            "metadata": METADATA_FILENAME,
            "tile_summary": TILE_SUMMARY_FILENAME,
            "timestep_summary": TIMESTEP_SUMMARY_FILENAME,
        },
        "trace_shapes": {
            "noise_std": list(trace.noise_std.shape),
            "fidelity_score": list(
                trace.fidelity_score.shape
            ),
            "dynamic_fidelity_class": list(
                trace.dynamic_fidelity_class.shape
            ),
            "available": list(trace.available.shape),
            "faulted": list(trace.faulted.shape),
        },
        "summary": dict(summary),
        "model_metadata": dict(trace.metadata),
    }


def print_run_summary(
    *,
    output_directory: Path,
    summary: Mapping[str, Any],
) -> None:
    """Print the primary experiment results."""

    rank_correlation = summary[
        "initial_to_final_rank_correlation"
    ]

    if rank_correlation is None:
        rank_text = "undefined"
    else:
        rank_text = f"{rank_correlation:.6f}"

    noise_change = summary["mean_noise_change_percent"]

    if noise_change is None:
        noise_change_text = "undefined"
    else:
        noise_change_text = f"{noise_change:.3f}%"

    LOGGER.info("Phase 2 fidelity simulation completed.")
    LOGGER.info(
        "Tiles: %d | Timesteps: %d",
        summary["num_tiles"],
        summary["num_timesteps"],
    )
    LOGGER.info(
        "Mean noise: %.6f -> %.6f (%s)",
        summary["initial_mean_noise_std"],
        summary["final_mean_noise_std"],
        noise_change_text,
    )
    LOGGER.info(
        "Mean fidelity score: %.6f -> %.6f",
        summary["initial_mean_fidelity_score"],
        summary["final_mean_fidelity_score"],
    )
    LOGGER.info(
        "Initial/final tile-rank correlation: %s",
        rank_text,
    )
    LOGGER.info(
        "Scheduled faults: %d | Final faulted tiles: %d",
        summary["scheduled_fault_count"],
        summary["final_faulted_tile_count"],
    )
    LOGGER.info(
        "Final available tiles: %d",
        summary["final_available_tile_count"],
    )
    LOGGER.info(
        "Results saved to: %s",
        output_directory,
    )


def _json_ready(value: Any) -> Any:
    """Recursively convert values into JSON/YAML-safe types."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        return value

    return value


def run_experiment(
    *,
    config_path: Path,
    seed_override: int | None,
    cli_output_dir: Path | None,
    overwrite: bool,
) -> Path:
    """Run one complete Phase 2 experiment."""

    config_path = config_path.expanduser().resolve()

    LOGGER.info(
        "Loading configuration from %s",
        config_path,
    )

    raw_config = load_yaml_config(config_path)

    model = TileFidelityModel.from_mapping(
        raw_config,
        seed=seed_override,
    )

    experiment_name = get_experiment_name(
        raw_config,
        config_path,
    )

    output_directory = resolve_output_directory(
        config=raw_config,
        config_path=config_path,
        experiment_name=experiment_name,
        seed=model.seed,
        cli_output_dir=cli_output_dir,
    )

    prepare_output_directory(
        output_directory,
        overwrite=overwrite,
    )

    LOGGER.info(
        "Generating trace with seed %d",
        model.seed,
    )

    trace = model.generate_trace()

    trace_path = output_directory / TRACE_FILENAME
    trace.save_npz(trace_path)

    effective_config = build_effective_config(
        raw_config,
        seed=model.seed,
        experiment_name=experiment_name,
        output_directory=output_directory,
    )

    save_yaml(
        effective_config,
        output_directory / CONFIG_FILENAME,
    )

    write_tile_summary(
        trace,
        output_directory / TILE_SUMMARY_FILENAME,
    )

    write_timestep_summary(
        trace,
        output_directory / TIMESTEP_SUMMARY_FILENAME,
    )

    summary = build_run_summary(trace)

    metadata = build_metadata(
        config_path=config_path,
        output_directory=output_directory,
        experiment_name=experiment_name,
        seed=model.seed,
        trace=trace,
        summary=summary,
    )

    save_json(
        metadata,
        output_directory / METADATA_FILENAME,
    )

    print_run_summary(
        output_directory=output_directory,
        summary=summary,
    )

    return output_directory


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """Command-line entry point."""

    parsed_arguments = parse_arguments(arguments)
    configure_logging(parsed_arguments.log_level)

    try:
        run_experiment(
            config_path=parsed_arguments.config,
            seed_override=parsed_arguments.seed,
            cli_output_dir=parsed_arguments.output_dir,
            overwrite=parsed_arguments.overwrite,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        yaml.YAMLError,
        HardwareConfigurationError,
        FidelityConfigurationError,
        FidelitySimulationError,
    ) as error:
        LOGGER.error("Phase 2 experiment failed: %s", error)
        return 2
    except Exception:
        LOGGER.exception(
            "Phase 2 experiment failed due to an unexpected error."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())