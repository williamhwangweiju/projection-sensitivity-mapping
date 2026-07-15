#!/usr/bin/env python3
"""Analyze Phase 1 AIHWKit projection-sensitivity results.

The analyzer consumes the JSON produced by ``run_aihwkit_profiling.py`` and
creates:

* console summaries and projection rankings;
* a machine-readable CSV projection table;
* a compact JSON analysis summary;
* sensitivity heatmap, distribution, and per-block plots;
* empirical AIHWKit programming-noise calibration plots.

The current Phase 1 workflow exports mean total DeltaPPL as the mapping score:

    sensitivity_score_for_mapping == sensitivity_mean
    sensitivity_score_unit == "delta_ppl_total"

The analyzer validates that contract by default while retaining limited
backward compatibility for older Phase 1 result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import numpy as np


DEFAULT_RESULT_PATTERNS = (
    "gpt2_aihwkit_calibrated_*.json",
    "gpt2_aihwkit_sensitivity_*.json",
    "programming_noise_sensitivity_*.json",
)

EXPECTED_MAPPING_FIELD = "sensitivity_score_for_mapping"
EXPECTED_MAPPING_UNIT = "delta_ppl_total"

PROJECTION_ORDER = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "lm_head",
)

PROJECTION_COLORS = {
    "attn.c_attn": "tab:red",
    "attn.c_proj": "tab:orange",
    "mlp.c_fc": "tab:blue",
    "mlp.c_proj": "tab:cyan",
    "lm_head": "tab:brown",
}


@dataclass(frozen=True, slots=True)
class ProjectionAnalysisRecord:
    """Normalized Phase 1 projection record used by the analyzer."""

    block_id: str
    projection_name: str
    projection_id: str
    hf_module_path: str
    sensitivity_rank: int | None

    sensitivity_mean: float
    sensitivity_std: float

    digital_nll: float | None
    digital_ppl: float | None
    reference_nll: float | None
    reference_ppl: float | None
    noisy_nll_mean: float | None
    noisy_ppl_mean: float | None

    delta_nll_total_mean: float | None
    delta_nll_programming_mean: float | None
    delta_ppl_preprocessing: float | None

    measured_noise_std_absolute: float | None
    measured_noise_rms_absolute: float | None
    noise_reference_scale: float | None
    reference_sigma_normalized: float | None
    num_calibration_seeds: int | None


@dataclass(frozen=True, slots=True)
class Phase1AnalysisData:
    """Validated and normalized Phase 1 analysis payload."""

    source_path: Path
    mapping_field: str
    mapping_unit: str
    digital_nll: float | None
    digital_ppl: float
    records: tuple[ProjectionAnalysisRecord, ...]


SensitivityTable = dict[str, dict[str, float]]


def find_repo_root(script_path: Path) -> Path:
    """Find the repository root from this script location."""
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "src" / "profilers" / "aihwkit_profiler.py").is_file():
            return candidate
    return resolved.parent


REPO_ROOT = find_repo_root(Path(__file__))


def _optional_finite_float(value: Any, *, field_name: str) -> float | None:
    """Parse an optional finite float."""
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, received {value!r}.")
    return parsed


def _required_finite_float(value: Any, *, field_name: str) -> float:
    """Parse a required finite float."""
    parsed = _optional_finite_float(value, field_name=field_name)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


def _optional_int(value: Any, *, field_name: str) -> int | None:
    """Parse an optional integer."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not boolean.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be nonnegative.")
    return parsed


def block_sort_key(block_id: str) -> tuple[int, int, str]:
    """Sort GPT-2 blocks numerically and place the LM head last."""
    if block_id.startswith("block_"):
        try:
            return (0, int(block_id.removeprefix("block_")), block_id)
        except ValueError:
            pass
    if block_id == "head":
        return (1, 0, block_id)
    return (2, 0, block_id)


def projection_sort_key(record: ProjectionAnalysisRecord) -> tuple[int, int, str]:
    """Sort records by block and canonical projection order."""
    try:
        projection_index = PROJECTION_ORDER.index(record.projection_name)
    except ValueError:
        projection_index = len(PROJECTION_ORDER)
    block_group, block_index, block_text = block_sort_key(record.block_id)
    return (block_group * 100_000 + block_index, projection_index, block_text)


def find_latest_results_file(
    *,
    results_dir: Path | None = None,
    patterns: Sequence[str] = DEFAULT_RESULT_PATTERNS,
) -> Path:
    """Return the newest Phase 1 result matching any configured pattern."""
    root = (
        results_dir.expanduser().resolve()
        if results_dir is not None
        else REPO_ROOT / "data" / "results" / "phase1_sensitivity"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    candidates: set[Path] = set()
    for pattern in patterns:
        candidates.update(path.resolve() for path in root.rglob(pattern) if path.is_file())

    if not candidates:
        joined = ", ".join(repr(pattern) for pattern in patterns)
        raise FileNotFoundError(
            f"No Phase 1 result matching any of [{joined}] was found under {root}. "
            "Pass --results-file to select a file explicitly."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and require an object at the root."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Results file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)

    if not isinstance(loaded, dict):
        raise TypeError("Phase 1 result JSON root must be an object.")
    return loaded


def _resolve_results_object(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested results object used by current and older runners."""
    results = payload.get("results")
    if isinstance(results, Mapping):
        return results

    # Very old files occasionally stored the result fields at the top level.
    if isinstance(payload.get("projections"), list):
        return payload

    raise ValueError("Phase 1 JSON must contain a 'results' object.")


def _resolve_mapping_contract(
    payload: Mapping[str, Any],
    results: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve the mapper sensitivity field and its declared unit."""
    mapping_field = str(
        results.get(
            "mapping_sensitivity_field",
            payload.get("mapping_sensitivity_field", EXPECTED_MAPPING_FIELD),
        )
    ).strip()
    mapping_unit = str(
        results.get(
            "mapping_sensitivity_unit",
            payload.get("mapping_sensitivity_unit", ""),
        )
    ).strip()

    if not mapping_field:
        mapping_field = EXPECTED_MAPPING_FIELD

    # Current projection rows also carry the unit. Use it when the aggregate
    # object does not declare one.
    projections = results.get("projections")
    if not mapping_unit and isinstance(projections, list):
        row_units = {
            str(row.get("sensitivity_score_unit", "")).strip()
            for row in projections
            if isinstance(row, Mapping) and row.get("sensitivity_score_unit") is not None
        }
        row_units.discard("")
        if len(row_units) == 1:
            mapping_unit = next(iter(row_units))
        elif len(row_units) > 1:
            raise ValueError(
                "Projection records declare inconsistent sensitivity_score_unit values: "
                f"{sorted(row_units)}"
            )

    # Backward compatibility: older DeltaPPL files did not declare a unit.
    if not mapping_unit and mapping_field in {
        "sensitivity_score_for_mapping",
        "sensitivity_mean",
        "delta_ppl_total_mean",
    }:
        mapping_unit = EXPECTED_MAPPING_UNIT

    return mapping_field, mapping_unit


def _calibration_lookup(
    payload: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Build a projection-id lookup for top-level calibration records."""
    raw_calibrations = payload.get("projection_noise_calibration")
    if not isinstance(raw_calibrations, list):
        raw_calibrations = results.get("projection_noise_calibration")
    if not isinstance(raw_calibrations, list):
        return {}

    lookup: dict[str, Mapping[str, Any]] = {}
    for index, calibration in enumerate(raw_calibrations):
        if not isinstance(calibration, Mapping):
            raise TypeError(f"Calibration record {index} is not an object.")
        projection_id = str(calibration.get("projection_id", "")).strip()
        if not projection_id:
            raise ValueError(f"Calibration record {index} is missing projection_id.")
        if projection_id in lookup:
            raise ValueError(f"Duplicate calibration record for {projection_id}.")
        lookup[projection_id] = calibration
    return lookup


def normalize_phase1_payload(
    *,
    payload: Mapping[str, Any],
    source_path: Path,
    expected_mapping_unit: str | None,
) -> Phase1AnalysisData:
    """Validate and normalize a Phase 1 result payload."""
    results = _resolve_results_object(payload)
    projections = results.get("projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError("results.projections must be a non-empty list.")

    mapping_field, mapping_unit = _resolve_mapping_contract(payload, results)
    if expected_mapping_unit is not None and mapping_unit != expected_mapping_unit:
        raise ValueError(
            "Unexpected Phase 1 mapping sensitivity unit: "
            f"{mapping_unit!r}; expected {expected_mapping_unit!r}. "
            "This analyzer is intended for the updated DeltaPPL Phase 1 workflow."
        )

    digital_ppl = _required_finite_float(
        results.get("digital_perplexity", results.get("baseline", {}).get("clean_ppl") if isinstance(results.get("baseline"), Mapping) else None),
        field_name="results.digital_perplexity",
    )
    digital_nll = _optional_finite_float(
        results.get("digital_nll", results.get("baseline", {}).get("clean_nll") if isinstance(results.get("baseline"), Mapping) else None),
        field_name="results.digital_nll",
    )

    calibrations = _calibration_lookup(payload, results)
    normalized_records: list[ProjectionAnalysisRecord] = []
    seen_projection_ids: set[str] = set()

    for index, row in enumerate(projections):
        if not isinstance(row, Mapping):
            raise TypeError(f"Projection record {index} is not an object.")

        block_id = str(row.get("block_id", "")).strip()
        projection_name = str(row.get("proj_name", row.get("projection_name", ""))).strip()
        if not block_id or not projection_name:
            raise ValueError(
                f"Projection record {index} must contain block_id and proj_name."
            )
        if projection_name not in PROJECTION_ORDER:
            raise ValueError(f"Unknown projection name: {projection_name!r}")

        projection_id = str(
            row.get("projection_id", f"{block_id}/{projection_name}")
        ).strip()
        if projection_id in seen_projection_ids:
            raise ValueError(f"Duplicate projection result: {projection_id}")
        seen_projection_ids.add(projection_id)

        row_unit = str(row.get("sensitivity_score_unit", mapping_unit)).strip()
        if row_unit and mapping_unit and row_unit != mapping_unit:
            raise ValueError(
                f"{projection_id} uses sensitivity unit {row_unit!r}, "
                f"but the result package declares {mapping_unit!r}."
            )

        if mapping_field in row:
            sensitivity_mean = _required_finite_float(
                row[mapping_field], field_name=f"{projection_id}.{mapping_field}"
            )
        elif "sensitivity_score_for_mapping" in row:
            sensitivity_mean = _required_finite_float(
                row["sensitivity_score_for_mapping"],
                field_name=f"{projection_id}.sensitivity_score_for_mapping",
            )
        elif "sensitivity_mean" in row:
            sensitivity_mean = _required_finite_float(
                row["sensitivity_mean"],
                field_name=f"{projection_id}.sensitivity_mean",
            )
        else:
            raise ValueError(
                f"{projection_id} does not contain mapping field {mapping_field!r}."
            )

        if mapping_unit == "delta_ppl_total":
            std_source = row.get("delta_ppl_total_std", row.get("sensitivity_std"))
        elif mapping_unit == "delta_nll_programming":
            std_source = row.get("delta_nll_programming_std")
        else:
            std_source = row.get("sensitivity_std", 0.0)

        sensitivity_std = _required_finite_float(
            0.0 if std_source is None else std_source,
            field_name=f"{projection_id}.sensitivity_std",
        )
        if sensitivity_std < 0.0:
            raise ValueError(f"{projection_id}.sensitivity_std cannot be negative.")

        embedded_calibration = row.get("noise_calibration")
        if embedded_calibration is not None and not isinstance(embedded_calibration, Mapping):
            raise TypeError(f"{projection_id}.noise_calibration must be an object.")
        calibration = (
            embedded_calibration
            if isinstance(embedded_calibration, Mapping)
            else calibrations.get(projection_id, {})
        )

        hf_module_path = str(
            row.get("hf_module_path", calibration.get("hf_module_path", ""))
        ).strip()

        rank_value = row.get("sensitivity_rank")
        sensitivity_rank = _optional_int(
            rank_value, field_name=f"{projection_id}.sensitivity_rank"
        )
        if sensitivity_rank == 0:
            raise ValueError(f"{projection_id}.sensitivity_rank must start at 1.")

        normalized_records.append(
            ProjectionAnalysisRecord(
                block_id=block_id,
                projection_name=projection_name,
                projection_id=projection_id,
                hf_module_path=hf_module_path,
                sensitivity_rank=sensitivity_rank,
                sensitivity_mean=sensitivity_mean,
                sensitivity_std=sensitivity_std,
                digital_nll=_optional_finite_float(
                    row.get("nll_clean", digital_nll),
                    field_name=f"{projection_id}.nll_clean",
                ),
                digital_ppl=_optional_finite_float(
                    row.get("ppl_clean", digital_ppl),
                    field_name=f"{projection_id}.ppl_clean",
                ),
                reference_nll=_optional_finite_float(
                    row.get("nll_reference"),
                    field_name=f"{projection_id}.nll_reference",
                ),
                reference_ppl=_optional_finite_float(
                    row.get("ppl_reference"),
                    field_name=f"{projection_id}.ppl_reference",
                ),
                noisy_nll_mean=_optional_finite_float(
                    row.get("nll_noisy_mean"),
                    field_name=f"{projection_id}.nll_noisy_mean",
                ),
                noisy_ppl_mean=_optional_finite_float(
                    row.get("ppl_noisy_mean"),
                    field_name=f"{projection_id}.ppl_noisy_mean",
                ),
                delta_nll_total_mean=_optional_finite_float(
                    row.get("delta_nll_total_mean"),
                    field_name=f"{projection_id}.delta_nll_total_mean",
                ),
                delta_nll_programming_mean=_optional_finite_float(
                    row.get("delta_nll_programming_mean"),
                    field_name=f"{projection_id}.delta_nll_programming_mean",
                ),
                delta_ppl_preprocessing=_optional_finite_float(
                    row.get("delta_ppl_preprocessing"),
                    field_name=f"{projection_id}.delta_ppl_preprocessing",
                ),
                measured_noise_std_absolute=_optional_finite_float(
                    calibration.get(
                        "measured_noise_std_absolute",
                        row.get("noise_std_absolute_mean"),
                    ),
                    field_name=f"{projection_id}.measured_noise_std_absolute",
                ),
                measured_noise_rms_absolute=_optional_finite_float(
                    calibration.get(
                        "measured_noise_rms_absolute",
                        row.get("noise_rms_absolute_mean"),
                    ),
                    field_name=f"{projection_id}.measured_noise_rms_absolute",
                ),
                noise_reference_scale=_optional_finite_float(
                    calibration.get(
                        "noise_reference_scale",
                        row.get("noise_reference_scale_mean"),
                    ),
                    field_name=f"{projection_id}.noise_reference_scale",
                ),
                reference_sigma_normalized=_optional_finite_float(
                    calibration.get("reference_sigma_normalized"),
                    field_name=f"{projection_id}.reference_sigma_normalized",
                ),
                num_calibration_seeds=_optional_int(
                    calibration.get("num_calibration_seeds"),
                    field_name=f"{projection_id}.num_calibration_seeds",
                ),
            )
        )

    # Verify ranks when they are present. The runner ranks descending DeltaPPL.
    ranked_records = sorted(
        normalized_records,
        key=lambda record: record.sensitivity_mean,
        reverse=True,
    )
    for expected_rank, record in enumerate(ranked_records, start=1):
        if record.sensitivity_rank is not None and record.sensitivity_rank != expected_rank:
            raise ValueError(
                f"Sensitivity rank mismatch for {record.projection_id}: "
                f"saved={record.sensitivity_rank}, expected={expected_rank}."
            )

    return Phase1AnalysisData(
        source_path=source_path.resolve(),
        mapping_field=mapping_field,
        mapping_unit=mapping_unit,
        digital_nll=digital_nll,
        digital_ppl=digital_ppl,
        records=tuple(sorted(normalized_records, key=projection_sort_key)),
    )


def configured_projection_order(records: Iterable[ProjectionAnalysisRecord]) -> list[str]:
    """Return canonical projection types present in the result."""
    present = {record.projection_name for record in records}
    return [name for name in PROJECTION_ORDER if name in present]


def sensitivity_tables(
    records: Sequence[ProjectionAnalysisRecord],
) -> tuple[SensitivityTable, SensitivityTable]:
    """Build mean and standard-deviation tables by block and projection."""
    means: SensitivityTable = {}
    stds: SensitivityTable = {}
    for record in records:
        means.setdefault(record.block_id, {})[record.projection_name] = (
            record.sensitivity_mean
        )
        stds.setdefault(record.block_id, {})[record.projection_name] = (
            record.sensitivity_std
        )
    return means, stds


def metric_axis_label(mapping_unit: str) -> str:
    """Return a readable y-axis label for the mapping metric."""
    labels = {
        "delta_ppl_total": "Mean total DeltaPPL",
        "delta_nll_programming": "Mean programming-only DeltaNLL",
        "delta_nll_total": "Mean total DeltaNLL",
    }
    return labels.get(mapping_unit, f"Mean sensitivity ({mapping_unit})")


def configure_signed_log_axis(ax: Any, values: Sequence[float]) -> None:
    """Use log scale for positive values and symlog when signs are mixed."""
    values_array = np.asarray(values, dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return

    if np.all(values_array > 0.0):
        dynamic_ratio = float(values_array.max() / values_array.min())
        if dynamic_ratio >= 100.0:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        return

    absolute_values = np.abs(values_array)
    nonzero_values = absolute_values[absolute_values > 0.0]
    if nonzero_values.size == 0:
        return
    linthresh = max(
        float(np.percentile(nonzero_values, 20)),
        float(absolute_values.max()) * 1e-4,
        1e-12,
    )
    ax.set_yscale("symlog", linthresh=linthresh, base=10)


def summarize_analysis(data: Phase1AnalysisData) -> dict[str, Any]:
    """Print Phase 1 statistics and return a serializable summary."""
    records = list(data.records)
    values = np.asarray(
        [record.sensitivity_mean for record in records], dtype=np.float64
    )
    projection_names = configured_projection_order(records)
    means, _ = sensitivity_tables(records)

    calibration_records = [
        record for record in records if record.noise_reference_scale is not None
    ]
    negative_records = [record for record in records if record.sensitivity_mean < 0.0]

    print("\n" + "=" * 78)
    print("PHASE 1 AIHWKIT PROJECTION-SENSITIVITY ANALYSIS")
    print("=" * 78)
    print(f"\nSource: {data.source_path}")
    print(f"Mapping field: {data.mapping_field}")
    print(f"Mapping unit:  {data.mapping_unit}")
    if data.digital_nll is not None:
        print(f"Digital FP32 NLL: {data.digital_nll:.8f}")
    print(f"Digital FP32 PPL: {data.digital_ppl:.6f}")
    print(f"Profiled projections: {len(records)}")
    print(f"Negative sensitivity estimates: {len(negative_records)}")

    print(f"\nOverall {metric_axis_label(data.mapping_unit)} statistics:")
    print(f"  Mean:   {values.mean():.8f}")
    print(f"  Median: {np.median(values):.8f}")
    print(f"  Std:    {values.std(ddof=0):.8f}")
    print(f"  Min:    {values.min():.8f}")
    print(f"  Max:    {values.max():.8f}")

    print("\nBlock-average mapping sensitivity:")
    block_summaries: dict[str, dict[str, float]] = {}
    for block_id in sorted(means, key=block_sort_key):
        block_values = np.asarray(list(means[block_id].values()), dtype=np.float64)
        block_summaries[block_id] = {
            "mean": float(block_values.mean()),
            "std": float(block_values.std(ddof=0)),
            "minimum": float(block_values.min()),
            "maximum": float(block_values.max()),
        }
        print(
            f"  {block_id:9s}: mean={block_values.mean():.8f}, "
            f"range=[{block_values.min():.8f}, {block_values.max():.8f}]"
        )

    print("\nProjection-type mapping sensitivity:")
    projection_type_summaries: dict[str, dict[str, float]] = {}
    for projection_name in projection_names:
        projection_values = np.asarray(
            [
                record.sensitivity_mean
                for record in records
                if record.projection_name == projection_name
            ],
            dtype=np.float64,
        )
        projection_type_summaries[projection_name] = {
            "mean": float(projection_values.mean()),
            "std": float(projection_values.std(ddof=0)),
            "minimum": float(projection_values.min()),
            "maximum": float(projection_values.max()),
        }
        print(
            f"  {projection_name:12s}: mean={projection_values.mean():.8f}, "
            f"std={projection_values.std(ddof=0):.8f}, "
            f"range=[{projection_values.min():.8f}, "
            f"{projection_values.max():.8f}]"
        )

    ranked = sorted(records, key=lambda record: record.sensitivity_mean, reverse=True)
    top_count = min(8, len(ranked))

    print(f"\nTop {top_count} most sensitive projections:")
    for rank, record in enumerate(ranked[:top_count], start=1):
        print(
            f"  {rank:2d}. {record.projection_id:28s}: "
            f"{record.sensitivity_mean:.8f} +/- {record.sensitivity_std:.8f}"
        )

    print(f"\nBottom {top_count} least sensitive projections:")
    for rank, record in enumerate(reversed(ranked[-top_count:]), start=1):
        print(
            f"  {rank:2d}. {record.projection_id:28s}: "
            f"{record.sensitivity_mean:.8f} +/- {record.sensitivity_std:.8f}"
        )

    calibration_summary: dict[str, Any] | None = None
    if calibration_records:
        scales = np.asarray(
            [record.noise_reference_scale for record in calibration_records],
            dtype=np.float64,
        )
        absolute_stds = np.asarray(
            [
                record.measured_noise_std_absolute
                for record in calibration_records
                if record.measured_noise_std_absolute is not None
            ],
            dtype=np.float64,
        )
        seed_counts = sorted(
            {
                record.num_calibration_seeds
                for record in calibration_records
                if record.num_calibration_seeds is not None
            }
        )
        calibration_summary = {
            "num_records": len(calibration_records),
            "noise_reference_scale_mean": float(scales.mean()),
            "noise_reference_scale_std": float(scales.std(ddof=0)),
            "noise_reference_scale_min": float(scales.min()),
            "noise_reference_scale_max": float(scales.max()),
            "calibration_seed_counts": seed_counts,
        }
        if absolute_stds.size:
            calibration_summary.update(
                {
                    "measured_noise_std_absolute_mean": float(
                        absolute_stds.mean()
                    ),
                    "measured_noise_std_absolute_min": float(
                        absolute_stds.min()
                    ),
                    "measured_noise_std_absolute_max": float(
                        absolute_stds.max()
                    ),
                }
            )

        print("\nAIHWKit programming-noise calibration:")
        print(f"  Records: {len(calibration_records)}")
        print(
            "  noise_reference_scale: "
            f"mean={scales.mean():.8e}, std={scales.std(ddof=0):.8e}, "
            f"range=[{scales.min():.8e}, {scales.max():.8e}]"
        )
        if absolute_stds.size:
            print(
                "  measured absolute noise std: "
                f"mean={absolute_stds.mean():.8e}, "
                f"range=[{absolute_stds.min():.8e}, "
                f"{absolute_stds.max():.8e}]"
            )
        if seed_counts:
            print(f"  Calibration seed counts: {seed_counts}")

    if negative_records:
        print(
            "\nNote: negative single-projection DeltaPPL estimates can occur "
            "from finite-sample and programming-noise variation. Use multiple "
            "Phase 1 seeds for the final ranking."
        )

    return {
        "source_file": str(data.source_path),
        "mapping_field": data.mapping_field,
        "mapping_unit": data.mapping_unit,
        "digital_nll": data.digital_nll,
        "digital_ppl": data.digital_ppl,
        "num_projections": len(records),
        "num_negative_sensitivities": len(negative_records),
        "overall": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        },
        "by_block": block_summaries,
        "by_projection_type": projection_type_summaries,
        "ranking": [
            {
                "rank": rank,
                "projection_id": record.projection_id,
                "sensitivity_mean": record.sensitivity_mean,
                "sensitivity_std": record.sensitivity_std,
            }
            for rank, record in enumerate(ranked, start=1)
        ],
        "calibration": calibration_summary,
    }


def write_projection_csv(
    records: Sequence[ProjectionAnalysisRecord],
    output_path: Path,
) -> None:
    """Write a normalized projection-level analysis table."""
    fieldnames = [
        "projection_id",
        "block_id",
        "projection_name",
        "hf_module_path",
        "sensitivity_rank",
        "sensitivity_mean",
        "sensitivity_std",
        "digital_nll",
        "digital_ppl",
        "reference_nll",
        "reference_ppl",
        "noisy_nll_mean",
        "noisy_ppl_mean",
        "delta_nll_total_mean",
        "delta_nll_programming_mean",
        "delta_ppl_preprocessing",
        "reference_sigma_normalized",
        "measured_noise_std_absolute",
        "measured_noise_rms_absolute",
        "noise_reference_scale",
        "num_calibration_seeds",
    ]

    ranked = {
        record.projection_id: rank
        for rank, record in enumerate(
            sorted(records, key=lambda item: item.sensitivity_mean, reverse=True),
            start=1,
        )
    }

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "projection_id": record.projection_id,
                    "block_id": record.block_id,
                    "projection_name": record.projection_name,
                    "hf_module_path": record.hf_module_path,
                    "sensitivity_rank": ranked[record.projection_id],
                    "sensitivity_mean": record.sensitivity_mean,
                    "sensitivity_std": record.sensitivity_std,
                    "digital_nll": record.digital_nll,
                    "digital_ppl": record.digital_ppl,
                    "reference_nll": record.reference_nll,
                    "reference_ppl": record.reference_ppl,
                    "noisy_nll_mean": record.noisy_nll_mean,
                    "noisy_ppl_mean": record.noisy_ppl_mean,
                    "delta_nll_total_mean": record.delta_nll_total_mean,
                    "delta_nll_programming_mean": (
                        record.delta_nll_programming_mean
                    ),
                    "delta_ppl_preprocessing": record.delta_ppl_preprocessing,
                    "reference_sigma_normalized": (
                        record.reference_sigma_normalized
                    ),
                    "measured_noise_std_absolute": (
                        record.measured_noise_std_absolute
                    ),
                    "measured_noise_rms_absolute": (
                        record.measured_noise_rms_absolute
                    ),
                    "noise_reference_scale": record.noise_reference_scale,
                    "num_calibration_seeds": record.num_calibration_seeds,
                }
            )


def _heatmap_norm(values: np.ndarray) -> TwoSlopeNorm | None:
    """Return a zero-centered color normalization when values cross zero."""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not (finite.min() < 0.0 < finite.max()):
        return None
    magnitude = max(abs(float(finite.min())), abs(float(finite.max())))
    return TwoSlopeNorm(vmin=-magnitude, vcenter=0.0, vmax=magnitude)


def plot_sensitivity_heatmap(
    data: Phase1AnalysisData,
    output_path: Path,
) -> None:
    """Plot block-by-projection mapping sensitivity."""
    means, _ = sensitivity_tables(data.records)
    blocks = sorted(means, key=block_sort_key)
    projection_names = configured_projection_order(data.records)

    matrix = np.full((len(blocks), len(projection_names)), np.nan, dtype=np.float64)
    for row_index, block_id in enumerate(blocks):
        for column_index, projection_name in enumerate(projection_names):
            matrix[row_index, column_index] = means[block_id].get(
                projection_name, np.nan
            )

    fig, ax = plt.subplots(
        figsize=(12, max(4.5, 0.45 * len(blocks) + 1.8))
    )
    norm = _heatmap_norm(matrix)
    cmap = "coolwarm" if norm is not None else "YlOrRd"
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )
    ax.set_xticks(range(len(projection_names)))
    ax.set_yticks(range(len(blocks)))
    ax.set_xticklabels(projection_names, rotation=20, ha="right")
    ax.set_yticklabels(blocks)
    ax.set_xlabel("Projection")
    ax.set_ylabel("GPT-2 block")
    ax.set_title("Phase 1 Projection Mapping Sensitivity")
    plt.colorbar(image, ax=ax, label=metric_axis_label(data.mapping_unit))

    finite = matrix[np.isfinite(matrix)]
    threshold = float(np.median(np.abs(finite))) if finite.size else 0.0
    for row_index in range(len(blocks)):
        for column_index in range(len(projection_names)):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                text = "-"
                text_color = "black"
            else:
                text = f"{value:.5g}"
                text_color = "white" if abs(value) > threshold else "black"
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sensitivity_distribution(
    data: Phase1AnalysisData,
    output_path: Path,
) -> None:
    """Plot sensitivity distributions by projection type."""
    projection_names = configured_projection_order(data.records)
    values_by_projection = [
        [
            record.sensitivity_mean
            for record in data.records
            if record.projection_name == projection_name
        ]
        for projection_name in projection_names
    ]
    all_values = [value for values in values_by_projection for value in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(values_by_projection, patch_artist=True)
    for patch, projection_name in zip(box["boxes"], projection_names):
        patch.set_facecolor(PROJECTION_COLORS.get(projection_name, "tab:gray"))
        patch.set_alpha(0.65)

    for index, values in enumerate(values_by_projection, start=1):
        x_values = np.full(len(values), index, dtype=np.float64)
        ax.scatter(x_values, values, s=20, alpha=0.65, zorder=3)

    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.6)
    ax.set_xticks(range(1, len(projection_names) + 1))
    ax.set_xticklabels(projection_names, rotation=20, ha="right")
    configure_signed_log_axis(ax, all_values)
    ax.set_ylabel(metric_axis_label(data.mapping_unit))
    ax.set_title("Sensitivity Distribution by Projection Type")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sensitivity_by_block(
    data: Phase1AnalysisData,
    output_path: Path,
) -> None:
    """Plot mean projection sensitivity with seed standard deviations."""
    means, stds = sensitivity_tables(data.records)
    blocks = sorted(means, key=block_sort_key)
    projection_names = configured_projection_order(data.records)
    all_values = [record.sensitivity_mean for record in data.records]

    fig, ax = plt.subplots(figsize=(max(11.0, 0.9 * len(blocks)), 6.5))
    x_positions = np.arange(len(blocks))
    width = min(0.8 / max(len(projection_names), 1), 0.19)
    center = (len(projection_names) - 1) / 2.0

    for index, projection_name in enumerate(projection_names):
        values = np.asarray(
            [means[block_id].get(projection_name, np.nan) for block_id in blocks],
            dtype=np.float64,
        )
        error_values = np.asarray(
            [stds[block_id].get(projection_name, 0.0) for block_id in blocks],
            dtype=np.float64,
        )
        ax.bar(
            x_positions + (index - center) * width,
            values,
            yerr=error_values,
            capsize=2,
            width=width,
            label=projection_name,
            color=PROJECTION_COLORS.get(projection_name, "tab:gray"),
            alpha=0.8,
        )

    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(blocks, rotation=30, ha="right")
    configure_signed_log_axis(ax, all_values)
    ax.set_ylabel(metric_axis_label(data.mapping_unit))
    ax.set_title("Per-Block Projection Sensitivity")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_calibration_scales(
    data: Phase1AnalysisData,
    output_path: Path,
) -> bool:
    """Plot per-projection AIHWKit-to-absolute-noise conversion scales."""
    records = [
        record for record in data.records if record.noise_reference_scale is not None
    ]
    if not records:
        return False

    labels = [record.projection_id for record in records]
    scales = np.asarray(
        [record.noise_reference_scale for record in records], dtype=np.float64
    )
    positions = np.arange(len(records))
    colors = [
        PROJECTION_COLORS.get(record.projection_name, "tab:gray")
        for record in records
    ]

    fig, ax = plt.subplots(figsize=(max(13.0, 0.28 * len(records)), 6.5))
    ax.bar(positions, scales, color=colors, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("noise_reference_scale")
    ax.set_title("Phase 1 Empirical AIHWKit Programming-Noise Calibration")
    ax.grid(True, alpha=0.3, axis="y")

    positive = scales[scales > 0.0]
    if positive.size and float(positive.max() / positive.min()) >= 100.0:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_noise_std_vs_sensitivity(
    data: Phase1AnalysisData,
    output_path: Path,
) -> bool:
    """Plot calibrated weight-noise magnitude against mapping sensitivity."""
    records = [
        record
        for record in data.records
        if record.measured_noise_std_absolute is not None
    ]
    if not records:
        return False

    x_values = np.asarray(
        [record.measured_noise_std_absolute for record in records],
        dtype=np.float64,
    )
    y_values = np.asarray(
        [record.sensitivity_mean for record in records], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for projection_name in configured_projection_order(records):
        selected = [
            record for record in records if record.projection_name == projection_name
        ]
        ax.scatter(
            [record.measured_noise_std_absolute for record in selected],
            [record.sensitivity_mean for record in selected],
            label=projection_name,
            s=45,
            alpha=0.8,
            color=PROJECTION_COLORS.get(projection_name, "tab:gray"),
        )

    if len(records) >= 2 and np.std(x_values) > 0.0 and np.std(y_values) > 0.0:
        correlation = float(np.corrcoef(x_values, y_values)[0, 1])
        ax.set_title(
            "Effective Programmed Weight Noise vs. Sensitivity\n"
            f"Pearson correlation = {correlation:.3f}"
        )
    else:
        ax.set_title("Effective Programmed Weight Noise vs. Sensitivity")

    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="black", alpha=0.6)
    ax.set_xlabel("Measured absolute programmed-weight noise std")
    ax.set_ylabel(metric_axis_label(data.mapping_unit))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def create_outputs(
    data: Phase1AnalysisData,
    output_dir: Path,
    summary: Mapping[str, Any],
) -> list[Path]:
    """Write tables, summaries, and all available plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    csv_path = output_dir / "phase1_projection_summary.csv"
    write_projection_csv(data.records, csv_path)
    output_files.append(csv_path)

    summary_path = output_dir / "phase1_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    output_files.append(summary_path)

    heatmap_path = output_dir / "phase1_sensitivity_heatmap.png"
    plot_sensitivity_heatmap(data, heatmap_path)
    output_files.append(heatmap_path)

    distribution_path = output_dir / "phase1_sensitivity_distribution.png"
    plot_sensitivity_distribution(data, distribution_path)
    output_files.append(distribution_path)

    block_path = output_dir / "phase1_sensitivity_by_block.png"
    plot_sensitivity_by_block(data, block_path)
    output_files.append(block_path)

    calibration_path = output_dir / "phase1_noise_reference_scale.png"
    if plot_calibration_scales(data, calibration_path):
        output_files.append(calibration_path)

    scatter_path = output_dir / "phase1_noise_std_vs_sensitivity.png"
    if plot_noise_std_vs_sensitivity(data, scatter_path):
        output_files.append(scatter_path)

    return output_files


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and plot Phase 1 AIHWKit sensitivity results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--results-file",
        type=Path,
        help="Specific Phase 1 JSON result.",
    )
    selection.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing Phase 1 JSON results.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Filename pattern used when selecting the newest result. May be "
            "passed more than once. Defaults to current and legacy patterns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for CSV, JSON summary, and plots.",
    )
    parser.add_argument(
        "--expected-mapping-unit",
        default=EXPECTED_MAPPING_UNIT,
        help=(
            "Require this Phase 1 mapping sensitivity unit. Pass an empty "
            "string to disable the unit check for legacy files."
        ),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Load one Phase 1 result, summarize it, and generate analysis outputs."""
    args = parse_args(arguments)
    patterns = tuple(args.patterns) if args.patterns else DEFAULT_RESULT_PATTERNS

    if args.results_file is not None:
        results_file = args.results_file.expanduser().resolve()
    else:
        results_file = find_latest_results_file(
            results_dir=args.results_dir,
            patterns=patterns,
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_file.parent
    )
    expected_unit = args.expected_mapping_unit.strip() or None

    print("\nLoading Phase 1 results:")
    print(f"  Results: {results_file}")
    print(f"  Outputs: {output_dir}")

    payload = load_json_object(results_file)
    data = normalize_phase1_payload(
        payload=payload,
        source_path=results_file,
        expected_mapping_unit=expected_unit,
    )
    summary = summarize_analysis(data)
    output_files = create_outputs(data, output_dir, summary)

    print("\nSaved analysis outputs:")
    for path in output_files:
        print(f"  {path}")
    print("\nPhase 1 analysis complete.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
