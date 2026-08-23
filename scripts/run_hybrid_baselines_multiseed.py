#!/usr/bin/env python3
"""Run and aggregate the digital/analog hybrid comparator across trace seeds.

For every trace seed in the multiseed manifest, this script reuses the frozen
Phase-2 trace and the archived all-analog Phase-4 table of that seed, runs
``experiments/phase4_quality/run_hybrid_baselines.py`` (which checkpoints every
condition to disk and resumes), records the finished seeds in a campaign
manifest (so a restarted campaign skips them), and finally aggregates with
each independent hardware trace as one inferential unit.

Outputs under ``<manifest dir>/<run_name>/``:

* ``hybrid_baselines_cross_trace_summary.csv`` -- per design: mean paired
  improvement versus the all-analog placement with the same remainder policy
  and versus all-analog ``static_sensitivity``, Student-t 95% CIs over traces,
  trace win fractions, digital MAC fraction;
* ``hybrid_baselines_cross_trace_by_timestep.csv`` -- the same per timestep;
* ``hybrid_baselines_paper_by_condition.csv`` -- compact per-condition rows for
  the paper's data folder;
* ``hybrid_baselines_run_manifest.yaml`` -- completed seeds and artifact paths.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.tabular import write_csv


T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

IMPROVEMENT_COLUMNS = (
    ("vs_all_analog", "nll_improvement_vs_all_analog"),
    ("vs_static_sensitivity", "nll_improvement_vs_static_sensitivity"),
)

PAPER_COLUMNS = (
    "trace_seed",
    "design",
    "digital_weight_mode",
    "policy",
    "timestep",
    "realization",
    "nll",
    "all_analog_nll",
    "nll_improvement_vs_all_analog",
    "static_sensitivity_nll",
    "nll_improvement_vs_static_sensitivity",
    "nominal_nll",
    "checkpoint_digital_nll",
    "checkpoint_clipped_digital_nll",
    "digital_projection_mac_fraction",
    "predicted_tokens",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _one_file(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {label} under {root} matching {pattern!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _trace_stats(per_trace: dict[int, float]) -> dict[str, Any]:
    """Mean, std, Student-t 95% CI and win fraction over independent traces."""
    values = [per_trace[seed] for seed in sorted(per_trace)]
    n = len(values)
    mean = _mean(values)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
        if n > 1
        else 0.0
    )
    sem = std / math.sqrt(n) if n > 1 else math.nan
    margin = T_975.get(n - 1, 1.96 + 2.4 / (n - 1)) * sem if n > 1 else math.nan
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - margin if n > 1 else math.nan,
        "ci_high": mean + margin if n > 1 else math.nan,
        "win_fraction": sum(value > 0 for value in values) / n,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def _load_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in run_rows:
        seed = int(run["trace_seed"])
        for row in _read_csv(Path(run["quality_path"])):
            if str(row.get("nll_improvement_vs_all_analog", "")) == "":
                raise ValueError(
                    f"{run['quality_path']} lacks paired all-analog improvements."
                )
            rows.append({"trace_seed": seed, **row})
    if not rows:
        raise ValueError("No completed hybrid baseline rows were found.")
    return rows


def aggregate(run_rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Treat each independent hardware trace as one inferential unit."""
    rows = _load_rows(run_rows)
    by_design: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_design.setdefault(str(row["design"]), []).append(row)
    output: list[dict[str, Any]] = []
    for design, group in sorted(by_design.items()):
        first = group[0]
        record: dict[str, Any] = {
            "design": design,
            "digital_weight_mode": first.get("digital_weight_mode", "unclipped"),
            "mapping_policy": first.get("policy", ""),
            "trace_seeds": ";".join(
                str(seed) for seed in sorted({int(row["trace_seed"]) for row in group})
            ),
            "n_traces": len({int(row["trace_seed"]) for row in group}),
        }
        for label, column in IMPROVEMENT_COLUMNS:
            per_trace: dict[int, list[float]] = {}
            for row in group:
                value = str(row.get(column, ""))
                if value == "":
                    continue
                per_trace.setdefault(int(row["trace_seed"]), []).append(float(value))
            if not per_trace:
                continue
            stats = _trace_stats({seed: _mean(values) for seed, values in per_trace.items()})
            if label == "vs_all_analog":
                # Historical column names (kept for downstream readers).
                record["mean_nll_improvement_vs_all_analog"] = stats["mean"]
                record["std_across_traces"] = stats["std"]
                record["t_ci95_low"] = stats["ci_low"]
                record["t_ci95_high"] = stats["ci_high"]
                record["trace_win_fraction"] = stats["win_fraction"]
                record["min_trace_improvement"] = stats["min"]
                record["max_trace_improvement"] = stats["max"]
            else:
                record[f"mean_nll_improvement_{label}"] = stats["mean"]
                record[f"std_across_traces_{label}"] = stats["std"]
                record[f"t_ci95_low_{label}"] = stats["ci_low"]
                record[f"t_ci95_high_{label}"] = stats["ci_high"]
                record[f"trace_win_fraction_{label}"] = stats["win_fraction"]
                record[f"min_trace_improvement_{label}"] = stats["min"]
                record[f"max_trace_improvement_{label}"] = stats["max"]
        record["digital_projection_mac_fraction"] = float(
            first["digital_projection_mac_fraction"]
        )
        nominal = str(first.get("nominal_nll", ""))
        record["nominal_nll"] = float(nominal) if nominal else ""
        clipped_reference = str(first.get("checkpoint_clipped_digital_nll", ""))
        record["checkpoint_clipped_digital_nll"] = (
            float(clipped_reference) if clipped_reference else ""
        )
        output.append(record)
    if not output:
        raise ValueError("No completed hybrid baseline rows were found.")
    return write_csv(output_path, output)


def aggregate_by_timestep(run_rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Per-(design, timestep) trace-level statistics of the paired improvements."""
    rows = _load_rows(run_rows)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["design"]), int(row["timestep"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (design, timestep), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "design": design,
            "timestep": timestep,
            "n_traces": len({int(row["trace_seed"]) for row in group}),
            "mean_nll": _mean([float(row["nll"]) for row in group]),
        }
        for label, column in IMPROVEMENT_COLUMNS:
            per_trace: dict[int, list[float]] = {}
            for row in group:
                value = str(row.get(column, ""))
                if value == "":
                    continue
                per_trace.setdefault(int(row["trace_seed"]), []).append(float(value))
            if not per_trace:
                continue
            stats = _trace_stats({seed: _mean(values) for seed, values in per_trace.items()})
            record[f"mean_nll_improvement_{label}"] = stats["mean"]
            record[f"t_ci95_low_{label}"] = stats["ci_low"]
            record[f"t_ci95_high_{label}"] = stats["ci_high"]
            record[f"trace_win_fraction_{label}"] = stats["win_fraction"]
        output.append(record)
    return write_csv(output_path, output)


def export_paper_rows(run_rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Compact per-condition rows (the columns the paper's data folder keeps)."""
    rows = _load_rows(run_rows)
    compact = [
        {column: row.get(column, "") for column in PAPER_COLUMNS}
        for row in sorted(
            rows,
            key=lambda row: (
                int(row["trace_seed"]),
                str(row["design"]),
                int(row["timestep"]),
                int(row["realization"]),
            ),
        )
    ]
    return write_csv(output_path, compact)


def main(
    base_config_path: Path,
    phase1_path: Path,
    manifest_path: Path,
    trace_seeds: list[int] | None,
    *,
    run_name: str | None = None,
    digital_weight_mode: str | None = None,
) -> Path:
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config.get("hybrid_baselines"), dict):
        raise ValueError(f"{base_config_path} does not define hybrid_baselines.")
    if run_name:
        base_config["hybrid_baselines"]["run_name"] = str(run_name)
    if digital_weight_mode:
        base_config["hybrid_baselines"]["digital_weight_mode"] = str(digital_weight_mode)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    runs = list(manifest.get("runs", []))
    selected = None if trace_seeds is None else set(trace_seeds)
    if selected is not None:
        runs = [run for run in runs if int(run["trace_seed"]) in selected]
        missing = selected - {int(run["trace_seed"]) for run in runs}
        if missing:
            raise ValueError(f"Trace seeds absent from the manifest: {sorted(missing)}")
    if not runs:
        raise ValueError("No trace runs selected.")

    run_name = str(base_config["hybrid_baselines"].get("run_name", "hybrid_baselines"))
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("hybrid_baselines.run_name must be one safe directory name.")
    campaign_root = manifest_path.parent / run_name
    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest_out = campaign_root / "hybrid_baselines_run_manifest.yaml"
    completed_by_seed: dict[int, dict[str, Any]] = {}
    if manifest_out.is_file():
        prior = yaml.safe_load(manifest_out.read_text(encoding="utf-8")) or {}
        for prior_run in prior.get("runs", []):
            quality_path = Path(str(prior_run.get("quality_path", "")))
            metadata_path = Path(str(prior_run.get("metadata_path", "")))
            if quality_path.is_file() and metadata_path.is_file():
                completed_by_seed[int(prior_run["trace_seed"])] = dict(prior_run)

    def save_campaign_manifest() -> None:
        manifest_out.write_text(
            yaml.safe_dump(
                {
                    "base_config": str(base_config_path.resolve()),
                    "phase1": str(phase1_path.resolve()),
                    "all_analog_manifest": str(manifest_path.resolve()),
                    "digital_weight_mode": str(
                        base_config["hybrid_baselines"].get("digital_weight_mode", "unclipped")
                    ),
                    "runs": [completed_by_seed[seed] for seed in sorted(completed_by_seed)],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    for run in sorted(runs, key=lambda value: int(value["trace_seed"])):
        seed = int(run["trace_seed"])
        prior = completed_by_seed.get(seed)
        if prior is not None:
            print(f"Trace seed {seed}: already complete ({prior['quality_path']}); skipping.")
            continue
        trace_root = Path(run["output_root"])
        organized_root = manifest_path.parent / f"trace_seed_{seed}"
        if organized_root.is_dir() and not (
            trace_root / "phase4" / "hybrid_quality_by_policy.csv"
        ).is_file():
            print(
                f"Trace seed {seed}: manifest root is stale; using reorganized Drive root "
                f"{organized_root}"
            )
            trace_root = organized_root
        trace_path = _one_file(trace_root, "phase2/**/trace.npz", "Phase-2 trace")
        all_analog_path = trace_root / "phase4" / "hybrid_quality_by_policy.csv"
        if not all_analog_path.is_file():
            raise FileNotFoundError(
                f"Missing all-analog paired reference for trace seed {seed}: {all_analog_path}"
            )
        runtime_config = dict(base_config)
        configured_runtime = Path(str(run.get("runtime_config", "")))
        if configured_runtime.is_file():
            runtime_config = yaml.safe_load(configured_runtime.read_text(encoding="utf-8"))
            runtime_config["hybrid_baselines"] = dict(base_config["hybrid_baselines"])
            # Archived per-trace configs predate the Drive reorganization and may
            # point at the removed seed_42 checkpoint tree.  The checkpoint is
            # shared across traces, so the current base config is authoritative.
            runtime_config["model"] = dict(base_config["model"])
        runtime_config.setdefault("experiment", {})["seed"] = seed
        runtime_config["experiment"]["placement_seed"] = seed
        output_root = trace_root / run_name
        runtime_config["hybrid_baselines"]["output_root"] = str(output_root)
        generated_config = trace_root / f"{run_name}_runtime_config.yaml"
        generated_config.write_text(
            yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8"
        )
        command = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "experiments/phase4_quality/run_hybrid_baselines.py"),
            "--config",
            str(generated_config),
            "--phase1",
            str(phase1_path),
            "--trace",
            str(trace_path),
            "--output-dir",
            str(output_root),
            "--all-analog-quality",
            str(all_analog_path),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        completed_by_seed[seed] = {
            "trace_seed": seed,
            "output_root": str(output_root),
            "quality_path": str(output_root / "hybrid_baselines_by_condition.csv"),
            "metadata_path": str(output_root / "hybrid_baselines_metadata.json"),
        }
        save_campaign_manifest()

    completed_runs = [completed_by_seed[seed] for seed in sorted(completed_by_seed)]
    summary = aggregate(
        completed_runs, campaign_root / "hybrid_baselines_cross_trace_summary.csv"
    )
    aggregate_by_timestep(
        completed_runs, campaign_root / "hybrid_baselines_cross_trace_by_timestep.csv"
    )
    export_paper_rows(
        completed_runs, campaign_root / "hybrid_baselines_paper_by_condition.csv"
    )
    print(f"Hybrid baseline campaign complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override hybrid_baselines.run_name (campaign directory name).",
    )
    parser.add_argument(
        "--digital-weight-mode",
        type=str,
        default=None,
        choices=["clipped", "unclipped"],
        help="Override hybrid_baselines.digital_weight_mode.",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.phase1,
        args.manifest,
        args.trace_seeds,
        run_name=args.run_name,
        digital_weight_mode=args.digital_weight_mode,
    )
