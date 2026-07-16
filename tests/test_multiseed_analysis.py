from pathlib import Path
import csv

from experiments.phase6_analysis.aggregate_multiseed_results import main
from src.common.config import save_json, load_json


def _write_quality(path: Path, trace_offset: float) -> None:
    rows = []
    for realization in range(2):
        rows.extend(
            [
                {
                    "digital_set_id": "digital_test",
                    "timestep": 0,
                    "realization": realization,
                    "policy": "hardware_only",
                    "delta_nll_tile": 0.30 + trace_offset,
                },
                {
                    "digital_set_id": "digital_test",
                    "timestep": 0,
                    "realization": realization,
                    "policy": "static_sensitivity",
                    "delta_nll_tile": 0.20 + trace_offset,
                },
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_multiseed_aggregation_bootstraps_trace_means(tmp_path: Path) -> None:
    metadata_paths = []
    for seed, offset in ((42, 0.0), (43, 0.05)):
        quality = tmp_path / f"quality_{seed}.csv"
        _write_quality(quality, offset)
        metadata = tmp_path / f"metadata_{seed}.json"
        save_json(
            metadata,
            {
                "experiment_seed": seed,
                "artifacts": {"quality": str(quality)},
            },
        )
        metadata_paths.append(metadata)
    manifest = main(metadata_paths, tmp_path / "out", 7)
    payload = load_json(manifest)
    summary_path = Path(payload["artifacts"]["paired_summary"])
    with summary_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = next(
        item
        for item in rows
        if item["baseline_policy"] == "hardware_only"
        and item["method_policy"] == "static_sensitivity"
    )
    assert int(row["independent_trace_count"]) == 2
    assert abs(float(row["mean_trace_level_nll_improvement"]) - 0.1) < 1e-12
