"""Cross-trace aggregation tests for hybrid baselines."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hybrid_baselines_multiseed import (
    aggregate,
    aggregate_by_timestep,
    export_paper_rows,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_uses_trace_means_as_independent_units(tmp_path: Path):
    runs = []
    for seed, values in [(41, [0.1, 0.3]), (42, [0.2, 0.4])]:
        path = tmp_path / f"seed_{seed}.csv"
        _write(
            path,
            [
                {
                    "design": "hybrid_lm_head",
                    "nll_improvement_vs_all_analog": value,
                    "digital_projection_mac_fraction": 0.45,
                }
                for value in values
            ],
        )
        runs.append({"trace_seed": seed, "quality_path": str(path)})
    output = aggregate(runs, tmp_path / "summary.csv")
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 1
    assert int(rows[0]["n_traces"]) == 2
    assert float(rows[0]["mean_nll_improvement_vs_all_analog"]) == pytest.approx(0.25)
    assert float(rows[0]["trace_win_fraction"]) == 1.0
    # Historical artifacts without the proposed-policy column still aggregate.
    assert "mean_nll_improvement_vs_static_sensitivity" not in rows[0] or (
        rows[0]["mean_nll_improvement_vs_static_sensitivity"] == ""
    )


def test_aggregate_reports_proposed_policy_and_timesteps(tmp_path: Path):
    runs = []
    for seed, (vs_hw, vs_ss) in {41: (0.10, -0.02), 42: (0.06, 0.01), 43: (0.08, -0.01)}.items():
        path = tmp_path / f"seed_{seed}.csv"
        _write(
            path,
            [
                {
                    "design": "hybrid_lm_head",
                    "digital_weight_mode": "clipped",
                    "policy": "hardware_only",
                    "timestep": timestep,
                    "realization": 0,
                    "nll": 3.6,
                    "all_analog_nll": 3.7,
                    "nll_improvement_vs_all_analog": vs_hw + (0.02 if timestep else 0.0),
                    "static_sensitivity_nll": 3.65,
                    "nll_improvement_vs_static_sensitivity": vs_ss,
                    "nominal_nll": 3.45,
                    "checkpoint_digital_nll": 3.76,
                    "checkpoint_clipped_digital_nll": 3.44,
                    "digital_projection_mac_fraction": 0.3124,
                    "predicted_tokens": 285417,
                }
                for timestep in (0, 119)
            ],
        )
        runs.append({"trace_seed": seed, "quality_path": str(path)})
    summary = list(csv.DictReader(aggregate(runs, tmp_path / "s.csv").open(encoding="utf-8")))
    assert len(summary) == 1
    row = summary[0]
    assert row["digital_weight_mode"] == "clipped"
    assert row["mapping_policy"] == "hardware_only"
    assert int(row["n_traces"]) == 3
    assert float(row["mean_nll_improvement_vs_all_analog"]) == pytest.approx(0.09)
    assert float(row["mean_nll_improvement_vs_static_sensitivity"]) == pytest.approx(-0.02 / 3)
    assert float(row["trace_win_fraction_vs_static_sensitivity"]) == pytest.approx(1 / 3)
    assert float(row["checkpoint_clipped_digital_nll"]) == pytest.approx(3.44)

    by_timestep = list(
        csv.DictReader(
            aggregate_by_timestep(runs, tmp_path / "t.csv").open(encoding="utf-8")
        )
    )
    assert [int(r["timestep"]) for r in by_timestep] == [0, 119]
    assert float(by_timestep[0]["mean_nll_improvement_vs_all_analog"]) == pytest.approx(0.08)
    assert float(by_timestep[1]["mean_nll_improvement_vs_all_analog"]) == pytest.approx(0.10)

    paper = list(
        csv.DictReader(export_paper_rows(runs, tmp_path / "p.csv").open(encoding="utf-8"))
    )
    assert len(paper) == 6
    assert paper[0]["trace_seed"] == "41" and paper[-1]["trace_seed"] == "43"
    assert "nll_improvement_vs_static_sensitivity" in paper[0]
