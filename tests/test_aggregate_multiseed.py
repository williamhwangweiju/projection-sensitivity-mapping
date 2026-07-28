"""Cross-trace paired aggregation arithmetic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_multiseed import per_trace_paired_means, t_quantile


def rows_for_trace(offset: float) -> list[dict[str, str]]:
    rows = []
    for timestep in (0, 1):
        for realization in (0, 1):
            for policy, delta in (
                ("random", 2.0 + offset),
                ("static_sensitivity", 1.0 + offset),
                ("static_fisher", 1.5 + offset),
            ):
                rows.append(
                    {
                        "digital_set_id": "set_a",
                        "timestep": str(timestep),
                        "realization": str(realization),
                        "policy": policy,
                        "delta_nll_tile": str(delta),
                    }
                )
    return rows


def test_per_trace_paired_means_are_offset_invariant():
    means = per_trace_paired_means(
        rows_for_trace(0.0),
        method_policies=["static_sensitivity", "static_fisher"],
        baseline_policies=["random"],
    )
    # The paired difference cancels any trace-wide offset, and the legacy
    # digital_set_id column in the synthetic rows is ignored.
    shifted = per_trace_paired_means(
        rows_for_trace(5.0),
        method_policies=["static_sensitivity", "static_fisher"],
        baseline_policies=["random"],
    )
    key = ("random", "static_sensitivity")
    assert means[key] == pytest.approx(1.0)
    assert shifted[key] == pytest.approx(1.0)
    assert means[("random", "static_fisher")] == pytest.approx(0.5)


def test_missing_policy_pairs_are_skipped():
    rows = [
        {
            "digital_set_id": "set_a",
            "timestep": "0",
            "realization": "0",
            "policy": "random",
            "delta_nll_tile": "2.0",
        }
    ]
    means = per_trace_paired_means(
        rows, method_policies=["static_sensitivity"], baseline_policies=["random"]
    )
    assert means == {}


def test_t_quantiles_cover_small_trace_counts():
    assert t_quantile(2) == pytest.approx(4.303)  # 3 traces
    assert t_quantile(1) == pytest.approx(12.706)  # 2 traces
    assert t_quantile(30) == pytest.approx(1.96, abs=0.15)
