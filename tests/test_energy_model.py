"""Hand-checked arithmetic for the analytical energy model."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.pareto import pareto_frontier
from src.cost.energy_model import (
    EnergyModelParams,
    all_digital_energy_pj,
    analog_shard_energy_pj,
    operating_point_energy,
)

PARAMS = EnergyModelParams(
    e_mac_digital_pj=0.25,
    e_analog_mac_pj=0.005,
    e_adc_pj=1.5,
    e_dac_pj=0.1,
)


def placement_row(row_start, row_end, col_start, col_end):
    return {
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end,
    }


def test_two_shard_operating_point_matches_hand_computation():
    # 4x6 projection split into two 4x3 shards; a second 2x2 projection is digital.
    point = {
        "digital_macs_per_token": 4,      # the 2x2 digital projection
        "total_macs_per_token": 4 + 24,   # plus the 4x6 analog projection
    }
    rows = [placement_row(0, 4, 0, 3), placement_row(0, 4, 3, 6)]
    energy = operating_point_energy(point, rows, PARAMS)
    assert energy["digital_energy_pj_per_token"] == pytest.approx(4 * 0.25)
    assert energy["analog_mac_energy_pj_per_token"] == pytest.approx(24 * 0.005)
    assert energy["adc_energy_pj_per_token"] == pytest.approx((4 + 4) * 1.5)
    assert energy["dac_energy_pj_per_token"] == pytest.approx((3 + 3) * 0.1)
    assert energy["total_energy_pj_per_token"] == pytest.approx(
        1.0 + 0.12 + 12.0 + 0.6
    )
    assert energy["placed_shards"] == 2.0


def test_moving_a_projection_digital_shifts_mac_energy():
    analog_point = {"digital_macs_per_token": 0, "total_macs_per_token": 24}
    digital_point = {"digital_macs_per_token": 24, "total_macs_per_token": 24}
    analog_rows = [placement_row(0, 4, 0, 3), placement_row(0, 4, 3, 6)]
    analog = operating_point_energy(analog_point, analog_rows, PARAMS)
    digital = operating_point_energy(digital_point, [], PARAMS)
    assert digital["digital_energy_pj_per_token"] == pytest.approx(24 * 0.25)
    assert digital["analog_energy_pj_per_token"] == 0.0
    assert analog["digital_energy_pj_per_token"] == 0.0
    assert analog["analog_mac_energy_pj_per_token"] == pytest.approx(24 * 0.005)
    assert digital["total_energy_pj_per_token"] == pytest.approx(
        all_digital_energy_pj(digital_point, PARAMS)
    )


def test_adc_dominates_for_row_heavy_sliver_shard():
    # Canonical [out, in]: 512 output rows x 1 input column. If rows/cols were
    # swapped, DAC energy would dominate instead — this guards orientation.
    breakdown = analog_shard_energy_pj(512, 1, PARAMS)
    assert breakdown["adc_pj"] == pytest.approx(512 * 1.5)
    assert breakdown["dac_pj"] == pytest.approx(0.1)
    assert breakdown["adc_pj"] > 100 * breakdown["dac_pj"]


def test_mac_coverage_mismatch_is_an_error():
    point = {"digital_macs_per_token": 0, "total_macs_per_token": 100}
    with pytest.raises(ValueError, match="analog MACs"):
        operating_point_energy(point, [placement_row(0, 4, 0, 6)], PARAMS)


def test_pareto_flagging_on_synthetic_rows():
    rows = [
        {"digital_set_id": "a", "total_energy_pj_per_token": 1.0, "mean_degraded_nll": 5.0},
        {"digital_set_id": "b", "total_energy_pj_per_token": 2.0, "mean_degraded_nll": 3.0},
        {"digital_set_id": "c", "total_energy_pj_per_token": 3.0, "mean_degraded_nll": 4.0},  # dominated by b
    ]
    optimal = pareto_frontier(
        rows,
        cost_field="total_energy_pj_per_token",
        quality_field="mean_degraded_nll",
    )
    assert [row["digital_set_id"] for row in optimal] == ["a", "b"]
