"""First-order analytical per-token energy model for the hybrid deployment.

The model separates a token's MAC work into the digital-protected fraction
and the analog crossbar fraction. Digital MACs cost a per-op constant. An
analog shard placed on one physical tier costs, per token:

- one DAC line drive per input column of the shard,
- one in-crossbar analog MAC per weight, and
- one ADC conversion per output row of the shard.

All constants are configurable (``cost_model``) with literature-anchored
defaults cited in the shipped configurations; see docs/REFERENCES.md. This
is deliberately a first-order accounting: it excludes data movement,
digital-side SRAM, peripheral control, and inter-tile communication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EnergyModelParams:
    e_mac_digital_pj: float
    e_analog_mac_pj: float
    e_adc_pj: float
    e_dac_pj: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EnergyModelParams":
        cost = config["cost_model"]
        return cls(
            e_mac_digital_pj=float(cost["e_mac_digital_pj"]),
            e_analog_mac_pj=float(cost["e_analog_mac_pj"]),
            e_adc_pj=float(cost["e_adc_pj"]),
            e_dac_pj=float(cost["e_dac_pj"]),
        )

    def validate(self) -> None:
        for name, value in (
            ("e_mac_digital_pj", self.e_mac_digital_pj),
            ("e_analog_mac_pj", self.e_analog_mac_pj),
            ("e_adc_pj", self.e_adc_pj),
            ("e_dac_pj", self.e_dac_pj),
        ):
            if not value >= 0.0:
                raise ValueError(f"{name} must be non-negative.")


def analog_shard_energy_pj(
    row_span: int, col_span: int, params: EnergyModelParams
) -> dict[str, float]:
    """Per-token energy breakdown for one placed shard.

    Shards live in canonical [out, in] coordinates: ``row_span`` output rows
    (each needing one ADC conversion) and ``col_span`` input columns (each
    needing one DAC drive).
    """
    if row_span <= 0 or col_span <= 0:
        raise ValueError("Shard spans must be positive.")
    mac = float(row_span * col_span) * params.e_analog_mac_pj
    adc = float(row_span) * params.e_adc_pj
    dac = float(col_span) * params.e_dac_pj
    return {
        "analog_mac_pj": mac,
        "adc_pj": adc,
        "dac_pj": dac,
        "total_pj": mac + adc + dac,
    }


def operating_point_energy(
    point: Mapping[str, Any],
    placement_rows: Sequence[Mapping[str, Any]],
    params: EnergyModelParams,
) -> dict[str, float]:
    """Per-token energy for one operating point under one placement.

    ``point`` is a Phase-1.5 operating-point record (supplies
    ``digital_macs_per_token``); ``placement_rows`` is one Phase-3 placement
    CSV (supplies shard row/col extents). The placement policy does not change
    the energy — every policy places the same shard set — but rows are taken
    from a placement so the accounting covers exactly the placed shards.
    """
    params.validate()
    digital_energy = float(point["digital_macs_per_token"]) * params.e_mac_digital_pj
    analog_mac = 0.0
    adc = 0.0
    dac = 0.0
    analog_macs = 0
    for row in placement_rows:
        row_span = int(row["row_end"]) - int(row["row_start"])
        col_span = int(row["col_end"]) - int(row["col_start"])
        breakdown = analog_shard_energy_pj(row_span, col_span, params)
        analog_mac += breakdown["analog_mac_pj"]
        adc += breakdown["adc_pj"]
        dac += breakdown["dac_pj"]
        analog_macs += row_span * col_span
    expected_analog_macs = int(point["total_macs_per_token"]) - int(
        point["digital_macs_per_token"]
    )
    if analog_macs != expected_analog_macs:
        raise ValueError(
            f"Placement covers {analog_macs} analog MACs but the operating "
            f"point expects {expected_analog_macs}."
        )
    analog_energy = analog_mac + adc + dac
    total = digital_energy + analog_energy
    total_macs = int(point["total_macs_per_token"])
    return {
        "digital_energy_pj_per_token": digital_energy,
        "analog_mac_energy_pj_per_token": analog_mac,
        "adc_energy_pj_per_token": adc,
        "dac_energy_pj_per_token": dac,
        "analog_energy_pj_per_token": analog_energy,
        "total_energy_pj_per_token": total,
        "energy_per_mac_pj": total / total_macs if total_macs else 0.0,
        "placed_shards": float(len(placement_rows)),
    }


def all_digital_energy_pj(point: Mapping[str, Any], params: EnergyModelParams) -> float:
    """Reference: the whole candidate universe executed digitally."""
    return float(point["total_macs_per_token"]) * params.e_mac_digital_pj
