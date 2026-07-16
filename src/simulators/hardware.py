"""Minimal structural model of the heterogeneous 3D-CIM substrate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HardwareConfig:
    num_tiles: int
    tiers_per_tile: int
    tier_rows: int
    tier_cols: int
    num_thermal_zones: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HardwareConfig":
        hw = config["hardware"]
        return cls(
            num_tiles=int(hw["num_tiles"]),
            tiers_per_tile=int(hw["tiers_per_tile"]),
            tier_rows=int(hw["tier_shape"]["rows"]),
            tier_cols=int(hw["tier_shape"]["cols"]),
            num_thermal_zones=int(hw.get("num_thermal_zones", 1)),
        )

    @property
    def total_tiers(self) -> int:
        return self.num_tiles * self.tiers_per_tile

    def validate(self) -> None:
        if min(self.num_tiles, self.tiers_per_tile, self.tier_rows, self.tier_cols, self.num_thermal_zones) <= 0:
            raise ValueError("Hardware dimensions must be positive.")
