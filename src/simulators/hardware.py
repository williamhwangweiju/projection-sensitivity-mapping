"""Core hardware data structures for Phase 2 fidelity simulation.

This module defines:

- The structural configuration of the simulated accelerator.
- The static and current state of each hardware tile.
- A validated collection of tile states.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


class HardwareConfigurationError(ValueError):
    """Raised when a hardware configuration is invalid."""

class TileStateError(ValueError):
    """Raised when a tile state is invalid."""

class FidelityClass(str, Enum):
    """Nominal fidelity class assigned to a tile at initialization.

    The class is a descriptive label. The continuous hardware-quality
    variable used by later phases should be ``current_noise_std``.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_value(cls, value: FidelityClass | str) -> FidelityClass:
        """Convert a string or enum instance into a FidelityClass."""
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TileStateError(
                "Fidelity class must be a string or FidelityClass, "
                f"received {type(value).__name__}."
            )

        normalized = value.strip().lower()

        try:
            return cls(normalized)
        except ValueError as exc:
            valid_values = ", ".join(item.value for item in cls)
            raise TileStateError(
                f"Unknown fidelity class {value!r}. "
                f"Expected one of: {valid_values}."
            ) from exc

@dataclass(frozen=True, slots=True)
class TileGeometry:
    """Shape of one analog-compute tier within a tile."""

    rows: int
    cols: int

    def __post_init__(self) -> None:
        _validate_positive_integer("rows", self.rows)
        _validate_positive_integer("cols", self.cols)

    @property
    def cells_per_tier(self) -> int:
        """Return the raw number of cells in one tier."""
        return self.rows * self.cols

    def to_dict(self) -> dict[str, int]:
        """Return a serializable representation."""
        return {
            "rows": self.rows,
            "cols": self.cols,
        }

@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """Structural configuration of the simulated accelerator.

    ``tiers_per_tile`` follows the terminology used by the IBM 3D-CIM
    simulator's ``AcceleratorConfig``. It is the number of compute tiers
    available inside each tile, not a tile's position in the stack.
    """

    num_tiles: int
    tiers_per_tile: int
    tile_geometry: TileGeometry
    num_thermal_zones: int = 1

    def __post_init__(self) -> None:
        _validate_positive_integer("num_tiles", self.num_tiles)
        _validate_positive_integer("tiers_per_tile", self.tiers_per_tile)
        _validate_positive_integer(
            "num_thermal_zones",
            self.num_thermal_zones,
        )

        if self.num_thermal_zones > self.num_tiles:
            raise HardwareConfigurationError(
                "num_thermal_zones cannot exceed num_tiles: "
                f"{self.num_thermal_zones} > {self.num_tiles}."
            )

        if not isinstance(self.tile_geometry, TileGeometry):
            raise HardwareConfigurationError(
                "tile_geometry must be a TileGeometry instance."
            )

    @property
    def tile_rows(self) -> int:
        """Return the number of rows in each tier."""
        return self.tile_geometry.rows

    @property
    def tile_cols(self) -> int:
        """Return the number of columns in each tier."""
        return self.tile_geometry.cols

    @property
    def cells_per_tile(self) -> int:
        """Return the raw cell count across all tiers of one tile.

        This is a physical cell count, not necessarily the number of
        model weights that can be stored. Differential encoding,
        redundancy, and reserved cells may reduce usable capacity.
        """
        return self.tiers_per_tile * self.tile_geometry.cells_per_tier

    @property
    def total_cells(self) -> int:
        """Return the raw cell count across the complete accelerator."""
        return self.num_tiles * self.cells_per_tile

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical serializable configuration."""
        return {
            "num_tiles": self.num_tiles,
            "tiers_per_tile": self.tiers_per_tile,
            "tier_shape": self.tile_geometry.to_dict(),
            "num_thermal_zones": self.num_thermal_zones,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> HardwareConfig:
        """Construct a configuration from a dictionary-like object.

        Both project-style and IBM 3D-CIM-style field names are accepted.

        Supported examples::

            {
                "num_tiles": 64,
                "tiers_per_tile": 1024,
                "tier_shape": {
                    "rows": 512,
                    "cols": 512,
                },
                "num_thermal_zones": 4,
            }

        and::

            {
                "tiles": 64,
                "tiers": 1024,
                "tier_shape": [512, 512],
                "num_thermal_zones": 4,
            }
        """
        if not isinstance(data, Mapping):
            raise HardwareConfigurationError(
                "Hardware configuration must be a mapping."
            )

        num_tiles = _get_required_alias(
            data,
            canonical_name="num_tiles",
            aliases=("tiles",),
        )
        tiers_per_tile = _get_required_alias(
            data,
            canonical_name="tiers_per_tile",
            aliases=("tiers",),
        )

        raw_shape = _get_required_alias(
            data,
            canonical_name="tier_shape",
            aliases=("tier_shape",),
        )
        tile_geometry = _parse_tile_geometry(raw_shape)

        num_thermal_zones = data.get("num_thermal_zones", 1)

        return cls(
            num_tiles=_to_integer("num_tiles", num_tiles),
            tiers_per_tile=_to_integer(
                "tiers_per_tile",
                tiers_per_tile,
            ),
            tile_geometry=tile_geometry,
            num_thermal_zones=_to_integer(
                "num_thermal_zones",
                num_thermal_zones,
            ),
        )

@dataclass(slots=True)
class TileState:
    """State of one hardware tile.

    ``fidelity_class`` is the tile's nominal initialization class. It
    should not be treated as the authoritative time-varying quality
    value. Later phases should use ``current_noise_std`` for mapping and
    quality estimation.

    ``drift_rate`` is stored as a tile-specific simulation parameter,
    but this class does not apply drift itself.
    """

    tile_id: int
    thermal_zone: int

    tiers_per_tile: int
    geometry: TileGeometry

    fidelity_class: FidelityClass

    base_noise_std: float
    current_noise_std: float
    drift_rate: float = 0.0

    available: bool = True
    faulted: bool = False
    fault_timestep: int | None = None

    def __post_init__(self) -> None:
        self.tile_id = _to_nonnegative_integer(
            "tile_id",
            self.tile_id,
            error_type=TileStateError,
        )
        self.thermal_zone = _to_nonnegative_integer(
            "thermal_zone",
            self.thermal_zone,
            error_type=TileStateError,
        )
        self.tiers_per_tile = _to_positive_integer(
            "tiers_per_tile",
            self.tiers_per_tile,
            error_type=TileStateError,
        )

        if not isinstance(self.geometry, TileGeometry):
            raise TileStateError(
                "geometry must be a TileGeometry instance."
            )

        self.fidelity_class = FidelityClass.from_value(
            self.fidelity_class
        )

        self.base_noise_std = _to_nonnegative_float(
            "base_noise_std",
            self.base_noise_std,
        )
        self.current_noise_std = _to_nonnegative_float(
            "current_noise_std",
            self.current_noise_std,
        )
        self.drift_rate = _to_nonnegative_float(
            "drift_rate",
            self.drift_rate,
        )

        if not isinstance(self.available, bool):
            raise TileStateError("available must be a boolean.")

        if not isinstance(self.faulted, bool):
            raise TileStateError("faulted must be a boolean.")

        if self.fault_timestep is not None:
            self.fault_timestep = _to_nonnegative_integer(
                "fault_timestep",
                self.fault_timestep,
                error_type=TileStateError,
            )

            if not self.faulted:
                raise TileStateError(
                    "fault_timestep cannot be set when faulted is False."
                )

    @property
    def rows(self) -> int:
        """Return the number of rows in one tier."""
        return self.geometry.rows

    @property
    def cols(self) -> int:
        """Return the number of columns in one tier."""
        return self.geometry.cols

    @property
    def cells_per_tier(self) -> int:
        """Return the raw number of cells in one tier."""
        return self.geometry.cells_per_tier

    @property
    def total_cells(self) -> int:
        """Return the raw number of cells across all tile tiers."""
        return self.tiers_per_tile * self.cells_per_tier

    def set_current_noise_std(self, noise_std: float) -> None:
        """Update the tile's effective noise standard deviation."""
        self.current_noise_std = _to_nonnegative_float(
            "current_noise_std",
            noise_std,
        )

    def mark_faulted(
        self,
        timestep: int,
        *,
        make_unavailable: bool = False,
    ) -> None:
        """Mark this tile as faulted.

        A faulted tile may remain available. This supports degradation
        events where the tile becomes noisier but can still execute
        operations.

        Args:
            timestep: Timestep at which the fault occurred.
            make_unavailable: Whether the fault removes the tile from
                the set of usable hardware resources.
        """
        self.faulted = True
        self.fault_timestep = _to_nonnegative_integer(
            "timestep",
            timestep,
            error_type=TileStateError,
        )

        if make_unavailable:
            self.available = False

    def clear_fault(self) -> None:
        """Clear fault metadata without changing the current noise."""
        self.faulted = False
        self.fault_timestep = None

    def reset_dynamic_state(self) -> None:
        """Restore this tile to its initial simulation state."""
        self.current_noise_std = self.base_noise_std
        self.available = True
        self.faulted = False
        self.fault_timestep = None

    def clone(self) -> TileState:
        """Return an independent copy of this tile state."""
        return replace(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serializable tile-state record."""
        return {
            "tile_id": self.tile_id,
            "thermal_zone": self.thermal_zone,
            "tiers_per_tile": self.tiers_per_tile,
            "tile_rows": self.rows,
            "tile_cols": self.cols,
            "fidelity_class": self.fidelity_class.value,
            "base_noise_std": self.base_noise_std,
            "current_noise_std": self.current_noise_std,
            "drift_rate": self.drift_rate,
            "available": self.available,
            "faulted": self.faulted,
            "fault_timestep": self.fault_timestep,
            "total_cells": self.total_cells,
        }

@dataclass(slots=True)
class HardwareState:
    """Validated collection of tile states."""

    config: HardwareConfig
    tiles: list[TileState]

    def __post_init__(self) -> None:
        if not isinstance(self.config, HardwareConfig):
            raise HardwareConfigurationError(
                "config must be a HardwareConfig instance."
            )

        self.tiles = list(self.tiles)

        if len(self.tiles) != self.config.num_tiles:
            raise HardwareConfigurationError(
                "Tile count does not match hardware configuration: "
                f"received {len(self.tiles)}, "
                f"expected {self.config.num_tiles}."
            )

        expected_ids = set(range(self.config.num_tiles))
        actual_ids = {tile.tile_id for tile in self.tiles}

        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)

            raise HardwareConfigurationError(
                "Tile IDs must be unique and cover "
                f"0 through {self.config.num_tiles - 1}. "
                f"Missing IDs: {missing}. "
                f"Unexpected IDs: {unexpected}."
            )

        for tile in self.tiles:
            self._validate_tile_against_config(tile)

        self.tiles.sort(key=lambda tile: tile.tile_id)

    def _validate_tile_against_config(self, tile: TileState) -> None:
        if not isinstance(tile, TileState):
            raise HardwareConfigurationError(
                "All hardware entries must be TileState instances."
            )

        if tile.geometry != self.config.tile_geometry:
            raise HardwareConfigurationError(
                f"Tile {tile.tile_id} geometry {tile.geometry} does not "
                f"match configured geometry "
                f"{self.config.tile_geometry}."
            )

        if tile.tiers_per_tile != self.config.tiers_per_tile:
            raise HardwareConfigurationError(
                f"Tile {tile.tile_id} has {tile.tiers_per_tile} tiers, "
                f"but the hardware configuration requires "
                f"{self.config.tiers_per_tile}."
            )

        if tile.thermal_zone >= self.config.num_thermal_zones:
            raise HardwareConfigurationError(
                f"Tile {tile.tile_id} uses thermal zone "
                f"{tile.thermal_zone}, but valid zones range from 0 to "
                f"{self.config.num_thermal_zones - 1}."
            )

    def get_tile(self, tile_id: int) -> TileState:
        """Return a tile by its integer identifier."""
        normalized_id = _to_nonnegative_integer(
            "tile_id",
            tile_id,
            error_type=TileStateError,
        )

        if normalized_id >= len(self.tiles):
            raise KeyError(f"Unknown tile ID: {normalized_id}.")

        return self.tiles[normalized_id]

    @property
    def available_tiles(self) -> list[TileState]:
        """Return all currently available tiles."""
        return [tile for tile in self.tiles if tile.available]

    @property
    def faulted_tiles(self) -> list[TileState]:
        """Return all currently faulted tiles."""
        return [tile for tile in self.tiles if tile.faulted]

    def reset_dynamic_state(self) -> None:
        """Reset every tile to its initial state."""
        for tile in self.tiles:
            tile.reset_dynamic_state()

    def clone(self) -> HardwareState:
        """Return an independent copy of the complete hardware state."""
        return HardwareState(
            config=self.config,
            tiles=[tile.clone() for tile in self.tiles],
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return tile states as flat serializable records."""
        return [tile.to_dict() for tile in self.tiles]

    @classmethod
    def from_assignments(
        cls,
        config: HardwareConfig,
        *,
        base_noise_std_by_tile: Sequence[float],
        fidelity_class_by_tile: Sequence[FidelityClass | str],
        drift_rate_by_tile: Sequence[float] | None = None,
        thermal_zone_by_tile: Sequence[int] | None = None,
    ) -> HardwareState:
        """Build a complete hardware state from per-tile assignments.

        This method performs no random sampling. Random fidelity-class,
        noise, and drift assignments should be generated externally and then passed into this factory.

        When thermal zones are not supplied, tiles are divided into
        contiguous, approximately equal-sized thermal zones.
        """
        _validate_sequence_length(
            "base_noise_std_by_tile",
            base_noise_std_by_tile,
            config.num_tiles,
        )
        _validate_sequence_length(
            "fidelity_class_by_tile",
            fidelity_class_by_tile,
            config.num_tiles,
        )

        if drift_rate_by_tile is None:
            drift_rate_by_tile = [0.0] * config.num_tiles
        else:
            _validate_sequence_length(
                "drift_rate_by_tile",
                drift_rate_by_tile,
                config.num_tiles,
            )

        if thermal_zone_by_tile is None:
            thermal_zone_by_tile = [
                _balanced_group_index(
                    item_index=tile_id,
                    item_count=config.num_tiles,
                    group_count=config.num_thermal_zones,
                )
                for tile_id in range(config.num_tiles)
            ]
        else:
            _validate_sequence_length(
                "thermal_zone_by_tile",
                thermal_zone_by_tile,
                config.num_tiles,
            )

        tiles: list[TileState] = []

        for tile_id in range(config.num_tiles):
            tile = TileState(
                tile_id=tile_id,
                thermal_zone=thermal_zone_by_tile[tile_id],
                tiers_per_tile=config.tiers_per_tile,
                geometry=config.tile_geometry,
                fidelity_class=fidelity_class_by_tile[tile_id],
                base_noise_std=base_noise_std_by_tile[tile_id],
                current_noise_std=base_noise_std_by_tile[tile_id],
                drift_rate=drift_rate_by_tile[tile_id],
            )
            tiles.append(tile)

        return cls(config=config, tiles=tiles)

def _parse_tile_geometry(raw_shape: Any) -> TileGeometry:
    """Parse a tile shape from a mapping or two-element sequence."""
    if isinstance(raw_shape, Mapping):
        try:
            rows = raw_shape["rows"]
            cols = raw_shape["cols"]
        except KeyError as exc:
            raise HardwareConfigurationError(
                "Tile-shape mapping must contain 'rows' and 'cols'."
            ) from exc

        return TileGeometry(
            rows=_to_integer("tier_shape.rows", rows),
            cols=_to_integer("tier_shape.cols", cols),
        )

    if (
        isinstance(raw_shape, Sequence)
        and not isinstance(raw_shape, (str, bytes))
    ):
        if len(raw_shape) != 2:
            raise HardwareConfigurationError(
                "Tier-shape sequence must contain exactly two values: "
                "[rows, cols]."
            )

        return TileGeometry(
            rows=_to_integer("tier_shape[0]", raw_shape[0]),
            cols=_to_integer("tier_shape[1]", raw_shape[1]),
        )

    raise HardwareConfigurationError(
        "Tier shape must be either a mapping with rows/cols or a "
        "two-element sequence."
    )

def _get_required_alias(
    data: Mapping[str, Any],
    *,
    canonical_name: str,
    aliases: Sequence[str],
) -> Any:
    """Get a required configuration field using supported aliases."""
    if canonical_name in data:
        return data[canonical_name]

    for alias in aliases:
        if alias in data:
            return data[alias]

    accepted = ", ".join(
        repr(name) for name in (canonical_name, *aliases)
    )
    raise HardwareConfigurationError(
        f"Missing required hardware configuration field. "
        f"Accepted names: {accepted}."
    )

def _balanced_group_index(
    *,
    item_index: int,
    item_count: int,
    group_count: int,
) -> int:
    """Assign contiguous items to approximately equal-sized groups."""
    return min(
        (item_index * group_count) // item_count,
        group_count - 1,
    )

def _validate_sequence_length(
    name: str,
    values: Sequence[Any],
    expected_length: int,
) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(
        values,
        Sequence,
    ):
        raise HardwareConfigurationError(
            f"{name} must be a sequence."
        )

    if len(values) != expected_length:
        raise HardwareConfigurationError(
            f"{name} contains {len(values)} values; "
            f"expected {expected_length}."
        )

def _validate_positive_integer(name: str, value: Any) -> None:
    _to_positive_integer(
        name,
        value,
        error_type=HardwareConfigurationError,
    )

def _to_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HardwareConfigurationError(
            f"{name} must be an integer, received {value!r}."
        )

    return int(value)

def _to_positive_integer(
    name: str,
    value: Any,
    *,
    error_type: type[ValueError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise error_type(
            f"{name} must be an integer, received {value!r}."
        )

    normalized = int(value)

    if normalized <= 0:
        raise error_type(
            f"{name} must be greater than zero, "
            f"received {normalized}."
        )

    return normalized

def _to_nonnegative_integer(
    name: str,
    value: Any,
    *,
    error_type: type[ValueError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise error_type(
            f"{name} must be an integer, received {value!r}."
        )

    normalized = int(value)

    if normalized < 0:
        raise error_type(
            f"{name} must be nonnegative, received {normalized}."
        )

    return normalized

def _to_nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TileStateError(
            f"{name} must be numeric, received {value!r}."
        )

    normalized = float(value)

    if not isfinite(normalized):
        raise TileStateError(
            f"{name} must be finite, received {normalized}."
        )

    if normalized < 0.0:
        raise TileStateError(
            f"{name} must be nonnegative, received {normalized}."
        )

    return normalized

__all__ = [
    "FidelityClass",
    "HardwareConfig",
    "HardwareConfigurationError",
    "HardwareState",
    "TileGeometry",
    "TileState",
    "TileStateError",
]