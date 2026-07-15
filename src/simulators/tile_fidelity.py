"""Time-varying tile-fidelity model for Phase 2.

This module is responsible for:

- Initializing heterogeneous high-, medium-, and low-fidelity tiles.
- Assigning a baseline effective noise standard deviation to every tile.
- Assigning nonuniform gradual drift rates.
- Simulating temporally correlated thermal variation.
- Scheduling permanent localized degradation events.
- Updating the complete hardware state one timestep at a time.
- Producing NumPy trace arrays for later mapping phases.

The primary hardware-quality variable is ``current_noise_std``. The
high/medium/low fidelity labels are primarily descriptive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hardware import (
    FidelityClass,
    HardwareConfig,
    HardwareState,
)

class FidelityConfigurationError(ValueError):
    """Raised when the tile-fidelity configuration is invalid."""

class FidelitySimulationError(RuntimeError):
    """Raised when the tile-fidelity simulation is used incorrectly."""

FIDELITY_CLASS_TO_CODE: dict[FidelityClass, int] = {
    FidelityClass.HIGH: 0,
    FidelityClass.MEDIUM: 1,
    FidelityClass.LOW: 2,
}

FIDELITY_CODE_TO_CLASS: dict[int, FidelityClass] = {
    code: fidelity_class
    for fidelity_class, code in FIDELITY_CLASS_TO_CODE.items()
}

@dataclass(frozen=True, slots=True)
class FloatRange:
    """Validated inclusive floating-point range."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _to_finite_float("minimum", self.minimum)
        maximum = _to_finite_float("maximum", self.maximum)

        if minimum > maximum:
            raise FidelityConfigurationError(
                "Range minimum cannot exceed range maximum: "
                f"{minimum} > {maximum}."
            )

        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        """Sample uniformly from this range."""
        if self.minimum == self.maximum:
            if size is None:
                return self.minimum

            return np.full(size, self.minimum, dtype=np.float64)

        return rng.uniform(
            low=self.minimum,
            high=self.maximum,
            size=size,
        )

    def to_list(self) -> list[float]:
        """Return a YAML/JSON-friendly representation."""
        return [self.minimum, self.maximum]

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        name: str,
        nonnegative: bool = False,
    ) -> FloatRange:
        """Parse a range from a sequence or mapping.

        Supported forms::

            [0.05, 0.20]

        or::

            {
                "minimum": 0.05,
                "maximum": 0.20,
            }

        The shorter ``min`` and ``max`` mapping keys are also accepted.
        """
        if isinstance(value, Mapping):
            if "minimum" in value:
                minimum = value["minimum"]
            elif "min" in value:
                minimum = value["min"]
            else:
                raise FidelityConfigurationError(
                    f"{name} mapping is missing 'minimum' or 'min'."
                )

            if "maximum" in value:
                maximum = value["maximum"]
            elif "max" in value:
                maximum = value["max"]
            else:
                raise FidelityConfigurationError(
                    f"{name} mapping is missing 'maximum' or 'max'."
                )

        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            if len(value) != 2:
                raise FidelityConfigurationError(
                    f"{name} must contain exactly two values."
                )

            minimum = value[0]
            maximum = value[1]

        else:
            raise FidelityConfigurationError(
                f"{name} must be a two-element sequence or a mapping."
            )

        parsed = cls(
            minimum=_to_finite_float(f"{name}.minimum", minimum),
            maximum=_to_finite_float(f"{name}.maximum", maximum),
        )

        if nonnegative and parsed.minimum < 0.0:
            raise FidelityConfigurationError(
                f"{name} must be nonnegative."
            )

        return parsed

@dataclass(frozen=True, slots=True)
class IntRange:
    """Validated inclusive integer range."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        minimum = _to_integer("minimum", self.minimum)
        maximum = _to_integer("maximum", self.maximum)

        if minimum > maximum:
            raise FidelityConfigurationError(
                "Integer range minimum cannot exceed maximum: "
                f"{minimum} > {maximum}."
            )

        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
    ) -> int | np.ndarray:
        """Sample integers from this inclusive range."""
        if self.minimum == self.maximum:
            if size is None:
                return self.minimum

            return np.full(size, self.minimum, dtype=np.int64)

        return rng.integers(
            low=self.minimum,
            high=self.maximum + 1,
            size=size,
        )

    def to_list(self) -> list[int]:
        """Return a YAML/JSON-friendly representation."""
        return [self.minimum, self.maximum]

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        name: str,
        nonnegative: bool = False,
    ) -> IntRange:
        """Parse an integer range from a sequence or mapping."""
        if isinstance(value, Mapping):
            if "minimum" in value:
                minimum = value["minimum"]
            elif "min" in value:
                minimum = value["min"]
            else:
                raise FidelityConfigurationError(
                    f"{name} mapping is missing 'minimum' or 'min'."
                )

            if "maximum" in value:
                maximum = value["maximum"]
            elif "max" in value:
                maximum = value["max"]
            else:
                raise FidelityConfigurationError(
                    f"{name} mapping is missing 'maximum' or 'max'."
                )

        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            if len(value) != 2:
                raise FidelityConfigurationError(
                    f"{name} must contain exactly two integers."
                )

            minimum = value[0]
            maximum = value[1]

        else:
            raise FidelityConfigurationError(
                f"{name} must be a two-element sequence or a mapping."
            )

        parsed = cls(
            minimum=_to_integer(f"{name}.minimum", minimum),
            maximum=_to_integer(f"{name}.maximum", maximum),
        )

        if nonnegative and parsed.minimum < 0:
            raise FidelityConfigurationError(
                f"{name} must be nonnegative."
            )

        return parsed

@dataclass(frozen=True, slots=True)
class FidelityClassConfig:
    """Initialization configuration for one nominal fidelity class."""

    fraction: float
    noise_multiplier_range: FloatRange

    def __post_init__(self) -> None:
        fraction = _to_finite_float("fraction", self.fraction)

        if fraction < 0.0 or fraction > 1.0:
            raise FidelityConfigurationError(
                "Fidelity-class fraction must be between 0 and 1, "
                f"received {fraction}."
            )

        if self.noise_multiplier_range.minimum < 0.0:
            raise FidelityConfigurationError(
                "Noise multipliers must be nonnegative."
            )

        object.__setattr__(self, "fraction", fraction)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "fraction": self.fraction,
            "noise_multiplier_range": (
                self.noise_multiplier_range.to_list()
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        class_name: str,
    ) -> FidelityClassConfig:
        """Construct a class configuration from a mapping."""
        if not isinstance(data, Mapping):
            raise FidelityConfigurationError(
                f"Configuration for class {class_name!r} "
                "must be a mapping."
            )

        if "fraction" not in data:
            raise FidelityConfigurationError(
                f"Fidelity class {class_name!r} is missing 'fraction'."
            )

        multiplier_value = data.get(
            "noise_multiplier_range",
            data.get("multiplier_range"),
        )

        if multiplier_value is None:
            raise FidelityConfigurationError(
                f"Fidelity class {class_name!r} is missing "
                "'noise_multiplier_range'."
            )

        return cls(
            fraction=_to_finite_float(
                f"{class_name}.fraction",
                data["fraction"],
            ),
            noise_multiplier_range=FloatRange.from_value(
                multiplier_value,
                name=f"{class_name}.noise_multiplier_range",
                nonnegative=True,
            ),
        )

@dataclass(frozen=True, slots=True)
class GradualDriftConfig:
    """Configuration for tile-specific gradual degradation."""

    enabled: bool
    total_increase_range: FloatRange

    def __post_init__(self) -> None:
        if self.total_increase_range.minimum < 0.0:
            raise FidelityConfigurationError(
                "Gradual-drift increase must be nonnegative."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "total_increase_range": (
                self.total_increase_range.to_list()
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> GradualDriftConfig:
        """Construct gradual-drift settings."""
        data = data or {}

        return cls(
            enabled=_to_boolean(
                "gradual_drift.enabled",
                data.get("enabled", False),
            ),
            total_increase_range=FloatRange.from_value(
                data.get("total_increase_range", [0.0, 0.0]),
                name="gradual_drift.total_increase_range",
                nonnegative=True,
            ),
        )

@dataclass(frozen=True, slots=True)
class ThermalVariationConfig:
    """Configuration for correlated thermal fluctuations.

    ``standard_deviation_fraction`` is interpreted as the approximate
    stationary standard deviation of the fractional noise variation.
    """

    enabled: bool
    correlation: float
    standard_deviation_fraction: float

    def __post_init__(self) -> None:
        correlation = _to_finite_float(
            "thermal correlation",
            self.correlation,
        )
        standard_deviation = _to_finite_float(
            "thermal standard deviation",
            self.standard_deviation_fraction,
        )

        if correlation < 0.0 or correlation >= 1.0:
            raise FidelityConfigurationError(
                "Thermal correlation must satisfy 0 <= correlation < 1, "
                f"received {correlation}."
            )

        if standard_deviation < 0.0:
            raise FidelityConfigurationError(
                "Thermal standard deviation must be nonnegative."
            )

        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(
            self,
            "standard_deviation_fraction",
            standard_deviation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "correlation": self.correlation,
            "standard_deviation_fraction": (
                self.standard_deviation_fraction
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> ThermalVariationConfig:
        """Construct thermal-variation settings."""
        data = data or {}

        standard_deviation = data.get(
            "standard_deviation_fraction",
            data.get("std_fraction", 0.0),
        )

        return cls(
            enabled=_to_boolean(
                "thermal_variation.enabled",
                data.get("enabled", False),
            ),
            correlation=_to_finite_float(
                "thermal_variation.correlation",
                data.get("correlation", 0.94),
            ),
            standard_deviation_fraction=_to_finite_float(
                "thermal_variation.standard_deviation_fraction",
                standard_deviation,
            ),
        )

@dataclass(frozen=True, slots=True)
class LocalizedFaultConfig:
    """Configuration for permanent localized tile degradation."""

    enabled: bool
    num_affected_tiles: int
    onset_timestep_range: IntRange
    noise_increase_range: FloatRange
    candidate_classes: tuple[FidelityClass, ...]
    make_unavailable: bool = False

    def __post_init__(self) -> None:
        if self.num_affected_tiles < 0:
            raise FidelityConfigurationError(
                "num_affected_tiles must be nonnegative."
            )

        if self.onset_timestep_range.minimum < 0:
            raise FidelityConfigurationError(
                "Fault onset timesteps must be nonnegative."
            )

        if self.noise_increase_range.minimum < 0.0:
            raise FidelityConfigurationError(
                "Fault noise increase must be nonnegative."
            )

        if not self.candidate_classes:
            raise FidelityConfigurationError(
                "candidate_classes cannot be empty."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "num_affected_tiles": self.num_affected_tiles,
            "onset_timestep_range": (
                self.onset_timestep_range.to_list()
            ),
            "noise_increase_range": (
                self.noise_increase_range.to_list()
            ),
            "candidate_classes": [
                fidelity_class.value
                for fidelity_class in self.candidate_classes
            ],
            "make_unavailable": self.make_unavailable,
            "permanent": True,
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> LocalizedFaultConfig:
        """Construct localized-fault settings."""
        data = data or {}

        permanent = _to_boolean(
            "localized_fault.permanent",
            data.get("permanent", True),
        )

        if not permanent:
            raise FidelityConfigurationError(
                "The current Phase 2 implementation supports only "
                "permanent localized faults. Set permanent: true."
            )

        raw_candidate_classes = data.get(
            "candidate_classes",
            [
                FidelityClass.HIGH.value,
                FidelityClass.MEDIUM.value,
                FidelityClass.LOW.value,
            ],
        )

        if (
            not isinstance(raw_candidate_classes, Sequence)
            or isinstance(raw_candidate_classes, (str, bytes))
        ):
            raise FidelityConfigurationError(
                "localized_fault.candidate_classes must be a sequence."
            )

        candidate_classes = tuple(
            FidelityClass.from_value(value)
            for value in raw_candidate_classes
        )

        return cls(
            enabled=_to_boolean(
                "localized_fault.enabled",
                data.get("enabled", False),
            ),
            num_affected_tiles=_to_nonnegative_integer(
                "localized_fault.num_affected_tiles",
                data.get("num_affected_tiles", 0),
            ),
            onset_timestep_range=IntRange.from_value(
                data.get("onset_timestep_range", [0, 0]),
                name="localized_fault.onset_timestep_range",
                nonnegative=True,
            ),
            noise_increase_range=FloatRange.from_value(
                data.get("noise_increase_range", [0.0, 0.0]),
                name="localized_fault.noise_increase_range",
                nonnegative=True,
            ),
            candidate_classes=candidate_classes,
            make_unavailable=_to_boolean(
                "localized_fault.make_unavailable",
                data.get("make_unavailable", False),
            ),
        )

@dataclass(frozen=True, slots=True)
class TileFidelityConfig:
    """Complete configuration for the tile-fidelity simulation."""

    num_timesteps: int

    reference_noise_std: float
    minimum_noise_std: float
    maximum_noise_std: float

    class_configs: Mapping[
        FidelityClass,
        FidelityClassConfig,
    ]

    gradual_drift: GradualDriftConfig
    thermal_variation: ThermalVariationConfig
    localized_fault: LocalizedFaultConfig

    def __post_init__(self) -> None:
        num_timesteps = _to_positive_integer(
            "num_timesteps",
            self.num_timesteps,
        )
        reference_noise = _to_positive_float(
            "reference_noise_std",
            self.reference_noise_std,
        )
        minimum_noise = _to_nonnegative_float(
            "minimum_noise_std",
            self.minimum_noise_std,
        )
        maximum_noise = _to_positive_float(
            "maximum_noise_std",
            self.maximum_noise_std,
        )

        if minimum_noise >= maximum_noise:
            raise FidelityConfigurationError(
                "minimum_noise_std must be smaller than "
                "maximum_noise_std."
            )

        if not minimum_noise <= reference_noise <= maximum_noise:
            raise FidelityConfigurationError(
                "reference_noise_std must be within the configured "
                "minimum and maximum noise bounds."
            )

        required_classes = set(FidelityClass)
        configured_classes = set(self.class_configs)

        if configured_classes != required_classes:
            missing = sorted(
                item.value
                for item in required_classes - configured_classes
            )
            unexpected = sorted(
                str(item)
                for item in configured_classes - required_classes
            )

            raise FidelityConfigurationError(
                "Configurations must be provided for high, medium, and "
                f"low fidelity classes. Missing: {missing}. "
                f"Unexpected: {unexpected}."
            )

        fraction_sum = sum(
            class_config.fraction
            for class_config in self.class_configs.values()
        )

        if not np.isclose(fraction_sum, 1.0, atol=1e-9):
            raise FidelityConfigurationError(
                "Fidelity-class fractions must sum to 1.0, "
                f"received {fraction_sum}."
            )

        class_midpoints = {
            fidelity_class: (
                class_config.noise_multiplier_range.minimum
                + class_config.noise_multiplier_range.maximum
            )
            / 2.0
            for fidelity_class, class_config
            in self.class_configs.items()
        }

        if not (
            class_midpoints[FidelityClass.HIGH]
            < class_midpoints[FidelityClass.MEDIUM]
            < class_midpoints[FidelityClass.LOW]
        ):
            raise FidelityConfigurationError(
                "Noise multiplier ranges must represent increasing noise "
                "from high to medium to low fidelity."
            )

        object.__setattr__(self, "num_timesteps", num_timesteps)
        object.__setattr__(
            self,
            "reference_noise_std",
            reference_noise,
        )
        object.__setattr__(
            self,
            "minimum_noise_std",
            minimum_noise,
        )
        object.__setattr__(
            self,
            "maximum_noise_std",
            maximum_noise,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "num_timesteps": self.num_timesteps,
            "reference_noise_std": self.reference_noise_std,
            "min_noise_std": self.minimum_noise_std,
            "max_noise_std": self.maximum_noise_std,
            "fidelity_classes": {
                fidelity_class.value: class_config.to_dict()
                for fidelity_class, class_config
                in self.class_configs.items()
            },
            "degradation": {
                "gradual_drift": self.gradual_drift.to_dict(),
                "thermal_variation": (
                    self.thermal_variation.to_dict()
                ),
                "localized_fault": self.localized_fault.to_dict(),
            },
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> TileFidelityConfig:
        """Construct a fidelity configuration from a full YAML mapping.

        This accepts the recommended structure::

            hardware:
              ...

            fidelity_model:
              num_timesteps: 120
              reference_noise_std: 0.023
              min_noise_std: 0.005
              max_noise_std: 0.060
              fidelity_classes:
                high:
                  fraction: 0.25
                  noise_multiplier_range: [0.55, 0.80]
                medium:
                  fraction: 0.50
                  noise_multiplier_range: [0.90, 1.10]
                low:
                  fraction: 0.25
                  noise_multiplier_range: [1.30, 1.70]

            degradation:
              gradual_drift:
                ...
        """
        if not isinstance(data, Mapping):
            raise FidelityConfigurationError(
                "Fidelity configuration must be a mapping."
            )

        model_data = data.get("fidelity_model", data)

        if not isinstance(model_data, Mapping):
            raise FidelityConfigurationError(
                "fidelity_model must be a mapping."
            )

        degradation_data = data.get(
            "degradation",
            model_data.get("degradation", {}),
        )

        if not isinstance(degradation_data, Mapping):
            raise FidelityConfigurationError(
                "degradation must be a mapping."
            )

        class_configs = _parse_fidelity_classes(model_data)

        minimum_noise = model_data.get(
            "min_noise_std",
            model_data.get("minimum_noise_std"),
        )
        maximum_noise = model_data.get(
            "max_noise_std",
            model_data.get("maximum_noise_std"),
        )

        if minimum_noise is None:
            raise FidelityConfigurationError(
                "fidelity_model is missing min_noise_std."
            )

        if maximum_noise is None:
            raise FidelityConfigurationError(
                "fidelity_model is missing max_noise_std."
            )

        return cls(
            num_timesteps=_to_positive_integer(
                "fidelity_model.num_timesteps",
                _get_required(
                    model_data,
                    "num_timesteps",
                    section_name="fidelity_model",
                ),
            ),
            reference_noise_std=_to_positive_float(
                "fidelity_model.reference_noise_std",
                _get_required(
                    model_data,
                    "reference_noise_std",
                    section_name="fidelity_model",
                ),
            ),
            minimum_noise_std=_to_nonnegative_float(
                "fidelity_model.min_noise_std",
                minimum_noise,
            ),
            maximum_noise_std=_to_positive_float(
                "fidelity_model.max_noise_std",
                maximum_noise,
            ),
            class_configs=class_configs,
            gradual_drift=GradualDriftConfig.from_mapping(
                degradation_data.get("gradual_drift")
            ),
            thermal_variation=ThermalVariationConfig.from_mapping(
                degradation_data.get(
                    "thermal_variation",
                    degradation_data.get("thermal"),
                )
            ),
            localized_fault=LocalizedFaultConfig.from_mapping(
                degradation_data.get(
                    "localized_fault",
                    degradation_data.get("localized_faults"),
                )
            ),
        )

@dataclass(frozen=True, slots=True)
class TileFidelityTrace:
    """Complete time-varying output of one fidelity simulation.

    Array dimensions:

    - Time-varying arrays: ``[num_timesteps, num_tiles]``
    - Static per-tile arrays: ``[num_tiles]``
    """

    timesteps: np.ndarray
    tile_ids: np.ndarray

    noise_std: np.ndarray
    fidelity_score: np.ndarray
    dynamic_fidelity_class: np.ndarray

    available: np.ndarray
    faulted: np.ndarray

    base_noise_std: np.ndarray
    drift_rate: np.ndarray
    initial_fidelity_class: np.ndarray
    thermal_zone: np.ndarray

    fault_onset_timestep: np.ndarray
    fault_noise_increase_fraction: np.ndarray

    metadata: Mapping[str, Any]

    @property
    def num_timesteps(self) -> int:
        """Return the number of simulated timesteps."""
        return int(self.noise_std.shape[0])

    @property
    def num_tiles(self) -> int:
        """Return the number of simulated tiles."""
        return int(self.noise_std.shape[1])

    def save_npz(self, output_path: str | Path) -> Path:
        """Save the complete trace as a compressed NPZ file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata_json = json.dumps(
            self.metadata,
            sort_keys=True,
        )

        np.savez_compressed(
            path,
            timesteps=self.timesteps,
            tile_ids=self.tile_ids,
            noise_std=self.noise_std,
            fidelity_score=self.fidelity_score,
            dynamic_fidelity_class=self.dynamic_fidelity_class,
            available=self.available,
            faulted=self.faulted,
            base_noise_std=self.base_noise_std,
            drift_rate=self.drift_rate,
            initial_fidelity_class=self.initial_fidelity_class,
            thermal_zone=self.thermal_zone,
            fault_onset_timestep=self.fault_onset_timestep,
            fault_noise_increase_fraction=(
                self.fault_noise_increase_fraction
            ),
            metadata_json=np.asarray(metadata_json),
        )

        return path

    @classmethod
    def load_npz(cls, input_path: str | Path) -> TileFidelityTrace:
        """Load a previously saved trace from a compressed NPZ file."""
        path = Path(input_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Trace file does not exist: {path}")

        with np.load(path, allow_pickle=False) as data:
            required_arrays = (
                "timesteps",
                "tile_ids",
                "noise_std",
                "fidelity_score",
                "dynamic_fidelity_class",
                "available",
                "faulted",
                "base_noise_std",
                "drift_rate",
                "initial_fidelity_class",
                "thermal_zone",
                "fault_onset_timestep",
                "fault_noise_increase_fraction",
            )

            missing = [
                array_name
                for array_name in required_arrays
                if array_name not in data.files
            ]
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise FidelityConfigurationError(
                    f"Trace NPZ is missing required arrays: {missing_text}."
                )

            metadata: dict[str, Any]
            if "metadata_json" in data.files:
                raw_metadata = data["metadata_json"]
                if isinstance(raw_metadata, np.ndarray):
                    if raw_metadata.shape == ():
                        raw_metadata = raw_metadata.item()
                    else:
                        raw_metadata = "".join(str(item) for item in raw_metadata.tolist())

                if not isinstance(raw_metadata, str):
                    raise FidelityConfigurationError(
                        "Trace metadata_json must contain a JSON string."
                    )

                parsed_metadata = json.loads(raw_metadata)
                if not isinstance(parsed_metadata, Mapping):
                    raise FidelityConfigurationError(
                        "Trace metadata_json must decode to a JSON mapping."
                    )
                metadata = dict(parsed_metadata)
            else:
                metadata = {}

            return cls(
                timesteps=np.asarray(data["timesteps"], dtype=np.int64),
                tile_ids=np.asarray(data["tile_ids"], dtype=np.int64),
                noise_std=np.asarray(data["noise_std"], dtype=np.float64),
                fidelity_score=np.asarray(data["fidelity_score"], dtype=np.float64),
                dynamic_fidelity_class=np.asarray(
                    data["dynamic_fidelity_class"],
                    dtype=np.uint8,
                ),
                available=np.asarray(data["available"], dtype=np.bool_),
                faulted=np.asarray(data["faulted"], dtype=np.bool_),
                base_noise_std=np.asarray(data["base_noise_std"], dtype=np.float64),
                drift_rate=np.asarray(data["drift_rate"], dtype=np.float64),
                initial_fidelity_class=np.asarray(
                    data["initial_fidelity_class"],
                    dtype=np.uint8,
                ),
                thermal_zone=np.asarray(data["thermal_zone"], dtype=np.int64),
                fault_onset_timestep=np.asarray(
                    data["fault_onset_timestep"],
                    dtype=np.int64,
                ),
                fault_noise_increase_fraction=np.asarray(
                    data["fault_noise_increase_fraction"],
                    dtype=np.float64,
                ),
                metadata=metadata,
            )

    def get_snapshot(self, timestep: int) -> dict[str, Any]:
        """Return one timestep snapshot for downstream mapping phases."""
        normalized_timestep = _to_nonnegative_integer("timestep", timestep)

        if normalized_timestep >= self.num_timesteps:
            raise FidelitySimulationError(
                "Requested timestep is outside trace bounds: "
                f"{normalized_timestep} not in [0, {self.num_timesteps - 1}]."
            )

        return {
            "timestep": normalized_timestep,
            "tile_ids": self.tile_ids.copy(),
            "noise_std": self.noise_std[normalized_timestep].copy(),
            "fidelity_score": self.fidelity_score[normalized_timestep].copy(),
            "dynamic_fidelity_class": (
                self.dynamic_fidelity_class[normalized_timestep].copy()
            ),
            "available": self.available[normalized_timestep].copy(),
            "faulted": self.faulted[normalized_timestep].copy(),
        }

class TileFidelityModel:
    """Simulate heterogeneous and time-varying tile quality."""

    def __init__(
        self,
        hardware_config: HardwareConfig,
        fidelity_config: TileFidelityConfig,
        *,
        seed: int = 0,
    ) -> None:
        if not isinstance(hardware_config, HardwareConfig):
            raise FidelityConfigurationError(
                "hardware_config must be a HardwareConfig instance."
            )

        if not isinstance(fidelity_config, TileFidelityConfig):
            raise FidelityConfigurationError(
                "fidelity_config must be a TileFidelityConfig instance."
            )

        self.hardware_config = hardware_config
        self.fidelity_config = fidelity_config
        self.seed = _to_nonnegative_integer("seed", seed)

        self._validate_combined_configuration()

        self._rng = np.random.default_rng(self.seed)
        self._hardware: HardwareState | None = None
        self._thermal_state = np.zeros(
            self.hardware_config.num_thermal_zones,
            dtype=np.float64,
        )

        self._fault_onset_timestep = np.full(
            self.hardware_config.num_tiles,
            fill_value=-1,
            dtype=np.int64,
        )
        self._fault_noise_increase_fraction = np.zeros(
            self.hardware_config.num_tiles,
            dtype=np.float64,
        )

        self._current_timestep = -1

        self.reset(self.seed)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        seed: int | None = None,
    ) -> TileFidelityModel:
        """Construct a complete model from a YAML-like mapping."""
        if not isinstance(data, Mapping):
            raise FidelityConfigurationError(
                "Phase 2 configuration must be a mapping."
            )

        hardware_data = data.get("hardware")

        if not isinstance(hardware_data, Mapping):
            raise FidelityConfigurationError(
                "Phase 2 configuration requires a hardware mapping."
            )

        hardware_config = HardwareConfig.from_mapping(hardware_data)
        fidelity_config = TileFidelityConfig.from_mapping(data)

        if seed is None:
            experiment_data = data.get("experiment", {})

            if not isinstance(experiment_data, Mapping):
                raise FidelityConfigurationError(
                    "experiment must be a mapping."
                )

            seed = experiment_data.get(
                "seed",
                data.get("seed", 0),
            )

        return cls(
            hardware_config=hardware_config,
            fidelity_config=fidelity_config,
            seed=_to_nonnegative_integer("seed", seed),
        )

    @property
    def hardware(self) -> HardwareState:
        """Return the current internal hardware state.

        The returned value is a clone so external code cannot mutate the
        simulation accidentally.
        """
        if self._hardware is None:
            raise FidelitySimulationError(
                "Hardware has not been initialized."
            )

        return self._hardware.clone()

    @property
    def current_timestep(self) -> int:
        """Return the most recently completed timestep."""
        return self._current_timestep

    @property
    def fault_onset_timestep(self) -> np.ndarray:
        """Return a copy of the per-tile fault schedule."""
        return self._fault_onset_timestep.copy()

    @property
    def fault_noise_increase_fraction(self) -> np.ndarray:
        """Return a copy of per-tile permanent fault severity."""
        return self._fault_noise_increase_fraction.copy()

    def reset(self, seed: int | None = None) -> HardwareState:
        """Reinitialize all random state and tile assignments.

        Reusing the same seed produces the same initial tile assignment,
        degradation rates, thermal trajectory, and fault schedule.
        """
        if seed is not None:
            self.seed = _to_nonnegative_integer("seed", seed)

        self._rng = np.random.default_rng(self.seed)
        self._current_timestep = -1

        initial_classes = self._sample_initial_fidelity_classes()
        base_noise = self._sample_base_noise(initial_classes)
        drift_rates = self._sample_drift_rates()

        self._hardware = HardwareState.from_assignments(
            config=self.hardware_config,
            base_noise_std_by_tile=base_noise.tolist(),
            fidelity_class_by_tile=initial_classes,
            drift_rate_by_tile=drift_rates.tolist(),
        )

        self._thermal_state = np.zeros(
            self.hardware_config.num_thermal_zones,
            dtype=np.float64,
        )

        self._schedule_localized_faults()

        return self._hardware.clone()

    def step(self, timestep: int) -> HardwareState:
        """Advance the model by exactly one sequential timestep.

        Timesteps must be supplied in order: 0, 1, 2, and so on. This
        is necessary because thermal variation follows a correlated
        autoregressive process.
        """
        timestep = _to_nonnegative_integer("timestep", timestep)

        expected_timestep = self._current_timestep + 1

        if timestep != expected_timestep:
            raise FidelitySimulationError(
                "TileFidelityModel must be stepped sequentially. "
                f"Expected timestep {expected_timestep}, "
                f"received {timestep}."
            )

        if timestep >= self.fidelity_config.num_timesteps:
            raise FidelitySimulationError(
                f"Timestep {timestep} is outside the configured range "
                f"0 through "
                f"{self.fidelity_config.num_timesteps - 1}."
            )

        if self._hardware is None:
            raise FidelitySimulationError(
                "Hardware has not been initialized."
            )

        self._advance_thermal_state(timestep)

        total_timesteps = self.fidelity_config.num_timesteps

        if total_timesteps <= 1:
            normalized_time = 0.0
        else:
            normalized_time = timestep / (total_timesteps - 1)

        for tile in self._hardware.tiles:
            drift_fraction = tile.drift_rate * normalized_time

            thermal_fraction = self._thermal_state[
                tile.thermal_zone
            ]

            fault_fraction = 0.0
            fault_timestep = self._fault_onset_timestep[
                tile.tile_id
            ]

            if fault_timestep >= 0 and timestep >= fault_timestep:
                fault_fraction = (
                    self._fault_noise_increase_fraction[
                        tile.tile_id
                    ]
                )

                if not tile.faulted:
                    tile.mark_faulted(
                        timestep=timestep,
                        make_unavailable=(
                            self.fidelity_config
                            .localized_fault
                            .make_unavailable
                        ),
                    )

            effective_multiplier = (
                1.0
                + drift_fraction
                + thermal_fraction
                + fault_fraction
            )

            raw_noise_std = (
                tile.base_noise_std * effective_multiplier
            )

            bounded_noise_std = float(
                np.clip(
                    raw_noise_std,
                    self.fidelity_config.minimum_noise_std,
                    self.fidelity_config.maximum_noise_std,
                )
            )

            tile.set_current_noise_std(bounded_noise_std)

        self._current_timestep = timestep

        return self._hardware.clone()

    def generate_trace(
        self,
        *,
        reset: bool = True,
    ) -> TileFidelityTrace:
        """Generate the complete configured fidelity trace."""
        if reset:
            self.reset(self.seed)
        elif self._current_timestep != -1:
            raise FidelitySimulationError(
                "generate_trace(reset=False) requires a newly reset "
                "model whose current timestep is -1."
            )

        num_timesteps = self.fidelity_config.num_timesteps
        num_tiles = self.hardware_config.num_tiles

        noise_std = np.empty(
            (num_timesteps, num_tiles),
            dtype=np.float64,
        )
        fidelity_score = np.empty(
            (num_timesteps, num_tiles),
            dtype=np.float64,
        )
        dynamic_fidelity_class = np.empty(
            (num_timesteps, num_tiles),
            dtype=np.uint8,
        )
        available = np.empty(
            (num_timesteps, num_tiles),
            dtype=np.bool_,
        )
        faulted = np.empty(
            (num_timesteps, num_tiles),
            dtype=np.bool_,
        )

        if self._hardware is None:
            raise FidelitySimulationError(
                "Hardware has not been initialized."
            )

        base_noise_std = np.asarray(
            [
                tile.base_noise_std
                for tile in self._hardware.tiles
            ],
            dtype=np.float64,
        )
        drift_rate = np.asarray(
            [
                tile.drift_rate
                for tile in self._hardware.tiles
            ],
            dtype=np.float64,
        )
        initial_fidelity_class = np.asarray(
            [
                FIDELITY_CLASS_TO_CODE[tile.fidelity_class]
                for tile in self._hardware.tiles
            ],
            dtype=np.uint8,
        )
        thermal_zone = np.asarray(
            [
                tile.thermal_zone
                for tile in self._hardware.tiles
            ],
            dtype=np.int64,
        )

        for timestep in range(num_timesteps):
            state = self.step(timestep)

            for tile in state.tiles:
                tile_id = tile.tile_id
                current_noise = tile.current_noise_std

                noise_std[timestep, tile_id] = current_noise
                fidelity_score[timestep, tile_id] = (
                    self.compute_fidelity_score(current_noise)
                )
                dynamic_fidelity_class[timestep, tile_id] = (
                    FIDELITY_CLASS_TO_CODE[
                        self.classify_noise_std(current_noise)
                    ]
                )
                available[timestep, tile_id] = tile.available
                faulted[timestep, tile_id] = tile.faulted

        metadata = {
            "seed": self.seed,
            "hardware": self.hardware_config.to_dict(),
            "noise_unit": "pcmlike_prog_noise_scale_equivalent",
            "reference_programming_noise_scale": (
                self.fidelity_config.reference_noise_std
            ),
            "fidelity_model": self.fidelity_config.to_dict(),
            "class_code_mapping": {
                str(code): fidelity_class.value
                for code, fidelity_class
                in FIDELITY_CODE_TO_CLASS.items()
            },
            "selected_fault_tile_ids": np.flatnonzero(
                self._fault_onset_timestep >= 0
            ).astype(int).tolist(),
        }

        return TileFidelityTrace(
            timesteps=np.arange(
                num_timesteps,
                dtype=np.int64,
            ),
            tile_ids=np.arange(
                num_tiles,
                dtype=np.int64,
            ),
            noise_std=noise_std,
            fidelity_score=fidelity_score,
            dynamic_fidelity_class=dynamic_fidelity_class,
            available=available,
            faulted=faulted,
            base_noise_std=base_noise_std,
            drift_rate=drift_rate,
            initial_fidelity_class=initial_fidelity_class,
            thermal_zone=thermal_zone,
            fault_onset_timestep=(
                self._fault_onset_timestep.copy()
            ),
            fault_noise_increase_fraction=(
                self._fault_noise_increase_fraction.copy()
            ),
            metadata=metadata,
        )

    def compute_fidelity_score(self, noise_std: float) -> float:
        """Convert effective noise into a bounded fidelity score.

        The score is:

            1 / (1 + (noise_std / reference_noise_std)^2)

        Higher scores indicate lower effective noise.
        """
        noise_std = _to_nonnegative_float(
            "noise_std",
            noise_std,
        )

        ratio = (
            noise_std
            / self.fidelity_config.reference_noise_std
        )

        return float(1.0 / (1.0 + ratio**2))

    def classify_noise_std(
        self,
        noise_std: float,
    ) -> FidelityClass:
        """Classify current noise into high, medium, or low fidelity.

        Dynamic classification thresholds are derived from the configured
        initialization multiplier ranges.
        """
        noise_std = _to_nonnegative_float(
            "noise_std",
            noise_std,
        )

        noise_ratio = (
            noise_std
            / self.fidelity_config.reference_noise_std
        )

        high_config = self.fidelity_config.class_configs[
            FidelityClass.HIGH
        ]
        medium_config = self.fidelity_config.class_configs[
            FidelityClass.MEDIUM
        ]
        low_config = self.fidelity_config.class_configs[
            FidelityClass.LOW
        ]

        high_medium_threshold = (
            high_config.noise_multiplier_range.maximum
            + medium_config.noise_multiplier_range.minimum
        ) / 2.0

        medium_low_threshold = (
            medium_config.noise_multiplier_range.maximum
            + low_config.noise_multiplier_range.minimum
        ) / 2.0

        if noise_ratio <= high_medium_threshold:
            return FidelityClass.HIGH

        if noise_ratio <= medium_low_threshold:
            return FidelityClass.MEDIUM

        return FidelityClass.LOW

    def _validate_combined_configuration(self) -> None:
        fault_config = self.fidelity_config.localized_fault

        if fault_config.num_affected_tiles > (
            self.hardware_config.num_tiles
        ):
            raise FidelityConfigurationError(
                "num_affected_tiles cannot exceed num_tiles."
            )

        if (
            fault_config.enabled
            and fault_config.onset_timestep_range.maximum
            >= self.fidelity_config.num_timesteps
        ):
            raise FidelityConfigurationError(
                "Fault onset timestep must be smaller than "
                "num_timesteps. Maximum configured onset is "
                f"{fault_config.onset_timestep_range.maximum}, while "
                f"num_timesteps is "
                f"{self.fidelity_config.num_timesteps}."
            )

    def _sample_initial_fidelity_classes(
        self,
    ) -> list[FidelityClass]:
        """Create exact class counts using largest-remainder rounding."""
        ordered_classes = [
            FidelityClass.HIGH,
            FidelityClass.MEDIUM,
            FidelityClass.LOW,
        ]

        fractions = np.asarray(
            [
                self.fidelity_config
                .class_configs[fidelity_class]
                .fraction
                for fidelity_class in ordered_classes
            ],
            dtype=np.float64,
        )

        raw_counts = fractions * self.hardware_config.num_tiles
        counts = np.floor(raw_counts).astype(np.int64)

        remaining = (
            self.hardware_config.num_tiles
            - int(counts.sum())
        )

        fractional_remainders = raw_counts - counts
        remainder_order = np.argsort(
            -fractional_remainders,
            kind="stable",
        )

        for index in remainder_order[:remaining]:
            counts[index] += 1

        assignments: list[FidelityClass] = []

        for fidelity_class, count in zip(
            ordered_classes,
            counts,
            strict=True,
        ):
            assignments.extend(
                [fidelity_class] * int(count)
            )

        permutation = self._rng.permutation(
            self.hardware_config.num_tiles
        )

        return [
            assignments[int(index)]
            for index in permutation
        ]

    def _sample_base_noise(
        self,
        initial_classes: Sequence[FidelityClass],
    ) -> np.ndarray:
        """Sample baseline noise according to each tile's class."""
        base_noise = np.empty(
            self.hardware_config.num_tiles,
            dtype=np.float64,
        )

        for tile_id, fidelity_class in enumerate(
            initial_classes
        ):
            class_config = (
                self.fidelity_config
                .class_configs[fidelity_class]
            )

            multiplier = float(
                class_config.noise_multiplier_range.sample(
                    self._rng
                )
            )

            raw_noise = (
                self.fidelity_config.reference_noise_std
                * multiplier
            )

            base_noise[tile_id] = np.clip(
                raw_noise,
                self.fidelity_config.minimum_noise_std,
                self.fidelity_config.maximum_noise_std,
            )

        return base_noise

    def _sample_drift_rates(self) -> np.ndarray:
        """Sample total fractional degradation over the full trace."""
        drift_config = self.fidelity_config.gradual_drift

        if not drift_config.enabled:
            return np.zeros(
                self.hardware_config.num_tiles,
                dtype=np.float64,
            )

        return np.asarray(
            drift_config.total_increase_range.sample(
                self._rng,
                size=self.hardware_config.num_tiles,
            ),
            dtype=np.float64,
        )

    def _schedule_localized_faults(self) -> None:
        """Select affected tiles and sample fault time/severity."""
        self._fault_onset_timestep = np.full(
            self.hardware_config.num_tiles,
            fill_value=-1,
            dtype=np.int64,
        )
        self._fault_noise_increase_fraction = np.zeros(
            self.hardware_config.num_tiles,
            dtype=np.float64,
        )

        fault_config = self.fidelity_config.localized_fault

        if (
            not fault_config.enabled
            or fault_config.num_affected_tiles == 0
        ):
            return

        if self._hardware is None:
            raise FidelitySimulationError(
                "Hardware must be initialized before faults are "
                "scheduled."
            )

        candidate_class_set = set(
            fault_config.candidate_classes
        )

        candidate_tile_ids = np.asarray(
            [
                tile.tile_id
                for tile in self._hardware.tiles
                if tile.fidelity_class in candidate_class_set
            ],
            dtype=np.int64,
        )

        if (
            fault_config.num_affected_tiles
            > candidate_tile_ids.size
        ):
            raise FidelityConfigurationError(
                "Not enough tiles match localized_fault."
                "candidate_classes. Requested "
                f"{fault_config.num_affected_tiles}, but only "
                f"{candidate_tile_ids.size} candidates are available."
            )

        selected_tile_ids = self._rng.choice(
            candidate_tile_ids,
            size=fault_config.num_affected_tiles,
            replace=False,
        )

        onset_timesteps = (
            fault_config.onset_timestep_range.sample(
                self._rng,
                size=fault_config.num_affected_tiles,
            )
        )
        noise_increases = (
            fault_config.noise_increase_range.sample(
                self._rng,
                size=fault_config.num_affected_tiles,
            )
        )

        self._fault_onset_timestep[selected_tile_ids] = (
            onset_timesteps
        )
        self._fault_noise_increase_fraction[
            selected_tile_ids
        ] = noise_increases

    def _advance_thermal_state(self, timestep: int) -> None:
        """Advance thermal-zone state using an AR(1) process."""
        thermal_config = (
            self.fidelity_config.thermal_variation
        )

        if not thermal_config.enabled:
            self._thermal_state.fill(0.0)
            return

        if timestep == 0:
            self._thermal_state.fill(0.0)
            return

        correlation = thermal_config.correlation

        innovation_std = (
            thermal_config.standard_deviation_fraction
            * sqrt(1.0 - correlation**2)
        )

        innovation = self._rng.normal(
            loc=0.0,
            scale=innovation_std,
            size=self.hardware_config.num_thermal_zones,
        )

        self._thermal_state = (
            correlation * self._thermal_state
            + innovation
        )

def _parse_fidelity_classes(
    model_data: Mapping[str, Any],
) -> dict[FidelityClass, FidelityClassConfig]:
    """Parse fidelity classes from supported configuration formats."""
    raw_classes = model_data.get("fidelity_classes")

    if raw_classes is not None:
        if not isinstance(raw_classes, Mapping):
            raise FidelityConfigurationError(
                "fidelity_classes must be a mapping."
            )

        class_configs: dict[
            FidelityClass,
            FidelityClassConfig,
        ] = {}

        for fidelity_class in FidelityClass:
            raw_class = raw_classes.get(
                fidelity_class.value
            )

            if raw_class is None:
                raise FidelityConfigurationError(
                    "fidelity_classes is missing "
                    f"{fidelity_class.value!r}."
                )

            class_configs[fidelity_class] = (
                FidelityClassConfig.from_mapping(
                    raw_class,
                    class_name=fidelity_class.value,
                )
            )

        return class_configs

    fractions = model_data.get("class_fractions")
    multipliers = model_data.get(
        "class_noise_multipliers"
    )

    if not isinstance(fractions, Mapping):
        raise FidelityConfigurationError(
            "Provide either fidelity_classes or class_fractions."
        )

    if not isinstance(multipliers, Mapping):
        raise FidelityConfigurationError(
            "class_noise_multipliers must be provided when using "
            "class_fractions."
        )

    class_configs = {}

    for fidelity_class in FidelityClass:
        class_name = fidelity_class.value

        if class_name not in fractions:
            raise FidelityConfigurationError(
                f"class_fractions is missing {class_name!r}."
            )

        if class_name not in multipliers:
            raise FidelityConfigurationError(
                "class_noise_multipliers is missing "
                f"{class_name!r}."
            )

        class_configs[fidelity_class] = FidelityClassConfig(
            fraction=_to_finite_float(
                f"class_fractions.{class_name}",
                fractions[class_name],
            ),
            noise_multiplier_range=FloatRange.from_value(
                multipliers[class_name],
                name=(
                    "class_noise_multipliers."
                    f"{class_name}"
                ),
                nonnegative=True,
            ),
        )

    return class_configs

def _get_required(
    data: Mapping[str, Any],
    key: str,
    *,
    section_name: str,
) -> Any:
    if key not in data:
        raise FidelityConfigurationError(
            f"{section_name} is missing required field {key!r}."
        )

    return data[key]

def _to_boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise FidelityConfigurationError(
            f"{name} must be a boolean, received {value!r}."
        )

    return value

def _to_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FidelityConfigurationError(
            f"{name} must be an integer, received {value!r}."
        )

    return int(value)

def _to_nonnegative_integer(
    name: str,
    value: Any,
) -> int:
    normalized = _to_integer(name, value)

    if normalized < 0:
        raise FidelityConfigurationError(
            f"{name} must be nonnegative, received {normalized}."
        )

    return normalized

def _to_positive_integer(
    name: str,
    value: Any,
) -> int:
    normalized = _to_integer(name, value)

    if normalized <= 0:
        raise FidelityConfigurationError(
            f"{name} must be greater than zero, "
            f"received {normalized}."
        )

    return normalized

def _to_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FidelityConfigurationError(
            f"{name} must be numeric, received {value!r}."
        )

    normalized = float(value)

    if not isfinite(normalized):
        raise FidelityConfigurationError(
            f"{name} must be finite, received {normalized}."
        )

    return normalized

def _to_nonnegative_float(
    name: str,
    value: Any,
) -> float:
    normalized = _to_finite_float(name, value)

    if normalized < 0.0:
        raise FidelityConfigurationError(
            f"{name} must be nonnegative, received {normalized}."
        )

    return normalized

def _to_positive_float(
    name: str,
    value: Any,
) -> float:
    normalized = _to_finite_float(name, value)

    if normalized <= 0.0:
        raise FidelityConfigurationError(
            f"{name} must be greater than zero, "
            f"received {normalized}."
        )

    return normalized

__all__ = [
    "FIDELITY_CLASS_TO_CODE",
    "FIDELITY_CODE_TO_CLASS",
    "FidelityClassConfig",
    "FidelityConfigurationError",
    "FidelitySimulationError",
    "FloatRange",
    "GradualDriftConfig",
    "IntRange",
    "LocalizedFaultConfig",
    "ThermalVariationConfig",
    "TileFidelityConfig",
    "TileFidelityModel",
    "TileFidelityTrace",
]