"""Time-varying tile-fidelity simulator for Phase 2.

The trace unit is explicit and shared with Phases 1 and 4:

    normalized programming-noise standard deviation
    / programmed projection weight range

A value of 0.023 therefore means that the logical-weight Gaussian noise
standard deviation is 2.3% of the manually clipped projection's programmed
range.  No AIHWKit PCMLike ``prog_noise_scale`` conversion is involved.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CLASS_ORDER = ("high", "medium", "low")


@dataclass(frozen=True)
class FaultEvent:
    tile_id: int
    onset_timestep: int
    noise_increase_fraction: float
    permanent: bool
    make_unavailable: bool


@dataclass(frozen=True)
class FidelityTrace:
    noise_std: np.ndarray
    fidelity_score: np.ndarray
    faulted: np.ndarray
    available: np.ndarray
    tile_class: np.ndarray
    thermal_zone: np.ndarray
    initial_noise_std: np.ndarray
    drift_fraction: np.ndarray
    fault_events: tuple[FaultEvent, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.noise_std.ndim != 2:
            raise ValueError("noise_std must have shape [timesteps, tiles].")
        shape = self.noise_std.shape
        for name, array in (
            ("fidelity_score", self.fidelity_score),
            ("faulted", self.faulted),
            ("available", self.available),
        ):
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} does not match {shape}.")
        if self.tile_class.shape != (shape[1],):
            raise ValueError("tile_class must have one entry per tile.")
        if self.thermal_zone.shape != (shape[1],):
            raise ValueError("thermal_zone must have one entry per tile.")
        if self.initial_noise_std.shape != (shape[1],):
            raise ValueError("initial_noise_std must have one entry per tile.")
        if self.drift_fraction.shape != (shape[1],):
            raise ValueError("drift_fraction must have one entry per tile.")
        if not np.isfinite(self.noise_std).all() or (self.noise_std < 0).any():
            raise ValueError("noise_std contains invalid values.")
        if not np.isfinite(self.fidelity_score).all():
            raise ValueError("fidelity_score contains invalid values.")

    def save(self, output_dir: Path) -> None:
        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "trace.npz",
            noise_std=self.noise_std.astype(np.float32),
            fidelity_score=self.fidelity_score.astype(np.float32),
            faulted=self.faulted.astype(np.bool_),
            available=self.available.astype(np.bool_),
            tile_class=self.tile_class.astype("U16"),
            thermal_zone=self.thermal_zone.astype(np.int32),
            initial_noise_std=self.initial_noise_std.astype(np.float32),
            drift_fraction=self.drift_fraction.astype(np.float32),
        )
        metadata = dict(self.metadata)
        metadata["fault_events"] = [asdict(event) for event in self.fault_events]
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )

        tile_rows: list[dict[str, Any]] = []
        for tile_id in range(self.noise_std.shape[1]):
            event = next(
                (event for event in self.fault_events if event.tile_id == tile_id),
                None,
            )
            tile_rows.append(
                {
                    "tile_id": tile_id,
                    "fidelity_class": str(self.tile_class[tile_id]),
                    "thermal_zone": int(self.thermal_zone[tile_id]),
                    "initial_noise_std": float(self.initial_noise_std[tile_id]),
                    "drift_fraction": float(self.drift_fraction[tile_id]),
                    "fault_onset": None if event is None else event.onset_timestep,
                    "fault_noise_increase_fraction": (
                        None if event is None else event.noise_increase_fraction
                    ),
                    "fault_permanent": None if event is None else event.permanent,
                    "make_unavailable": (
                        None if event is None else event.make_unavailable
                    ),
                }
            )
        pd.DataFrame(tile_rows).to_csv(output_dir / "tile_metadata.csv", index=False)

        timestep_rows = []
        for timestep in range(self.noise_std.shape[0]):
            timestep_rows.append(
                {
                    "timestep": timestep,
                    "mean_noise_std": float(self.noise_std[timestep].mean()),
                    "std_noise_std": float(self.noise_std[timestep].std()),
                    "min_noise_std": float(self.noise_std[timestep].min()),
                    "max_noise_std": float(self.noise_std[timestep].max()),
                    "mean_fidelity_score": float(
                        self.fidelity_score[timestep].mean()
                    ),
                    "num_faulted_tiles": int(self.faulted[timestep].sum()),
                    "num_available_tiles": int(self.available[timestep].sum()),
                }
            )
        pd.DataFrame(timestep_rows).to_csv(
            output_dir / "timestep_summary.csv", index=False
        )


def _require_fraction_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range.")
    low, high = float(value[0]), float(value[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
        raise ValueError(f"Invalid range for {name}: {value!r}")
    return low, high


def _allocate_classes(
    num_tiles: int,
    class_config: Mapping[str, Mapping[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    fractions = np.asarray(
        [float(class_config[name]["fraction"]) for name in CLASS_ORDER],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-9):
        raise ValueError("Fidelity-class fractions must be non-negative and sum to 1.")
    raw_counts = fractions * num_tiles
    counts = np.floor(raw_counts).astype(int)
    remainder = num_tiles - int(counts.sum())
    if remainder:
        fractional = raw_counts - counts
        for index in np.argsort(-fractional)[:remainder]:
            counts[index] += 1
    labels = np.concatenate(
        [np.repeat(name, count) for name, count in zip(CLASS_ORDER, counts)]
    ).astype("U16")
    if labels.size != num_tiles:
        raise RuntimeError("Class allocation did not produce num_tiles labels.")
    rng.shuffle(labels)
    return labels


def _thermal_process(
    num_timesteps: int,
    num_zones: int,
    correlation: float,
    std_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (-0.999 < correlation < 0.999):
        raise ValueError("Thermal correlation must be strictly between -0.999 and 0.999.")
    if std_fraction < 0:
        raise ValueError("Thermal standard_deviation_fraction cannot be negative.")
    thermal = np.zeros((num_timesteps, num_zones), dtype=np.float64)
    if num_timesteps <= 1 or std_fraction == 0:
        return thermal
    innovation_std = std_fraction * math.sqrt(1.0 - correlation**2)
    thermal[0] = rng.normal(0.0, std_fraction, size=num_zones)
    for timestep in range(1, num_timesteps):
        innovation = rng.normal(0.0, innovation_std, size=num_zones)
        thermal[timestep] = correlation * thermal[timestep - 1] + innovation
    return thermal


def generate_fidelity_trace(config: Mapping[str, Any], seed: int) -> FidelityTrace:
    hardware = config["hardware"]
    model = config["fidelity_model"]
    degradation = config["degradation"]

    num_tiles = int(hardware["num_tiles"])
    num_zones = int(hardware["num_thermal_zones"])
    num_timesteps = int(model["num_timesteps"])
    reference_noise_std = float(model["reference_noise_std"])
    min_noise_std = float(model["min_noise_std"])
    max_noise_std = float(model["max_noise_std"])
    if num_tiles <= 0 or num_zones <= 0 or num_timesteps <= 0:
        raise ValueError("Tiles, thermal zones, and timesteps must be positive.")
    if not (0 <= min_noise_std <= reference_noise_std <= max_noise_std):
        raise ValueError(
            "Expected min_noise_std <= reference_noise_std <= max_noise_std."
        )

    rng = np.random.default_rng(int(seed))
    tile_class = _allocate_classes(num_tiles, model["fidelity_classes"], rng)
    thermal_zone = np.arange(num_tiles, dtype=np.int32) % num_zones
    rng.shuffle(thermal_zone)

    initial_noise = np.empty(num_tiles, dtype=np.float64)
    for tile_id, class_name in enumerate(tile_class):
        low, high = _require_fraction_range(
            model["fidelity_classes"][str(class_name)]["noise_multiplier_range"],
            f"{class_name}.noise_multiplier_range",
        )
        initial_noise[tile_id] = reference_noise_std * rng.uniform(low, high)

    drift_cfg = degradation["gradual_drift"]
    if bool(drift_cfg.get("enabled", True)):
        drift_low, drift_high = _require_fraction_range(
            drift_cfg["total_increase_range"],
            "degradation.gradual_drift.total_increase_range",
        )
        drift_fraction = rng.uniform(drift_low, drift_high, size=num_tiles)
    else:
        drift_fraction = np.zeros(num_tiles, dtype=np.float64)

    if num_timesteps == 1:
        progress = np.zeros(1, dtype=np.float64)
    else:
        progress = np.linspace(0.0, 1.0, num_timesteps, dtype=np.float64)
    noise_std = initial_noise[None, :] * (
        1.0 + progress[:, None] * drift_fraction[None, :]
    )

    thermal_cfg = degradation["thermal_variation"]
    if bool(thermal_cfg.get("enabled", True)):
        thermal = _thermal_process(
            num_timesteps=num_timesteps,
            num_zones=num_zones,
            correlation=float(thermal_cfg["correlation"]),
            std_fraction=float(thermal_cfg["standard_deviation_fraction"]),
            rng=rng,
        )
        noise_std *= np.clip(1.0 + thermal[:, thermal_zone], 0.05, None)

    faulted = np.zeros((num_timesteps, num_tiles), dtype=bool)
    available = np.ones((num_timesteps, num_tiles), dtype=bool)
    events: list[FaultEvent] = []
    fault_cfg = degradation["localized_fault"]
    if bool(fault_cfg.get("enabled", True)):
        num_affected = int(fault_cfg["num_affected_tiles"])
        if not 0 <= num_affected <= num_tiles:
            raise ValueError("num_affected_tiles must be in [0, num_tiles].")
        onset_low, onset_high = [
            int(value) for value in fault_cfg["onset_timestep_range"]
        ]
        if not 0 <= onset_low <= onset_high < num_timesteps:
            raise ValueError("Fault onset range must lie inside the trace.")
        increase_low, increase_high = _require_fraction_range(
            fault_cfg["noise_increase_range"],
            "degradation.localized_fault.noise_increase_range",
        )
        candidate_classes = {
            str(value) for value in fault_cfg.get("candidate_classes", CLASS_ORDER)
        }
        candidates = np.asarray(
            [
                tile_id
                for tile_id, class_name in enumerate(tile_class)
                if str(class_name) in candidate_classes
            ],
            dtype=np.int32,
        )
        if candidates.size < num_affected:
            raise ValueError("Not enough fault candidates for num_affected_tiles.")
        selected = rng.choice(candidates, size=num_affected, replace=False)
        permanent = bool(fault_cfg.get("permanent", True))
        make_unavailable = bool(fault_cfg.get("make_unavailable", False))
        for tile_id in sorted(int(value) for value in selected):
            onset = int(rng.integers(onset_low, onset_high + 1))
            increase = float(rng.uniform(increase_low, increase_high))
            event = FaultEvent(
                tile_id=tile_id,
                onset_timestep=onset,
                noise_increase_fraction=increase,
                permanent=permanent,
                make_unavailable=make_unavailable,
            )
            events.append(event)
            if permanent:
                active = np.arange(num_timesteps) >= onset
            else:
                duration = max(1, int(math.ceil(num_timesteps * 0.1)))
                active = (np.arange(num_timesteps) >= onset) & (
                    np.arange(num_timesteps) < onset + duration
                )
            faulted[active, tile_id] = True
            noise_std[active, tile_id] *= 1.0 + increase
            if make_unavailable:
                available[active, tile_id] = False

    noise_std = np.clip(noise_std, min_noise_std, max_noise_std)
    # A bounded monotonic fidelity score: 1 at zero noise and 0.5 at reference.
    fidelity_score = 1.0 / (1.0 + noise_std / reference_noise_std)

    rank_correlation = float(
        pd.Series(noise_std[0]).corr(pd.Series(noise_std[-1]), method="spearman")
    )
    metadata = {
        "seed": int(seed),
        "num_tiles": num_tiles,
        "num_timesteps": num_timesteps,
        "num_thermal_zones": num_zones,
        "noise_unit": "normalized_std_fraction_of_programmed_projection_range",
        "reference_noise_std": reference_noise_std,
        "initial_mean_noise_std": float(noise_std[0].mean()),
        "final_mean_noise_std": float(noise_std[-1].mean()),
        "mean_noise_increase_fraction": float(
            noise_std[-1].mean() / noise_std[0].mean() - 1.0
        ),
        "initial_mean_fidelity_score": float(fidelity_score[0].mean()),
        "final_mean_fidelity_score": float(fidelity_score[-1].mean()),
        "initial_final_tile_rank_spearman": rank_correlation,
        "num_scheduled_faults": len(events),
        "num_final_faulted_tiles": int(faulted[-1].sum()),
        "num_final_available_tiles": int(available[-1].sum()),
    }
    trace = FidelityTrace(
        noise_std=noise_std,
        fidelity_score=fidelity_score,
        faulted=faulted,
        available=available,
        tile_class=tile_class,
        thermal_zone=thermal_zone,
        initial_noise_std=initial_noise,
        drift_fraction=drift_fraction,
        fault_events=tuple(events),
        metadata=metadata,
    )
    trace.validate()
    return trace


def load_fidelity_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"noise_std", "fidelity_score", "faulted", "available"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"Fidelity trace is missing arrays: {missing}")
    return arrays
