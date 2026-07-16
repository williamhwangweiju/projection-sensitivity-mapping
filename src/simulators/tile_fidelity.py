"""Deterministic heterogeneous and time-varying tile-fidelity model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.simulators.hardware import HardwareConfig


@dataclass(frozen=True)
class TileFidelityTrace:
    noise_std: np.ndarray
    fidelity_score: np.ndarray
    available: np.ndarray
    faulted: np.ndarray
    class_code: np.ndarray
    thermal_zone: np.ndarray
    fault_onset: np.ndarray

    def validate(self) -> None:
        if self.noise_std.ndim != 2:
            raise ValueError("noise_std must have shape [timesteps, tiles].")
        shape = self.noise_std.shape
        for name in ("fidelity_score", "available", "faulted"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape mismatch.")
        if not np.isfinite(self.noise_std).all() or (self.noise_std < 0).any():
            raise ValueError("Invalid noise values.")


class TileFidelityModel:
    def __init__(self, hardware: HardwareConfig, config: Mapping[str, Any], seed: int) -> None:
        self.hardware = hardware
        self.cfg = config["fidelity_model"]
        self.seed = int(seed)

    def generate(self) -> TileFidelityTrace:
        rng = np.random.default_rng(self.seed)
        n_tiles = self.hardware.num_tiles
        timesteps = int(self.cfg["num_timesteps"])
        reference = float(self.cfg["reference_noise_std"])
        min_noise = float(self.cfg["min_noise_std"])
        max_noise = float(self.cfg["max_noise_std"])
        classes = self.cfg["fidelity_classes"]

        labels: list[str] = []
        for name, definition in classes.items():
            labels.extend([name] * int(round(float(definition["fraction"]) * n_tiles)))
        labels = (labels + ["medium"] * n_tiles)[:n_tiles]
        rng.shuffle(labels)
        class_names = sorted(classes)
        class_to_code = {name: index for index, name in enumerate(class_names)}
        class_code = np.array([class_to_code[name] for name in labels], dtype=np.int16)
        base = np.empty(n_tiles, dtype=np.float64)
        for tile, label in enumerate(labels):
            low, high = map(float, classes[label]["noise_multiplier_range"])
            base[tile] = reference * rng.uniform(low, high)

        drift_cfg = self.cfg["degradation"]["gradual_drift"]
        if bool(drift_cfg.get("enabled", True)):
            low, high = map(float, drift_cfg["total_increase_range"])
            drift_total = rng.uniform(low, high, size=n_tiles)
        else:
            drift_total = np.zeros(n_tiles)

        zones = np.arange(n_tiles, dtype=np.int16) % self.hardware.num_thermal_zones
        rng.shuffle(zones)
        thermal_cfg = self.cfg["degradation"]["thermal_variation"]
        rho = float(thermal_cfg.get("correlation", 0.94))
        thermal_std = float(thermal_cfg.get("standard_deviation_fraction", 0.05))
        zone_state = np.zeros(self.hardware.num_thermal_zones, dtype=np.float64)

        fault_cfg = self.cfg["degradation"]["localized_fault"]
        fault_count = int(fault_cfg.get("num_affected_tiles", 0)) if bool(fault_cfg.get("enabled", True)) else 0
        candidates = [index for index, label in enumerate(labels) if label in set(fault_cfg.get("candidate_classes", class_names))]
        chosen = rng.choice(candidates, size=min(fault_count, len(candidates)), replace=False) if candidates else np.array([], dtype=int)
        onset_low, onset_high = map(int, fault_cfg.get("onset_timestep_range", [timesteps, timesteps]))
        fault_onset = np.full(n_tiles, -1, dtype=np.int32)
        fault_multiplier = np.zeros(n_tiles, dtype=np.float64)
        for tile in chosen:
            fault_onset[tile] = int(rng.integers(onset_low, onset_high + 1))
            low, high = map(float, fault_cfg.get("noise_increase_range", [0.2, 0.6]))
            fault_multiplier[tile] = rng.uniform(low, high)

        noise = np.empty((timesteps, n_tiles), dtype=np.float64)
        faulted = np.zeros((timesteps, n_tiles), dtype=bool)
        available = np.ones((timesteps, n_tiles), dtype=bool)
        for timestep in range(timesteps):
            progress = 0.0 if timesteps <= 1 else timestep / (timesteps - 1)
            if bool(thermal_cfg.get("enabled", True)):
                innovation_std = thermal_std * np.sqrt(max(1.0 - rho**2, 0.0))
                zone_state = rho * zone_state + rng.normal(0.0, innovation_std, size=len(zone_state))
            thermal = zone_state[zones]
            current = base * (1.0 + drift_total * progress + thermal)
            for tile in chosen:
                if timestep >= fault_onset[tile]:
                    current[tile] *= 1.0 + fault_multiplier[tile]
                    faulted[timestep, tile] = True
                    if bool(fault_cfg.get("make_unavailable", False)):
                        available[timestep, tile] = False
            noise[timestep] = np.clip(current, min_noise, max_noise)

        fidelity = 1.0 / (1.0 + noise / max(reference, 1e-12))
        trace = TileFidelityTrace(
            noise_std=noise.astype(np.float32),
            fidelity_score=fidelity.astype(np.float32),
            available=available,
            faulted=faulted,
            class_code=class_code,
            thermal_zone=zones,
            fault_onset=fault_onset,
        )
        trace.validate()
        return trace


def save_trace(path: str, trace: TileFidelityTrace) -> None:
    np.savez_compressed(
        path,
        noise_std=trace.noise_std,
        fidelity_score=trace.fidelity_score,
        available=trace.available,
        faulted=trace.faulted,
        class_code=trace.class_code,
        thermal_zone=trace.thermal_zone,
        fault_onset=trace.fault_onset,
    )


def load_trace(path: str) -> TileFidelityTrace:
    data = np.load(path)
    trace = TileFidelityTrace(**{name: data[name] for name in (
        "noise_std", "fidelity_score", "available", "faulted", "class_code", "thermal_zone", "fault_onset"
    )})
    trace.validate()
    return trace
