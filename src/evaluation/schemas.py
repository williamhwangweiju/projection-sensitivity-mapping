"""Data schemas for Phase 4 tile-level quality evaluation.

Canonical weight convention
---------------------------
All projection weights use::

    W_canonical[out_features, in_features]

Hugging Face GPT-2 ``Conv1D`` stores the transpose, so::

    W_canonical = module.weight.T
    module.weight = W_canonical.T
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectionNoiseCalibration:
    """Empirical Phase-1 calibration for one GPT-2 projection."""

    projection_id: str
    hf_module_path: str
    reference_sigma_normalized: float
    measured_noise_std_absolute: float
    measured_noise_rms_absolute: float
    noise_reference_scale: float
    calibration_source: str
    num_calibration_seeds: int

    def __post_init__(self) -> None:
        if not self.projection_id:
            raise ValueError("projection_id cannot be empty.")
        if not self.hf_module_path:
            raise ValueError("hf_module_path cannot be empty.")
        for name, value, allow_zero in (
            ("reference_sigma_normalized", self.reference_sigma_normalized, False),
            ("measured_noise_std_absolute", self.measured_noise_std_absolute, True),
            ("measured_noise_rms_absolute", self.measured_noise_rms_absolute, True),
            ("noise_reference_scale", self.noise_reference_scale, False),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if allow_zero:
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative.")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.num_calibration_seeds <= 0:
            raise ValueError("num_calibration_seeds must be positive.")

        expected = self.measured_noise_std_absolute / self.reference_sigma_normalized
        if not math.isclose(
            self.noise_reference_scale,
            expected,
            rel_tol=1e-5,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "noise_reference_scale is inconsistent with "
                "measured_noise_std_absolute/reference_sigma_normalized: "
                f"stored={self.noise_reference_scale:.12e}, "
                f"expected={expected:.12e}."
            )

    def to_row(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "hf_module_path": self.hf_module_path,
            "reference_sigma_normalized": self.reference_sigma_normalized,
            "measured_noise_std_absolute": self.measured_noise_std_absolute,
            "measured_noise_rms_absolute": self.measured_noise_rms_absolute,
            "noise_reference_scale": self.noise_reference_scale,
            "calibration_source": self.calibration_source,
            "num_calibration_seeds": self.num_calibration_seeds,
        }


@dataclass(frozen=True, slots=True)
class GPT2ShardAssignment:
    """Map one Phase-3 shard to exact GPT-2 canonical coordinates."""

    shard_id: str
    projection_id: str
    hf_module_path: str
    sim_module_path: str
    sim_layer: int
    target_gpt2_block: int
    qkv_component: str | None

    # Simulator orientation: [input_features, output_features].
    sim_input_start: int
    sim_input_end: int
    sim_output_start: int
    sim_output_end: int

    # Canonical orientation: [output_features, input_features].
    canonical_row_start: int
    canonical_row_end: int
    canonical_col_start: int
    canonical_col_end: int

    tile_id: int
    tier_id: int
    policy: str
    placement_seed: int

    weights_in_shard: int
    weights_in_projection: int
    shard_weight: float
    sensitivity_score: float

    def __post_init__(self) -> None:
        if not self.shard_id:
            raise ValueError("shard_id cannot be empty.")
        if not self.projection_id:
            raise ValueError("projection_id cannot be empty.")
        if not self.hf_module_path:
            raise ValueError("hf_module_path cannot be empty.")
        if self.qkv_component not in {None, "Q", "K", "V"}:
            raise ValueError(f"Invalid qkv_component: {self.qkv_component!r}.")
        if self.sim_layer < 0 or self.target_gpt2_block < 0:
            raise ValueError("Layer indices cannot be negative.")
        if not (0 <= self.sim_input_start < self.sim_input_end):
            raise ValueError(f"Invalid simulator input range for {self.shard_id}.")
        if not (0 <= self.sim_output_start < self.sim_output_end):
            raise ValueError(f"Invalid simulator output range for {self.shard_id}.")
        if not (0 <= self.canonical_row_start < self.canonical_row_end):
            raise ValueError(f"Invalid canonical row range for {self.shard_id}.")
        if not (0 <= self.canonical_col_start < self.canonical_col_end):
            raise ValueError(f"Invalid canonical column range for {self.shard_id}.")
        if self.tile_id < 0 or self.tier_id < 0:
            raise ValueError("tile_id and tier_id cannot be negative.")
        if self.weights_in_shard <= 0 or self.weights_in_projection <= 0:
            raise ValueError("Weight counts must be positive.")

        canonical_count = (
            (self.canonical_row_end - self.canonical_row_start)
            * (self.canonical_col_end - self.canonical_col_start)
        )
        simulator_count = (
            (self.sim_input_end - self.sim_input_start)
            * (self.sim_output_end - self.sim_output_start)
        )
        if canonical_count != self.weights_in_shard:
            raise ValueError(
                f"{self.shard_id}: canonical slice has {canonical_count} weights, "
                f"but weights_in_shard={self.weights_in_shard}."
            )
        if simulator_count != self.weights_in_shard:
            raise ValueError(
                f"{self.shard_id}: simulator slice has {simulator_count} weights, "
                f"but weights_in_shard={self.weights_in_shard}."
            )
        if not math.isfinite(self.shard_weight) or not 0.0 < self.shard_weight <= 1.0:
            raise ValueError("shard_weight must be finite and in (0, 1].")
        expected_weight = self.weights_in_shard / self.weights_in_projection
        if not math.isclose(self.shard_weight, expected_weight, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(
                f"{self.shard_id}: shard_weight={self.shard_weight} does not match "
                f"weights_in_shard/weights_in_projection={expected_weight}."
            )
        if not math.isfinite(self.sensitivity_score):
            raise ValueError("sensitivity_score must be finite.")

    def to_row(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "projection_id": self.projection_id,
            "hf_module_path": self.hf_module_path,
            "sim_module_path": self.sim_module_path,
            "sim_layer": self.sim_layer,
            "target_gpt2_block": self.target_gpt2_block,
            "qkv_component": self.qkv_component or "",
            "sim_input_start": self.sim_input_start,
            "sim_input_end": self.sim_input_end,
            "sim_output_start": self.sim_output_start,
            "sim_output_end": self.sim_output_end,
            "canonical_row_start": self.canonical_row_start,
            "canonical_row_end": self.canonical_row_end,
            "canonical_col_start": self.canonical_col_start,
            "canonical_col_end": self.canonical_col_end,
            "tile_id": self.tile_id,
            "tier_id": self.tier_id,
            "policy": self.policy,
            "placement_seed": self.placement_seed,
            "weights_in_shard": self.weights_in_shard,
            "weights_in_projection": self.weights_in_projection,
            "shard_weight": self.shard_weight,
            "sensitivity_score": self.sensitivity_score,
        }


@dataclass(frozen=True, slots=True)
class NoiseInjectionRecord:
    """Record the noise assigned to one shard for one evaluation."""

    shard_id: str
    projection_id: str
    hf_module_path: str
    tile_id: int
    tier_id: int
    policy: str
    timestep: int
    noise_realization: int
    noise_realization_seed: int
    tile_noise_std_normalized: float
    noise_reference_scale: float
    tile_noise_std_absolute: float
    noise_reference_source: str
    is_faulted: bool
    is_available: bool
    unavailable_action: str

    def to_row(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "projection_id": self.projection_id,
            "hf_module_path": self.hf_module_path,
            "tile_id": self.tile_id,
            "tier_id": self.tier_id,
            "policy": self.policy,
            "timestep": self.timestep,
            "noise_realization": self.noise_realization,
            "noise_realization_seed": self.noise_realization_seed,
            "tile_noise_std_normalized": self.tile_noise_std_normalized,
            "noise_reference_scale": self.noise_reference_scale,
            "tile_noise_std_absolute": self.tile_noise_std_absolute,
            "noise_reference_source": self.noise_reference_source,
            "is_faulted": self.is_faulted,
            "is_available": self.is_available,
            "unavailable_action": self.unavailable_action,
        }


@dataclass(slots=True)
class QualityMetrics:
    """Quality for one policy, hardware snapshot, and noise realization."""

    policy: str
    timestep: int
    noise_realization: int
    noise_realization_seed: int
    trace_seed: int
    placement_seed: int

    digital_nll: float
    digital_ppl: float
    reference_nll: float
    reference_ppl: float
    nll: float
    ppl: float
    delta_nll: float
    delta_ppl: float
    kl_divergence: float
    next_token_agreement: float

    num_faulted_shards: int = 0
    num_unavailable_shards: int = 0
    mean_tile_noise_normalized: float = 0.0
    hardware_state_mode: str = "operational_degradation"
    unavailable_action: str = "error"

    def to_row(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "timestep": self.timestep,
            "noise_realization": self.noise_realization,
            "noise_realization_seed": self.noise_realization_seed,
            "trace_seed": self.trace_seed,
            "placement_seed": self.placement_seed,
            "digital_nll": self.digital_nll,
            "digital_ppl": self.digital_ppl,
            "reference_nll": self.reference_nll,
            "reference_ppl": self.reference_ppl,
            "nll": self.nll,
            "ppl": self.ppl,
            "delta_nll": self.delta_nll,
            "delta_ppl": self.delta_ppl,
            "kl_divergence": self.kl_divergence,
            "next_token_agreement": self.next_token_agreement,
            "num_faulted_shards": self.num_faulted_shards,
            "num_unavailable_shards": self.num_unavailable_shards,
            "mean_tile_noise_normalized": self.mean_tile_noise_normalized,
            "hardware_state_mode": self.hardware_state_mode,
            "unavailable_action": self.unavailable_action,
        }


@dataclass(slots=True)
class PairedPolicyDifference:
    """Paired DeltaNLL difference under an identical random realization."""

    timestep: int
    noise_realization: int
    noise_realization_seed: int
    trace_seed: int
    placement_seed: int
    policy_a: str
    policy_b: str
    delta_nll_a: float
    delta_nll_b: float
    delta_ppl_a: float
    delta_ppl_b: float
    difference: float  # Positive means policy_b has lower DeltaNLL.
    difference_delta_ppl: float

    def to_row(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "noise_realization": self.noise_realization,
            "noise_realization_seed": self.noise_realization_seed,
            "trace_seed": self.trace_seed,
            "placement_seed": self.placement_seed,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "delta_nll_a": self.delta_nll_a,
            "delta_nll_b": self.delta_nll_b,
            "delta_ppl_a": self.delta_ppl_a,
            "delta_ppl_b": self.delta_ppl_b,
            "difference": self.difference,
            "difference_delta_ppl": self.difference_delta_ppl,
        }


@dataclass(slots=True)
class WeightChecksum:
    """SHA-256 checksums used to verify exact weight restoration."""

    hf_module_path: str
    checksum_before: str
    checksum_after_noise: str
    checksum_restored: str
    weights_match_original: bool

    def to_row(self) -> dict[str, Any]:
        return {
            "hf_module_path": self.hf_module_path,
            "checksum_before": self.checksum_before,
            "checksum_after_noise": self.checksum_after_noise,
            "checksum_restored": self.checksum_restored,
            "weights_match_original": self.weights_match_original,
        }


@dataclass(slots=True)
class ProxyCorrelationRecord:
    """Correlation between a Phase-3 proxy and measured Phase-4 quality."""

    timestep: int
    proxy_vs_metric: str
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n_samples: int

    def to_row(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "proxy_vs_metric": self.proxy_vs_metric,
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "n_samples": self.n_samples,
        }
