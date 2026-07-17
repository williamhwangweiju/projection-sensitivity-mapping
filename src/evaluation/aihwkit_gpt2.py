"""Hybrid GPT-2 conversion utilities for projection-selective analog execution."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterable, Iterator, Mapping
import torch
from torch import Tensor

from src.common.analog import (
    ManualAnalogSettings,
    get_analog_weights_exact,
    make_rpu_config,
    prepare_projection_weight,
    set_analog_weights_exact,
)
from src.common.projections import (
    ProjectionHandle,
    canonical_weight_bias,
    iter_gpt2_projections,
    linear_from_canonical,
)


@dataclass
class AnalogProjectionState:
    handle: ProjectionHandle
    analog_module: Any
    clipped_weight: Tensor
    bias: Tensor | None
    preprocessing: dict[str, Any]

    @property
    def programmed_range(self) -> float:
        return float(self.preprocessing["programmed_range"])

    @property
    def clipped_fraction(self) -> float:
        return float(self.preprocessing["fraction_clipped"])


def _phase1_map(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    return {} if rows is None else {str(row["projection_id"]): row for row in rows}


def _validate_phase1_preprocessing(
    projection_id: str,
    actual: Mapping[str, Any],
    expected_row: Mapping[str, Any] | None,
) -> None:
    if expected_row is None:
        return
    expected = expected_row.get("preprocessing")
    if not isinstance(expected, Mapping):
        # Older profiles did not persist checksums. Their numeric metadata can
        # still be used, but strict cross-phase checksum validation is unavailable.
        return
    for field in ("original_checksum", "clipped_checksum", "range_mode"):
        if str(expected[field]) != str(actual[field]):
            raise ValueError(
                f"{projection_id}: Phase-1/hybrid preprocessing mismatch for {field}."
            )
    for field in ("original_std", "clip_threshold", "programmed_range"):
        wanted = float(expected[field])
        observed = float(actual[field])
        if not math.isclose(wanted, observed, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f"{projection_id}: Phase-1/hybrid {field} mismatch: "
                f"{wanted} vs {observed}."
            )


class HybridAnalogModel:
    """Convert only the analog projection set while protected projections stay digital."""

    def __init__(
        self,
        model: Any,
        *,
        digital_projection_ids: Iterable[str],
        settings: ManualAnalogSettings,
        include_lm_head_candidate: bool,
        phase1_projection_rows: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.digital_projection_ids = frozenset(digital_projection_ids)
        self.include_lm_head_candidate = include_lm_head_candidate
        self.phase1_by_id = _phase1_map(phase1_projection_rows)
        self.states: dict[str, AnalogProjectionState] = {}
        self.original_modules: dict[str, Any] = {}

    def convert(self) -> "HybridAnalogModel":
        try:
            from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
        except ImportError as exc:
            raise RuntimeError(
                "AIHWKit 1.1.0 is required for Phase 1/4/5 quality runs."
            ) from exc

        device = next(self.model.parameters()).device
        handles = list(
            iter_gpt2_projections(
                self.model, include_lm_head=self.include_lm_head_candidate
            )
        )
        known_model_ids = {handle.projection_id for handle in handles}

        # The Phase-1 artifact is the authoritative candidate universe. This is
        # essential for reduced smoke profiles: projections that were not
        # profiled are left digital rather than being analogized without a
        # sensitivity score or a Phase-3 physical placement. Full paper runs
        # profile all 48 transformer projections plus the optional LM head.
        candidate_ids = (
            set(self.phase1_by_id) if self.phase1_by_id else set(known_model_ids)
        )
        unknown_candidates = candidate_ids - known_model_ids
        if unknown_candidates:
            raise ValueError(
                "Phase-1 artifact contains projections absent from this model: "
                f"{sorted(unknown_candidates)}"
            )
        unknown_digital = set(self.digital_projection_ids) - candidate_ids
        if unknown_digital:
            raise ValueError(
                "Digital set contains projections outside the Phase-1 candidate "
                f"universe: {sorted(unknown_digital)}"
            )

        for handle in handles:
            if (
                handle.projection_id not in candidate_ids
                or handle.projection_id in self.digital_projection_ids
            ):
                continue

            original_weight, bias = canonical_weight_bias(handle.module)

            prepared = prepare_projection_weight(
                original_weight,
                self.settings,
            )

            preprocessing = prepared.preprocessing.to_dict()

            _validate_phase1_preprocessing(
                handle.projection_id,
                preprocessing,
                self.phase1_by_id.get(handle.projection_id),
            )

            # AIHWKit requires CPU-backed source weights during mapped-layer
            # construction.
            cpu_device = torch.device("cpu")

            cpu_weight = prepared.clipped_weight.detach().to(
                device=cpu_device,
                dtype=torch.float32,
            )

            cpu_bias = (
                None
                if bias is None
                else bias.detach().to(
                    device=cpu_device,
                    dtype=torch.float32,
                )
            )

            digital_linear = linear_from_canonical(
                cpu_weight,
                cpu_bias,
                cpu_device,
            )

            analog = AnalogLinearMapped.from_digital(
                digital_linear,
                rpu_config=make_rpu_config(self.settings),
            )

            # Move only after construction is complete.
            analog = analog.to(device)
            analog.eval()

            runtime_weight = prepared.clipped_weight.detach().to(
                device=device,
                dtype=torch.float32,
            )

            runtime_bias = (
                None
                if bias is None
                else bias.detach().to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            set_analog_weights_exact(
                analog,
                runtime_weight,
                runtime_bias,
                verify=True,
            )

            self.original_modules[handle.projection_id] = handle.module

            setattr(
                handle.parent,
                handle.attribute,
                analog,
            )

            self.states[handle.projection_id] = AnalogProjectionState(
                handle=handle,
                analog_module=analog,
                clipped_weight=runtime_weight.detach().clone(),
                bias=(
                    None
                    if runtime_bias is None
                    else runtime_bias.detach().clone()
                ),
                preprocessing=preprocessing,
            )
        return self

    @property
    def analog_projection_ids(self) -> tuple[str, ...]:
        return tuple(self.states)

    def restore_nominal_weights(self) -> None:
        for state in self.states.values():
            set_analog_weights_exact(
                state.analog_module,
                state.clipped_weight,
                state.bias,
                verify=False,
            )

    def snapshot_weights(self) -> dict[str, Tensor]:
        return {
            projection_id: get_analog_weights_exact(state.analog_module)[0].clone()
            for projection_id, state in self.states.items()
        }

    def assert_nominal_restored(self, atol: float = 3e-6) -> None:
        for projection_id, state in self.states.items():
            actual, _ = get_analog_weights_exact(state.analog_module)
            error = float((actual - state.clipped_weight).abs().max().item())
            if error > atol:
                raise RuntimeError(
                    f"{projection_id} was not restored exactly; max error={error:.3e}."
                )

    def swap_to_digital(self, projection_ids: Iterable[str]) -> list[str]:
        """Temporarily route converted projections through their original digital modules.

        Projections that are permanently digital (in ``digital_projection_ids``)
        are ignored, so callers may pass a full candidate digital set. The
        analog modules and their programmed weights are preserved and can be
        reinstalled with :meth:`swap_back_to_analog`. This makes greedy
        digital-set search O(1) module swaps per trial instead of rebuilding
        the whole hybrid conversion.
        """
        swapped: list[str] = []
        for raw_id in projection_ids:
            projection_id = str(raw_id)
            if projection_id in self.digital_projection_ids:
                continue
            state = self.states.get(projection_id)
            if state is None:
                raise KeyError(
                    f"{projection_id} is not part of the converted analog set."
                )
            setattr(
                state.handle.parent,
                state.handle.attribute,
                self.original_modules[projection_id],
            )
            swapped.append(projection_id)
        return swapped

    def swap_back_to_analog(self, projection_ids: Iterable[str]) -> None:
        """Reinstall the analog modules for projections swapped out by swap_to_digital."""
        for raw_id in projection_ids:
            projection_id = str(raw_id)
            state = self.states[projection_id]
            setattr(state.handle.parent, state.handle.attribute, state.analog_module)

    @contextmanager
    def temporarily_digital(self, projection_ids: Iterable[str]) -> Iterator[list[str]]:
        """Context manager wrapping swap_to_digital / swap_back_to_analog."""
        swapped = self.swap_to_digital(projection_ids)
        try:
            yield swapped
        finally:
            self.swap_back_to_analog(swapped)

    def restore_digital_modules(self) -> None:
        for projection_id, original in self.original_modules.items():
            state = self.states[projection_id]
            setattr(state.handle.parent, state.handle.attribute, original)
        self.states.clear()
        self.original_modules.clear()

    def metadata(self) -> dict[str, Any]:
        return {
            "digital_projection_ids": sorted(self.digital_projection_ids),
            "analog_projection_ids": sorted(self.states),
            "analog_projection_count": len(self.states),
            "preprocessing_by_projection": {
                key: value.preprocessing for key, value in self.states.items()
            },
        }


__all__ = [
    "AnalogProjectionState",
    "HybridAnalogModel",
    "get_analog_weights_exact",
    "set_analog_weights_exact",
]
