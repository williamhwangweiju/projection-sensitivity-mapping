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

from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def cpu_default_device() -> Iterator[None]:
    """Force implicit PyTorch allocations onto CPU temporarily."""
    previous_device = torch.get_default_device()

    try:
        torch.set_default_device("cpu")
        yield
    finally:
        torch.set_default_device(previous_device)


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


@dataclass
class DigitalProjectionState:
    """A projection retained in digital compute with *clipped* float weights."""

    handle: ProjectionHandle
    digital_module: Any
    preprocessing: dict[str, Any]


#: Accepted values for ``HybridAnalogModel(digital_weight_mode=...)``.
DIGITAL_WEIGHT_MODES = ("unclipped", "clipped")


def swap_in_clipped_digital_module(
    handle: ProjectionHandle,
    settings: ManualAnalogSettings,
    device: torch.device,
) -> DigitalProjectionState:
    """Replace ``handle``'s module by a float32 ``nn.Linear`` holding the
    clipped weights (clip at ``settings.clip_sigma`` population std, exactly
    as the analog conversion path), with no quantization and no noise.

    The original module object is left untouched so it can be swapped back
    with ``setattr(handle.parent, handle.attribute, handle.module)``. For the
    LM head this deliberately breaks weight tying with the token embedding:
    the digital head computes ``x @ clip(W)^T`` while the embedding lookup keeps
    the checkpoint's unclipped table, mirroring the all-analog deployment in
    which the head tiles hold ``clip(W)`` and the embedding lookup is digital.
    """
    original_weight, bias = canonical_weight_bias(handle.module)
    prepared = prepare_projection_weight(original_weight, settings)
    digital_linear = linear_from_canonical(prepared.clipped_weight, bias, device)
    digital_linear.eval()
    for parameter in digital_linear.parameters():
        parameter.requires_grad_(False)
    setattr(handle.parent, handle.attribute, digital_linear)
    return DigitalProjectionState(
        handle=handle,
        digital_module=digital_linear,
        preprocessing=prepared.preprocessing.to_dict(),
    )


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
    """Convert only the analog projection set while protected projections stay digital.

    ``digital_weight_mode`` controls how the protected (digital) projections
    hold their weights: ``"unclipped"`` (default, historical behaviour) leaves
    the checkpoint's floating-point modules untouched; ``"clipped"`` swaps each
    protected projection for a float32 module holding the 2.5-sigma-clipped
    weights the checkpoint was trained through (no quantization, no noise), the
    deployment-relevant digital comparator. Both swaps are undone by
    :meth:`restore_digital_modules`.
    """

    def __init__(
        self,
        model: Any,
        *,
        digital_projection_ids: Iterable[str],
        settings: ManualAnalogSettings,
        include_lm_head_candidate: bool,
        phase1_projection_rows: Iterable[Mapping[str, Any]] | None = None,
        digital_weight_mode: str = "unclipped",
    ) -> None:
        mode = str(digital_weight_mode).strip().lower()
        if mode not in DIGITAL_WEIGHT_MODES:
            raise ValueError(
                f"digital_weight_mode must be one of {DIGITAL_WEIGHT_MODES}; got {mode!r}."
            )
        self.model = model
        self.settings = settings
        self.digital_projection_ids = frozenset(digital_projection_ids)
        self.include_lm_head_candidate = include_lm_head_candidate
        self.phase1_by_id = _phase1_map(phase1_projection_rows)
        self.digital_weight_mode = mode
        self.states: dict[str, AnalogProjectionState] = {}
        self.digital_states: dict[str, DigitalProjectionState] = {}
        self.original_modules: dict[str, Any] = {}
        self.handles: dict[str, ProjectionHandle] = {}

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
            if handle.projection_id not in candidate_ids:
                continue
            if handle.projection_id in self.digital_projection_ids:
                if self.digital_weight_mode == "clipped":
                    state = swap_in_clipped_digital_module(
                        handle, self.settings, device
                    )
                    _validate_phase1_preprocessing(
                        handle.projection_id,
                        state.preprocessing,
                        self.phase1_by_id.get(handle.projection_id),
                    )
                    self.original_modules[handle.projection_id] = handle.module
                    self.handles[handle.projection_id] = handle
                    self.digital_states[handle.projection_id] = state
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

            cpu_weight = prepared.clipped_weight.detach().to(
                device="cpu",
                dtype=torch.float32,
            )

            cpu_bias = (
                None
                if bias is None
                else bias.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )

            with cpu_default_device():
                digital_linear = linear_from_canonical(
                    cpu_weight,
                    cpu_bias,
                    torch.device("cpu"),
                )

                if digital_linear.weight.device.type != "cpu":
                    raise RuntimeError(
                        f"{handle.projection_id}: temporary digital layer "
                        f"was created on {digital_linear.weight.device}, expected CPU."
                    )

                analog = AnalogLinearMapped.from_digital(
                    digital_linear,
                    rpu_config=make_rpu_config(self.settings),
                )

            # from_digital completed successfully on CPU. Clear allocator
            # fragmentation before creating this projection's CUDA tiles;
            # converting 49 mapped projections in one long-lived process can
            # otherwise fail CUBLAS initialization mid-way.
            if device.type == "cuda":
                import gc

                gc.collect()
                torch.cuda.empty_cache()
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
            self.handles[handle.projection_id] = handle

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
            expected = state.clipped_weight.to(
                device=actual.device,
                dtype=actual.dtype,
            )
            error = float((actual - expected).abs().max().item())
            if error > atol:
                raise RuntimeError(
                    f"{projection_id} was not restored exactly; max error={error:.3e}."
                )

    @property
    def clipped_digital_projection_ids(self) -> tuple[str, ...]:
        return tuple(self.digital_states)

    def restore_digital_modules(self) -> None:
        """Undo every module swap (analog tiles and clipped-digital copies)."""
        for projection_id, original in self.original_modules.items():
            handle = self.handles[projection_id]
            setattr(handle.parent, handle.attribute, original)
        self.states.clear()
        self.digital_states.clear()
        self.original_modules.clear()
        self.handles.clear()

    def metadata(self) -> dict[str, Any]:
        return {
            "digital_projection_ids": sorted(self.digital_projection_ids),
            "digital_weight_mode": self.digital_weight_mode,
            "clipped_digital_projection_ids": sorted(self.digital_states),
            "analog_projection_ids": sorted(self.states),
            "analog_projection_count": len(self.states),
            "preprocessing_by_projection": {
                key: value.preprocessing for key, value in self.states.items()
            },
            "digital_preprocessing_by_projection": {
                key: value.preprocessing for key, value in self.digital_states.items()
            },
        }


__all__ = [
    "AnalogProjectionState",
    "DIGITAL_WEIGHT_MODES",
    "DigitalProjectionState",
    "HybridAnalogModel",
    "get_analog_weights_exact",
    "set_analog_weights_exact",
    "swap_in_clipped_digital_module",
]
