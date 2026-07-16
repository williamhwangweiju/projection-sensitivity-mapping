"""All-projection AIHWKit conversion using the shared manual weight pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped

from src.common.analog import (
    ManualAnalogSettings,
    get_analog_weights_exact,
    make_rpu_config,
    prepare_projection_weight,
    set_analog_weights_exact,
    tensor_checksum,
)
from src.common.projections import (
    ProjectionHandle,
    canonical_weight_bias,
    iter_gpt2_projections,
    linear_from_canonical,
)


@dataclass
class AnalogProjectionReference:
    projection_id: str
    block_index: int
    projection_name: str
    hf_module_path: str
    parent: Any
    attribute: str
    original_module: Any
    analog_module: AnalogLinearMapped
    clipped_weight: Tensor
    bias: Tensor | None
    preprocessing: dict[str, Any]

    def restore_reference(self) -> None:
        set_analog_weights_exact(
            self.analog_module,
            self.clipped_weight,
            self.bias,
            verify=False,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "block_index": self.block_index,
            "projection_name": self.projection_name,
            "hf_module_path": self.hf_module_path,
            "weight_shape_out_in": list(self.clipped_weight.shape),
            "preprocessing": self.preprocessing,
        }


def _phase1_projection_map(phase1_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    results = phase1_payload.get("results", phase1_payload)
    records = results.get("projections")
    if not isinstance(records, list):
        raise ValueError("Phase-1 payload does not contain results.projections.")
    return {str(record["projection_id"]): record for record in records}


def convert_all_transformer_projections(
    model: Any,
    config: Mapping[str, Any],
    phase1_payload: Mapping[str, Any],
) -> dict[str, AnalogProjectionReference]:
    """Replace all 48 GPT-2 projections by identical clipped AIHWKit layers."""
    device = torch.device(str(config["model"]["device"]))
    settings = ManualAnalogSettings.from_config(config)
    phase1_map = _phase1_projection_map(phase1_payload)
    references: dict[str, AnalogProjectionReference] = {}

    handles = list(iter_gpt2_projections(model))
    if len(handles) != 48:
        raise RuntimeError(f"Expected 48 GPT-2 transformer projections, found {len(handles)}.")

    for handle in handles:
        original_weight, bias = canonical_weight_bias(handle.module)
        prepared = prepare_projection_weight(original_weight, settings)
        phase1_record = phase1_map.get(handle.projection_id)
        if phase1_record is None:
            raise ValueError(f"Phase-1 result is missing {handle.projection_id}.")
        phase1_preprocessing = phase1_record.get("preprocessing")
        if not isinstance(phase1_preprocessing, Mapping):
            raise ValueError(f"Phase-1 preprocessing is missing for {handle.projection_id}.")
        for field in ("original_checksum", "clipped_checksum", "range_mode"):
            if str(phase1_preprocessing[field]) != str(
                getattr(prepared.preprocessing, field)
            ):
                raise ValueError(
                    f"{handle.projection_id}: Phase-1/Phase-4 preprocessing mismatch "
                    f"for {field}."
                )
        for field in ("clip_threshold", "programmed_range", "original_std"):
            expected = float(phase1_preprocessing[field])
            actual = float(getattr(prepared.preprocessing, field))
            if abs(expected - actual) > max(1e-9, abs(expected) * 1e-6):
                raise ValueError(
                    f"{handle.projection_id}: Phase-1/Phase-4 {field} mismatch: "
                    f"{expected} vs {actual}."
                )

        digital_linear = linear_from_canonical(prepared.clipped_weight, bias, device)
        analog = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=make_rpu_config(settings),
        ).to(device)
        analog.eval()
        readback, _ = get_analog_weights_exact(analog)
        max_error = float((readback - prepared.clipped_weight).abs().max().item())
        if max_error > 3e-6:
            raise RuntimeError(
                f"{handle.projection_id}: clean clipped conversion changed logical "
                f"weights; max_error={max_error:.8e}."
            )
        setattr(handle.parent, handle.attribute, analog)
        references[handle.projection_id] = AnalogProjectionReference(
            projection_id=handle.projection_id,
            block_index=handle.block_index,
            projection_name=handle.projection_name,
            hf_module_path=handle.hf_module_path,
            parent=handle.parent,
            attribute=handle.attribute,
            original_module=handle.module,
            analog_module=analog,
            clipped_weight=prepared.clipped_weight,
            bias=bias,
            preprocessing=prepared.preprocessing.to_dict(),
        )

    if set(references) != set(phase1_map):
        extra = sorted(set(phase1_map) - set(references))
        if extra:
            raise ValueError(f"Phase-1 payload contains unexpected projections: {extra}")
    return references


def restore_all_references(
    references: Mapping[str, AnalogProjectionReference],
) -> None:
    for reference in references.values():
        reference.restore_reference()


def restore_original_digital_modules(
    references: Mapping[str, AnalogProjectionReference],
) -> None:
    for reference in references.values():
        setattr(reference.parent, reference.attribute, reference.original_module)


def reference_checksums(
    references: Mapping[str, AnalogProjectionReference],
) -> list[dict[str, Any]]:
    rows = []
    for reference in references.values():
        actual, _ = get_analog_weights_exact(reference.analog_module)
        rows.append(
            {
                "projection_id": reference.projection_id,
                "expected_clipped_checksum": reference.preprocessing[
                    "clipped_checksum"
                ],
                "actual_logical_checksum": tensor_checksum(actual),
                "max_reference_error": float(
                    (actual - reference.clipped_weight).abs().max().item()
                ),
            }
        )
    return rows
