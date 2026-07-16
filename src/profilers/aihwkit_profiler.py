"""Manual, cross-phase-consistent GPT-2 projection sensitivity profiling."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped

from src.common.analog import (
    ManualAnalogSettings,
    analog_configuration,
    get_analog_weights_exact,
    make_rpu_config,
    materialize_manual_noise,
    prepare_projection_weight,
    projection_noise_seed,
    set_analog_weights_exact,
    set_seed,
)
from src.common.metrics import evaluate_nll_ppl, summarize
from src.common.projections import (
    ProjectionHandle,
    canonical_weight_bias,
    iter_gpt2_projections,
    linear_from_canonical,
)

Batch = Mapping[str, Tensor]


class AIHWKITSensitivityProfiler:
    """Profile one analog projection at a time with explicit manual noise."""

    def __init__(self, model: Any, config: Mapping[str, Any]) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(str(config["model"]["device"]))
        self.settings = ManualAnalogSettings.from_config(config)
        self.settings.validate()
        profile = config["profiling"]
        self.num_seeds = int(profile["num_seeds"])
        self.seed_stride = int(profile.get("seed_stride", 1))
        self.profile_blocks = tuple(int(value) for value in profile["profile_blocks"])
        self.antithetic = bool(profile.get("antithetic", False))
        self.mapping_sensitivity_field = str(
            profile.get("sensitivity_field", "delta_nll_noise_mean")
        )
        self.base_seed = int(config["experiment"]["seed"])
        if self.num_seeds <= 0 or self.seed_stride <= 0:
            raise ValueError("num_seeds and seed_stride must be positive.")
        self.model.to(self.device, dtype=torch.float32)
        self.model.eval()

    def analog_configuration(self) -> dict[str, Any]:
        result = analog_configuration(self.settings)
        result.update(
            {
                "phase1_mode": "one_projection_analog_rest_digital",
                "num_seeds": self.num_seeds,
                "seed_stride": self.seed_stride,
                "antithetic": self.antithetic,
                "mapping_sensitivity_field": self.mapping_sensitivity_field,
            }
        )
        return result

    def _selected_handles(self) -> list[ProjectionHandle]:
        handles = [
            handle
            for handle in iter_gpt2_projections(self.model)
            if handle.block_index in self.profile_blocks
        ]
        expected = len(self.profile_blocks) * 4
        if len(handles) != expected:
            raise RuntimeError(f"Expected {expected} projections, found {len(handles)}.")
        return handles

    def _convert_projection(
        self, handle: ProjectionHandle
    ) -> tuple[AnalogLinearMapped, Tensor, Tensor | None, dict[str, Any]]:
        original_weight, bias = canonical_weight_bias(handle.module)
        prepared = prepare_projection_weight(original_weight, self.settings)
        digital_linear = linear_from_canonical(
            prepared.clipped_weight, bias, self.device
        )
        analog_linear = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=make_rpu_config(self.settings),
        )
        analog_linear = analog_linear.to(self.device)
        analog_linear.eval()
        actual, _ = get_analog_weights_exact(analog_linear)
        max_error = float((actual - prepared.clipped_weight).abs().max().item())
        if max_error > 3e-6:
            raise RuntimeError(
                f"{handle.projection_id}: clipped reference changed during conversion; "
                f"max_error={max_error:.8e}."
            )
        return (
            analog_linear,
            prepared.clipped_weight,
            bias,
            prepared.preprocessing.to_dict(),
        )

    def profile_projection(
        self,
        batches: Iterable[Batch],
        handle: ProjectionHandle,
        digital_nll: float,
        digital_ppl: float,
    ) -> dict[str, Any]:
        original_module = handle.module
        analog_linear, clipped_weight, bias, preprocessing = self._convert_projection(
            handle
        )
        setattr(handle.parent, handle.attribute, analog_linear)
        try:
            reference_nll, reference_ppl, token_count = evaluate_nll_ppl(
                self.model, batches, self.device
            )
            realization_rows: list[dict[str, Any]] = []
            for realization in range(self.num_seeds):
                realization_seed = self.base_seed + realization * self.seed_stride
                seed = projection_noise_seed(realization_seed, handle.projection_id)
                set_seed(seed)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                z = torch.randn(
                    clipped_weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                signs = (1.0, -1.0) if self.antithetic else (1.0,)
                sign_metrics: list[dict[str, float]] = []
                for sign in signs:
                    noisy_weight, noise = materialize_manual_noise(
                        clipped_weight,
                        self.settings.reference_noise_std,
                        float(preprocessing["programmed_range"]),
                        z * sign,
                    )
                    set_analog_weights_exact(analog_linear, noisy_weight, bias, verify=False)
                    nll, ppl, _ = evaluate_nll_ppl(self.model, batches, self.device)
                    sign_metrics.append(
                        {
                            "nll": nll,
                            "ppl": ppl,
                            "noise_std_absolute": float(
                                noise.std(unbiased=False).item()
                            ),
                            "noise_mean_absolute": float(noise.abs().mean().item()),
                            "noise_max_absolute": float(noise.abs().max().item()),
                        }
                    )
                nll = float(sum(row["nll"] for row in sign_metrics) / len(sign_metrics))
                ppl = float(math.exp(nll))
                realization_rows.append(
                    {
                        "realization": realization,
                        "realization_seed": realization_seed,
                        "projection_noise_seed": seed,
                        "antithetic_count": len(signs),
                        "nll": nll,
                        "ppl": ppl,
                        "delta_nll_total": nll - digital_nll,
                        "delta_ppl_total": ppl - digital_ppl,
                        "delta_nll_noise": nll - reference_nll,
                        "delta_ppl_noise": ppl - reference_ppl,
                        "noise_std_absolute": float(
                            sum(row["noise_std_absolute"] for row in sign_metrics)
                            / len(sign_metrics)
                        ),
                        "noise_mean_absolute": float(
                            sum(row["noise_mean_absolute"] for row in sign_metrics)
                            / len(sign_metrics)
                        ),
                        "noise_max_absolute": float(
                            max(row["noise_max_absolute"] for row in sign_metrics)
                        ),
                    }
                )
                set_analog_weights_exact(analog_linear, clipped_weight, bias, verify=False)
        finally:
            setattr(handle.parent, handle.attribute, original_module)

        fields = {
            name: summarize([float(row[name]) for row in realization_rows])
            for name in (
                "nll",
                "ppl",
                "delta_nll_total",
                "delta_ppl_total",
                "delta_nll_noise",
                "delta_ppl_noise",
                "noise_std_absolute",
            )
        }
        sensitivity_value = float(fields[self.mapping_sensitivity_field]["mean"])
        result = {
            "projection_id": handle.projection_id,
            "block_index": handle.block_index,
            "projection_name": handle.projection_name,
            "block_label": f"block_{handle.block_index}",
            "projection_label": handle.projection_name,
            "hf_module_path": handle.hf_module_path,
            "weight_shape_out_in": list(clipped_weight.shape),
            "num_weights": int(clipped_weight.numel()),
            "token_count": int(token_count),
            "nll_clean": digital_nll,
            "ppl_clean": digital_ppl,
            "nll_analog_reference": reference_nll,
            "ppl_analog_reference": reference_ppl,
            "delta_nll_analog_reference": reference_nll - digital_nll,
            "delta_ppl_analog_reference": reference_ppl - digital_ppl,
            "preprocessing": preprocessing,
            "reference_noise_std_normalized": self.settings.reference_noise_std,
            "realizations": realization_rows,
            "sensitivity_score_for_mapping": sensitivity_value,
            "sensitivity_score_unit": self.mapping_sensitivity_field.removesuffix("_mean"),
        }
        for field_name, stats in fields.items():
            for statistic, value in stats.items():
                result[f"{field_name}_{statistic}"] = float(value)
        return result

    def profile_all(self, batches: list[Batch]) -> dict[str, Any]:
        digital_nll, digital_ppl, token_count = evaluate_nll_ppl(
            self.model, batches, self.device
        )
        projections: list[dict[str, Any]] = []
        handles = self._selected_handles()
        for index, handle in enumerate(handles, start=1):
            print(f"Profiling {index}/{len(handles)}: {handle.projection_id}", flush=True)
            result = self.profile_projection(
                batches, handle, digital_nll, digital_ppl
            )
            projections.append(result)
            print(
                f"{handle.projection_id}: reference PPL={result['ppl_analog_reference']:.6f}, "
                f"noise DeltaNLL={result['delta_nll_noise_mean']:.8f}, "
                f"noise DeltaPPL={result['delta_ppl_noise_mean']:.6f}",
                flush=True,
            )
        return {
            "digital_nll": digital_nll,
            "digital_perplexity": digital_ppl,
            "token_count": token_count,
            "mapping_sensitivity_field": "sensitivity_score_for_mapping",
            "mapping_sensitivity_unit": self.mapping_sensitivity_field.removesuffix(
                "_mean"
            ),
            "projections": projections,
        }
