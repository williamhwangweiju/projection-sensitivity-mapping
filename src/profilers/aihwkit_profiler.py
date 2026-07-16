"""Projection-level GPT-2 sensitivity profiler with optional LM-head profiling."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor

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
    canonical_weight_bias,
    iter_gpt2_projections,
    linear_from_canonical,
)

Batch = Mapping[str, Tensor]


class AIHWKITSensitivityProfiler:
    """Evaluate one analog projection at a time while all others remain digital."""

    def __init__(self, model: Any, config: Mapping[str, Any]) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(str(config["model"]["device"]))
        self.settings = ManualAnalogSettings.from_config(config)
        self.settings.validate()
        profile = config["profiling"]
        self.num_seeds = int(profile["num_seeds"])
        self.seed_stride = int(profile.get("seed_stride", 1))
        self.profile_blocks = tuple(int(v) for v in profile.get("profile_blocks", []))
        self.include_lm_head = bool(profile.get("include_lm_head", True))
        self.antithetic = bool(profile.get("antithetic", True))
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
                "include_lm_head": self.include_lm_head,
            }
        )
        return result

    def _handles(self) -> list[Any]:
        handles = list(
            iter_gpt2_projections(
                self.model, include_lm_head=self.include_lm_head
            )
        )
        if self.profile_blocks:
            handles = [
                handle
                for handle in handles
                if handle.block_index is None or handle.block_index in self.profile_blocks
            ]
        return handles

    def profile_all(self, batches: list[Batch]) -> list[dict[str, Any]]:
        clean_nll, clean_ppl, token_count = evaluate_nll_ppl(
            self.model, batches, self.device
        )
        handles = self._handles()
        results: list[dict[str, Any]] = []
        for index, handle in enumerate(handles, start=1):
            print(f"Profiling {index}/{len(handles)}: {handle.projection_id}", flush=True)
            results.append(
                self._profile_one(
                    handle, batches, clean_nll, clean_ppl, token_count
                )
            )
        return results

    def _profile_one(
        self,
        handle: Any,
        batches: list[Batch],
        clean_nll: float,
        clean_ppl: float,
        token_count: int,
    ) -> dict[str, Any]:
        try:
            from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
        except ImportError as exc:
            raise RuntimeError("AIHWKit 1.1.0 is required.") from exc

        original_module = handle.module
        weight, bias = canonical_weight_bias(original_module)
        prepared = prepare_projection_weight(weight, self.settings)
        preprocessing = prepared.preprocessing.to_dict()
        digital_linear = linear_from_canonical(
            prepared.clipped_weight, bias, self.device
        )
        analog = AnalogLinearMapped.from_digital(
            digital_linear, rpu_config=make_rpu_config(self.settings)
        ).to(self.device)
        analog.eval()
        set_analog_weights_exact(
            analog,
            prepared.clipped_weight,
            bias,
            verify=True,
        )
        readback, _ = get_analog_weights_exact(analog)
        if float((readback - prepared.clipped_weight).abs().max().item()) > 3e-6:
            raise RuntimeError(f"{handle.projection_id}: clean conversion mismatch.")
        setattr(handle.parent, handle.attribute, analog)
        try:
            reference_nll, reference_ppl, _ = evaluate_nll_ppl(
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
                    prepared.clipped_weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                signs = (1.0, -1.0) if self.antithetic else (1.0,)
                sign_metrics: list[dict[str, float]] = []
                for sign in signs:
                    noisy_weight, noise = materialize_manual_noise(
                        prepared.clipped_weight,
                        self.settings.reference_noise_std,
                        float(preprocessing["programmed_range"]),
                        z * sign,
                    )
                    set_analog_weights_exact(
                        analog, noisy_weight, bias, verify=False
                    )
                    nll, ppl, _ = evaluate_nll_ppl(
                        self.model, batches, self.device
                    )
                    sign_metrics.append(
                        {
                            "nll": nll,
                            "ppl": ppl,
                            "noise_std_absolute": float(
                                noise.std(unbiased=False).item()
                            ),
                        }
                    )
                mean_nll = sum(row["nll"] for row in sign_metrics) / len(sign_metrics)
                mean_ppl_from_nll = math.exp(mean_nll)
                realization_rows.append(
                    {
                        "realization": realization,
                        "realization_seed": realization_seed,
                        "projection_noise_seed": seed,
                        "antithetic_count": len(signs),
                        "nll": mean_nll,
                        "ppl": mean_ppl_from_nll,
                        "delta_nll_total": mean_nll - clean_nll,
                        "delta_ppl_total": mean_ppl_from_nll - clean_ppl,
                        "delta_nll_noise": mean_nll - reference_nll,
                        "delta_ppl_noise": mean_ppl_from_nll - reference_ppl,
                        "noise_std_absolute": sum(
                            row["noise_std_absolute"] for row in sign_metrics
                        ) / len(sign_metrics),
                    }
                )
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
            sensitivity = float(fields["delta_nll_noise"]["mean"])
            result: dict[str, Any] = {
                "projection_id": handle.projection_id,
                "module_path": handle.module_path,
                "role": handle.role,
                "block_index": handle.block_index,
                "in_features": handle.in_features,
                "out_features": handle.out_features,
                "parameter_count": handle.parameter_count,
                "macs_per_token": handle.macs_per_token,
                "tied_to_embedding": handle.tied_to_embedding,
                "nll_clean": clean_nll,
                "ppl_clean": clean_ppl,
                "nll_analog_reference": reference_nll,
                "ppl_analog_reference": reference_ppl,
                "delta_nll_analog_reference": reference_nll - clean_nll,
                "delta_ppl_analog_reference": reference_ppl - clean_ppl,
                "preprocessing": preprocessing,
                "realizations": realization_rows,
                "sensitivity_score_for_mapping": sensitivity,
                "sensitivity_score_unit": "delta_nll_noise",
                "sensitivity_per_parameter": sensitivity / max(handle.parameter_count, 1),
                "sensitivity_per_mac": sensitivity / max(handle.macs_per_token, 1),
                "clip_value": float(preprocessing["clip_threshold"]),
                "programmed_range": float(preprocessing["programmed_range"]),
                "clipped_fraction": float(preprocessing["fraction_clipped"]),
                "reference_noise_std": self.settings.reference_noise_std,
                "realization_count": self.num_seeds,
                "predicted_tokens": token_count,
            }
            for field_name, stats in fields.items():
                for statistic, value in stats.items():
                    result[f"{field_name}_{statistic}"] = float(value)
            return result
        finally:
            set_analog_weights_exact(
                analog, prepared.clipped_weight, bias, verify=False
            )
            setattr(handle.parent, handle.attribute, original_module)
