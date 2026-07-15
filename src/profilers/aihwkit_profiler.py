"""AIHWKit-native sensitivity profiling for GPT-2 projections.

The profiler intentionally preserves the original Phase-1 procedure:

1. Convert one GPT-2 projection to ``AnalogLinearMapped``.
2. Apply AIHWKit ``LAYER_GAUSSIAN`` clipping by calling
   ``post_update_step()`` on every mapped analog tile.
3. Program one fixed PCM-noise realization.
4. Evaluate the model while only that projection is analog.
5. Restore the original digital projection.

In addition to the original DeltaPPL sensitivity fields, the profiler now
records token-weighted NLL, the deterministic pre-programming analog reference,
and empirical logical-weight programming-noise statistics.  These additional
fields make the output compatible with the unified Phase-1 runner and provide
the calibration records consumed by later phases without changing the original
sensitivity experiment.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D

from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter, WeightClipParameter
from aihwkit.simulator.parameters.enums import (
    BoundManagementType,
    NoiseManagementType,
    WeightClipType,
    WeightNoiseType,
)

Batch = Mapping[str, Tensor]
DatasetLike = Iterable[Batch]
BlockId = Union[int, str]


class AIHWKITSensitivityProfiler:
    """Profile GPT-2 projection sensitivity under AIHWKit analog effects."""

    PROJECTION_PATHS: Tuple[str, ...] = (
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    )

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is None:
            raise ValueError("A YAML configuration dictionary is required.")

        model_cfg = config["model"]
        profiling_cfg = config["profiling"]
        experiment_cfg = config["experiment"]

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(model_cfg["device"])

        self.include_lm_head = bool(profiling_cfg["include_lm_head"])
        self.clip_sigma = float(profiling_cfg["clip_sigma"])
        self.tile_size = int(profiling_cfg["tile_size"])
        self.adc_dac_bits = int(profiling_cfg["adc_dac_bits"])
        self.programming_noise_scale = float(
            profiling_cfg["programming_noise_scale"]
        )
        self.num_seeds = int(profiling_cfg["num_seeds"])
        self.seed_stride = int(profiling_cfg["seed_stride"])
        self.profile_blocks = tuple(
            int(index) for index in profiling_cfg["profile_blocks"]
        )
        self.seed = int(experiment_cfg["seed"])
        self.weight_scaling_omega = float(
            profiling_cfg.get("weight_scaling_omega", 0.0)
        )
        self.weight_scaling_columnwise = bool(
            profiling_cfg.get("weight_scaling_columnwise", False)
        )

        self._validate_configuration()
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()

    def _validate_configuration(self) -> None:
        if self.clip_sigma <= 0.0 or not math.isfinite(self.clip_sigma):
            raise ValueError("clip_sigma must be finite and positive.")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if self.adc_dac_bits < 2:
            raise ValueError("adc_dac_bits must be at least 2.")
        if (
            self.programming_noise_scale <= 0.0
            or not math.isfinite(self.programming_noise_scale)
        ):
            raise ValueError(
                "programming_noise_scale must be finite and positive because "
                "the Phase-4 calibration divides by this value."
            )
        if self.num_seeds <= 0:
            raise ValueError("num_seeds must be positive.")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive.")
        if not self.profile_blocks:
            raise ValueError("profile_blocks must contain at least one block index.")

    @property
    def projection_order(self) -> Tuple[Tuple[str, str], ...]:
        """Return configured GPT-2 projections in deterministic order."""
        num_blocks = len(self.model.transformer.h)
        invalid = [
            index
            for index in self.profile_blocks
            if index < 0 or index >= num_blocks
        ]
        if invalid:
            raise ValueError(
                f"Invalid GPT-2 block indices {invalid}; "
                f"model has {num_blocks} blocks."
            )

        order = [
            (f"block_{block_index}", projection_path)
            for block_index in self.profile_blocks
            for projection_path in self.PROJECTION_PATHS
        ]
        if self.include_lm_head:
            order.append(("head", "lm_head"))
        return tuple(order)

    @staticmethod
    def set_seed(seed: int) -> None:
        """Seed all random sources used by one analog realization."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def make_rpu_config(self) -> InferenceRPUConfig:
        """Create the AIHWKit analog configuration."""
        resolution = 1.0 / float(2**self.adc_dac_bits - 2)
        rpu_config = InferenceRPUConfig()

        rpu_config.clip = WeightClipParameter(
            type=WeightClipType.LAYER_GAUSSIAN,
            sigma=self.clip_sigma,
        )
        rpu_config.mapping = MappingParameter(
            max_input_size=self.tile_size,
            max_output_size=self.tile_size,
            digital_bias=True,
            weight_scaling_omega=self.weight_scaling_omega,
            weight_scaling_columnwise=self.weight_scaling_columnwise,
        )

        rpu_config.forward.is_perfect = False
        rpu_config.forward.inp_res = resolution
        rpu_config.forward.out_res = resolution
        rpu_config.forward.out_bound = 12.0
        rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
        rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX

        rpu_config.forward.inp_noise = 0.0
        rpu_config.forward.out_noise = 0.0
        rpu_config.forward.w_noise = 0.0
        rpu_config.forward.w_noise_type = WeightNoiseType.NONE
        rpu_config.noise_model = PCMLikeNoiseModel(
            prog_noise_scale=self.programming_noise_scale,
            read_noise_scale=0.0,
            drift_scale=0.0,
        )
        rpu_config.drift_compensation = None
        return rpu_config

    def analog_configuration(self) -> Dict[str, Any]:
        """Return metadata saved with the sensitivity profile."""
        resolution = 1.0 / float(2**self.adc_dac_bits - 2)
        return {
            "clip_type": "LAYER_GAUSSIAN",
            "clip_sigma": self.clip_sigma,
            "clipping_scope": "aihwkit_mapped_tile_post_update_step",
            "post_update_step_applied": True,
            "tile_size": self.tile_size,
            "adc_dac_bits": self.adc_dac_bits,
            "input_resolution": resolution,
            "output_resolution": resolution,
            "output_bound": 12.0,
            "bound_management": "ITERATIVE",
            "noise_management": "ABS_MAX",
            "programming_noise_model": "PCMLikeNoiseModel",
            "programming_noise_scale": self.programming_noise_scale,
            "phase2_noise_unit": "pcmlike_prog_noise_scale_equivalent",
            "phase4_noise_calibration": (
                "empirical_logical_weight_std_per_prog_noise_scale"
            ),
            "weight_noise_calibration_metric": "std_unbiased_false",
            "num_seeds": self.num_seeds,
            "seed_stride": self.seed_stride,
            "seed": self.seed,
            "profile_blocks": list(self.profile_blocks),
            "include_lm_head": self.include_lm_head,
            "weight_scaling_omega": self.weight_scaling_omega,
            "weight_scaling_columnwise": self.weight_scaling_columnwise,
            "read_noise_enabled": False,
            "drift_enabled": False,
        }
    
    @staticmethod
    def _parent_and_attribute(root: Any, path: str) -> Tuple[Any, str]:
        """Return the parent module and final attribute for a dotted path."""
        parts = path.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def resolve_projection(
        self,
        block_id: BlockId,
        projection_name: str,
    ) -> Tuple[nn.Module, str, nn.Module, str, str]:
        """Locate a projection and return handles needed for replacement."""
        if str(block_id) == "head":
            return self.model, "lm_head", self.model.lm_head, "head", "lm_head"

        block_index = int(str(block_id).removeprefix("block_"))
        block = self.model.transformer.h[block_index]
        parent, attribute = self._parent_and_attribute(block, projection_name)
        original = getattr(parent, attribute)
        return (
            parent,
            attribute,
            original,
            f"block_{block_index}",
            projection_name,
        )

    @staticmethod
    def _hf_module_path(block_label: str, projection_label: str) -> str:
        """Return the Hugging Face module path consumed by Phase 4."""
        if block_label == "head":
            return "lm_head"
        block_index = int(block_label.removeprefix("block_"))
        return f"transformer.h.{block_index}.{projection_label}"

    def to_linear(self, module: nn.Module) -> nn.Linear:
        """Copy a GPT-2 Conv1D or Linear projection into nn.Linear format."""
        if isinstance(module, Conv1D):
            weight = module.weight.detach().transpose(0, 1).contiguous()
            bias = module.bias.detach()
        elif isinstance(module, nn.Linear):
            weight = module.weight.detach()
            bias = None if module.bias is None else module.bias.detach()
        else:
            raise TypeError(
                f"Expected Conv1D or Linear, got {type(module).__name__}."
            )

        weight = weight.to(device=self.device, dtype=torch.float32)
        bias = (
            None
            if bias is None
            else bias.to(device=self.device, dtype=torch.float32)
        )

        linear = nn.Linear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bias=bias is not None,
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            linear.weight.copy_(weight)
            if bias is not None:
                linear.bias.copy_(bias)
        return linear

    @staticmethod
    def _get_analog_weights_exact(
        analog_linear: AnalogLinearMapped,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Read logical AIHWKit weights with compatibility fallbacks."""
        try:
            weight, bias = analog_linear.get_weights(
                force_exact=True,
                apply_weight_scaling=True,
            )
        except TypeError:
            weight, bias = analog_linear.get_weights()

        weight = weight.detach().cpu().float().contiguous()
        bias = None if bias is None else bias.detach().cpu().float().contiguous()
        return weight, bias

    def compute_quality(self, dataset: DatasetLike) -> Tuple[float, float]:
        """Compute token-weighted causal-LM NLL and perplexity."""
        total_nll = 0.0
        total_tokens = 0
        self.model.eval()

        with torch.inference_mode():
            for batch in dataset:
                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                labels = batch.get("labels")
                labels = (
                    input_ids.clone()
                    if labels is None
                    else labels.to(self.device).clone()
                )
                if attention_mask is not None:
                    labels.masked_fill_(attention_mask == 0, -100)

                token_count = int((labels[..., 1:] != -100).sum().item())
                if token_count == 0:
                    continue

                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                total_nll += float(output.loss.item()) * token_count
                total_tokens += token_count

        if total_tokens == 0:
            raise ValueError("The dataset contains no valid prediction tokens.")

        nll = total_nll / total_tokens
        return float(nll), float(math.exp(nll))

    def compute_perplexity(self, dataset: DatasetLike) -> float:
        """Backward-compatible wrapper returning only perplexity."""
        _, perplexity = self.compute_quality(dataset)
        return perplexity

    def evaluate_realization(
        self,
        dataset: DatasetLike,
        block_id: BlockId,
        projection_name: str,
        seed: int,
        *,
        evaluate_reference: bool,
    ) -> Tuple[float, float, str, str, Dict[str, Any]]:
        """Program and evaluate one fixed analog realization.

        This method deliberately recreates the analog projection for every seed,
        matching the original implementation.  The deterministic reference is
        measured only for the first seed to avoid repeated dataset passes.
        """
        (
            parent,
            attribute,
            original,
            block_label,
            projection_label,
        ) = self.resolve_projection(block_id, projection_name)

        digital_linear = self.to_linear(original)
        analog_linear = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=self.make_rpu_config(),
        )
        analog_linear.eval()

        # Apply the configured 2.5-sigma clipping.
        for analog_tile in analog_linear.analog_tiles():
            analog_tile.post_update_step()

        reference_weight, _ = self._get_analog_weights_exact(analog_linear)
        original_weight = digital_linear.weight.detach().cpu().float().contiguous()
        if reference_weight.shape != original_weight.shape:
            raise ValueError(
                f"{block_label}/{projection_label}: AIHWKit reference shape "
                f"{tuple(reference_weight.shape)} does not match digital shape "
                f"{tuple(original_weight.shape)}."
            )

        reference_weight_delta = reference_weight - original_weight
        reference_metrics = {
            "reference_weight_mean_absolute_change": float(
                reference_weight_delta.abs().mean().item()
            ),
            "reference_weight_max_absolute_change": float(
                reference_weight_delta.abs().max().item()
            ),
            "reference_weight_std_absolute": float(
                reference_weight.std(unbiased=False).item()
            ),
            "reference_weight_rms_absolute": float(
                torch.sqrt(reference_weight.square().mean()).item()
            ),
            "digital_weight_std_absolute": float(
                original_weight.std(unbiased=False).item()
            ),
        }

        setattr(parent, attribute, analog_linear)
        try:
            reference_nll: Optional[float] = None
            reference_ppl: Optional[float] = None
            if evaluate_reference:
                reference_nll, reference_ppl = self.compute_quality(dataset)

            # Preserve the original fixed-realization programming procedure.
            self.set_seed(seed)
            analog_linear.program_analog_weights()

            programmed_weight, _ = self._get_analog_weights_exact(analog_linear)
            if programmed_weight.shape != reference_weight.shape:
                raise ValueError(
                    f"{block_label}/{projection_label}: programmed weight shape "
                    f"{tuple(programmed_weight.shape)} does not match reference "
                    f"shape {tuple(reference_weight.shape)}."
                )

            weight_noise = programmed_weight - reference_weight
            noise_mean_absolute = float(weight_noise.mean().item())
            noise_std_absolute = float(
                weight_noise.std(unbiased=False).item()
            )
            noise_rms_absolute = float(
                torch.sqrt(weight_noise.square().mean()).item()
            )
            noise_max_absolute = float(weight_noise.abs().max().item())
            noise_reference_scale = (
                noise_std_absolute / self.programming_noise_scale
            )

            noisy_nll, noisy_perplexity = self.compute_quality(dataset)
        finally:
            setattr(parent, attribute, original)
            self.model.eval()

        metrics: Dict[str, Any] = {
            "in_features": int(digital_linear.in_features),
            "out_features": int(digital_linear.out_features),
            "analog_tile_count": int(analog_linear.analog_tile_count()),
            "reference_nll": reference_nll,
            "reference_ppl": reference_ppl,
            "noise_mean_absolute": noise_mean_absolute,
            "noise_std_absolute": noise_std_absolute,
            "noise_rms_absolute": noise_rms_absolute,
            "noise_max_absolute": noise_max_absolute,
            "noise_reference_scale": noise_reference_scale,
            **reference_metrics,
        }

        return (
            noisy_nll,
            noisy_perplexity,
            block_label,
            projection_label,
            metrics,
        )

    @staticmethod
    def _as_float_array(values: List[float]) -> np.ndarray:
        """Convert a list of scalar measurements to float64 NumPy."""
        return np.asarray(values, dtype=np.float64)

    def profile_projection(
        self,
        block_id: BlockId,
        projection_name: str,
        dataset: DatasetLike,
        clean_nll: float,
        clean_perplexity: float,
    ) -> Dict[str, Any]:
        """Profile one projection over all configured noise realizations."""
        noisy_nlls: List[float] = []
        noisy_perplexities: List[float] = []
        delta_ppl_total_values: List[float] = []
        delta_nll_total_values: List[float] = []
        delta_nll_programming_values: List[float] = []

        noise_mean_absolute_values: List[float] = []
        noise_std_absolute_values: List[float] = []
        noise_rms_absolute_values: List[float] = []
        noise_max_absolute_values: List[float] = []
        noise_reference_scale_values: List[float] = []
        realization_seeds: List[int] = []

        projection_metadata: Dict[str, Any] = {}
        reference_nll: Optional[float] = None
        reference_ppl: Optional[float] = None
        block_label = str(block_id)
        projection_label = projection_name

        for realization_index in range(self.num_seeds):
            realization_seed = self.seed + realization_index * self.seed_stride
            (
                noisy_nll,
                noisy_perplexity,
                block_label,
                projection_label,
                metrics,
            ) = self.evaluate_realization(
                dataset=dataset,
                block_id=block_id,
                projection_name=projection_name,
                seed=realization_seed,
                evaluate_reference=realization_index == 0,
            )

            if realization_index == 0:
                if metrics["reference_nll"] is None or metrics["reference_ppl"] is None:
                    raise RuntimeError(
                        "The first realization did not produce reference quality."
                    )
                reference_nll = float(metrics["reference_nll"])
                reference_ppl = float(metrics["reference_ppl"])

            if reference_nll is None or reference_ppl is None:
                raise RuntimeError("Reference quality was not initialized.")

            delta_ppl_total = noisy_perplexity - clean_perplexity
            delta_nll_total = noisy_nll - clean_nll
            delta_nll_programming = noisy_nll - reference_nll

            noisy_nlls.append(noisy_nll)
            noisy_perplexities.append(noisy_perplexity)
            delta_ppl_total_values.append(delta_ppl_total)
            delta_nll_total_values.append(delta_nll_total)
            delta_nll_programming_values.append(delta_nll_programming)

            noise_mean_absolute_values.append(metrics["noise_mean_absolute"])
            noise_std_absolute_values.append(metrics["noise_std_absolute"])
            noise_rms_absolute_values.append(metrics["noise_rms_absolute"])
            noise_max_absolute_values.append(metrics["noise_max_absolute"])
            noise_reference_scale_values.append(metrics["noise_reference_scale"])
            realization_seeds.append(realization_seed)

            print(
                f"{block_label}/{projection_label} "
                f"seed={realization_seed}: "
                f"PPL={noisy_perplexity:.6f}, "
                f"DeltaPPL={delta_ppl_total:.6f}, "
                f"DeltaNLL_total={delta_nll_total:.8f}, "
                f"DeltaNLL_programming={delta_nll_programming:.8f}, "
                f"noise_std_abs={metrics['noise_std_absolute']:.8e}, "
                f"noise_rms_abs={metrics['noise_rms_absolute']:.8e}",
                flush=True,
            )

            if not projection_metadata:
                projection_metadata = {
                    "in_features": metrics["in_features"],
                    "out_features": metrics["out_features"],
                    "analog_tile_count": metrics["analog_tile_count"],
                    "digital_weight_std_absolute": metrics[
                        "digital_weight_std_absolute"
                    ],
                    "reference_weight_std_absolute": metrics[
                        "reference_weight_std_absolute"
                    ],
                    "reference_weight_rms_absolute": metrics[
                        "reference_weight_rms_absolute"
                    ],
                    "reference_weight_mean_absolute_change": metrics[
                        "reference_weight_mean_absolute_change"
                    ],
                    "reference_weight_max_absolute_change": metrics[
                        "reference_weight_max_absolute_change"
                    ],
                }

        noisy_nll_array = self._as_float_array(noisy_nlls)
        noisy_ppl_array = self._as_float_array(noisy_perplexities)
        delta_ppl_array = self._as_float_array(delta_ppl_total_values)
        delta_nll_total_array = self._as_float_array(delta_nll_total_values)
        delta_nll_programming_array = self._as_float_array(
            delta_nll_programming_values
        )
        noise_mean_array = self._as_float_array(noise_mean_absolute_values)
        noise_std_array = self._as_float_array(noise_std_absolute_values)
        noise_rms_array = self._as_float_array(noise_rms_absolute_values)
        noise_max_array = self._as_float_array(noise_max_absolute_values)
        noise_reference_scale_array = self._as_float_array(
            noise_reference_scale_values
        )

        measured_noise_std_absolute = float(noise_std_array.mean())
        measured_noise_rms_absolute = float(noise_rms_array.mean())
        mean_noise_reference_scale = float(noise_reference_scale_array.mean())

        projection_id = f"{block_label}/{projection_label}"
        hf_module_path = self._hf_module_path(block_label, projection_label)

        return {
            "block_id": block_label,
            "proj_name": projection_label,
            "projection_label": projection_id,
            "projection_id": projection_id,
            "hf_module_path": hf_module_path,
            **projection_metadata,
            "nll_clean": float(clean_nll),
            "ppl_clean": float(clean_perplexity),
            "nll_reference": float(reference_nll),
            "ppl_reference": float(reference_ppl),
            "delta_nll_preprocessing": float(reference_nll - clean_nll),
            "delta_ppl_preprocessing": float(
                reference_ppl - clean_perplexity
            ),
            "nll_noisy_mean": float(noisy_nll_array.mean()),
            "nll_noisy_std": float(noisy_nll_array.std(ddof=0)),
            "ppl_noisy_mean": float(noisy_ppl_array.mean()),
            "ppl_noisy_std": float(noisy_ppl_array.std(ddof=0)),
            # Original Phase-1 sensitivity fields remain unchanged.
            "sensitivity_mean": float(delta_ppl_array.mean()),
            "sensitivity_std": float(delta_ppl_array.std(ddof=0)),
            "sensitivity_per_seed": delta_ppl_array.tolist(),
            # Explicit aliases used by the unified runner and analyzer.
            "delta_ppl_total_mean": float(delta_ppl_array.mean()),
            "delta_ppl_total_std": float(delta_ppl_array.std(ddof=0)),
            "delta_ppl_total_per_seed": delta_ppl_array.tolist(),
            "delta_nll_total_mean": float(delta_nll_total_array.mean()),
            "delta_nll_total_std": float(
                delta_nll_total_array.std(ddof=0)
            ),
            "delta_nll_total_per_seed": delta_nll_total_array.tolist(),
            "delta_nll_programming_mean": float(
                delta_nll_programming_array.mean()
            ),
            "delta_nll_programming_std": float(
                delta_nll_programming_array.std(ddof=0)
            ),
            "delta_nll_programming_per_seed": (
                delta_nll_programming_array.tolist()
            ),
            # Empirical AIHWKit logical-weight programming analytics.
            "noise_mean_absolute_mean": float(noise_mean_array.mean()),
            "noise_mean_absolute_std": float(noise_mean_array.std(ddof=0)),
            "noise_mean_absolute_per_seed": noise_mean_array.tolist(),
            "noise_std_absolute_mean": measured_noise_std_absolute,
            "noise_std_absolute_std": float(noise_std_array.std(ddof=0)),
            "noise_std_absolute_per_seed": noise_std_array.tolist(),
            "noise_rms_absolute_mean": measured_noise_rms_absolute,
            "noise_rms_absolute_std": float(noise_rms_array.std(ddof=0)),
            "noise_rms_absolute_per_seed": noise_rms_array.tolist(),
            "noise_max_absolute_mean": float(noise_max_array.mean()),
            "noise_max_absolute_std": float(noise_max_array.std(ddof=0)),
            "noise_max_absolute_per_seed": noise_max_array.tolist(),
            "noise_reference_scale_mean": mean_noise_reference_scale,
            "noise_reference_scale_std": float(
                noise_reference_scale_array.std(ddof=0)
            ),
            "noise_reference_scale_per_seed": (
                noise_reference_scale_array.tolist()
            ),
            "realization_seeds": realization_seeds,
            "noise_calibration": {
                "projection_id": projection_id,
                "hf_module_path": hf_module_path,
                "reference_sigma_normalized": float(
                    self.programming_noise_scale
                ),
                "measured_noise_std_absolute": measured_noise_std_absolute,
                "measured_noise_rms_absolute": measured_noise_rms_absolute,
                "noise_reference_scale": mean_noise_reference_scale,
                "calibration_source": (
                    "phase1_aihwkit_post_update_programming_readback"
                ),
                "num_calibration_seeds": self.num_seeds,
            },
        }

    @staticmethod
    def add_sensitivity_ranks(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add descending total-DeltaPPL ranks for Phase 3 mapping."""
        sorted_indices = sorted(
            range(len(results)),
            key=lambda index: results[index]["sensitivity_mean"],
            reverse=True,
        )
        for rank, index in enumerate(sorted_indices, start=1):
            results[index]["sensitivity_rank"] = rank
            results[index]["sensitivity_score_for_mapping"] = results[index][
                "sensitivity_mean"
            ]
            results[index]["sensitivity_score_unit"] = "delta_ppl_total"
        return results

    def profile_all(self, dataset: DatasetLike) -> List[Dict[str, Any]]:
        """Profile all configured projections and return mapper-ready metrics."""
        batches = dataset if isinstance(dataset, (list, tuple)) else tuple(dataset)
        clean_nll, clean_perplexity = self.compute_quality(batches)

        results: List[Dict[str, Any]] = []
        order = self.projection_order

        print(
            f"Digital FP32 NLL: {clean_nll:.8f} | "
            f"PPL: {clean_perplexity:.6f}",
            flush=True,
        )

        for index, (block_id, projection_name) in enumerate(order, start=1):
            print(
                f"\nProfiling {index}/{len(order)}: "
                f"{block_id}/{projection_name}",
                flush=True,
            )

            result = self.profile_projection(
                block_id=block_id,
                projection_name=projection_name,
                dataset=batches,
                clean_nll=clean_nll,
                clean_perplexity=clean_perplexity,
            )
            results.append(result)

            print(
                f"Completed {result['projection_label']}: "
                f"DeltaPPL={result['sensitivity_mean']:.6f} +/- "
                f"{result['sensitivity_std']:.6f}; "
                f"DeltaNLL_programming="
                f"{result['delta_nll_programming_mean']:.8f} +/- "
                f"{result['delta_nll_programming_std']:.8f}",
                flush=True,
            )

        return self.add_sensitivity_ranks(results)
