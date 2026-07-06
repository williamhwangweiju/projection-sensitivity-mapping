"""Projection-level AIHWKit sensitivity profiler for GPT-2.

This implementation follows the experiment described by Lammie (2026):

* clip each complete projection at 2.5 standard deviations before tiling;
* map the clipped projection to 512 x 512 analog tiles;
* apply approximately 8-bit DAC/ADC quantization;
* inject one fixed additive Gaussian programming-noise realization with
  standard deviation 0.023 in the programmed analog-weight domain;
* keep read noise, drift, and other forward-pass noise disabled.

Only one projection is converted at a time. Every programming realization is
held fixed while perplexity is evaluated over the complete dataset.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D

from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter, WeightClipParameter
from aihwkit.simulator.parameters.enums import WeightClipType, WeightNoiseType

logger = logging.getLogger(__name__)

Batch = Mapping[str, Tensor]
DatasetLike = Iterable[Batch]
BlockId = Union[int, str]


class AdditiveGaussianProgrammingNoiseModel(BaseNoiseModel):
    """Apply fixed additive Gaussian noise once during programming.

    The analog weights are explicitly remapped to ``weight_scaling_omega``
    before programming. With the recommended omega of 1.0, ``noise_std=0.023``
    therefore means a standard deviation of 0.023 relative to the programmed
    analog full-scale weight magnitude.

    This differs from ``PCMLikeNoiseModel(prog_noise_scale=0.023)``. In that
    model, ``prog_noise_scale`` multiplies an existing conductance-dependent
    PCM polynomial and is not itself the final normalized weight-noise
    standard deviation.
    """

    def __init__(self, noise_std: float) -> None:
        super().__init__()
        noise_std = float(noise_std)
        if not math.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError(
                "Programming-noise standard deviation must be finite and "
                f"nonnegative, received {noise_std}."
            )
        self.noise_std = noise_std

    @torch.no_grad()
    def apply_programming_noise(
        self,
        weights: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Return one programmed-weight realization and no drift parameters."""
        if self.noise_std == 0.0:
            return weights.detach().clone(), []

        noisy_weights = (
            weights.detach()
            + self.noise_std * torch.randn_like(weights)
        )
        return noisy_weights, []

    @torch.no_grad()
    def apply_programming_noise_to_conductance(
        self,
        g_target: Tensor,
    ) -> Tensor:
        """Required base hook; direct weight-domain programming is used."""
        return g_target

    @torch.no_grad()
    def apply_drift_noise_to_conductance(
        self,
        g_prog: Tensor,
        drift_noise_param: Optional[Tensor],
        t_inference: float,
    ) -> Tensor:
        """Return unchanged conductance because drift/read noise are disabled."""
        del drift_noise_param, t_inference
        return g_prog


class AIHWKITSensitivityProfiler:
    """Profile GPT-2 projections using a controlled AIHWKit configuration."""

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
        **_: Any,
    ) -> None:
        config = config or {}
        model_cfg = config.get("model", {})
        profiling_cfg = config.get("profiling", {})
        experiment_cfg = config.get("experiment", {})

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(model_cfg.get("device", "cpu"))

        self.clip_sigma = float(profiling_cfg.get("clip_sigma", 2.5))
        self.tile_size = int(profiling_cfg.get("tile_size", 512))
        self.adc_dac_bits = int(profiling_cfg.get("adc_dac_bits", 8))

        # Keep the old YAML key as an accepted alias, but name the quantity
        # correctly: it is the final normalized programming-noise std.
        self.programming_noise_std = float(
            profiling_cfg.get(
                "programming_noise_std",
                profiling_cfg.get("programming_noise_scale", 0.023),
            )
        )
        self.programming_noise_scale = self.programming_noise_std

        self.weight_scaling_omega = float(
            profiling_cfg.get("weight_scaling_omega", 1.0)
        )
        self.weight_scaling_columnwise = bool(
            profiling_cfg.get("weight_scaling_columnwise", True)
        )

        self.num_seeds = int(profiling_cfg.get("num_seeds", 10))
        self.seed_stride = int(profiling_cfg.get("seed_stride", 1))
        self.include_lm_head = bool(
            profiling_cfg.get("include_lm_head", True)
        )
        self.seed = int(experiment_cfg.get("seed", 42))

        requested_blocks = profiling_cfg.get("profile_blocks")
        self.profile_blocks: Optional[Tuple[int, ...]] = (
            None
            if requested_blocks is None
            else tuple(int(index) for index in requested_blocks)
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
        if self.programming_noise_std < 0.0 or not math.isfinite(
            self.programming_noise_std
        ):
            raise ValueError(
                "programming_noise_std must be finite and nonnegative."
            )
        if self.weight_scaling_omega <= 0.0 or not math.isfinite(
            self.weight_scaling_omega
        ):
            raise ValueError(
                "weight_scaling_omega must be finite and positive so that "
                "programming_noise_std has a defined normalized scale."
            )
        if self.num_seeds <= 0:
            raise ValueError("num_seeds must be positive.")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive.")

    @property
    def projection_order(self) -> Tuple[Tuple[str, str], ...]:
        """Return block projections followed by the language-model head."""
        num_blocks = len(self.model.transformer.h)

        if self.profile_blocks is None:
            block_indices = list(range(num_blocks))
        else:
            block_indices = [
                index if index >= 0 else num_blocks + index
                for index in self.profile_blocks
            ]

        invalid = [
            index
            for index in block_indices
            if index < 0 or index >= num_blocks
        ]
        if invalid:
            raise ValueError(
                f"Invalid GPT-2 block indices {invalid}; model has "
                f"{num_blocks} blocks."
            )

        order = [
            (f"block_{block_index}", projection_path)
            for block_index in block_indices
            for projection_path in self.PROJECTION_PATHS
        ]

        if self.include_lm_head:
            order.append(("head", "lm_head"))

        return tuple(order)

    @staticmethod
    def set_seed(seed: int) -> None:
        """Seed Python, NumPy, and PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def make_rpu_config(self) -> InferenceRPUConfig:
        """Create the controlled analog-forward configuration.

        Layer clipping is performed explicitly before tiling, so AIHWKit's
        tile-local post-update clipping is disabled here. All forward noise
        sources except DAC/ADC quantization are explicitly disabled.
        """
        resolution = 1.0 / float(2**self.adc_dac_bits - 2)

        rpu_config = InferenceRPUConfig()
        rpu_config.clip = WeightClipParameter(
            type=WeightClipType.NONE,
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

        # Explicitly remove default or accidental nonidealities that are not
        # part of the t=0 programming-noise experiment.
        rpu_config.forward.inp_noise = 0.0
        rpu_config.forward.out_noise = 0.0
        rpu_config.forward.out_noise_std = 0.0
        rpu_config.forward.w_noise = 0.0
        rpu_config.forward.w_noise_type = WeightNoiseType.NONE
        rpu_config.forward.ir_drop = 0.0
        rpu_config.forward.out_nonlinearity = 0.0
        rpu_config.forward.out_nonlinearity_std = 0.0
        rpu_config.forward.v_offset_std = 0.0
        rpu_config.forward.r_series = 0.0
        rpu_config.forward.w_read_asymmetry_dtod = 0.0

        rpu_config.noise_model = AdditiveGaussianProgrammingNoiseModel(
            noise_std=self.programming_noise_std,
        )
        rpu_config.drift_compensation = None

        return rpu_config

    def analog_configuration(self) -> Dict[str, Any]:
        """Return metadata that matches the configuration actually simulated."""
        rpu_config = self.make_rpu_config()
        return {
            "clip_type": "global_layer_gaussian_before_tiling",
            "clip_sigma": self.clip_sigma,
            "tile_size": self.tile_size,
            "adc_dac_bits": self.adc_dac_bits,
            "input_resolution": float(rpu_config.forward.inp_res),
            "output_resolution": float(rpu_config.forward.out_res),
            "programming_noise_model": (
                "fixed_additive_gaussian_weight_domain"
            ),
            "programming_noise_std": self.programming_noise_std,
            "programming_noise_reference": (
                "programmed_analog_weight_full_scale"
            ),
            "weight_scaling_omega": self.weight_scaling_omega,
            "weight_scaling_columnwise": self.weight_scaling_columnwise,
            "forward_input_noise": float(rpu_config.forward.inp_noise),
            "forward_output_noise": float(rpu_config.forward.out_noise),
            "forward_weight_noise": float(rpu_config.forward.w_noise),
            "read_noise_enabled": False,
            "drift_enabled": False,
            "digital_bias": True,
        }

    @staticmethod
    def _parent_and_attribute(root: Any, path: str) -> Tuple[Any, str]:
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
        """Resolve one GPT-2 projection and its parent module."""
        if str(block_id) == "head":
            return (
                self.model,
                "lm_head",
                self.model.lm_head,
                "head",
                "lm_head",
            )

        block_index = int(str(block_id).removeprefix("block_"))
        block = self.model.transformer.h[block_index]
        parent, attribute = self._parent_and_attribute(
            block,
            projection_name,
        )

        return (
            parent,
            attribute,
            getattr(parent, attribute),
            f"block_{block_index}",
            projection_name,
        )

    def to_linear(self, module: nn.Module) -> nn.Linear:
        """Copy a GPT-2 Conv1D or Linear projection into ``nn.Linear``."""
        if isinstance(module, Conv1D):
            weight = module.weight.detach().transpose(0, 1).contiguous()
            bias = module.bias.detach()
        elif isinstance(module, nn.Linear):
            weight = module.weight.detach()
            bias = None if module.bias is None else module.bias.detach()
        else:
            raise TypeError(
                "Expected Conv1D or Linear, got "
                f"{type(module).__name__}."
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

    def clip_projection_before_tiling(
        self,
        linear: nn.Linear,
    ) -> Dict[str, float]:
        """Apply one global Gaussian clipping threshold to a projection."""
        with torch.no_grad():
            weight = linear.weight
            weight_std = float(weight.std(unbiased=False).item())
            threshold = self.clip_sigma * weight_std

            if threshold <= 0.0:
                clipped_fraction = 0.0
            else:
                clipped_mask = weight.abs() > threshold
                clipped_fraction = float(
                    clipped_mask.to(dtype=torch.float32).mean().item()
                )
                weight.clamp_(min=-threshold, max=threshold)

            programmed_abs_max = float(weight.abs().max().item())
            programmed_peak_to_peak = float(
                (weight.max() - weight.min()).item()
            )

        return {
            "weight_std_before_clipping": weight_std,
            "clip_threshold": threshold,
            "clipped_weight_fraction": clipped_fraction,
            "programmed_abs_max_before_mapping": programmed_abs_max,
            "programmed_peak_to_peak_before_mapping": (
                programmed_peak_to_peak
            ),
        }

    @staticmethod
    def programming_error_diagnostics(
        analog_linear: AnalogLinearMapped,
    ) -> Dict[str, float]:
        """Measure the actual fixed programming error stored in the tiles."""
        squared_error_sum = 0.0
        element_count = 0
        normalized_rms_values: List[float] = []
        reference_abs_max_values: List[float] = []

        for tile in analog_linear.analog_tiles():
            reference = getattr(tile, "reference_combined_weights", None)
            programmed = getattr(tile, "programmed_weights", None)

            if reference is None or programmed is None:
                continue

            error = programmed.detach() - reference.detach()
            squared_error_sum += float(error.square().sum().item())
            element_count += int(error.numel())

            reference_abs_max = float(reference.abs().max().item())
            reference_abs_max_values.append(reference_abs_max)
            error_rms = float(error.square().mean().sqrt().item())
            if reference_abs_max > 0.0:
                normalized_rms_values.append(
                    error_rms / reference_abs_max
                )

        if element_count == 0:
            raise RuntimeError(
                "AIHWKit did not expose programmed/reference tile weights; "
                "cannot verify programming noise."
            )

        global_rms = math.sqrt(squared_error_sum / element_count)
        return {
            "analog_tile_count": float(
                analog_linear.analog_tile_count()
            ),
            "programming_error_rms": global_rms,
            "programming_error_normalized_rms_mean": float(
                np.mean(normalized_rms_values)
            ),
            "programming_error_normalized_rms_min": float(
                np.min(normalized_rms_values)
            ),
            "programming_error_normalized_rms_max": float(
                np.max(normalized_rms_values)
            ),
            "programmed_tile_abs_max_mean": float(
                np.mean(reference_abs_max_values)
            ),
        }

    def compute_perplexity(self, dataset: DatasetLike) -> float:
        """Compute token-weighted causal-language-model perplexity."""
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

                token_count = int(
                    (labels[..., 1:] != -100).sum().item()
                )
                if token_count == 0:
                    continue

                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                ).loss

                total_nll += float(loss.item()) * token_count
                total_tokens += token_count

        if total_tokens == 0:
            raise ValueError(
                "The dataset contains no valid prediction tokens."
            )

        return float(math.exp(total_nll / total_tokens))

    def evaluate_realization(
        self,
        dataset: DatasetLike,
        block_id: BlockId,
        projection_name: str,
        seed: int,
    ) -> Tuple[float, str, str, Dict[str, float]]:
        """Program and evaluate one fixed additive-noise realization."""
        (
            parent,
            attribute,
            original,
            canonical_block,
            canonical_projection,
        ) = self.resolve_projection(block_id, projection_name)

        # The experiment order in the paper is clipping -> tiling ->
        # quantized analog execution -> fixed programming noise.
        digital_linear = self.to_linear(original)
        clipping_diagnostics = self.clip_projection_before_tiling(
            digital_linear
        )

        self.set_seed(seed)
        analog_linear = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=self.make_rpu_config(),
        )
        analog_linear.eval()

        # Programming noise is sampled exactly once. No forward-pass random
        # noise remains enabled, so the weights stay fixed for all batches.
        self.set_seed(seed)
        analog_linear.program_analog_weights()

        programming_diagnostics = self.programming_error_diagnostics(
            analog_linear
        )
        diagnostics = {
            **clipping_diagnostics,
            **programming_diagnostics,
        }

        logger.info(
            "%s/%s: clip_sigma=%.3f, clipped=%.5f%%, "
            "tile=%dx%d, io_bits=%d, prog_noise_std=%.6f, "
            "measured_norm_rms=%.6f, tiles=%d",
            canonical_block,
            canonical_projection,
            self.clip_sigma,
            100.0 * diagnostics["clipped_weight_fraction"],
            self.tile_size,
            self.tile_size,
            self.adc_dac_bits,
            self.programming_noise_std,
            diagnostics["programming_error_normalized_rms_mean"],
            int(diagnostics["analog_tile_count"]),
        )

        setattr(parent, attribute, analog_linear)
        try:
            noisy_perplexity = self.compute_perplexity(dataset)
        finally:
            setattr(parent, attribute, original)
            self.model.eval()

        return (
            noisy_perplexity,
            canonical_block,
            canonical_projection,
            diagnostics,
        )

    def profile_projection(
        self,
        block_id: BlockId,
        projection_name: str,
        dataset: DatasetLike,
        clean_perplexity: float,
    ) -> Dict[str, Any]:
        """Profile one projection over all configured realizations."""
        noisy_perplexities: List[float] = []
        realization_seeds: List[int] = []
        programming_normalized_rms: List[float] = []
        programming_rms: List[float] = []
        canonical_block = str(block_id)
        canonical_projection = projection_name
        constant_diagnostics: Optional[Dict[str, float]] = None

        for realization_index in range(self.num_seeds):
            realization_seed = (
                self.seed
                + realization_index * self.seed_stride
            )
            (
                noisy_perplexity,
                canonical_block,
                canonical_projection,
                diagnostics,
            ) = self.evaluate_realization(
                dataset=dataset,
                block_id=block_id,
                projection_name=projection_name,
                seed=realization_seed,
            )

            if constant_diagnostics is None:
                constant_diagnostics = {
                    key: diagnostics[key]
                    for key in (
                        "weight_std_before_clipping",
                        "clip_threshold",
                        "clipped_weight_fraction",
                        "programmed_abs_max_before_mapping",
                        "programmed_peak_to_peak_before_mapping",
                        "analog_tile_count",
                        "programmed_tile_abs_max_mean",
                    )
                }

            noisy_perplexities.append(noisy_perplexity)
            realization_seeds.append(realization_seed)
            programming_rms.append(
                diagnostics["programming_error_rms"]
            )
            programming_normalized_rms.append(
                diagnostics[
                    "programming_error_normalized_rms_mean"
                ]
            )

            logger.info(
                "%s/%s realization %d/%d: PPL=%.6f, Delta=%.6f",
                canonical_block,
                canonical_projection,
                realization_index + 1,
                self.num_seeds,
                noisy_perplexity,
                noisy_perplexity - clean_perplexity,
            )

        noisy_array = np.asarray(
            noisy_perplexities,
            dtype=np.float64,
        )
        sensitivity_array = noisy_array - float(clean_perplexity)
        normalized_rms_array = np.asarray(
            programming_normalized_rms,
            dtype=np.float64,
        )
        programming_rms_array = np.asarray(
            programming_rms,
            dtype=np.float64,
        )

        return {
            "block_id": canonical_block,
            "proj_name": canonical_projection,
            "projection_label": (
                f"{canonical_block}/{canonical_projection}"
            ),
            "ppl_clean": float(clean_perplexity),
            "ppl_noisy_mean": float(noisy_array.mean()),
            "sensitivity_mean": float(sensitivity_array.mean()),
            "sensitivity_std": float(
                sensitivity_array.std(ddof=0)
            ),
            "sensitivity_per_seed": sensitivity_array.tolist(),
            "realization_seeds": realization_seeds,
            "programming_error_rms_mean": float(
                programming_rms_array.mean()
            ),
            "programming_error_normalized_rms_mean": float(
                normalized_rms_array.mean()
            ),
            "programming_error_normalized_rms_std": float(
                normalized_rms_array.std(ddof=0)
            ),
            "programming_error_normalized_rms_per_seed": (
                normalized_rms_array.tolist()
            ),
            **(constant_diagnostics or {}),
        }

    def profile_all(
        self,
        dataset: DatasetLike,
        projection_order: Optional[
            Sequence[Tuple[str, str]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Profile every requested projection in deterministic order."""
        batches = (
            dataset
            if isinstance(dataset, (list, tuple))
            else tuple(dataset)
        )
        order = tuple(projection_order or self.projection_order)
        clean_perplexity = self.compute_perplexity(batches)

        logger.info(
            "Digital FP32 baseline PPL: %.6f",
            clean_perplexity,
        )
        logger.info(
            "Actual analog configuration: %s",
            self.analog_configuration(),
        )

        results: List[Dict[str, Any]] = []

        for index, (block_id, projection_name) in enumerate(
            order,
            start=1,
        ):
            logger.info(
                "Profiling projection %d/%d: %s/%s",
                index,
                len(order),
                block_id,
                projection_name,
            )
            results.append(
                self.profile_projection(
                    block_id=block_id,
                    projection_name=projection_name,
                    dataset=batches,
                    clean_perplexity=clean_perplexity,
                )
            )

        return results
