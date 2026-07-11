"""AIHWKit-native sensitivity profiling for GPT-2 projections.

The profiler converts one configured projection to an AIHWKit analog module,
measures perplexity degradation, then restores the original digital projection.
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
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
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
        self.programming_noise_scale = float(profiling_cfg["programming_noise_scale"])
        self.num_seeds = int(profiling_cfg["num_seeds"])
        self.seed_stride = int(profiling_cfg["seed_stride"])
        self.profile_blocks = tuple(int(index) for index in profiling_cfg["profile_blocks"])
        self.seed = int(experiment_cfg["seed"])
        self.weight_scaling_omega = float(
            profiling_cfg.get("weight_scaling_omega", 1.0)
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
        if self.programming_noise_scale < 0.0 or not math.isfinite(
            self.programming_noise_scale
        ):
            raise ValueError("programming_noise_scale must be finite and nonnegative.")
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
                f"Invalid GPT-2 block indices {invalid}; model has {num_blocks} blocks."
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
        """Create the AIHWKit analog configuration for the paper setup."""
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
            "tile_size": self.tile_size,
            "adc_dac_bits": self.adc_dac_bits,
            "input_resolution": resolution,
            "output_resolution": resolution,
            "programming_noise_model": "PCMLikeNoiseModel",
            "programming_noise_scale": self.programming_noise_scale,
            "num_seeds": self.num_seeds,
            "seed_stride": self.seed_stride,
            "seed": self.seed,
            "profile_blocks": list(self.profile_blocks),
            "include_lm_head": self.include_lm_head,
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
        return parent, attribute, original, f"block_{block_index}", projection_name

    def to_linear(self, module: nn.Module) -> nn.Linear:
        """Copy a GPT-2 Conv1D or Linear projection into nn.Linear format."""
        if isinstance(module, Conv1D):
            weight = module.weight.detach().transpose(0, 1).contiguous()
            bias = module.bias.detach()
        elif isinstance(module, nn.Linear):
            weight = module.weight.detach()
            bias = None if module.bias is None else module.bias.detach()
        else:
            raise TypeError(f"Expected Conv1D or Linear, got {type(module).__name__}.")

        weight = weight.to(device=self.device, dtype=torch.float32)
        bias = None if bias is None else bias.to(device=self.device, dtype=torch.float32)

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

    def compute_perplexity(self, dataset: DatasetLike) -> float:
        """Compute token-weighted causal language-model perplexity."""
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
                labels = input_ids.clone() if labels is None else labels.to(self.device).clone()
                if attention_mask is not None:
                    labels.masked_fill_(attention_mask == 0, -100)

                token_count = int((labels[..., 1:] != -100).sum().item())
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
            raise ValueError("The dataset contains no valid prediction tokens.")
        return float(math.exp(total_nll / total_tokens))

    def evaluate_realization(
        self,
        dataset: DatasetLike,
        block_id: BlockId,
        projection_name: str,
        seed: int,
    ) -> Tuple[float, str, str, Dict[str, Any]]:
        """Program and evaluate one fixed analog realization."""
        parent, attribute, original, block_label, projection_label = self.resolve_projection(
            block_id,
            projection_name,
        )

        digital_linear = self.to_linear(original)

        analog_linear = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=self.make_rpu_config(),
        )
        analog_linear.eval()

        # Apply the configured 2.5-sigma clipping.
        for analog_tile in analog_linear.analog_tiles():
            analog_tile.post_update_step()

        # Sample one fixed programming realization.
        self.set_seed(seed)
        analog_linear.program_analog_weights()

        metrics = {
            "in_features": int(digital_linear.in_features),
            "out_features": int(digital_linear.out_features),
            "analog_tile_count": int(analog_linear.analog_tile_count()),
        }

        setattr(parent, attribute, analog_linear)
        try:
            noisy_perplexity = self.compute_perplexity(dataset)
        finally:
            setattr(parent, attribute, original)
            self.model.eval()

        return noisy_perplexity, block_label, projection_label, metrics

    def profile_projection(
        self,
        block_id: BlockId,
        projection_name: str,
        dataset: DatasetLike,
        clean_perplexity: float,
    ) -> Dict[str, Any]:
        """Profile one projection over all configured noise realizations."""
        noisy_perplexities: List[float] = []
        sensitivity_values: List[float] = []
        realization_seeds: List[int] = []
        projection_metadata: Dict[str, Any] = {}

        block_label = str(block_id)
        projection_label = projection_name

        for realization_index in range(self.num_seeds):
            realization_seed = self.seed + realization_index * self.seed_stride
            noisy_perplexity, block_label, projection_label, metrics = (
                self.evaluate_realization(
                    dataset=dataset,
                    block_id=block_id,
                    projection_name=projection_name,
                    seed=realization_seed,
                )
            )

            delta_ppl = noisy_perplexity - clean_perplexity

            noisy_perplexities.append(noisy_perplexity)
            sensitivity_values.append(delta_ppl)
            realization_seeds.append(realization_seed)

            print(
                f"{block_label}/{projection_label} "
                f"seed={realization_seed}: "
                f"PPL={noisy_perplexity:.6f}, "
                f"DeltaPPL={delta_ppl:.6f}",
                flush=True,
            )

            if not projection_metadata:
                projection_metadata = {
                    "in_features": metrics["in_features"],
                    "out_features": metrics["out_features"],
                    "analog_tile_count": metrics["analog_tile_count"],
                }

        noisy_array = np.asarray(noisy_perplexities, dtype=np.float64)
        sensitivity_array = np.asarray(sensitivity_values, dtype=np.float64)

        return {
            "block_id": block_label,
            "proj_name": projection_label,
            "projection_label": f"{block_label}/{projection_label}",
            **projection_metadata,
            "ppl_clean": float(clean_perplexity),
            "ppl_noisy_mean": float(noisy_array.mean()),
            "ppl_noisy_std": float(noisy_array.std(ddof=0)),
            "sensitivity_mean": float(sensitivity_array.mean()),
            "sensitivity_std": float(sensitivity_array.std(ddof=0)),
            "sensitivity_per_seed": sensitivity_array.tolist(),
            "realization_seeds": realization_seeds,
        }

    @staticmethod
    def add_sensitivity_ranks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add descending sensitivity ranks for Phase 3 mapping."""
        sorted_indices = sorted(
            range(len(results)),
            key=lambda index: results[index]["sensitivity_mean"],
            reverse=True,
        )
        for rank, index in enumerate(sorted_indices, start=1):
            results[index]["sensitivity_rank"] = rank
        return results

    def profile_all(self, dataset: DatasetLike) -> List[Dict[str, Any]]:
        """Profile all configured projections and return mapper-ready metrics."""
        batches = dataset if isinstance(dataset, (list, tuple)) else tuple(dataset)
        clean_perplexity = self.compute_perplexity(batches)

        results: List[Dict[str, Any]] = []
        order = self.projection_order

        print(f"Digital FP32 PPL: {clean_perplexity:.6f}", flush=True)

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
                clean_perplexity=clean_perplexity,
            )
            results.append(result)

            print(
                f"Completed {result['projection_label']}: "
                f"DeltaPPL={result['sensitivity_mean']:.6f} "
                f"+/- {result['sensitivity_std']:.6f}",
                flush=True,
            )

        return self.add_sensitivity_ranks(results)
