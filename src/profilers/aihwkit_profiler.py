"""Projection-level AIHWKit sensitivity profiler for GPT-2.

Uses Corey Lammie's analog configuration:
- layer-Gaussian clipping at 2.5 sigma
- 512 x 512 mapped tiles
- 8-bit input/output resolution
- PCMLikeNoiseModel with programming-noise scale 0.023
- no read noise

Each seed creates one programmed-noise realization and evaluates that fixed
realization over the complete dataset.
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

from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter, WeightClipParameter
from aihwkit.simulator.parameters.enums import WeightClipType

logger = logging.getLogger(__name__)

Batch = Mapping[str, Tensor]
DatasetLike = Iterable[Batch]
BlockId = Union[int, str]


class AIHWKITSensitivityProfiler:
    """Profile GPT-2 projections using Lammie's AIHWKit configuration."""

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
        self.programming_noise_scale = float(
            profiling_cfg.get("programming_noise_sigma", 0.023)
        )

        self.num_seeds = int(profiling_cfg.get("num_seeds", 10))
        self.seed_stride = int(profiling_cfg.get("seed_stride", 1))
        self.include_lm_head = bool(profiling_cfg.get("include_lm_head", True))
        self.seed = int(experiment_cfg.get("seed", 42))

        requested_blocks = profiling_cfg.get("profile_blocks")
        self.profile_blocks: Optional[Tuple[int, ...]] = (
            None
            if requested_blocks is None
            else tuple(int(index) for index in requested_blocks)
        )

        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()

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
        """Create Lammie's AIHWKit inference configuration."""
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
        )
        rpu_config.forward.inp_res = resolution
        rpu_config.forward.out_res = resolution

        rpu_config.noise_model = PCMLikeNoiseModel(
            prog_noise_scale=self.programming_noise_scale,
            read_noise_scale=0.0,
            drift_scale=0.0,
        )

        return rpu_config

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
            return self.model, "lm_head", self.model.lm_head, "head", "lm_head"

        block_index = int(str(block_id).removeprefix("block_"))
        block = self.model.transformer.h[block_index]
        parent, attribute = self._parent_and_attribute(block, projection_name)

        return (
            parent,
            attribute,
            getattr(parent, attribute),
            f"block_{block_index}",
            projection_name,
        )

    def to_linear(self, module: nn.Module) -> nn.Linear:
        """Copy a GPT-2 Conv1D or Linear projection into nn.Linear."""
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
    ) -> Tuple[float, str, str]:
        """Program and evaluate one fixed PCM-noise realization."""
        (
            parent,
            attribute,
            original,
            canonical_block,
            canonical_projection,
        ) = self.resolve_projection(block_id, projection_name)

        self.set_seed(seed)
        analog_linear = AnalogLinearMapped.from_digital(
            self.to_linear(original),
            rpu_config=self.make_rpu_config(),
        )
        analog_linear.eval()

        # WeightClipParameter is applied by the tile post-update hook. Since
        # this profiler starts from pretrained weights and performs no
        # training update, invoke that hook once before programming.
        for tile in analog_linear.analog_tiles():
            tile.post_update_step()

        # Sample programming noise once, then keep the realization fixed for
        # the complete perplexity evaluation.
        self.set_seed(seed)
        analog_linear.program_analog_weights()

        logger.info(
            "%s/%s: clip_sigma=%.3f, tile=%dx%d, io_bits=%d, "
            "prog_noise_scale=%.6f, read_noise_scale=0.0",
            canonical_block,
            canonical_projection,
            self.clip_sigma,
            self.tile_size,
            self.tile_size,
            self.adc_dac_bits,
            self.programming_noise_scale,
        )

        setattr(parent, attribute, analog_linear)
        try:
            noisy_perplexity = self.compute_perplexity(dataset)
        finally:
            setattr(parent, attribute, original)
            self.model.eval()

        return noisy_perplexity, canonical_block, canonical_projection

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
        canonical_block = str(block_id)
        canonical_projection = projection_name

        for realization_index in range(self.num_seeds):
            realization_seed = self.seed + realization_index * self.seed_stride
            (
                noisy_perplexity,
                canonical_block,
                canonical_projection,
            ) = self.evaluate_realization(
                dataset=dataset,
                block_id=block_id,
                projection_name=projection_name,
                seed=realization_seed,
            )

            noisy_perplexities.append(noisy_perplexity)
            realization_seeds.append(realization_seed)

            logger.info(
                "%s/%s realization %d/%d: PPL=%.6f, Delta=%.6f",
                canonical_block,
                canonical_projection,
                realization_index + 1,
                self.num_seeds,
                noisy_perplexity,
                noisy_perplexity - clean_perplexity,
            )

        noisy_array = np.asarray(noisy_perplexities, dtype=np.float64)
        sensitivity_array = noisy_array - float(clean_perplexity)

        return {
            "block_id": canonical_block,
            "proj_name": canonical_projection,
            "projection_label": f"{canonical_block}/{canonical_projection}",
            "ppl_clean": float(clean_perplexity),
            "ppl_noisy_mean": float(noisy_array.mean()),
            "sensitivity_mean": float(sensitivity_array.mean()),
            "sensitivity_std": float(sensitivity_array.std(ddof=0)),
            "sensitivity_per_seed": sensitivity_array.tolist(),
            "realization_seeds": realization_seeds,
        }

    def profile_all(
        self,
        dataset: DatasetLike,
        projection_order: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Profile every requested projection in deterministic order."""
        batches = dataset if isinstance(dataset, (list, tuple)) else tuple(dataset)
        order = tuple(projection_order or self.projection_order)
        clean_perplexity = self.compute_perplexity(batches)

        logger.info("Digital FP32 baseline PPL: %.6f", clean_perplexity)
        results: List[Dict[str, Any]] = []

        for index, (block_id, projection_name) in enumerate(order, start=1):
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
