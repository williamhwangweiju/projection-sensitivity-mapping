"""Lammie 2026 GPT-2 Stage 1-2 sensitivity profiler using AIHWKIT 1.1.0.

Faithful reconstruction of the methodology disclosed in:

    C. Lammie, "Heterogeneous Mapping for Analog In-Memory Computing
    Accelerators: A Unified Workflow," IEEE Computer Architecture Letters,
    2026. arXiv:2606.02672v1

For each of GPT-2-small's 49 weight projections, this implementation:

1. Keeps the complete model in FP32.
2. Applies AIHWKIT-style layer-Gaussian clipping at 2.5 * population std
   of the layer weight matrix (std with correction=0).
3. Replaces exactly one GPT-2 projection with AnalogLinearMapped.
4. Maps the projection onto 512 x 512 physical analog tiles.
5. Applies approximately 8-bit DAC and ADC discretization in the analog MVM
   with BoundManagementType.NONE so ADC clipping error is not suppressed.
6. Programs the selected projection once with additive Gaussian programming
   noise having sigma_w = 0.023 relative to the programmed weight magnitude.
7. Keeps that programmed realization fixed for the complete WikiText-103 pass.
8. Repeats the experiment for 10 independent programming realizations.
9. Reports Delta PPL = PPL_analog - PPL_digital.

Paper-verified design decisions
--------------------------------
* The 49 projections are: 12 blocks x {c_attn (fused Q/K/V), attn.c_proj,
  mlp.c_fc, mlp.c_proj} + lm_head. The paper explicitly counts 49 (=12x4+1)
  and notes that "separating Q, K, and V ... would enable finer-grained
  profiling" as future work, confirming c_attn is treated as a single unit.
  (Section IV, paragraph 3 footnote.)
* "Gaussian weight clipping (2.5σ)" where σ is the population standard
  deviation of the weight matrix (not the RMS). This is AIHWKIT's
  WeightClipType.LAYER_GAUSSIAN convention.
* sigma_w = 0.023 is relative to the programmed weight range (absmax).
* WikiText-103 evaluation, n=10 noise realizations, t=0 (no drift/read noise).
* 490 total forward passes (49 projections x 10 realizations).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import no_grad
from transformers.pytorch_utils import Conv1D

from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
)
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.enums import (
    BoundManagementType,
    NoiseManagementType,
    WeightClipType,
    WeightModifierType,
    WeightNoiseType,
    WeightRemapType,
)

logger = logging.getLogger(__name__)

Batch = Mapping[str, Tensor]
DatasetLike = Iterable[Batch]
BlockId = Union[int, str]


class RelativeGaussianProgrammingNoise(BaseNoiseModel):
    """One-time Gaussian programming noise expressed in programmed-weight units.

    AIHWKIT calls ``apply_programming_noise()`` once for each physical tile when
    ``program_analog_weights()`` is invoked. Overriding this method keeps the
    paper's noise definition exact in the weight domain while using AIHWKIT's
    official inference-programming path and physical tiling.

    With ``range_mode="absmax"``::

        noise_std = sigma_w * max(abs(programmed_tile_weights))

    This matches the paper: sigma_w = 0.023 relative to the programmed weight
    range (Section IV). Because weight_scaling_omega = 1.0 is used below,
    the programmed tile range is normalised to approximately [-1, 1].

    ``range_mode="peak_to_peak"`` is provided for sensitivity checks.
    """

    def __init__(
        self,
        sigma_w: float = 0.023,
        range_mode: str = "absmax",
    ) -> None:
        super().__init__(
            g_converter=SinglePairConductanceConverter(g_max=25.0, g_min=0.0)
        )
        if sigma_w < 0.0:
            raise ValueError("sigma_w must be non-negative.")
        if range_mode not in {"absmax", "peak_to_peak"}:
            raise ValueError("range_mode must be 'absmax' or 'peak_to_peak'.")

        self.sigma_w = float(sigma_w)
        self.range_mode = range_mode

    @no_grad()
    def apply_programming_noise(
        self,
        weights: Tensor,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Program one fixed noisy copy of an AIHWKIT tile's weights."""
        if self.range_mode == "absmax":
            programmed_range = weights.detach().abs().amax()
        else:
            programmed_range = weights.detach().amax() - weights.detach().amin()

        if not torch.isfinite(programmed_range):
            raise FloatingPointError("The programmed weight range is not finite.")

        if programmed_range.item() == 0.0 or self.sigma_w == 0.0:
            return weights.detach().clone(), [None]

        noise_std = self.sigma_w * programmed_range
        programmed = weights.detach() + torch.randn_like(weights) * noise_std

        # Drift parameters unused: paper evaluates t=0.
        return programmed, [None]

    @no_grad()
    def apply_drift_noise(
        self,
        weights: Tensor,
        drift_noise_parameters: List[Optional[Tensor]],
        t_inference: float,
    ) -> Tensor:
        """No drift or read noise: paper evaluates t=0."""
        del drift_noise_parameters
        if float(t_inference) != 0.0:
            raise ValueError(
                "This reconstruction supports only t_inference=0. "
                "The paper's Stage 1 experiment excludes drift and read noise."
            )
        return weights

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Prevent accidental use through the conductance-polynomial API."""
        del g_target
        raise RuntimeError(
            "Use apply_programming_noise(). sigma_w is defined relative to "
            "each programmed tile's weight range, not in conductance units."
        )

    @no_grad()
    def apply_drift_noise_to_conductance(
        self,
        g_prog: Tensor,
        drift_noise_param: Optional[Tensor],
        t_inference: float,
    ) -> Tensor:
        """No conductance drift/read noise for the t=0 model."""
        del drift_noise_param
        if float(t_inference) != 0.0:
            raise ValueError("Conductance drift/read noise disabled at t=0.")
        return g_prog


@dataclass(frozen=True)
class ProjectionReference:
    """Location and metadata for one GPT-2 weight projection."""

    block_id: str
    projection_name: str
    parent: nn.Module
    attribute: str
    module: nn.Module


@dataclass(frozen=True)
class ConversionMetadata:
    """Metadata produced while converting one digital projection."""

    in_features: int
    out_features: int
    clip_value: float
    clipped_fraction: float
    source_type: str
    lm_head_was_tied: bool


class AIHWKITLammieSensitivityProfiler:
    """Projection-wise GPT-2 AIMC sensitivity profiler.

    Implements the Stage 1-2 methodology of Lammie (2026), arXiv:2606.02672v1.

    49 projections: 12 blocks x {c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj}
    plus lm_head. The fused c_attn is treated as a single projection, exactly
    as in the paper (490 total forward passes = 49 x 10 realizations).
    """

    # Paper Section IV: "12 blocks x {fused Q/K/V (c_attn), attention output
    # (c_proj), FFN up (c_fc), FFN down (mlp.c_proj)} + language model head"
    # = 49 projections total.
    PROJECTION_ORDER: Tuple[Tuple[str, str], ...] = tuple(
        (f"block_{b}", proj)
        for b in range(12)
        for proj in ("c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    ) + (("head", "lm_head"),)

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        config: Optional[Dict] = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        """Initialise the profiler.

        Args:
            model:
                Hugging Face GPT2LMHeadModel loaded from ``gpt2``.
            tokenizer:
                Retained for API compatibility; not used internally.
            config:
                Configuration dict with an optional ``profiling`` section.
            device:
                ``"cpu"`` or ``"cuda"``. AIHWKIT 1.1.0 does not support MPS.
            seed:
                Base RNG seed for the 10 independent programming realizations.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.device = torch.device(device)

        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "Use device='cpu' or a CUDA-enabled AIHWKIT installation. "
                "AIHWKIT analog tiles do not support Apple's MPS backend."
            )

        config = config or {}
        profiling = config.get("profiling", {})

        self.weight_clip_sigma = float(profiling.get("weight_clipping_std_multiple", 2.5))
        self.crossbar_rows = int(profiling.get("crossbar_rows", 512))
        self.crossbar_cols = int(profiling.get("crossbar_cols", 512))
        self.adc_dac_bits = int(profiling.get("adc_dac_bits", 8))
        self.programming_noise_std = float(profiling.get("programming_noise_std", 0.023))
        self.programming_noise_range_mode = str(
            profiling.get("programming_noise_range_mode", "absmax")
        )
        self.t_inference_seconds = float(profiling.get("t_inference_seconds", 0.0))
        self.num_seeds = int(profiling.get("num_seeds", 10))
        self.sensitivity_metric = str(profiling.get("sensitivity_metric", "perplexity"))
        self.input_bound = float(profiling.get("input_bound", 1.0))
        self.output_bound = float(profiling.get("output_bound", 12.0))
        self.weight_scaling_omega = float(profiling.get("weight_scaling_omega", 1.0))
        self.strict_paper_config = bool(profiling.get("strict_paper_config", True))

        self._validate_configuration()

        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()

        self.noise_model = RelativeGaussianProgrammingNoise(
            sigma_w=self.programming_noise_std,
            range_mode=self.programming_noise_range_mode,
        )

    def _validate_configuration(self) -> None:
        """Reject accidental departures from the paper's disclosed setup."""
        if self.weight_clip_sigma <= 0.0:
            raise ValueError("weight_clipping_std_multiple must be positive.")
        if self.crossbar_rows <= 0 or self.crossbar_cols <= 0:
            raise ValueError("Crossbar dimensions must be positive.")
        if self.adc_dac_bits <= 0:
            raise ValueError("adc_dac_bits must be positive.")
        if self.num_seeds <= 0:
            raise ValueError("num_seeds must be positive.")
        if self.sensitivity_metric != "perplexity":
            raise ValueError("The paper's Stage 2 metric is perplexity.")
        if self.t_inference_seconds != 0.0:
            raise ValueError("The paper evaluates at t=0 (immediately after programming).")

        if not self.strict_paper_config:
            return

        expected = {
            "weight clipping sigma": (self.weight_clip_sigma, 2.5),
            "crossbar rows": (self.crossbar_rows, 512),
            "crossbar columns": (self.crossbar_cols, 512),
            "ADC/DAC bits": (self.adc_dac_bits, 8),
            "programming noise sigma": (self.programming_noise_std, 0.023),
            "noise realizations": (self.num_seeds, 10),
        }

        mismatches = [
            f"{name}: got {actual!r}, expected {target!r}"
            for name, (actual, target) in expected.items()
            if actual != target
        ]
        if mismatches:
            raise ValueError(
                "strict_paper_config=True but values differ from Lammie 2026:\n- "
                + "\n- ".join(mismatches)
            )

    def _make_rpu_config(self) -> InferenceRPUConfig:
        """Build the AIHWKIT analog-tile configuration for one projection."""
        rpu_config = InferenceRPUConfig()

        # Physical 512 x 512 tiling (Section IV).
        rpu_config.mapping.max_input_size = self.crossbar_cols
        rpu_config.mapping.max_output_size = self.crossbar_rows
        rpu_config.mapping.digital_bias = True
        rpu_config.mapping.weight_scaling_omega = self.weight_scaling_omega
        rpu_config.mapping.weight_scaling_columnwise = False

        # ~8-bit ADC/DAC: AIHWKIT resolution = normalised step size (Section IV).
        converter_resolution = 1.0 / float(2 ** self.adc_dac_bits)

        rpu_config.forward.is_perfect = False
        rpu_config.forward.inp_bound = self.input_bound
        rpu_config.forward.out_bound = self.output_bound
        rpu_config.forward.inp_res = converter_resolution
        rpu_config.forward.out_res = converter_resolution
        rpu_config.forward.inp_sto_round = False
        rpu_config.forward.out_sto_round = False

        # Disable all non-idealities not mentioned in the paper.
        rpu_config.forward.inp_noise = 0.0
        rpu_config.forward.out_noise = 0.0
        rpu_config.forward.w_noise = 0.0
        rpu_config.forward.w_noise_type = WeightNoiseType.NONE
        rpu_config.forward.ir_drop = 0.0
        rpu_config.forward.inp_asymmetry = 0.0
        rpu_config.forward.out_asymmetry = 0.0
        rpu_config.forward.out_nonlinearity = 0.0

        # ABS_MAX: scale DAC input by the batch's absolute maximum.
        rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX

        # FIX-5: NONE (not ITERATIVE) so ADC output clipping is a genuine
        # error source rather than being silently corrected by a second MVM pass.
        rpu_config.forward.bound_management = BoundManagementType.NONE

        # Clipping is applied manually before tile creation (see
        # _clip_weight_aihwkit_layer_gaussian), so disable in-tile clipping.
        rpu_config.clip.type = WeightClipType.NONE
        rpu_config.remap.type = WeightRemapType.NONE
        rpu_config.modifier.type = WeightModifierType.NONE

        # FIX-3: noise model is embedded here; program_analog_weights() uses
        # it with no arguments in AIHWKIT 1.1.0.
        rpu_config.noise_model = self.noise_model
        rpu_config.drift_compensation = None

        return rpu_config

    @staticmethod
    def _require_reiterable(dataset: DatasetLike) -> None:
        """Reject one-shot generators that exhaust after one pass."""
        if iter(dataset) is dataset:
            raise TypeError(
                "dataset must be re-iterable (list, tuple, or DataLoader). "
                "It is evaluated up to 490 times (49 projections x 10 seeds)."
            )

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Seed all RNGs used by the reconstruction."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_projection(
        self,
        block_id: BlockId,
        proj_name: str,
    ) -> ProjectionReference:
        """Resolve a paper projection name to its parent module and attribute."""
        if block_id == "head":
            if proj_name != "lm_head":
                raise KeyError(f"Unknown head projection: {proj_name!r}.")
            return ProjectionReference(
                block_id="head",
                projection_name="lm_head",
                parent=self.model,
                attribute="lm_head",
                module=self.model.lm_head,
            )

        if isinstance(block_id, int):
            block_idx = block_id
        elif isinstance(block_id, str) and block_id.startswith("block_"):
            try:
                block_idx = int(block_id.split("_", maxsplit=1)[1])
            except ValueError as exc:
                raise KeyError(f"Invalid block identifier: {block_id!r}.") from exc
        else:
            raise KeyError(f"Invalid block identifier: {block_id!r}.")

        if block_idx < 0 or block_idx >= len(self.model.transformer.h):
            raise IndexError(f"GPT-2 block index out of range: {block_idx}.")

        block = self.model.transformer.h[block_idx]

        # Canonical alias map (including common alternative spellings).
        aliases = {
            "c_attn": "c_attn",
            "q_proj": "c_attn",   # fused — maps to same c_attn module
            "k_proj": "c_attn",
            "v_proj": "c_attn",
            "out_proj": "attn.c_proj",
            "c_proj": "attn.c_proj",
            "attn.c_proj": "attn.c_proj",
            "fc1": "mlp.c_fc",
            "c_fc": "mlp.c_fc",
            "mlp.c_fc": "mlp.c_fc",
            "fc2": "mlp.c_proj",
            "mlp.c_proj": "mlp.c_proj",
        }

        if proj_name not in aliases:
            raise KeyError(f"Unknown GPT-2 projection name: {proj_name!r}.")

        canonical = aliases[proj_name]

        if canonical == "c_attn":
            parent, attribute = block.attn, "c_attn"
        elif canonical == "attn.c_proj":
            parent, attribute = block.attn, "c_proj"
        elif canonical == "mlp.c_fc":
            parent, attribute = block.mlp, "c_fc"
        else:
            parent, attribute = block.mlp, "c_proj"

        return ProjectionReference(
            block_id=f"block_{block_idx}",
            projection_name=canonical,
            parent=parent,
            attribute=attribute,
            module=getattr(parent, attribute),
        )

    def _clip_weight_aihwkit_layer_gaussian(
        self,
        weight: Tensor,
    ) -> Tuple[Tensor, float, float]:
        """Apply AIHWKIT LayerGaussian clipping to the logical weight matrix.

        FIX-4: The paper says "Gaussian weight clipping (2.5σ)" where σ is the
        standard deviation of the weight matrix. AIHWKIT's
        WeightClipType.LAYER_GAUSSIAN uses population std (correction=0), NOT
        the RMS (sqrt(mean(w^2))). These differ when the weight mean != 0.

            clip_value = clip_sigma * std(weight, correction=0)

        Applied once over the full logical projection before physical tiling,
        consistent with the paper (Section IV, Stage 1).
        """
        weight_fp32 = weight.detach().to(dtype=torch.float32)

        # FIX-4: population standard deviation, not RMS.
        pop_std = weight_fp32.std(correction=0)
        clip_value_tensor = self.weight_clip_sigma * pop_std

        if not torch.isfinite(clip_value_tensor):
            raise FloatingPointError("Computed clipping threshold is not finite.")

        clip_value = float(clip_value_tensor.item())
        if clip_value == 0.0:
            return weight_fp32.clone(), 0.0, 0.0

        clipped = torch.clamp(weight_fp32, min=-clip_value, max=clip_value)
        clipped_fraction = float(
            (weight_fp32.abs() > clip_value).float().mean().item()
        )
        return clipped, clip_value, clipped_fraction

    def _to_clipped_digital_linear(
        self,
        reference: ProjectionReference,
    ) -> Tuple[nn.Linear, ConversionMetadata]:
        """Convert a GPT-2 Conv1D/Linear to a clipped FP32 nn.Linear.

        FIX-2: For lm_head, GPT-2 ties the weight to transformer.wte.weight.
        The clipped_weight produced here is already a fresh detached+clamped
        tensor, so assigning it to digital_linear.weight breaks the tie at the
        data level. The original model's embedding weight is never modified.
        """
        module = reference.module

        if isinstance(module, Conv1D):
            # HuggingFace Conv1D stores weights as [in_features, out_features].
            source_weight = module.weight.detach().transpose(0, 1).contiguous()
            source_bias = module.bias.detach()
            source_type = "transformers.Conv1D"
        elif isinstance(module, nn.Linear):
            # torch.nn.Linear stores weights as [out_features, in_features].
            source_weight = module.weight.detach()
            source_bias = module.bias.detach() if module.bias is not None else None
            source_type = "torch.nn.Linear"
        else:
            raise TypeError(
                "Expected transformers.Conv1D or torch.nn.Linear, "
                f"but found {type(module).__name__}."
            )

        out_features, in_features = source_weight.shape

        clipped_weight, clip_value, clipped_fraction = (
            self._clip_weight_aihwkit_layer_gaussian(source_weight)
        )

        # FIX-2: detect tied lm_head before building the independent copy.
        lm_head_was_tied = False
        if reference.block_id == "head" and isinstance(module, nn.Linear):
            embedding_weight = self.model.transformer.wte.weight
            lm_head_was_tied = (
                module.weight.data_ptr() == embedding_weight.data_ptr()
            )
            # clipped_weight is already an independent tensor (detach+clamp),
            # so no additional copy step is needed to break the tie.

        digital_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=source_bias is not None,
            device=self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            digital_linear.weight.copy_(clipped_weight.to(self.device))
            if source_bias is not None:
                digital_linear.bias.copy_(
                    source_bias.to(device=self.device, dtype=torch.float32)
                )

        return digital_linear, ConversionMetadata(
            in_features=in_features,
            out_features=out_features,
            clip_value=clip_value,
            clipped_fraction=clipped_fraction,
            source_type=source_type,
            lm_head_was_tied=lm_head_was_tied,
        )

    def _compute_perplexity(self, dataset: DatasetLike) -> float:
        """Compute token-weighted causal-LM perplexity on WikiText-103.

        FIX-6: predicted_tokens is counted from labels only (positions where
        labels[..., 1:] != -100). The original code additionally gated on
        attention_mask[..., 1:], which double-counted the masking already
        applied by HuggingFace's cross-entropy via the -100 ignore index.
        """
        self._require_reiterable(dataset)

        total_nll = 0.0
        total_tokens = 0

        self.model.eval()
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataset):
                if "input_ids" not in batch:
                    raise KeyError(f"Batch {batch_idx} has no 'input_ids'.")

                input_ids = batch["input_ids"].to(self.device)
                if input_ids.ndim != 2:
                    raise ValueError(
                        "input_ids must have shape [batch, sequence]."
                    )

                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                supplied_labels = batch.get("labels")
                if supplied_labels is None:
                    labels = input_ids.clone()
                    if attention_mask is not None:
                        # Mark padding positions so HuggingFace excludes them.
                        labels = labels.masked_fill(attention_mask == 0, -100)
                else:
                    labels = supplied_labels.to(self.device).clone()

                # FIX-6: count valid targets from labels only.
                # GPT-2 predicts labels[..., 1:] from logits[..., :-1].
                valid_targets = labels[..., 1:] != -100
                n_tokens = int(valid_targets.sum().item())
                if n_tokens == 0:
                    continue

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss

                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite LM loss in batch {batch_idx}."
                    )

                # loss is mean NLL over the n_tokens valid positions.
                total_nll += float(loss.item()) * n_tokens
                total_tokens += n_tokens

        if total_tokens == 0:
            raise ValueError("No valid next-token targets found in dataset.")

        return float(math.exp(total_nll / total_tokens))

    def compute_digital_perplexity(self, dataset: DatasetLike) -> float:
        """Public helper: compute the all-FP32 WikiText-103 baseline PPL."""
        return self._compute_perplexity(dataset)

    def _evaluate_programming_realization(
        self,
        dataset: DatasetLike,
        block_id: BlockId,
        proj_name: str,
        realization_seed: int,
    ) -> Tuple[float, int, ConversionMetadata, str, str]:
        """Evaluate one fixed programming realization for one projection.

        FIX-3: program_analog_weights() is called with no arguments; the noise
        model is embedded in rpu_config at construction time.
        """
        reference = self._resolve_projection(block_id, proj_name)
        original_module = reference.module

        digital_linear, metadata = self._to_clipped_digital_linear(reference)
        rpu_config = self._make_rpu_config()

        analog_linear = AnalogLinearMapped.from_digital(
            digital_linear,
            rpu_config=rpu_config,
        )
        analog_linear.eval()

        analog_tile_count = analog_linear.analog_tile_count()

        setattr(reference.parent, reference.attribute, analog_linear)

        try:
            self.model.eval()

            # Seed immediately before programming so that AnalogLinearMapped
            # construction RNG use cannot shift the realization index.
            self._set_seed(realization_seed)

            # FIX-3: no arguments — noise_model is already in rpu_config.
            analog_linear.program_analog_weights()

            perplexity = self._compute_perplexity(dataset)
        finally:
            # Always restore the original module, even on exception.
            setattr(reference.parent, reference.attribute, original_module)
            self.model.eval()

        return (
            perplexity,
            analog_tile_count,
            metadata,
            reference.block_id,
            reference.projection_name,
        )

    def profile_projection(
        self,
        block_id: BlockId,
        proj_name: str,
        dataset: DatasetLike,
        num_seeds: Optional[int] = None,
        ppl_clean: Optional[float] = None,
    ) -> Dict:
        """Profile one GPT-2 projection using the paper's Stage 1-2 method.

        Args:
            block_id:  ``"block_0"`` … ``"block_11"`` or ``"head"``.
            proj_name: ``"c_attn"``, ``"attn.c_proj"``, ``"mlp.c_fc"``,
                       ``"mlp.c_proj"``, or ``"lm_head"``.
            dataset:   Re-iterable WikiText-103 batches with ``input_ids``.
            num_seeds: Override self.num_seeds (default 10).
            ppl_clean: Pre-computed digital baseline PPL (avoids recomputation).
        """
        self._require_reiterable(dataset)

        realizations = self.num_seeds if num_seeds is None else int(num_seeds)
        if realizations <= 0:
            raise ValueError("num_seeds must be positive.")

        if ppl_clean is None:
            ppl_clean = self._compute_perplexity(dataset)

        noisy_perplexities: List[float] = []
        realization_seeds: List[int] = []
        analog_tile_count: Optional[int] = None
        conversion_metadata: Optional[ConversionMetadata] = None
        canonical_block_id = ""
        canonical_projection_name = ""

        for realization_index in range(realizations):
            realization_seed = self.seed + realization_index
            (
                noisy_ppl,
                current_tile_count,
                current_metadata,
                canonical_block_id,
                canonical_projection_name,
            ) = self._evaluate_programming_realization(
                dataset=dataset,
                block_id=block_id,
                proj_name=proj_name,
                realization_seed=realization_seed,
            )

            if analog_tile_count is not None and analog_tile_count != current_tile_count:
                raise RuntimeError("AIHWKIT tile count changed between realizations.")

            analog_tile_count = current_tile_count
            conversion_metadata = current_metadata
            noisy_perplexities.append(noisy_ppl)
            realization_seeds.append(realization_seed)

            logger.info(
                "%s/%s realization %d/%d: PPL=%.6f, Delta=%.6f",
                canonical_block_id,
                canonical_projection_name,
                realization_index + 1,
                realizations,
                noisy_ppl,
                noisy_ppl - ppl_clean,
            )

        noisy_array = np.asarray(noisy_perplexities, dtype=np.float64)
        delta_array = noisy_array - float(ppl_clean)

        if conversion_metadata is None or analog_tile_count is None:
            raise RuntimeError("No programming realizations were evaluated.")

        result = {
            "block_id": canonical_block_id,
            "proj_name": canonical_projection_name,
            "ppl_clean": float(ppl_clean),
            "ppl_noisy_per_seed": noisy_array.tolist(),
            "ppl_noisy_mean": float(noisy_array.mean()),
            "ppl_noisy_std": float(noisy_array.std(ddof=0)),
            "sensitivity_per_seed": delta_array.tolist(),
            "sensitivity_mean": float(delta_array.mean()),
            "sensitivity_std": float(delta_array.std(ddof=0)),
            "num_noise_realizations": realizations,
            "realization_seeds": realization_seeds,
            "analog_tile_count": int(analog_tile_count),
            "in_features": conversion_metadata.in_features,
            "out_features": conversion_metadata.out_features,
            "source_module_type": conversion_metadata.source_type,
            "clip_value": conversion_metadata.clip_value,
            "clipped_fraction": conversion_metadata.clipped_fraction,
            "lm_head_was_tied": conversion_metadata.lm_head_was_tied,
            "paper_configuration": {
                "weight_clip_sigma": self.weight_clip_sigma,
                "clip_formula": "clip_sigma * std(weight, correction=0)",
                "crossbar_rows": self.crossbar_rows,
                "crossbar_cols": self.crossbar_cols,
                "adc_dac_bits": self.adc_dac_bits,
                "programming_noise_std": self.programming_noise_std,
                "programming_noise_range_mode": self.programming_noise_range_mode,
                "t_inference_seconds": self.t_inference_seconds,
                "bound_management": "NONE",
                "other_analog_noise_disabled": True,
                "num_projections": 49,
                "c_attn_treatment": "fused_qkv_single_projection",
            },
        }

        logger.info(
            "%s/%s: clean PPL=%.6f, Delta PPL=%.6f +/- %.6f",
            canonical_block_id,
            canonical_projection_name,
            ppl_clean,
            result["sensitivity_mean"],
            result["sensitivity_std"],
        )
        return result

    def profile_all(
        self,
        dataset: DatasetLike,
        projection_order: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[Dict]:
        """Profile all 49 GPT-2-small projections (490 forward passes total).

        The digital FP32 baseline is computed once and reused for all 49
        projections, as in the paper.
        """
        self._require_reiterable(dataset)

        order = (
            tuple(projection_order)
            if projection_order is not None
            else self.PROJECTION_ORDER
        )

        ppl_clean = self._compute_perplexity(dataset)
        logger.info("Digital FP32 baseline PPL: %.6f", ppl_clean)

        results: List[Dict] = []
        for i, (block_id, proj_name) in enumerate(order):
            logger.info(
                "Profiling projection %d/%d: %s/%s",
                i + 1, len(order), block_id, proj_name,
            )
            results.append(
                self.profile_projection(
                    block_id=block_id,
                    proj_name=proj_name,
                    dataset=dataset,
                    num_seeds=self.num_seeds,
                    ppl_clean=ppl_clean,
                )
            )

        return results


# Backward-compatible alias.
AIHWKITSensitivityProfiler = AIHWKITLammieSensitivityProfiler

__all__ = [
    "AIHWKITLammieSensitivityProfiler",
    "AIHWKITSensitivityProfiler",
    "RelativeGaussianProgrammingNoise",
]
