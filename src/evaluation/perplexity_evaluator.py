"""Streaming GPT-2 quality evaluation for Phase 4.

The evaluator never stores all dataset logits. Reference and noisy models are
run batch by batch, allowing exact token-weighted NLL, KL(clean || noisy), and
next-token agreement without a dataset-sized logit cache.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


Batch = Mapping[str, Tensor]
DatasetLike = Iterable[Batch]


def _prepare_batch(
    batch: Batch,
    device: torch.device,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor, int]:
    """Move a batch to device and preserve any Phase-1 overlap-masked labels."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    supplied_labels = batch.get("labels")
    labels = (
        input_ids.clone()
        if supplied_labels is None
        else supplied_labels.to(device).clone()
    )
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0, -100)

    valid_mask = labels[..., 1:] != -100
    token_count = int(valid_mask.sum().item())
    return input_ids, attention_mask, labels, valid_mask, token_count


def _forward(
    model: Any,
    *,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    labels: Tensor,
) -> tuple[float, Tensor]:
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    return float(output.loss.item()), output.logits


def compute_nll(
    model: Any,
    dataset: DatasetLike,
    device: torch.device,
) -> tuple[float, float]:
    """Compute token-weighted NLL and perplexity."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for batch in dataset:
            input_ids, attention_mask, labels, _, token_count = _prepare_batch(
                batch,
                device,
            )
            if token_count == 0:
                continue
            loss, _ = _forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_nll += loss * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No valid prediction tokens were found in the dataset.")
    nll = total_nll / total_tokens
    return float(nll), float(math.exp(nll))


def evaluate_quality_pair(
    reference_model: Any,
    noisy_model: Any,
    dataset: DatasetLike,
    device: torch.device,
    *,
    compute_kl: bool = True,
    compute_agreement: bool = True,
) -> dict[str, float]:
    """Evaluate a fixed noisy snapshot against a fixed reference model."""
    reference_model.eval()
    noisy_model.eval()

    reference_nll_sum = 0.0
    noisy_nll_sum = 0.0
    total_tokens = 0
    kl_sum = 0.0
    agreement_count = 0.0
    agreement_tokens = 0

    with torch.inference_mode():
        for batch in dataset:
            input_ids, attention_mask, labels, valid_mask, token_count = _prepare_batch(
                batch,
                device,
            )
            if token_count == 0:
                continue

            reference_loss, reference_logits = _forward(
                reference_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            noisy_loss, noisy_logits = _forward(
                noisy_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            reference_nll_sum += reference_loss * token_count
            noisy_nll_sum += noisy_loss * token_count
            total_tokens += token_count

            reference_shifted = reference_logits[:, :-1, :]
            noisy_shifted = noisy_logits[:, :-1, :]
            if reference_shifted.shape != noisy_shifted.shape:
                raise ValueError(
                    "Reference and noisy logits have different shapes: "
                    f"{tuple(reference_shifted.shape)} vs "
                    f"{tuple(noisy_shifted.shape)}."
                )

            if compute_kl:
                kl_mean = _compute_kl_batch(
                    reference_shifted,
                    noisy_shifted,
                    valid_mask,
                )
                kl_sum += kl_mean * token_count
            if compute_agreement:
                batch_agreements, batch_tokens = _compute_agreement_batch(
                    reference_shifted,
                    noisy_shifted,
                    valid_mask,
                )
                agreement_count += batch_agreements
                agreement_tokens += batch_tokens

            del reference_logits, noisy_logits, reference_shifted, noisy_shifted

    if total_tokens == 0:
        raise ValueError("No valid prediction tokens were found in the dataset.")

    reference_nll = reference_nll_sum / total_tokens
    noisy_nll = noisy_nll_sum / total_tokens
    return {
        "reference_nll": float(reference_nll),
        "reference_ppl": float(math.exp(reference_nll)),
        "nll": float(noisy_nll),
        "ppl": float(math.exp(noisy_nll)),
        "delta_nll": float(noisy_nll - reference_nll),
        "delta_ppl": float(math.exp(noisy_nll) - math.exp(reference_nll)),
        "kl_divergence": (
            float(kl_sum / total_tokens) if compute_kl else 0.0
        ),
        "next_token_agreement": (
            float(agreement_count / agreement_tokens)
            if compute_agreement and agreement_tokens > 0
            else 0.0
        ),
        "total_tokens": float(total_tokens),
    }


def _compute_kl_batch(
    reference_logits: Tensor,
    noisy_logits: Tensor,
    valid_mask: Tensor,
) -> float:
    """Compute exact KL(reference || noisy), averaged over valid positions."""
    if reference_logits.ndim != 3 or noisy_logits.ndim != 3:
        raise ValueError("Logits must have shape [batch, sequence, vocabulary].")
    if reference_logits.shape != noisy_logits.shape:
        raise ValueError("Reference and noisy logits must have identical shapes.")
    if valid_mask.shape != reference_logits.shape[:2]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} does not match logits "
            f"positions {tuple(reference_logits.shape[:2])}."
        )

    vocabulary = reference_logits.shape[-1]
    mask = valid_mask.reshape(-1)
    if not bool(mask.any().item()):
        return 0.0
    reference = reference_logits.reshape(-1, vocabulary)[mask].float()
    noisy = noisy_logits.reshape(-1, vocabulary)[mask].float()

    log_p = F.log_softmax(reference, dim=-1)
    log_q = F.log_softmax(noisy, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())


def _compute_agreement_batch(
    reference_logits: Tensor,
    noisy_logits: Tensor,
    valid_mask: Tensor,
) -> tuple[float, int]:
    """Return the number of matching argmax tokens and valid positions."""
    if reference_logits.shape != noisy_logits.shape:
        raise ValueError("Reference and noisy logits must have identical shapes.")
    if valid_mask.shape != reference_logits.shape[:2]:
        raise ValueError("valid_mask does not align with the logit positions.")

    vocabulary = reference_logits.shape[-1]
    mask = valid_mask.reshape(-1)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return 0.0, 0
    reference = reference_logits.reshape(-1, vocabulary)[mask]
    noisy = noisy_logits.reshape(-1, vocabulary)[mask]
    matches = reference.argmax(dim=-1) == noisy.argmax(dim=-1)
    return float(matches.sum().item()), n_valid


def compute_preservation_gain(
    delta_nll_hardware: float,
    delta_nll_static: float,
    *,
    denominator_threshold: float = 1e-8,
) -> float | None:
    """Return the fraction of hardware-only DeltaNLL removed by static mapping."""
    if abs(delta_nll_hardware) < denominator_threshold:
        return None
    return (delta_nll_hardware - delta_nll_static) / delta_nll_hardware


def compute_ppl_preservation_ratio(
    ppl_reference: float,
    ppl_hardware: float,
    ppl_static: float,
    *,
    denominator_threshold: float = 1e-8,
) -> float | None:
    """Return the fraction of hardware-only PPL degradation removed."""
    denominator = ppl_hardware - ppl_reference
    if abs(denominator) < denominator_threshold:
        return None
    return (ppl_hardware - ppl_static) / denominator
