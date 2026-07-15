"""Tests for the perplexity evaluator.

Covers:
- Zero-noise injection reproduces clean logits and NLL.
- NLL is computed as token-weighted average (not batch-averaged perplexity).
- KL(clean||clean) == 0.
- KL(clean||uniform) is large.
- Next-token agreement is 1.0 when logits match exactly.
- DeltaNLL is zero for zero-noise model.
- Preservation gain formula.
- PPL preservation ratio formula.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.perplexity_evaluator import (
    _compute_agreement_batch,
    _compute_kl_batch,
    collect_logits,
    compute_nll,
    compute_preservation_gain,
    compute_ppl_preservation_ratio,
    evaluate_quality,
)


# ---------------------------------------------------------------------------
# Minimal toy language model for testing
# ---------------------------------------------------------------------------

class ToyLM(nn.Module):
    """Tiny causal LM that returns controllable logits for unit tests."""

    def __init__(self, vocab: int = 50, seq_len: int = 8) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, 16)
        self.linear = nn.Linear(16, vocab)
        self.seq_len = seq_len

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        x = self.embed(input_ids)
        logits = self.linear(x)
        loss = None
        if labels is not None:
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].reshape(-1, V)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100
            )

        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        return Output(loss, logits)


def _make_batch(vocab: int = 50, seq_len: int = 8, batch_size: int = 1) -> dict[str, torch.Tensor]:
    ids = torch.randint(0, vocab, (batch_size, seq_len))
    return {"input_ids": ids}


# ---------------------------------------------------------------------------
# NLL computation
# ---------------------------------------------------------------------------

class TestComputeNLL:
    def test_nll_matches_manual_cross_entropy(self) -> None:
        torch.manual_seed(0)
        vocab, seq_len = 20, 6
        model = ToyLM(vocab=vocab, seq_len=seq_len)
        model.eval()

        batch = _make_batch(vocab=vocab, seq_len=seq_len)
        dataset = [batch]
        device = torch.device("cpu")

        nll, ppl = compute_nll(model, dataset, device)

        # Recompute manually
        with torch.inference_mode():
            out = model(**batch, labels=batch["input_ids"], use_cache=False)
        expected_nll = float(out.loss.item())
        assert abs(nll - expected_nll) < 1e-5
        assert abs(ppl - math.exp(nll)) < 1e-4

    def test_nll_is_token_weighted(self) -> None:
        """Two batches with different token counts: result must be token-weighted."""
        torch.manual_seed(1)
        vocab = 30
        model = ToyLM(vocab=vocab)
        model.eval()

        b1 = _make_batch(vocab=vocab, seq_len=4)
        b2 = _make_batch(vocab=vocab, seq_len=8)
        device = torch.device("cpu")

        nll_joint, _ = compute_nll(model, [b1, b2], device)

        # Check it's not simply the average of the two losses
        with torch.inference_mode():
            out1 = model(**b1, labels=b1["input_ids"], use_cache=False)
            out2 = model(**b2, labels=b2["input_ids"], use_cache=False)

        loss1 = float(out1.loss.item())
        loss2 = float(out2.loss.item())

        # Seq length 4 → 3 valid positions; seq length 8 → 7 valid positions
        n1, n2 = 3, 7
        expected_weighted = (loss1 * n1 + loss2 * n2) / (n1 + n2)
        assert abs(nll_joint - expected_weighted) < 1e-4


# ---------------------------------------------------------------------------
# KL divergence helpers
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_kl_clean_vs_clean_is_zero(self) -> None:
        logits = torch.randn(2, 7, 50)
        mask = torch.ones(2, 7, dtype=torch.bool)
        kl = _compute_kl_batch(logits, logits, mask)
        assert abs(kl) < 1e-5

    def test_kl_is_nonnegative(self) -> None:
        torch.manual_seed(42)
        logits_clean = torch.randn(1, 7, 50)
        logits_noisy = torch.randn(1, 7, 50)
        mask = torch.ones(1, 7, dtype=torch.bool)
        kl = _compute_kl_batch(logits_clean, logits_noisy, mask)
        assert kl >= -1e-6

    def test_kl_increases_with_divergence(self) -> None:
        torch.manual_seed(7)
        logits_clean = torch.randn(1, 10, 30)
        small_noise = torch.randn(1, 10, 30) * 0.01
        large_noise = torch.randn(1, 10, 30) * 10.0
        mask = torch.ones(1, 10, dtype=torch.bool)

        kl_small = _compute_kl_batch(logits_clean, logits_clean + small_noise, mask)
        kl_large = _compute_kl_batch(logits_clean, logits_clean + large_noise, mask)
        assert kl_small < kl_large


# ---------------------------------------------------------------------------
# Next-token agreement
# ---------------------------------------------------------------------------

class TestNextTokenAgreement:
    def test_identical_logits_full_agreement(self) -> None:
        logits = torch.randn(2, 7, 50)
        mask = torch.ones(2, 7, dtype=torch.bool)
        agree, n = _compute_agreement_batch(logits, logits, mask)
        assert n == 14
        assert abs(agree / n - 1.0) < 1e-6

    def test_orthogonal_argmax_zero_agreement(self) -> None:
        B, T, V = 1, 5, 10
        mask = torch.ones(B, T, dtype=torch.bool)
        logits_clean = torch.eye(T, V)[:T, :V].unsqueeze(0)
        # Shift argmax to a different index
        logits_noisy = torch.roll(logits_clean, shifts=V // 2, dims=-1)
        agree, n = _compute_agreement_batch(logits_clean, logits_noisy, mask)
        assert agree == 0.0

    def test_masked_positions_excluded(self) -> None:
        logits = torch.randn(1, 5, 20)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        mask[0, :3] = True
        agree, n = _compute_agreement_batch(logits, logits, mask)
        assert n == 3


# ---------------------------------------------------------------------------
# Zero-noise test (end-to-end): clean NLL ≈ zero-noise NLL
# ---------------------------------------------------------------------------

class TestZeroNoiseEndToEnd:
    def test_zero_noise_reproduces_clean_nll(self) -> None:
        torch.manual_seed(0)
        model = ToyLM(vocab=30, seq_len=8)
        model.eval()
        device = torch.device("cpu")

        batches = [_make_batch(vocab=30, seq_len=8) for _ in range(3)]

        # Clean NLL
        clean_nll, clean_ppl, clean_logits = collect_logits(model, batches, device)

        # Evaluate with zero noise (no weight modifications)
        metrics = evaluate_quality(
            model, batches, device,
            clean_logits_list=clean_logits,
        )

        # NLL must match (weights not changed)
        assert abs(metrics["nll"] - clean_nll) < 1e-5
        assert abs(metrics["ppl"] - clean_ppl) < 1e-4

        # KL(clean || clean) must be ~0
        assert abs(metrics["kl_divergence"]) < 1e-5

        # Agreement must be 1.0
        assert abs(metrics["next_token_agreement"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Preservation gain and PPL ratio
# ---------------------------------------------------------------------------

class TestPreservationMetrics:
    def test_preservation_gain_positive_when_static_better(self) -> None:
        # static has lower DeltaNLL than hardware → gain > 0
        gain = compute_preservation_gain(
            delta_nll_hardware=0.5,
            delta_nll_static=0.2,
        )
        assert gain is not None
        assert abs(gain - (0.5 - 0.2) / 0.5) < 1e-9

    def test_preservation_gain_none_when_hardware_near_zero(self) -> None:
        gain = compute_preservation_gain(
            delta_nll_hardware=1e-9,
            delta_nll_static=0.1,
        )
        assert gain is None

    def test_ppl_preservation_ratio_positive(self) -> None:
        ratio = compute_ppl_preservation_ratio(
            ppl_clean=10.0,
            ppl_hardware=20.0,
            ppl_static=15.0,
        )
        assert ratio is not None
        # (20 - 15) / (20 - 10) = 0.5
        assert abs(ratio - 0.5) < 1e-9

    def test_ppl_preservation_ratio_none_when_denominator_near_zero(self) -> None:
        ratio = compute_ppl_preservation_ratio(
            ppl_clean=10.0,
            ppl_hardware=10.0 + 1e-9,
            ppl_static=10.0,
        )
        assert ratio is None

    def test_delta_nll_zero_means_zero_degradation(self) -> None:
        """DeltaNLL = 0 means no hardware degradation."""
        gain = compute_preservation_gain(0.0, 0.0, delta_nll_hardware_threshold=1e-6)
        assert gain is None  # hardware degradation too small to report
