"""Reusable hybrid quality-evaluation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from src.common.metrics import evaluate_nll_ppl
from src.common.tabular import write_csv
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel
from src.evaluation.noise_materialization import apply_tile_noise



def evaluate_nominal_hybrid(
    hybrid: HybridAnalogModel,
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, int]:
    hybrid.restore_nominal_weights()
    result = evaluate_nll_ppl(hybrid.model, batches, device)
    hybrid.assert_nominal_restored()
    return result


def evaluate_noisy_placement(
    hybrid: HybridAnalogModel,
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    placement_rows: Iterable[Mapping[str, Any]],
    *,
    base_seed: int,
    realization: int,
    antithetic: bool = False,
) -> dict[str, float]:
    signs = (1.0, -1.0) if antithetic else (1.0,)
    nll_values: list[float] = []
    ppl_values: list[float] = []
    injected_rms: list[float] = []
    token_count = 0
    try:
        for sign in signs:
            hybrid.restore_nominal_weights()
            diagnostics = apply_tile_noise(
                hybrid,
                placement_rows,
                base_seed=base_seed,
                realization=realization,
                sign=sign,
            )
            nll, ppl, token_count = evaluate_nll_ppl(hybrid.model, batches, device)
            nll_values.append(nll)
            ppl_values.append(ppl)
            injected_rms.append(float(diagnostics["injected_noise_rms"]))
    finally:
        hybrid.restore_nominal_weights()
        hybrid.assert_nominal_restored()
    mean_nll = sum(nll_values) / len(nll_values)
    return {
        "nll": mean_nll,
        "ppl_from_mean_nll": float(torch.exp(torch.tensor(mean_nll)).item()),
        "ppl_mean": sum(ppl_values) / len(ppl_values),
        "predicted_tokens": float(token_count),
        "injected_noise_rms": sum(injected_rms) / len(injected_rms),
        "antithetic_evaluations": float(len(signs)),
    }
