#!/usr/bin/env python3
"""Cheap proxy sensitivity scores: diagonal Fisher and magnitude baselines.

The measured Phase-1 sensitivity requires hundreds of noisy dataset passes.
This script computes two standard cheap alternatives from the same calibration
data with a handful of gradient passes and no AIHWKit dependency:

- ``fisher_score``: second-order expected NLL increase under the deployment
  noise model, ``sigma_abs^2 x trace(diagonal empirical Fisher)`` with
  ``sigma_abs = reference_noise_std x programmed_range``.
- ``magnitude_score``: noise-to-signal energy ratio,
  ``sigma_abs^2 x parameter_count / ||W_clipped||_F^2``.

The output is a sidecar JSON keyed by projection ID; Phase 1.5 merges it into
candidate scoring and Phase 3 can use ``fisher_score`` as an alternative
placement-importance channel (``static_fisher``).
"""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import (
    file_sha256,
    git_commit,
    load_json,
    load_yaml,
    resolve_path,
    save_json,
)


def main(config_path: Path, phase1_path: Path | None = None) -> Path:
    # Heavy dependencies are imported lazily so --help stays dependency-light.
    import torch
    from torch import nn

    from src.analysis.pareto import spearman_correlation
    from src.common.dataset import build_causal_lm_batches
    from src.common.manual_weights import ManualAnalogSettings, prepare_projection_weight
    from src.common.model_loading import load_model_and_tokenizer
    from src.common.projections import (
        canonical_weight_bias,
        iter_gpt2_projections,
    )
    from src.common.tabular import write_csv

    config = load_yaml(config_path)
    proxy_cfg = config.get("profiling", {}).get("proxy", {})
    max_batches = proxy_cfg.get("max_batches")

    device = torch.device(str(config["model"]["device"]))
    model, tokenizer, model_source = load_model_and_tokenizer(config, device=device)
    settings = ManualAnalogSettings.from_config(config)
    settings.validate()

    batches, dataset_metadata = build_causal_lm_batches(config, tokenizer)
    if max_batches is not None:
        batches = batches[: int(max_batches)]
    if not batches:
        raise ValueError("Proxy sensitivity requires at least one calibration batch.")

    include_lm_head = bool(config["profiling"].get("include_lm_head", True))
    handles = list(iter_gpt2_projections(model, include_lm_head=include_lm_head))

    # GPT-2 ties lm_head.weight to the token embedding. Untie onto a cloned
    # Parameter for the gradient pass so the LM-head matmul contribution is
    # separable from the embedding-lookup contribution.
    original_lm_head = None
    untied = None
    if include_lm_head:
        original_lm_head = model.lm_head
        untied = nn.Linear(
            original_lm_head.in_features,
            original_lm_head.out_features,
            bias=original_lm_head.bias is not None,
            device=device,
        )
        with torch.no_grad():
            untied.weight.copy_(original_lm_head.weight)
            if original_lm_head.bias is not None:
                untied.bias.copy_(original_lm_head.bias)
        model.lm_head = untied

    parameters = {
        handle.projection_id: (
            untied.weight
            if handle.projection_id == "lm_head" and untied is not None
            else handle.module.weight
        )
        for handle in handles
    }
    accumulators = {
        projection_id: torch.zeros_like(parameter, dtype=torch.float64)
        for projection_id, parameter in parameters.items()
    }

    # Token-weighted empirical diagonal Fisher: one backward per batch through
    # the batch-mean NLL, squared gradients weighted by predicted tokens.
    total_weight = 0.0
    for index, batch in enumerate(batches):
        moved = {key: value.to(device) for key, value in batch.items()}
        weight = float((moved["labels"][..., 1:] != -100).sum().item())
        if weight <= 0:
            continue
        model.zero_grad(set_to_none=True)
        loss = model(**moved).loss
        loss.backward()
        for projection_id, parameter in parameters.items():
            if parameter.grad is not None:
                accumulators[projection_id] += (
                    parameter.grad.detach().double() ** 2
                ) * weight
        total_weight += weight
        print(f"proxy batch {index + 1}/{len(batches)}")
    model.zero_grad(set_to_none=True)
    if total_weight <= 0:
        raise ValueError("No predicted tokens in the calibration batches.")

    if original_lm_head is not None:
        model.lm_head = original_lm_head

    rows: list[dict[str, Any]] = []
    for handle in handles:
        # handle.module is always the original module (handles were collected
        # before the lm_head untie; the clone's weights are identical).
        canonical, _ = canonical_weight_bias(handle.module)
        prepared = prepare_projection_weight(canonical, settings)
        programmed_range = prepared.preprocessing.programmed_range
        sigma_abs = settings.reference_noise_std * programmed_range
        fisher_trace = float(accumulators[handle.projection_id].sum().item()) / total_weight
        clipped_energy = float(
            (prepared.clipped_weight.double() ** 2).sum().item()
        )
        rows.append(
            {
                "projection_id": handle.projection_id,
                "role": handle.role,
                "parameter_count": handle.parameter_count,
                "macs_per_token": handle.macs_per_token,
                "tied_to_embedding": handle.tied_to_embedding,
                "programmed_range": programmed_range,
                "sigma_abs": sigma_abs,
                "fisher_trace": fisher_trace,
                "fisher_score": (sigma_abs**2) * fisher_trace,
                "magnitude_score": (sigma_abs**2)
                * handle.parameter_count
                / max(clipped_energy, 1e-30),
            }
        )

    rank_correlation: dict[str, Any] = {}
    if phase1_path is not None:
        profile = load_json(phase1_path)
        measured = {
            str(row["projection_id"]): float(row["sensitivity_score_for_mapping"])
            for row in profile["projections"]
        }
        shared = [row for row in rows if row["projection_id"] in measured]
        measured_values = [measured[row["projection_id"]] for row in shared]
        for proxy_name in ("fisher_score", "magnitude_score"):
            proxy_values = [float(row[proxy_name]) for row in shared]
            rank_correlation[f"{proxy_name}_vs_measured_spearman"] = (
                spearman_correlation(proxy_values, measured_values)
            )
        rank_correlation["n_projections"] = len(shared)
        rank_correlation["phase1_path"] = str(phase1_path)

    output_root = resolve_path(config["phase1"]["output_root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_root / f"proxy_sensitivity_{timestamp}.json"
    save_json(
        output_path,
        {
            "metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "repository_commit": git_commit(REPO_ROOT),
                "config_path": str(config_path.resolve()),
                "config_sha256": file_sha256(config_path),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "model_source": model_source,
                "dataset": dataset_metadata,
                "batches_used": len(batches),
                "reference_noise_std": settings.reference_noise_std,
                "fisher_estimator": "token_weighted_empirical_diagonal_fisher",
            },
            "projections": rows,
            "rank_correlation": rank_correlation,
        },
    )
    if rank_correlation:
        write_csv(
            output_root / "proxy_rank_correlation.csv",
            [
                {
                    "proxy": proxy_name,
                    "spearman_vs_measured": rank_correlation[
                        f"{proxy_name}_vs_measured_spearman"
                    ],
                    "n_projections": rank_correlation["n_projections"],
                }
                for proxy_name in ("fisher_score", "magnitude_score")
            ],
        )
    print(f"Proxy sensitivity saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml",
    )
    parser.add_argument(
        "--phase1",
        type=Path,
        help="Optional measured Phase-1 profile for rank-correlation reporting.",
    )
    args = parser.parse_args()
    main(args.config, args.phase1)
