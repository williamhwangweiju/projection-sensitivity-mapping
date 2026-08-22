#!/usr/bin/env python3
"""Leave-one-out check of the Phase-1 sensitivity profile (all 49 projections analog).

Measures s_loo(p) = E[NLL(all analog, noisy) - NLL(all analog, noisy, p restored)]
with exactly the Phase-1 noise fields, and reports the rank agreement (Spearman,
Kendall, top-k overlap) with the isolated Phase-1 profile.  Cost is comparable to
Phase 1 itself (~10 + 49 x 10 forward passes over the calibration split).

Usage (Colab, after Phase 1):
  python experiments/phase1_sensitivity/run_leave_one_out.py \
      --config configs/full_pipeline/gpt2_hybrid_3dcim.yaml \
      --phase1 <phase1 JSON> [--restore-mode nominal|digital] [--max-batches N] [--fresh]

Resumable: every forward-pass result is checkpointed to
<output dir>/leave_one_out_<mode>_partial.json as soon as it is measured; rerunning
the same command after a runtime disconnect skips what is already cached. The
checkpoint carries a signature (config/Phase-1 hashes, batch count, noise
settings, ...) and is set aside, not reused, when any of that differs. --fresh
discards it.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_json, load_yaml, resolve_path, save_json
from src.common.dataset import build_causal_lm_batches
from src.common.model_loading import load_model_and_tokenizer
from src.profilers.leave_one_out import LeaveOneOutProfiler


def main(config_path: Path, phase1_path: Path, restore_mode: str, max_batches: int | None,
         output_dir: Path | None, projection_ids: list[str] | None, fresh: bool = False,
         batch_stride: int = 1) -> Path:
    config = load_yaml(config_path)
    profile = load_json(phase1_path)
    phase1_rows = list(profile["projections"])
    model, tokenizer, model_source = load_model_and_tokenizer(config)
    batches, dataset_metadata = build_causal_lm_batches(config, tokenizer)
    total_windows = len(batches)
    if int(batch_stride) < 1:
        raise ValueError("--batch-stride must be >= 1")
    if int(batch_stride) > 1:
        # Evenly spaced subset of the calibration windows (every k-th window),
        # which samples the whole validation split instead of its first pages.
        batches = batches[:: int(batch_stride)]
    if max_batches is not None:
        batches = batches[: int(max_batches)]
    dataset_metadata = dict(dataset_metadata) | {
        "windows_available": total_windows,
        "batch_stride": int(batch_stride),
        "max_batches": None if max_batches is None else int(max_batches),
        "windows_used": len(batches),
    }
    print(f"Calibration windows: {len(batches)} of {total_windows} "
          f"(stride {int(batch_stride)}, max {max_batches})", flush=True)
    profiler = LeaveOneOutProfiler(model, config, phase1_rows, restore_mode=restore_mode)

    phase = config.get("phase1", {})
    out_dir = output_dir or (resolve_path(phase.get("output_root", "data/results/phase1_sensitivity")) / "leave_one_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-evaluation checkpoint (resumable across Colab disconnects). The
    # signature pins everything that changes the measured numbers, so a
    # checkpoint from a different run (e.g. a --max-batches smoke test) is
    # set aside rather than reused.
    checkpoint_path = out_dir / f"leave_one_out_{restore_mode}_partial.json"
    checkpoint_signature = {
        "restore_mode": restore_mode,
        "config_sha256": file_sha256(config_path),
        "phase1_sha256": file_sha256(phase1_path),
        "model_checkpoint": str(model_source.get("loaded_from") or model_source.get("checkpoint") or ""),
        "batches_used": len(batches),
        "batch_stride": int(batch_stride),
        "num_seeds": profiler.num_seeds,
        "seed_stride": profiler.seed_stride,
        "antithetic": profiler.antithetic,
        "base_seed": profiler.base_seed,
        "reference_noise_std": float(profiler.settings.reference_noise_std),
        "projection_ids": sorted(projection_ids) if projection_ids else None,
    }
    if fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"--fresh: discarded {checkpoint_path}")
    print(f"Checkpoint (resumable): {checkpoint_path}")
    result = profiler.run(batches, projection_ids=projection_ids,
                          checkpoint_path=checkpoint_path,
                          checkpoint_signature=checkpoint_signature)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload: dict[str, Any] = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "phase1_path": str(phase1_path.resolve()),
            "phase1_sha256": file_sha256(phase1_path),
            "model": model_source,
            "dataset": dataset_metadata,
            "batches_used": len(batches),
            "device": str(config["model"]["device"]),
            "torch": torch.__version__,
            "analog_configuration": profiler.hybrid.metadata() | {
                "reference_noise_std": profiler.settings.reference_noise_std,
                "num_seeds": profiler.num_seeds,
                "antithetic": profiler.antithetic,
                "restore_mode": restore_mode,
            },
        },
        **result,
    }
    json_path = out_dir / f"leave_one_out_{restore_mode}_{stamp}.json"
    save_json(json_path, payload)
    csv_path = out_dir / f"leave_one_out_{restore_mode}_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["projection_id", "role", "block_index", "sensitivity_isolated",
                  "sensitivity_loo_mean", "sensitivity_loo_std", "sensitivity_loo_sem"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["projections"]:
            writer.writerow({k: row[k] for k in fields})
    print(f"Leave-one-out profile saved to: {json_path}")
    print("Rank agreement vs isolated Phase-1 profile:", result["rank_agreement"])
    subset = "" if result["profiled_projection_count"] == result["analog_projection_count"] else \
        f" (over the {result['profiled_projection_count']}-projection subset only)"
    print(f"sum isolated = {result['sum_isolated_sensitivity']:.4f}  sum LOO = {result['sum_loo_sensitivity']:.4f}{subset};  "
          f"measured all-analog noise penalty = {result['total_noise_penalty_all_analog']:.4f}")
    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--restore-mode", choices=["nominal", "digital"], default="nominal")
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke-test limit on calibration batches.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--projection-ids", nargs="*", default=None, help="Optional subset (smoke tests).")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard the resumable checkpoint in the output dir and start over.")
    parser.add_argument("--batch-stride", type=int, default=1,
                        help="Use every k-th calibration window (evenly spaced subset); "
                             "k=5 cuts the ~15 h full run to ~3 h on a T4.")
    args = parser.parse_args()
    main(args.config, args.phase1, args.restore_mode, args.max_batches, args.output_dir,
         args.projection_ids, fresh=args.fresh, batch_stride=args.batch_stride)
