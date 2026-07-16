#!/usr/bin/env python3
"""Run Phase 1: one projection analog at a time with manual clip/noise."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "src" / "profilers" / "aihwkit_profiler.py").is_file():
            return candidate
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_yaml, resolve_path, with_seed
from src.common.dataset import build_lm_batches
from src.profilers.aihwkit_profiler import AIHWKITSensitivityProfiler


def package_versions() -> dict[str, Any]:
    import aihwkit

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "aihwkit": getattr(aihwkit, "__version__", None),
    }


def main(config_path: Path, seed: int | None = None) -> Path:
    config = load_yaml(config_path)
    if seed is not None:
        config = with_seed(config, seed)
    model_name = str(config["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    batches, dataset_metadata = build_lm_batches(config, tokenizer)
    profiler = AIHWKITSensitivityProfiler(model, config)
    results = profiler.profile_all(batches)

    phase1 = config["phase1"]
    output_root = resolve_path(REPO_ROOT, phase1["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = str(phase1.get("results_filename_prefix", "gpt2_manual_noise"))
    output_path = output_root / f"{prefix}_{timestamp}.json"

    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "manual_clip_manual_gaussian_noise_aihwkit_nonperfect_forward",
            "software_versions": package_versions(),
            "model": {
                "name": model.config._name_or_path,
                "n_layer": model.config.n_layer,
                "n_embd": model.config.n_embd,
                "vocab_size": model.config.vocab_size,
            },
            "dataset": dataset_metadata,
            "analog_configuration": profiler.analog_configuration(),
        },
        "requested_config": config,
        "baseline": {
            "digital_nll": results["digital_nll"],
            "digital_ppl": results["digital_perplexity"],
            "token_count": results["token_count"],
        },
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)

    print("\nPHASE 1 COMPLETE")
    print(f"Digital NLL/PPL: {results['digital_nll']:.8f} / {results['digital_perplexity']:.6f}")
    print(f"Profiled projections: {len(results['projections'])}")
    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "full_pipeline" / "gpt2_3dcim.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.seed)
